import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointpillars.model.anchors import Anchors, anchor_target, anchors2bboxes
from pointpillars.model.pointpillars import Backbone, Neck, Head
from pointpillars.ops import Voxelization, nms_cuda
from pointpillars.utils import limit_period

# batch_size=1 variants of PillarLayer/PillarEncoder for ONNX/TensorRT export:
# the multi-batch scatter in pointpillars.py uses advanced indexing
# (batched_canvas[b, :, y, x] = ...), which isn't exportable, so these use a
# flatten-index scatter instead and drop the batch-index column from coors.


class PillarLayerExport(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for pts in batched_pts:
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)
        coors = torch.cat(coors, dim=0)
        return pillars, coors, npoints_per_pillar


class PillarEncoderExport(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = math.ceil((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = math.ceil((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + ... + pb, 3), no batch-index column (batch_size=1 only)
        npoints_per_pillar: (p1 + ... + pb, )
        return: (1, out_channel, y_l, x_l)
        '''
        device = pillars.device
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]

        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 0:1] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 1:2] * self.vy + self.y_offset)

        features = torch.cat([pillars, offset_pt_center], dim=-1)
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center

        voxel_ids = torch.arange(0, pillars.size(1)).to(device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]

        # Note: only batch_size=1 is supported for TensorRT
        canvas = torch.zeros((self.x_l * self.y_l, self.out_channel), dtype=torch.float32, device=device)
        coors_flat = coors_batch[:, 1] * self.x_l + coors_batch[:, 0]
        canvas[coors_flat] = pooling_features
        canvas = canvas.view(self.y_l, self.x_l, self.out_channel)
        canvas = canvas.permute(2, 0, 1).contiguous()
        return canvas.unsqueeze(0)


class PointPillarsPre(nn.Module):
    '''batched_pts -> pillars/coors/npoints. Separate from PointPillarsCore because voxelization isn't ONNX/TensorRT-exportable.'''
    def __init__(self,
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.pillar_layer = PillarLayerExport(voxel_size=voxel_size,
                                              point_cloud_range=point_cloud_range,
                                              max_num_points=max_num_points,
                                              max_voxels=max_voxels)

    def forward(self, batched_pts):
        return self.pillar_layer(batched_pts)


class PointPillarsCore(nn.Module):
    '''The ONNX-exportable section: encoder -> backbone -> neck -> head -> anchor decode. Only batch_size=1 is supported.'''
    def __init__(self,
                 nclasses=3,
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 prefix='avikus'):
        super().__init__()
        self.nclasses = nclasses
        self.pillar_encoder = PillarEncoderExport(voxel_size=voxel_size,
                                                  point_cloud_range=point_cloud_range,
                                                  in_channel=7,
                                                  out_channel=48)
        self.backbone = Backbone(in_channel=48,
                                 out_channels=[64, 128, 192],
                                 layer_nums=[3, 4, 4])
        self.neck = Neck(in_channels=[64, 128, 192],
                         upsample_strides=[1, 2, 4],
                         out_channels=[96, 96, 96])
        self.head = Head(in_channel=288, n_anchors=2 * nclasses, n_classes=nclasses)

        ranges = [point_cloud_range for _ in range(nclasses)]
        sizes = [[3.37, 1.51, 1.32],
                [12.0, 4.14, 4.84],  # label[l, w, h] -> anchor_size[w, l, h]
                [27.53, 7.44, 9.81],
                [1.14, 2.02, 4.37],
                [11.79, 5.40, 13.13],
                [0.74, 0.72, 3.88],
                [3.02, 1.41, 1.00],
                [55.20, 12.06, 16.11],
                [18.59, 18.02, 14.01],
                [0.42, 0.45, 0.38]]
        rotations = [0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rotations)

        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35} for _ in range(nclasses)
        ]

        self.nms_pre = 100

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216)
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: (k, 7 + nclasses + 1) raw predictions (score filter/NMS applied separately by PointPillarsPos)
        '''
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds].float()
        anchors = anchors[inds]

        bbox_pred = anchors2bboxes(anchors, bbox_pred)
        return torch.cat([bbox_pred, bbox_cls_pred, bbox_dir_cls_pred[:, None]], 1)

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                       bbox_pred=bbox_pred[i],
                                                       bbox_dir_cls_pred=bbox_dir_cls_pred[i],
                                                       anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, pillars, coors_batch, npoints_per_pillar, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        xs = self.backbone(pillar_features)
        x = self.neck(xs)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batch_size = pillar_features.shape[0]
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors,
                                               batched_gt_bboxes=batched_gt_bboxes,
                                               batched_gt_labels=batched_gt_labels,
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode in ('val', 'test'):
            return self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred,
                                             bbox_pred=bbox_pred,
                                             bbox_dir_cls_pred=bbox_dir_cls_pred,
                                             batched_anchors=batched_anchors)
        else:
            raise ValueError


class PointPillarsPos(nn.Module):
    '''Score filter + NMS applied to PointPillarsCore's raw output, run outside the ONNX/TensorRT graph.'''
    def __init__(self, nclasses=3):
        super().__init__()
        self.nclasses = nclasses
        self.nms_thr = 0.1
        self.score_thr = 0.1
        self.max_num = 50

    def nms_filter(self, bbox_pred, bbox_cls_pred, bbox_dir_cls_pred):
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]

            keep_inds = nms_cuda(boxes=cur_bbox_pred2d,
                                 scores=cur_bbox_cls_pred,
                                 thresh=self.nms_thr,
                                 pre_maxsize=None,
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred)
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        if len(ret_bboxes) == 0:
            return None
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        return {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }

    def forward(self, results):
        pos_results = []
        for result in results:
            bbox_pred, bbox_cls_pred, bbox_dir_cls_pred = result[:, :7], result[:, 7:7 + self.nclasses], result[:, 7 + self.nclasses]
            pos_result = self.nms_filter(bbox_pred, bbox_cls_pred, bbox_dir_cls_pred)
            if pos_result is not None:
                pos_results.append(pos_result)
        return pos_results
