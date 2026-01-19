import argparse
import os
import torch
from tqdm import tqdm
import pdb
import cv2
import yaml
import numpy as np

from pointpillars.utils import setup_seed, vis_pc, read_calib, keep_bbox_from_lidar_range, RunningMetrics, iou3d_fn_lidar, iou_bev_fn_lidar, PRAccumulator
from pointpillars.dataset import Avikus, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.utils.process import pillars_to_bev_rgb_with_bboxes
from pointpillars.ops.iou3d_module import boxes_iou_bev

import random

random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


CLASSES = Avikus.CLASSES
ID2NAME = {int(v): k for k, v in CLASSES.items()}
IOU_THRESHOLDS  = Avikus.IOU_THRESHOLDS 
SCORE_THRESHOLD = 0.1
VISUALIZE = False
POINT_CLOUD_RANGE = [5, -72., -10., 180., 72., 30.]

def iou_bev_vec(cur_np: np.ndarray, boxes_np: np.ndarray) -> np.ndarray:
    """
    cur_np: (7,)  [x,y,z,l,w,h,yaw]  (혹은 너 코드에서 w,l 순서면 맞춰줘)
    boxes_np:(N,7)
    return: (N,) IoU
    """
    if boxes_np.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    cur = torch.from_numpy(cur_np.astype(np.float32)).view(1, 7)
    boxes = torch.from_numpy(boxes_np.astype(np.float32))

    # (x,y,z,l,w,h,yaw) -> (x1,y1,x2,y2,yaw)
    def to_bev5(b):
        x, y = b[:, 0], b[:, 1]
        l, w = b[:, 3], b[:, 4]
        yaw = b[:, 6]
        x1 = x - l / 2
        y1 = y - w / 2
        x2 = x + l / 2
        y2 = y + w / 2
        return torch.stack([x1, y1, x2, y2, yaw], dim=1)

    cur5 = to_bev5(cur)      # (1,5)
    boxes5 = to_bev5(boxes)  # (N,5)

    ious = boxes_iou_bev(cur5, boxes5)  # (1,N)
    return ious.squeeze(0).numpy()

def draw_center_separator(img, color=(255, 255, 255), thickness=2):
    h, w, _ = img.shape
    x = w // 2
    cv2.line(
        img,
        (x, 0),
        (x, h),
        color,
        thickness,
        cv2.LINE_AA
    )

def draw_bottom_center_text(img, text, x_center, color=(255, 255, 255)):
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = int(x_center - tw / 2)
    y = h - 15  # 하단 여백

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def main(args):
    setup_seed()

    point_cloud_range=POINT_CLOUD_RANGE
    pcd_limit_range = np.array(point_cloud_range, dtype=np.float32)
    z_span = point_cloud_range[5] - point_cloud_range[2]
    voxel_size=[0.25, 0.25, z_span]

    train_dataset = Avikus(data_root=args.data_root,
                        split='train', point_cloud_range=point_cloud_range)
    val_dataset = Avikus(data_root=args.data_root,
                        split='val', point_cloud_range=point_cloud_range)

    val_dataloader = get_dataloader(dataset=train_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    num_cls = len(CLASSES)
    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=num_cls, point_cloud_range=point_cloud_range, voxel_size=voxel_size, prefix='avikus').cuda()
    else:
        pointpillars = PointPillars(nclasses=num_cls, point_cloud_range=point_cloud_range, voxel_size = voxel_size, prefix='avikus')

    # load pretrained weight 
    print(f'weight loaded from {args.weight}..')
    checkpoint = torch.load(args.weight)
    model_dict = pointpillars.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                pretrained_dict[k] = v
            else:
                pretrained_dict[k] = model_dict[k]
                pretrained_dict[k][:v.shape[0]] = v

    for k, v in pretrained_dict.items():
        if torch.isnan(v).any():
            print(f'k {k} and v {v}')

    model_dict.update(pretrained_dict)
    pointpillars.load_state_dict(model_dict)

    CLASS_LIST = [ID2NAME[i] for i in sorted(CLASSES.values())]
    acc3d = PRAccumulator(CLASS_LIST, ID2NAME, iou3d_fn_lidar, IOU_THRESHOLDS)
    accbev = PRAccumulator(CLASS_LIST, ID2NAME, iou_bev_fn_lidar, IOU_THRESHOLDS)

    data_name = [dir for dir in os.listdir(args.data_root) \
                if os.path.isdir(os.path.join(args.data_root, dir)) and \
                os.path.isdir(os.path.join(args.data_root, dir, "label"))][0]

    bev_dir = os.path.join(args.out_dir, data_name, 'bev')
    bev_points_dir = os.path.join(bev_dir, 'points')
    bev_GT_pred = os.path.join(bev_dir, 'gt_pred')

    bev_list = [bev_dir, bev_points_dir, bev_GT_pred]
    for bev_path in bev_list:
        os.makedirs(bev_path, exist_ok=True)

    pointpillars.eval()
    with torch.no_grad():
        val_step = 0
        for _, data_dict in enumerate(tqdm(val_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_pcd_path = data_dict['batched_pcd_info']

            pillars, coors_batch, npoints_per_pillar, results = pointpillars(batched_pts=batched_pts, 
                            batched_gt_bboxes=batched_gt_bboxes, 
                            batched_gt_labels=batched_labels, mode='val')

            batch_size = len(results)
            for i in range(batch_size):
                res_filter = keep_bbox_from_lidar_range(results[i], pcd_limit_range)
                lidar_bboxes = res_filter['lidar_bboxes']
                labels, scores = res_filter['labels'], res_filter['scores']
                pcd_ts = os.path.basename(batched_pcd_path[i]).split('.')[0]

                acc3d.add_frame(lidar_bboxes, scores, labels, batched_gt_bboxes[i].cpu().numpy(), batched_labels[i].cpu().numpy(), collect_errors=True)
                accbev.add_frame(lidar_bboxes, scores, labels, batched_gt_bboxes[i].cpu().numpy(), batched_labels[i].cpu().numpy(), collect_errors=False)

                mask = scores >= SCORE_THRESHOLD
                lidar_bboxes = lidar_bboxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                if len(scores) == 0:
                    lidar_bboxes = np.zeros((0, 7), dtype=np.float32)
                    scores = np.zeros((0,1), dtype=np.float32)
                    labels = np.zeros((0,1), dtype=np.int64)
                else:
                    order = np.argsort(-scores)
                    boxes = lidar_bboxes[order]
                    scores = scores[order]
                    labels = labels[order]

                    keep = []
                    iou = iou_bev_fn_lidar(boxes, boxes)
                    np.fill_diagonal(iou, 0.0)

                    suppressed = np.zeros(boxes.shape[0], dtype=bool)
                    keep = []

                    for j in range(boxes.shape[0]):
                        if suppressed[j]:
                            continue
                        keep.append(j)
                        suppressed |= (iou[j] >= 0.3)
                        suppressed[j] = False

                    keep = np.array(keep, dtype=np.int64)
                    lidar_bboxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                bev_GT = pillars_to_bev_rgb_with_bboxes(
                    pillars,
                    coors_batch,
                    npoints_per_pillar,
                    point_cloud_range,
                    voxel_size,
                    bboxes_lidar=batched_gt_bboxes[i].cpu().numpy(),
                    labels=batched_labels[i].cpu().numpy(),
                    scores=scores,
                    batch_idx=i,
                    gt=True
                )

                # bev pred
                bev_pred = pillars_to_bev_rgb_with_bboxes(
                    pillars,
                    coors_batch,
                    npoints_per_pillar,
                    point_cloud_range,
                    voxel_size,
                    bboxes_lidar=lidar_bboxes,
                    labels=labels,
                    scores=scores,
                    batch_idx=i,
                )
                # bev points
                bev_points = pillars_to_bev_rgb_with_bboxes(
                    pillars,
                    coors_batch,
                    npoints_per_pillar,
                    point_cloud_range,
                    voxel_size,
                    bboxes_lidar=np.zeros((batched_gt_bboxes[i].shape[0], 7)),
                    labels=np.zeros((batched_gt_bboxes[i].shape[0])),
                    scores=np.zeros((batched_gt_bboxes[i].shape[0])),
                    batch_idx=i,
                )
                cv2.imwrite(f"{bev_points_dir}/{pcd_ts}.png", bev_points)

                # vis_pc(batched_pts[i].cpu().numpy(), bboxes=batched_gt_bboxes[i].cpu().numpy(), labels=batched_labels[i].cpu().numpy())
                # vis_pc(batched_pts[i].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)

                # 1. GT -> label만 뜨도록
                # 2. pred -> 가장 score 높은 bbox만 뜨도록
                # 3. 

                # 1) 가운데 구분선
                bev_side_by_side = np.hstack([bev_GT, bev_pred])
                half_w  = bev_GT.shape[1]
                draw_center_separator(bev_side_by_side, color=(200, 200, 200), thickness=2)

                # 2) 하단 텍스트
                draw_bottom_center_text(bev_side_by_side, "[GT]",   x_center=half_w // 2, color=(0, 255, 0))
                draw_bottom_center_text(bev_side_by_side, "[PRED]", x_center=half_w + half_w // 2, color=(0, 255, 255))
                cv2.imwrite(f"{bev_GT_pred}/{pcd_ts}.png", bev_side_by_side)

            val_step += 1

            print('GT count:', sum(acc3d.n_gt.values()))
            print('Pred scored count:', sum(len(v) for v in acc3d.scores.values()))
            print('matched_errs so far:', len(acc3d.errs))

    per_class_ap3d, matched_errs = acc3d.compute_map()
    per_class_apbev, _ = accbev.compute_map()

    metrics = RunningMetrics(class_names=CLASS_LIST)
    metrics.update_from_batch({"ap3d" : per_class_ap3d,
                                "apbev" : per_class_apbev, 
                                "matched_errors" : matched_errs})
    summary = metrics.compute()
    print(f"[VAL] score = {summary['score']} | mAP3D = {summary['mAP_3D']} | mAPBEV = {summary['mAP_BEV']} | ATE = {summary['ATE']} | AOE_deg = {summary['AOE_deg']} | ASE = {summary['ASE']}")

    metric_dir = os.path.join(args.out_dir, data_name, 'metric')
    os.makedirs(metric_dir, exist_ok=True)
    metric_path = os.path.join(metric_dir, "metric.txt")

    with open(metric_path, "w") as f:
        f.write(f"[metric values]\n")
        f.write(f"score={summary['score']}\n")
        f.write(f"mAP3D={summary['mAP_3D']}\n")
        f.write(f"mAPBEV={summary['mAP_BEV']}\n")
        f.write(f"ATE={summary['ATE']}\n")
        f.write(f"AOE_deg={summary['AOE_deg']}\n")
        f.write(f"ASE={summary['ASE']}\n")

        # 2) per-class AP3D
        f.write("\n[per_class_ap3d]\n")
        for cname in CLASS_LIST:   # CLASS_LIST 순서 고정
            ap = float(per_class_ap3d.get(cname, 0.0))
            f.write(f"{cname}={ap}\n")

        # 3) per-class APBEV
        f.write("\n[per_class_apbev]\n")
        for cname in CLASS_LIST:
            ap = float(per_class_apbev.get(cname, 0.0))
            f.write(f"{cname}={ap}\n")

        f.write("\n[GT per class]\n")
        for c in CLASS_LIST:
            f.write(f"{c}={acc3d.n_gt[c]}\n")

        f.write("\n[Pred count per class]\n")
        for c in CLASS_LIST:
            f.write(f"{c}={len(acc3d.scores[c])}\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', required=True, default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--imu', type=bool, default=False)
    parser.add_argument('--weight', type=str, default="pretrained/best.pth", help='pretrained weight')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)


# result_날짜_시간
# label이 있는 각 디렉토리들에 대해 수행
# 각 디렉토리 안에 summary 안의 각 값들 기록
# pred BEV 영상
# GT BEV 영상
# train dataset이 되겠구만.. 
# 얾.. 그러면 각 디렉토리에 대해 train / val / test pkl 만들어서 돌려야하는건데.. 