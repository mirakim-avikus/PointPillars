import argparse
import os
import torch
from tqdm import tqdm
import pdb
import cv2
import yaml
import numpy as np

from pointpillars.utils import setup_seed, vis_pc, keep_bbox_from_image_range, bbox3d2corners_camera, vis_img_3d, read_calib, keep_bbox_from_lidar_range, read_label, RunningMetrics, iou3d_fn_lidar, iou_bev_fn_lidar, PRAccumulator
from pointpillars.dataset import Avikus, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter

import random

random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

CLASSES = Avikus.CLASSES
ID2NAME = {int(v): k for k, v in CLASSES.items()}
IOU_THRESHOLDS  = Avikus.IOU_THRESHOLDS 
VISUALIZE = False

def find_closest_lidar(lidar_dir, data_name):
    lidar_list = sorted([lidar for lidar in os.listdir(lidar_dir) if lidar.endswith('.pcd')])
    lidar_ts_list = [int(f.split('.')[0]) for f in lidar_list]
    img_ts = int(data_name)

    if not lidar_ts_list:
        raise ValueError("LiDAR Directory is Empty.")

    if img_ts <= lidar_ts_list[0]:
        return str(lidar_ts_list[0])
    if img_ts >= lidar_ts_list[-1]:
        return str(lidar_ts_list[-1])

    closest_ts = min(lidar_ts_list, key=lambda x: abs(x - img_ts))
    return str(closest_ts)
    
def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)

def get_parameters(calib_path_yaml, calib_info):
    with open(calib_path_yaml, 'rb') as f:
        calib = yaml.safe_load(f)
    cam = calib['camera']
    K = np.array([
        [cam['fx'], cam['skew'], cam['cx']],
        [0, cam['fy'], cam['cy']],
        [0, 0, 1]
    ], dtype = np.float32)
    D = np.array([cam['k1'], cam['k2'], cam['k3'], cam['k4']], dtype=np.float32)

    rvec = np.array([calib['camera2lidar']['rvec_1'], calib['camera2lidar']['rvec_2'], calib['camera2lidar']['rvec_3']])
    tvec = np.array([calib['camera2lidar']['tvec_1'], calib['camera2lidar']['tvec_2'], calib['camera2lidar']['tvec_3']])

    R, _ = cv2.Rodrigues(rvec)
    tr_velo_to_cam = np.identity(4)
    tr_velo_to_cam[:3, :3] = R
    tr_velo_to_cam[:3, -1] = tvec

    r0_rect = calib_info['R0_rect'].astype(np.float32)
    P2 = calib_info['P2'].astype(np.float32)
    return tr_velo_to_cam, r0_rect, P2, K, D

def get_parameters(calib_path_yaml, calib_info):
    with open(calib_path_yaml, 'rb') as f:
        calib = yaml.safe_load(f)
    cam = calib['camera']
    K = np.array([
        [cam['fx'], cam['skew'], cam['cx']],
        [0, cam['fy'], cam['cy']],
        [0, 0, 1]
    ], dtype = np.float32)
    D = np.array([cam['k1'], cam['k2'], cam['k3'], cam['k4']], dtype=np.float32)

    rvec = np.array([calib['camera2lidar']['rvec_1'], calib['camera2lidar']['rvec_2'], calib['camera2lidar']['rvec_3']])
    tvec = np.array([calib['camera2lidar']['tvec_1'], calib['camera2lidar']['tvec_2'], calib['camera2lidar']['tvec_3']])

    R, _ = cv2.Rodrigues(rvec)
    tr_velo_to_cam = np.identity(4)
    lidar2avikus = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    tr_velo_to_cam[:3, :3] = R@lidar2avikus
    tr_velo_to_cam[:3, -1] = tvec

    r0_rect = calib_info['R0_rect'].astype(np.float32)
    P2 = calib_info['P2'].astype(np.float32)
    return tr_velo_to_cam, r0_rect, P2, K, D


def main(args):
    setup_seed()

    point_cloud_range=[4, -144., -10., 180., 144., 30.]
    pcd_limit_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size=[0.25, 0.25, 4]

    train_dataset = Avikus(data_root=args.data_root,
                        split='train', point_cloud_range=point_cloud_range)
    val_dataset = Avikus(data_root=args.data_root,
                        split='val', point_cloud_range=point_cloud_range)

    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    num_cls = sorted(CLASSES.values())[-1] + 1
    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=num_cls, point_cloud_range=point_cloud_range, voxel_size=voxel_size, prefix='avikus').cuda()
    else:
        pointpillars = PointPillars(nclasses=num_cls, point_cloud_range=point_cloud_range, voxel_size = voxel_size, prefix='avikus')
    loss_func = Loss()

    if args.pretrained:
        # load pretrained weight 
        print(f'weight loaded from {args.pretrained_weight}..')
        checkpoint = torch.load(args.pretrained_weight)
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
    else:
        print(f'training from scratch..')

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    for epoch in range(args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            optimizer.zero_grad()
            data_key = os.path.normpath(data_dict['batched_img_info'][0]['image_path']).split('/')[0]
            data_name = os.path.basename(os.path.normpath(data_dict['batched_img_info'][0]['image_path'])).split('.')[0]

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']

            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                pointpillars(batched_pts=batched_pts, 
                             mode='train',
                             batched_gt_bboxes=batched_gt_bboxes, 
                             batched_gt_labels=batched_labels)

            device = bbox_cls_pred.device
            feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
            anchors = pointpillars.anchors_generator.get_multi_anchors(feature_map_size)
            batch_size = len(batched_pts)
            batched_anchors = [anchors for _ in range(batch_size)]
            result_filter = pointpillars.get_predicted_bboxes(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors)[0]

            if result_filter == None:
                print(f'prediction is invalid in {data_name}.png')
                continue

            calib_info = read_calib(f"{os.path.normpath(args.data_root)}/{data_key}/calib_{data_key}.txt")
            calib_dir = os.path.join(*os.path.normpath(args.data_root).split('/'), data_key)
            new_yaml = os.path.join(calib_dir, "new_lidar.yaml")
            old_yaml = os.path.join(calib_dir, "lidar.yaml")
        
            # TODO: temporary fix — replace with permanent calibration loader
            calib_path_yaml = new_yaml if os.path.exists(new_yaml) else old_yaml
            tr_velo_to_cam, r0_rect, P2, K, D = get_parameters(calib_path_yaml, calib_info)

            parent_path = os.path.dirname(os.path.normpath(args.data_root))
            img_path = os.path.join(os.path.normpath(args.data_root), *parent_path.split('/'), data_dict['batched_img_info'][0]['image_path'])
            img = cv2.imread(img_path)
            image_shape = img.shape[:2]

            result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape, K=K, D=D, prefix='avikus')
            result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']
            if VISUALIZE:
                vis_pc(batched_pts[0].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)
            
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, num_cls)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < num_cls)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < num_cls).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = num_cls
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
            print(' | '.join(f'train loss {k}: {v:.4f}' for k, v in loss_dict.items()))
            loss = loss_dict['total_loss']
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        if epoch % 2 == 0:
            continue
        CLASS_LIST = [ID2NAME[i] for i in sorted(CLASSES.values())]
        acc3d = PRAccumulator(CLASS_LIST, ID2NAME, iou3d_fn_lidar, IOU_THRESHOLDS)
        accbev = PRAccumulator(CLASS_LIST, ID2NAME, iou_bev_fn_lidar, IOU_THRESHOLDS)

        pointpillars.eval()
        with torch.no_grad():
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

                results = pointpillars(batched_pts=batched_pts, 
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)

                batch_size = len(results)
                for i in range(batch_size):
                    data_name = os.path.normpath(data_dict['batched_img_info'][i]['image_path'])
                    if results[i] == None:
                        print(f'prediction is invalid in {data_name}')
                        continue

                # visualize image
                for i in range(batch_size):
                    data_key = os.path.normpath(data_dict['batched_img_info'][i]['image_path']).split('/')[0]
                    data_name = os.path.basename(os.path.normpath(data_dict['batched_img_info'][i]['image_path'])).split('.')[0]

                    parent_path = os.path.dirname(os.path.normpath(args.data_root))
                    img_path = os.path.join(args.data_root, *parent_path.split('/'), data_dict['batched_img_info'][i]['image_path'])

                    data_root = args.data_root
                    calib_info = read_calib(f"{os.path.normpath(data_root)}/{data_key}/calib_{data_key}.txt")
                    calib_dir = os.path.join(*os.path.normpath(data_root).split('/'), data_key)
                    new_yaml = os.path.join(calib_dir, "new_lidar.yaml")
                    old_yaml = os.path.join(calib_dir, "lidar.yaml")

                    # TODO: temporary fix — replace with permanent calibration loader
                    calib_path_yaml = new_yaml if os.path.exists(new_yaml) else old_yaml
                    tr_velo_to_cam, r0_rect, P2, K, D = get_parameters(calib_path_yaml, calib_info)

                    img = cv2.imread(img_path, 1)
                    image_shape = img.shape[:2]

                    res_filter = keep_bbox_from_image_range(results[i], tr_velo_to_cam, r0_rect, P2, image_shape, K=K, D=D, prefix='avikus')
                    res_filter = keep_bbox_from_lidar_range(res_filter, pcd_limit_range)
                    lidar_bboxes = res_filter['lidar_bboxes']
                    labels, scores = res_filter['labels'], res_filter['scores']

                    if VISUALIZE:
                        vis_pc(batched_pts[i].cpu().numpy(), bboxes=batched_gt_bboxes[i].cpu().numpy(), labels=batched_labels[i].cpu().numpy())
                        vis_pc(batched_pts[i].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)

                    acc3d.add_frame(lidar_bboxes, scores, labels, batched_gt_bboxes[i].cpu().numpy(), batched_labels[i].cpu().numpy(), collect_errors=True)
                    accbev.add_frame(lidar_bboxes, scores, labels, batched_gt_bboxes[i].cpu().numpy(), batched_labels[i].cpu().numpy(), collect_errors=False)

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
                
        writer.add_scalar("val/score", summary['score'], epoch)
        writer.add_scalar("val/mAP_3D", summary['mAP_3D'], epoch)
        writer.add_scalar("val/mAP_BEV", summary['mAP_BEV'], epoch)
        writer.add_scalar("val/ATE", summary['ATE'], epoch)
        writer.add_scalar("val/AOE_deg", summary['AOE_deg'], epoch)
        writer.add_scalar("val/ASE", summary['ASE'], epoch)
        writer.add_scalar("val/eval_loss", summary['eval_loss'], epoch)

        best_path = os.path.join(saved_ckpt_path, "best.pth")
        best_score_txt = os.path.join(saved_ckpt_path, "best_score.txt")
        prev_best = -1.0
        if os.path.exists(best_score_txt):
            with open(best_score_txt, "r") as f:
                prev_best = float(f.read().strip() or -1.0)
        
        if summary['score'] > prev_best:
            torch.save(pointpillars.state_dict(), best_path)
            with open(best_score_txt, "w") as f:
                f.write(str(summary["score"]))
            print(f"[VAL] New best! score={summary['score']} -> saved to {best_path}")
        else:
             print(f"[VAL] score={summary['score']:.4f} (best={prev_best:.4f})")
    
        metrics.reset()
        pointpillars.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', required=True, default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained weight or not')
    parser.add_argument('--pretrained_weight', type=str, default="pretrained/epoch_160.pth", help='pretrained weight')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
