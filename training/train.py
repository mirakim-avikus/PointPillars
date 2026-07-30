import argparse
import os
import torch
from tqdm import tqdm
import pdb
import cv2
import yaml
import numpy as np
import re

from pointpillars.utils import setup_seed, vis_pc, read_calib, keep_bbox_from_lidar_range, RunningMetrics, iou3d_fn_lidar, iou_bev_fn_lidar, PRAccumulator
from pointpillars.dataset import Avikus, get_dataloader, POINT_CLOUD_RANGE
from pointpillars.model import PointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter

import random

random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

CLASSES = Avikus.CLASSES
ID2NAME = {int(v): k for k, v in CLASSES.items()}
IOU_THRESHOLDS  = Avikus.IOU_THRESHOLDS
VISUALIZE = False

# ---- 10 -> 4 class mapping (orig_id:0~9 -> mapped_id:0~3) ----
CLASS_MAP_10_TO_4 = np.array([
    1,  # 0: jetski       -> 1 (jetski)
    0,  # 1: smallboat    -> 0 (motorboat)
    0,  # 2: mediumboat   -> 0
    2,  # 3: c-marker     -> 2 (cmarker group)
    0,  # 4: yacht        -> 0
    2,  # 5: pole         -> 2
    0,  # 6: dinghyboat   -> 0
    0,  # 7: bigboat      -> 0
    3,  # 8: bridgepillar -> 3
    2,  # 9: buoy         -> 2
], dtype=np.int64)

MAPPED_CLASS_NAMES = ["motorboat", "jetski", "cmarker", "bridgepillar"]
MAPPED_ID2NAME = {i: n for i, n in enumerate(MAPPED_CLASS_NAMES)}


def load_metric_from_section(path: str, section: str = "metric values_m", key: str = "mAPBEV") -> float:
    """
    로그 파일의 특정 섹션에서 지정된 key(예: mAPBEV)의 값을 읽어온다.
    없으면 -1.0 반환.
    """
    if not os.path.exists(path):
        return -1.0

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. 해당 섹션 블록([section] ... 다음 [ 전까지) 추출
    pattern = rf"\[{re.escape(section)}\]\s*(.*?)(?=\n\[|\Z)"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        return -1.0

    block = m.group(1)

    # 2. 섹션 블록 내부에서 key=value 패턴 찾기
    # key 뒤에 바로 = 이 오거나 공백이 있는 경우 모두 대응
    key_pattern = rf"^{re.escape(key)}\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
    m_metric = re.search(key_pattern, block, flags=re.MULTILINE)
    
    return float(m_metric.group(1)) if m_metric else -1.0

def remap_labels_to_4(labels: np.ndarray) -> np.ndarray:
    """
    labels: (N,) int array. supports:
      - already-mapped labels in [0..3]
      - original labels in [0..9]
    returns: (N,) labels in [0..3]
    """
    if labels.size == 0:
        return labels.astype(np.int64)

    labels = labels.astype(np.int64)

    # original 0~9 assumed
    if labels.min() < 0 or labels.max() >= len(CLASS_MAP_10_TO_4):
        raise ValueError(f"labels out of expected range. min={labels.min()} max={labels.max()}")

    return CLASS_MAP_10_TO_4[labels]

# original class names in your current setup (10-class)
ORIG_CLASS_LIST = [ID2NAME[i] for i in sorted(CLASSES.values())]  # e.g. ["jetski","smallboat",...]

GROUPS = {
    "motorboat":   ["smallboat", "mediumboat", "yacht", "dinghyboat", "bigboat"],
    "jetski":      ["jetski"],
    "cmarker":     ["c-marker", "pole", "buoy"],
    "bridgepillar":["bridgepillar"],
}

def build_mapped_iou_thresholds(IOU_THRESHOLDS):
    """
    returns thresholds keyed by mapped class name, same type style as your PRAccumulator expects.
    - if IOU_THRESHOLDS is dict[name->thr] => dict[mapped_name->thr]
    - if IOU_THRESHOLDS is list/tuple aligned with ORIG_CLASS_LIST order => dict[mapped_name->thr]
    """
    # case A) dict
    if isinstance(IOU_THRESHOLDS, dict):
        orig_thr = IOU_THRESHOLDS
        mapped_thr = {}
        for mname, members in GROUPS.items():
            vals = [orig_thr[mem] for mem in members if mem in orig_thr]
            if len(vals) == 0:
                raise KeyError(f"IOU_THRESHOLDS missing members for {mname}: {members}")
            mapped_thr[mname] = float(min(vals))
        return mapped_thr

    # case B) list/tuple aligned with ORIG_CLASS_LIST
    if isinstance(IOU_THRESHOLDS, (list, tuple, np.ndarray)):
        orig_thr = {c: float(IOU_THRESHOLDS[idx]) for idx, c in enumerate(ORIG_CLASS_LIST)}
        mapped_thr = {}
        for mname, members in GROUPS.items():
            vals = [orig_thr[mem] for mem in members if mem in orig_thr]
            if len(vals) == 0:
                raise KeyError(f"IOU_THRESHOLDS missing members for {mname}: {members}")
            mapped_thr[mname] = float(min(vals))
        return mapped_thr

    raise TypeError(f"Unsupported IOU_THRESHOLDS type: {type(IOU_THRESHOLDS)}")

MAPPED_IOU_THRESHOLDS = build_mapped_iou_thresholds(IOU_THRESHOLDS)

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

    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=train_dataset, # TODO replace train dataset with test dataset
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    num_cls = len(CLASSES)
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
                                                    max_lr=init_lr*3, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10,
                                                    final_div_factor=100,)

    saved_ckpt_path = os.path.join(args.saved_path, args.ckpt_name, 'weights')
    os.makedirs(saved_ckpt_path, exist_ok=True)
    saved_logs_path = os.path.join(args.saved_path, args.ckpt_name, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].cuda(non_blocking=True)
                    else:
                        # batched lists
                        new_list = []
                        for item in data_dict[key]:
                            if torch.is_tensor(item):
                                new_list.append(item.cuda(non_blocking=True))
                            else:
                                new_list.append(item)
                        data_dict[key] = new_list
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']

            with torch.cuda.amp.autocast():
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(
                        batched_pts=batched_pts,
                        mode='train',
                        batched_gt_bboxes=batched_gt_bboxes,
                        batched_gt_labels=batched_labels)

                batch_size = len(batched_pts)
                for i in range(batch_size):
                    data_key = os.path.normpath(data_dict['batched_img_info'][i]['image_path']).split('/')[0]

                    if VISUALIZE:
                        calib_info = read_calib(f"{os.path.normpath(args.data_root)}/{data_key}/calib_{data_key}.txt")
                        calib_dir = os.path.join(os.path.normpath(args.data_root), data_key)
                        calib_path_yaml = os.path.join(calib_dir, "lidar.yaml")
                        tr_velo_to_cam, r0_rect, P2, K, D = get_parameters(calib_path_yaml, calib_info)

                        device = bbox_cls_pred.device
                        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
                        anchors = pointpillars.anchors_generator.get_multi_anchors(feature_map_size)
                        batched_anchors = [anchors for _ in range(batch_size)]
                        result_filter = pointpillars.get_predicted_bboxes(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors, mode="train")   # visualize above threshold

                        res_filter = keep_bbox_from_lidar_range(result_filter[i], pcd_limit_range)
                        lidar_bboxes = res_filter['lidar_bboxes']
                        labels, scores = res_filter['labels'], res_filter['scores']


                        if result_filter[i]['lidar_bboxes'].shape[0] == 0:
                            data_name = os.path.basename(os.path.normpath(data_dict['batched_img_info'][i]['image_path'])).split('.')[0]
                            print(f'visualize pass! prediction above score threshold is empty in {data_name}.png')
                            continue

                        vis_pc(batched_pts[i].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)
                        vis_pc(batched_pts[i].cpu().numpy(), bboxes=batched_gt_bboxes[i].cpu().numpy(), labels=batched_labels[i].cpu().numpy())

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
                loss = loss_dict['total_loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        if epoch % args.val_freq_epoch != 0:
            continue
        CLASS_LIST = [ID2NAME[i] for i in sorted(CLASSES.values())]
        acc3d = PRAccumulator(CLASS_LIST, ID2NAME, iou3d_fn_lidar, IOU_THRESHOLDS)
        accbev = PRAccumulator(CLASS_LIST, ID2NAME, iou_bev_fn_lidar, IOU_THRESHOLDS)

        CLASS_LIST_m = MAPPED_CLASS_NAMES
        ID2NAME_FOR_ACC_m = MAPPED_ID2NAME
        acc3d_m = PRAccumulator(CLASS_LIST_m, ID2NAME_FOR_ACC_m, iou3d_fn_lidar, MAPPED_IOU_THRESHOLDS)
        accbev_m = PRAccumulator(CLASS_LIST_m, ID2NAME_FOR_ACC_m, iou_bev_fn_lidar, MAPPED_IOU_THRESHOLDS)

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
                    data_key = os.path.normpath(data_dict['batched_img_info'][i]['image_path']).split('/')[0]

                    data_root = args.data_root
                    calib_info = read_calib(f"{os.path.normpath(data_root)}/{data_key}/calib_{data_key}.txt")
                    calib_dir = os.path.join(os.path.normpath(data_root), data_key)
                    calib_path_yaml = os.path.join(calib_dir, "lidar.yaml")
                    tr_velo_to_cam, r0_rect, P2, K, D = get_parameters(calib_path_yaml, calib_info)

                    res_filter = keep_bbox_from_lidar_range(results[i], pcd_limit_range)
                    lidar_bboxes = res_filter['lidar_bboxes']
                    labels, scores = res_filter['labels'], res_filter['scores']

                    if VISUALIZE:
                        vis_pc(batched_pts[i].cpu().numpy(), bboxes=batched_gt_bboxes[i].cpu().numpy(), labels=batched_labels[i].cpu().numpy())
                        vis_pc(batched_pts[i].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)

                    acc3d.add_frame(lidar_bboxes, scores, labels, batched_gt_bboxes[i].cpu().numpy(), batched_labels[i].cpu().numpy(), collect_errors=True)
                    accbev.add_frame(lidar_bboxes, scores, labels, batched_gt_bboxes[i].cpu().numpy(), batched_labels[i].cpu().numpy(), collect_errors=False)

                    lidar_bboxes = res_filter['lidar_bboxes']
                    labels, scores = res_filter['labels'], res_filter['scores']

                    # --- remap pred/gt labels to 4 classes ---
                    labels_m = remap_labels_to_4(labels)
                    gt_labels_np = batched_labels[i].cpu().numpy()
                    gt_labels_m = remap_labels_to_4(gt_labels_np)

                    gt_boxes_np = batched_gt_bboxes[i].cpu().numpy()

                    acc3d_m.add_frame(lidar_bboxes, scores, labels_m, gt_boxes_np, gt_labels_m, collect_errors=True)
                    accbev_m.add_frame(lidar_bboxes, scores, labels_m, gt_boxes_np, gt_labels_m, collect_errors=False)

                val_step += 1

        per_class_ap3d, matched_errs = acc3d.compute_map()
        per_class_apbev, _ = accbev.compute_map()


        metrics = RunningMetrics(class_names=CLASS_LIST)
        metrics.update_from_batch({"ap3d" : per_class_ap3d,
                                    "apbev" : per_class_apbev, 
                                    "matched_errors" : matched_errs})
        summary = metrics.compute()
        print(f"[VAL] score = {summary['score']} | mAP3D = {summary['mAP_3D']} | mAPBEV = {summary['mAP_BEV']} | ATE = {summary['ATE']} | AOE_deg = {summary['AOE_deg']} | ASE = {summary['ASE']}")

        per_class_ap3d_m, matched_errs_m = acc3d_m.compute_map()
        per_class_apbev_m, _ = accbev_m.compute_map()

        metrics_m = RunningMetrics(class_names=CLASS_LIST_m)
        metrics_m.update_from_batch({"ap3d" : per_class_ap3d_m,
                                    "apbev" : per_class_apbev_m, 
                                    "matched_errors" : matched_errs_m})
        summary_m = metrics_m.compute()
        print(f"[VAL] score = {summary_m['score']} | mAP3D = {summary_m['mAP_3D']} | mAPBEV = {summary_m['mAP_BEV']} | ATE = {summary_m['ATE']} | AOE_deg = {summary_m['AOE_deg']} | ASE = {summary_m['ASE']}")


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
                prev_best = load_metric_from_section(best_score_txt, "metric values_m", "mAPBEV")
        
        if summary_m['mAP_BEV'] > prev_best:
            torch.save(pointpillars.state_dict(), best_path)
            with open(best_score_txt, "w") as f:
                f.write(f"[metric values]\n")
                # 1) all classes -> metric values
                f.write(f"score={summary['score']}\n")
                f.write(f"mAP3D={summary['mAP_3D']}\n")
                f.write(f"mAPBEV={summary['mAP_BEV']}\n")
                f.write(f"ATE={summary['ATE']}\n")
                f.write(f"AOE_deg={summary['AOE_deg']}\n")
                f.write(f"ASE={summary['ASE']}\n")

                # 2) all classes -> per-class AP3D
                f.write("\n[per_class_ap3d]\n")
                for cname in CLASS_LIST:   # CLASS_LIST 순서 고정
                    ap = float(per_class_ap3d.get(cname, 0.0))
                    f.write(f"{cname}={ap}\n")

                # 3) all classes -> per-class APBEV
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

                f.write(f"\n====================\n")
                # 1) mapped class -> metric values
                f.write(f"[metric values_m]\n")
                f.write(f"score={summary_m['score']}\n")
                f.write(f"mAP3D={summary_m['mAP_3D']}\n")
                f.write(f"mAPBEV={summary_m['mAP_BEV']}\n")
                f.write(f"ATE={summary_m['ATE']}\n")
                f.write(f"AOE_deg={summary_m['AOE_deg']}\n")
                f.write(f"ASE={summary_m['ASE']}\n")

                # 2) mapped class -> per-class AP3D
                f.write("\n[per_class_ap3d_m]\n")
                for cname in CLASS_LIST_m:   # CLASS_LIST 순서 고정
                    ap = float(per_class_ap3d_m.get(cname, 0.0))
                    f.write(f"{cname}={ap}\n")

                # 3) mapped class -> per-class APBEV
                f.write("\n[per_class_apbev_m]\n")
                for cname in CLASS_LIST_m:
                    ap = float(per_class_apbev_m.get(cname, 0.0))
                    f.write(f"{cname}={ap}\n")

                f.write("\n[GT per class_m]\n")
                for c in CLASS_LIST_m:
                    f.write(f"{c}={acc3d_m.n_gt[c]}\n")

                f.write("\n[Pred count per class_m]\n")
                for c in CLASS_LIST_m:
                    f.write(f"{c}={len(acc3d_m.scores[c])}\n")

            print(f"[VAL] New best! mAPBEV={summary_m['mAP_BEV']} -> saved to {best_path}")
        else:
            print(f"[VAL] mAPBEV={summary_m['mAP_BEV']:.4f} (best={prev_best:.4f})")
    
        metrics.reset()
        pointpillars.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', required=True, default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--init_lr', type=float, default=2e-4)
    parser.add_argument('--max_epoch', type=int, default=280)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--val_freq_epoch', type=int, default=10)
    parser.add_argument('--ckpt_name', type=str, required=True)
    parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained weight or not')
    parser.add_argument('--pretrained_weight', type=str, default="pretrained/epoch_160.pth", help='pretrained weight')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
