import argparse
import os
import sys
import torch
from tqdm import tqdm
import pdb
import cv2
import yaml
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE, '../')))

from pointpillars.utils import setup_seed, vis_pc, read_calib, keep_bbox_from_lidar_range, RunningMetrics, iou3d_fn_lidar, iou_bev_fn_lidar, PRAccumulator
from pointpillars.dataset import Avikus, get_dataloader, POINT_CLOUD_RANGE
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
SCORE_THRESHOLD = [0.1, 0.2, 0.3]
VISUALIZE = False
BEV = True

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

    CLASS_LIST_m = MAPPED_CLASS_NAMES
    ID2NAME_FOR_ACC_m = MAPPED_ID2NAME
    acc3d_m = PRAccumulator(CLASS_LIST_m, ID2NAME_FOR_ACC_m, iou3d_fn_lidar, MAPPED_IOU_THRESHOLDS)
    accbev_m = PRAccumulator(CLASS_LIST_m, ID2NAME_FOR_ACC_m, iou_bev_fn_lidar, MAPPED_IOU_THRESHOLDS)

    data_name = [dir for dir in os.listdir(args.data_root) \
                if os.path.isdir(os.path.join(args.data_root, dir)) and \
                os.path.isdir(os.path.join(args.data_root, dir, "label"))][0]

    bev_dir = os.path.join(args.out_dir, data_name, 'bev')
    os.makedirs(bev_dir, exist_ok=True)

    for SCORE in SCORE_THRESHOLD:
        bev_GT_pred = os.path.join(bev_dir, f"{int(SCORE * 10):02d}", 'gt_pred')
        os.makedirs(bev_GT_pred, exist_ok=True)

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

                lidar_bboxes = res_filter['lidar_bboxes']
                labels, scores = res_filter['labels'], res_filter['scores']

                # --- remap pred/gt labels to 4 classes ---
                labels_m = remap_labels_to_4(labels)
                gt_labels_np = batched_labels[i].cpu().numpy()
                gt_labels_m = remap_labels_to_4(gt_labels_np)

                gt_boxes_np = batched_gt_bboxes[i].cpu().numpy()

                acc3d_m.add_frame(lidar_bboxes, scores, labels_m, gt_boxes_np, gt_labels_m, collect_errors=True)
                accbev_m.add_frame(lidar_bboxes, scores, labels_m, gt_boxes_np, gt_labels_m, collect_errors=False)

                if BEV:
                    # --- 추가: GT 라벨도 4클래스로 매핑 ---
                    gt_labels_np = batched_labels[i].cpu().numpy()
                    gt_labels_m = remap_labels_to_4(gt_labels_np) # 4클래스로 변환

                    bev_GT = pillars_to_bev_rgb_with_bboxes(
                        pillars,
                        coors_batch,
                        npoints_per_pillar,
                        point_cloud_range,
                        voxel_size,
                        bboxes_lidar=batched_gt_bboxes[i].cpu().numpy(),
                        labels=gt_labels_m,
                        scores=batched_labels[i].cpu().numpy(),
                        batch_idx=i,
                        gt=True
                    )

                    for SCORE in SCORE_THRESHOLD:
                        lidar_bboxes = res_filter['lidar_bboxes']
                        labels, scores = res_filter['labels'], res_filter['scores']
                        labels_m = remap_labels_to_4(labels)

                        mask = scores >= SCORE
                        lidar_bboxes = lidar_bboxes[mask]
                        scores = scores[mask]
                        labels_m = labels_m[mask]
                        labels = labels[mask]

                        if len(scores) == 0:
                            lidar_bboxes = np.zeros((0, 7), dtype=np.float32)
                            scores = np.zeros((0,1), dtype=np.float32)
                            labels = np.zeros((0,1), dtype=np.int64)
                            labels_m = np.zeros((0,1), dtype=np.int64)
                        else:
                            order = np.argsort(-scores)
                            boxes = lidar_bboxes[order]
                            scores = scores[order]
                            labels = labels[order]
                            labels_m = labels_m[order]

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
                            lidar_bboxes, scores, labels, labels_m = boxes[keep], scores[keep], labels[keep], labels_m[keep]

                        # bev pred
                        bev_pred = pillars_to_bev_rgb_with_bboxes(
                            pillars,
                            coors_batch,
                            npoints_per_pillar,
                            point_cloud_range,
                            voxel_size,
                            bboxes_lidar=lidar_bboxes,
                            labels=labels_m,
                            scores=scores,
                            batch_idx=i,
                        )

                        # 0) original image path
                        image_path = os.path.join(args.out_dir, data_dict['batched_img_info'][i]['image_path'])
                        original_img = cv2.imread(image_path)

                        # 1) BEV 이미지들의 가로 합치기 (기존 코드)
                        bev_side_by_side = np.hstack([bev_GT, bev_pred])
                        bev_h, bev_w = bev_side_by_side.shape[:2]

                        # 2) 원본 이미지를 BEV 높이에 맞춰 리사이징 (비율 유지)
                        img_h, img_w = original_img.shape[:2]
                        new_w = int(img_w * (bev_h / img_h))
                        original_img_resized = cv2.resize(original_img, (new_w, bev_h), interpolation=cv2.INTER_LANCZOS4)

                        # 3) 전체 이미지 합치기 (Original + BEV_side_by_side)
                        # 가로로 순서대로: [Original | bev_GT | bev_pred]
                        final_combined = np.hstack([original_img_resized, bev_side_by_side])

                        # 4) 구분선 긋기 (기존 구분선 함수 활용)
                        half_w = bev_GT.shape[1]
                        # 첫 번째 구분선: Original과 GT 사이
                        cv2.line(final_combined, (new_w, 0), (new_w, bev_h), (200, 200, 200), thickness=2)
                        # 두 번째 구분선: GT와 PRED 사이 (이미 bev_side_by_side에 그려져 있지 않다면 아래 좌표로 그림)
                        cv2.line(final_combined, (new_w + half_w, 0), (new_w + half_w, bev_h), (200, 200, 200), thickness=2)

                        # 5) 텍스트 추가 (위치 오프셋 반영)
                        # Original 이미지 텍스트
                        draw_bottom_center_text(final_combined, "[IMAGE]", x_center=new_w // 2, color=(255, 255, 255))
                        # GT 텍스트 (new_w 만큼 오른쪽으로 밀림)
                        draw_bottom_center_text(final_combined, "[GT]", x_center=new_w + (half_w // 2), color=(0, 255, 0))
                        # PRED 텍스트
                        draw_bottom_center_text(final_combined, "[PRED]", x_center=new_w + half_w + (half_w // 2), color=(0, 255, 255))

                        # 6) 저장
                        bev_GT_pred = os.path.join(bev_dir, f"{int(SCORE * 10):02d}", 'gt_pred')
                        cv2.imwrite(f"{bev_GT_pred}/{pcd_ts}.png", final_combined)

                        if VISUALIZE:
                            vis_pc(batched_pts[i].cpu().numpy(), bboxes=batched_gt_bboxes[i].cpu().numpy(), labels=batched_labels[i].cpu().numpy())
                            vis_pc(batched_pts[i].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)

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

    per_class_ap3d_m, matched_errs_m = acc3d_m.compute_map()
    per_class_apbev_m, _ = accbev_m.compute_map()

    metrics_m = RunningMetrics(class_names=CLASS_LIST_m)
    metrics_m.update_from_batch({"ap3d" : per_class_ap3d_m,
                                "apbev" : per_class_apbev_m, 
                                "matched_errors" : matched_errs_m})
    summary_m = metrics_m.compute()
    print(f"[VAL] score = {summary_m['score']} | mAP3D = {summary_m['mAP_3D']} | mAPBEV = {summary_m['mAP_BEV']} | ATE = {summary_m['ATE']} | AOE_deg = {summary_m['AOE_deg']} | ASE = {summary_m['ASE']}")


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

        f.write(f"\n====================\n")
        f.write(f"[metric values_m]\n")
        f.write(f"score={summary_m['score']}\n")
        f.write(f"mAP3D={summary_m['mAP_3D']}\n")
        f.write(f"mAPBEV={summary_m['mAP_BEV']}\n")
        f.write(f"ATE={summary_m['ATE']}\n")
        f.write(f"AOE_deg={summary_m['AOE_deg']}\n")
        f.write(f"ASE={summary_m['ASE']}\n")

        # 2) per-class AP3D
        f.write("\n[per_class_ap3d_m]\n")
        for cname in CLASS_LIST_m:   # CLASS_LIST 순서 고정
            ap = float(per_class_ap3d_m.get(cname, 0.0))
            f.write(f"{cname}={ap}\n")

        # 3) per-class APBEV
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
