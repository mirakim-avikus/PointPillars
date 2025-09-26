from collections import defaultdict
import numpy as np
import math
import torch

# -------------------------------
# 회전 사각형 코너, 폴리곤 클리핑, 면적
# -------------------------------
def _rect_corners_xy(x, y, l, w, yaw):
    """Return (4,2) corners in XY-plane for a rotated box centered at (x,y)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = l * 0.5, w * 0.5
    # local corners (x forward, y left) → LiDAR XY 평면 기준
    pts = np.array([[ dx,  dy],
                    [ dx, -dy],
                    [-dx, -dy],
                    [-dx,  dy]], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (pts @ R.T) + np.array([x, y], dtype=np.float32)

def _poly_area(poly):
    """Shoelace formula; poly shape (K,2). If K<3, area=0."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _line_intersect(p1, p2, q1, q2):
    """Line p1->p2 with q1->q2; return intersection point or None."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
    return np.array([px, py], dtype=np.float32)

def _is_inside(p, a, b):
    """Check if point p is to the left of edge a->b (keeping polygon CCW)."""
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-9

def _sutherland_hodgman(subject, clipper):
    """Convex polygon clipping: returns intersection polygon vertices (K,2)."""
    if subject is None or len(subject) == 0:
        return None
    output = subject
    for i in range(len(clipper)):
        input_list = output
        if input_list is None or len(input_list) == 0:
            return None
        output = []
        A = clipper[i]
        B = clipper[(i+1) % len(clipper)]
        S = input_list[-1]
        for E in input_list:
            if _is_inside(E, A, B):
                if not _is_inside(S, A, B):
                    I = _line_intersect(S, E, A, B)
                    if I is not None:
                        output.append(I)
                output.append(E)
            elif _is_inside(S, A, B):
                I = _line_intersect(S, E, A, B)
                if I is not None:
                    output.append(I)
            S = E
        output = np.array(output, dtype=np.float32) if len(output) > 0 else None
    return output

def _inter_area_bev(a, b):
    """Intersection area between two rotated rectangles in XY-plane.
       a,b: (5,) as [x,y,l,w,yaw] or (7,) [x,y,z,l,w,h,yaw] (z,h ignored).
    """
    ax, ay, al, aw, ayaw = float(a[0]), float(a[1]), float(a[3]), float(a[4]), float(a[-1])
    bx, by, bl, bw, byaw = float(b[0]), float(b[1]), float(b[3]), float(b[4]), float(b[-1])

    pa = _rect_corners_xy(ax, ay, al, aw, ayaw)
    pb = _rect_corners_xy(bx, by, bl, bw, byaw)
    inter_poly = _sutherland_hodgman(pa, pb)
    return _poly_area(inter_poly)

# -------------------------------
# BEV IoU (LiDAR): [x,y,z,l,w,h,yaw]
# -------------------------------
def boxes_iou_bev_gpu(boxes_a, boxes_b):
    """
    boxes_a: (N,7) torch [x,y,z,l,w,h,yaw]
    boxes_b: (M,7) torch
    return : (N,M) torch IoU in BEV (XY-plane with rotation)
    """
    # 안전하게 CPU numpy로 계산 후 입력 device로 되돌림
    dev = boxes_a.device
    A = boxes_a.detach().cpu().numpy()
    B = boxes_b.detach().cpu().numpy()

    N, M = A.shape[0], B.shape[0]
    iou = np.zeros((N, M), dtype=np.float32)

    # 박스 면적 (BEV)
    area_a = (A[:, 3] * A[:, 4]).astype(np.float32)  # l*w
    area_b = (B[:, 3] * B[:, 4]).astype(np.float32)

    for i in range(N):
        for j in range(M):
            inter = _inter_area_bev(A[i], B[j])
            if inter <= 0.0:
                iou[i, j] = 0.0
            else:
                ua = area_a[i] + area_b[j] - inter
                iou[i, j] = inter / max(ua, 1e-9)

    return torch.from_numpy(iou).to(dev)

# -------------------------------
# 3D IoU (LiDAR): [x,y,z,l,w,h,yaw]
# -------------------------------
def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    boxes_a: (N,7) torch [x,y,z,l,w,h,yaw]
    boxes_b: (M,7) torch
    return : (N,M) torch IoU in full 3D (rotated boxes)
    """
    dev = boxes_a.device
    A = boxes_a.detach().cpu().numpy()
    B = boxes_b.detach().cpu().numpy()

    N, M = A.shape[0], B.shape[0]
    iou3d = np.zeros((N, M), dtype=np.float32)

    vol_a = (A[:, 3] * A[:, 4] * A[:, 5]).astype(np.float32)  # l*w*h
    vol_b = (B[:, 3] * B[:, 4] * B[:, 5]).astype(np.float32)

    # z-interval: [z - h/2, z + h/2]  (LiDAR에서 z가 수직축)
    zmin_a = A[:, 2] - A[:, 5] * 0.5
    zmax_a = A[:, 2] + A[:, 5] * 0.5

    zmin_b = B[:, 2] - B[:, 5] * 0.5
    zmax_b = B[:, 2] + B[:, 5] * 0.5

    for i in range(N):
        for j in range(M):
            # BEV 교집합 면적
            inter_bev = _inter_area_bev(A[i], B[j])
            if inter_bev <= 0.0:
                iou3d[i, j] = 0.0
                continue

            # 높이 교집합
            h_int = min(zmax_a[i], zmax_b[j]) - max(zmin_a[i], zmin_b[j])
            if h_int <= 1e-9:
                iou3d[i, j] = 0.0
                continue

            inter_vol = inter_bev * h_int
            union_vol = vol_a[i] + vol_b[j] - inter_vol
            iou3d[i, j] = inter_vol / max(union_vol, 1e-9)

    return torch.from_numpy(iou3d).to(dev)

def compute_ap(tp, fp, n_gt):
    """
    tp, fp : score 내림차순 정렬된 예측에 대한 0 / 1 배열
    n_gt : 클래스의 GT 갯수
    """
    if n_gt == 0:
        return 0.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / (n_gt + 1e-9)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)

    mpre = np.concatenate(([0.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    
    idx = np.where(mrec[1:] != mrec[:-1][0])
    ap = float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx + 1]))
    return ap

def yaw_diff_deg(yaw_pred, yaw_gt):
    d = abs(yaw_pred - yaw_gt)
    d = min(d, 2*np.pi - d)          # [0, pi]
    return d * 180.0 / np.pi

def compute_ate_one(p, g):
    # p,g: [x,y,z,l,w,h,yaw]
    return float(np.linalg.norm(p[:3] - g[:3]))

def compute_ase_one(p, g, eps=1e-6):
    # 상대오차: |Δsize| / size_gt 를 l,w,h 평균
    rel = np.abs(p[3:6] - g[3:6]) / (g[3:6] + eps)
    return float(np.mean(rel))

def iou3d_fn_lidar(P, G):
    """
    P, G: (N,7) [x,y,z,l,w,h,yaw]  (LiDAR 좌표계, rad)
    return: (Np x Ng) numpy IoU matrix
    """
    P_t = torch.from_numpy(P).float().cuda()
    G_t = torch.from_numpy(G).float().cuda()
    # 많은 라이브러리가 [x,y,z,l,w,h,yaw] 그대로 받습니다.
    iou = boxes_iou3d_gpu(P_t, G_t)            # (Np, Ng) torch
    return iou.cpu().numpy()

def iou_bev_fn_lidar(P, G):
    """
    P, G: (N,7) [x,y,z,l,w,h,yaw]  (LiDAR 좌표계)
    BEV 투영은 (x,y,l,w,yaw) 사용 (카메라일 때 x,z였던 것과 다름!)
    """
    P_bev = np.concatenate([P[:, [0, 1]], P[:, [3, 4]], P[:, 6]], axis=-1) 
    G_bev = np.concatenate([G[:, [0, 1]], G[:, [3, 4]], G[:, 6]], axis=-1)
    P_t = torch.from_numpy(P_bev).float().cuda()
    G_t = torch.from_numpy(G_bev).float().cuda()
    iou = boxes_iou_bev_gpu(P_t, G_t)          # (Np, Ng) torch
    return iou.cpu().numpy()

def eval_per_class(pred_bboxes, pred_scores, pred_labels,
                    gt_bboxes, gt_labels,
                        class_name, iou_thres_3d, iou_thres_bev = None,
                        iou3d_fn=None, iou_bev_fn=None):
    """
    pred_bboxes : (N_pred, 7) [x, y, z, l, w, h, yaw]
    gt_bboxes : (N_pred, 7) [x, y, z, l, w, h, yaw]
    """
    p_idx = np.where(pred_labels == class_name)[0]
    g_idx = np.where(gt_labels == class_name)[0]

    if len(p_idx) == 0 and len(g_idx) == 0:
        return 0.0, 0.0, []
    
    P = pred_bboxes[p_idx]
    S = pred_scores[p_idx]
    G = gt_bboxes[g_idx]

    order = np.argsort(-S)
    P = P[order]
    S = S[order]

    ap3d, apbev = 0.0, 0.0
    matched_errs = []

    # 3D AP
    if len(P) and len(G) and iou3d_fn is not None:
        IoU3D = iou3d_fn(P, G)
        tp = np.zeros(len(P), dtype=int)
        fp = np.zeros(len(P), dtype=int)
        gt_used = np.zeros(len(G), dtype=bool)

        for i in range(len(P)):
            j = int(np.argmax(IoU3D[i]))
            iou = IoU3D[i, j]
            if iou >= iou_thres_3d and not gt_used[j]:
                tp[i] = 1
                gt_used[j] = True
                ate = compute_ate_one(P[i], G[j])
                aoe = yaw_diff_deg(P[i, -1], G[j, -1])
                ase = compute_ase_one(P[i], G[j])
                matched_errs.append({"ate": ate, "aoe_deg": aoe, "ase": ase})
            else:
                fp[i] = 1
        
        ap3d = compute_ap(tp, fp, len(G))
    
    if iou_thres_bev is not None and len(P) and len(G) and (iou_bev_fn is not None):
        IoUBEV = iou_bev_fn(P, G)
        tp = np.zeros(len(P), dtype=int)
        fp = np.zeros(len(P), dtype=int)
        gt_used_bev = np.zeros(len(G), dtype=bool)

        for i in range(len(P)):
            j = int(np.argmax(IoUBEV[i]))
            iou = IoUBEV[i, j]
            if iou >= iou_thres_bev and not gt_used_bev[j]:
                tp[i] = 1
                gt_used_bev[j] = True
            else:
                fp[i] = 1
        apbev = compute_ap(tp, fp, len(G))
    return ap3d, apbev, matched_errs

def evaluate_predictions(lidar_bboxes, labels, scores,
                            gt_lidar_bboxes, gt_labels,
                                id2name, class_iou_3d_thres, class_iou_bev_thres = None,
                                iou3d_fn = None, iou_bev_fn = None):
    """
    bboxes / gt_bboxes : (N, 7)
    labels / gt_labesl : (N, )
    id2name : {id:int -> name:str}
    """
    pred_names = np.array([id2name[int(i)] for i in labels], dtype=object)
    gt_names = np.array([id2name[int(i)] for i in gt_labels], dtype=object)

    class_list = sorted(set(list(pred_names) + list(gt_names)))
    per_class_ap3d = {}
    per_class_apbev = {} if class_iou_bev_thres is not None else None
    matched_errs_all = []

    for cname in class_list:
        thr3d = class_iou_3d_thres.get(cname, 0.5)
        thrbev = None if class_iou_bev_thres is None else class_iou_bev_thres.get(cname, 0.5)

        ap3d, apbev, errs = eval_per_class(
            pred_bboxes=lidar_bboxes, pred_scores=scores, pred_labels=pred_names,
            gt_bboxes=gt_lidar_bboxes, gt_labels=gt_names,
            class_name=cname, iou_thres_3d=thr3d, iou_thres_bev=thrbev,
            iou3d_fn = iou3d_fn_lidar, iou_bev_fn=iou_bev_fn_lidar
        )
        per_class_ap3d[cname] = ap3d
        if per_class_apbev is not None:
            per_class_apbev[cname] = apbev
        matched_errs_all.extend(errs)
    return per_class_ap3d, per_class_apbev, matched_errs_all

def mean_or_zero(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else 0.0

class RunningMetrics:
    def __init__(self, class_names):
        self.cls = class_names                      # e.g., ["jetski","smallboat","yacht",...]
        self.ap3d = defaultdict(list)               # per-class AP@IoU_3D
        self.apbev = defaultdict(list)              # per-class AP@IoU_BEV (optional)
        self.ate = []                               # matched pairs에서의 |Δt|
        self.aoe_deg = []                           # matched pairs에서의 |Δyaw| in degrees
        self.ase = []                               # matched pairs에서의 scale relative error (0~+)

    def update_from_batch(self, batch_eval_out):
        """
        batch_eval_out example
        {
            "ap3d": {"jetski": 0.62, "smallboat": 0.55, ...},
            "apbev": {"jetski": 0.71, "smallboat": 0.63, ...},
            "matched_errors": [
                {"ate": 0.42, "aoe_deg": 7.3, "ase": 0.11}, 
                ...
            ]
        }
        """
        for c, v in batch_eval_out.get("ap3d", {}).items():
            self.ap3d[c].append(float(v))
        for c in batch_eval_out.get("apbev", {}):
            self.apbev[c].append(float(v))
        for e in batch_eval_out.get("matched_errors", {}):
            self.ate.append(float(e["ate"]))
            self.aoe_deg.append(float(e["aoe_deg"]))
            self.ase.append(float(e["ase"]))
        
    def compute(self, class_weights=None, 
                    w3d=0.7, wbev = 0.1, werr = 0.2, 
                    ATE_cap = 1.0, AOE_cap_deg = 15.0, ASE_cap = 0.20,
                    a = 0.5, b = 0.3, g = 0.2):
        # per-class average AP
        ap3d_mean = {c: mean_or_zero(self.ap3d[c]) for c in self.cls}
        has_bev = len(any(v) > 0 for v in self.apbev.values())
        apbev_mean = {c: (mean_or_zero(self.apbev[c]) if has_bev else 0.0) for c in self.cls}

        if class_weights is None:
            class_weights = {c: 1.0 for c in self.cls}
        sw = sum(class_weights.values()) or 1.0
        mAP_3D = sum(ap3d_mean[c] * class_weights[c] for c in self.cls) / sw
        mAP_BEV = sum(apbev_mean[c] * class_weights[c] for c in self.cls) / sw

        ATE = mean_or_zero(self.ate)
        AOE_deg = mean_or_zero(self.aoe_deg)
        ASE = mean_or_zero(self.ase)

        ATE_n = min(1.0, ATE / ATE_cap)
        AOE_n = min(1.0, AOE_deg / AOE_cap_deg)
        ASE_n = min(1.0, ASE / ASE_cap)
        penalty = a*ATE_n + b*AOE_n + g*ASE_n

        score = w3d*mAP_3D + wbev*mAP_BEV + werr*(1.0-penalty)
        score = max(0.0, min(1.0, score))
        eval_loss = 1.0 - score
        summary = {
            "mAP_3D": mAP_3D,
            "mAP_BEV": mAP_BEV, 
            "ATE": ATE,
            "AOE_deg": AOE_deg,
            "ASE": ASE,
            "ATE_n": ATE_n,
            "AOE_n": AOE_n,
            "ASE_n": ASE_n,
            "penalty": penalty,
            "score": score,
            "eval_loss": eval_loss,
            "ap3d_per_class": ap3d_mean
        }
        return summary


