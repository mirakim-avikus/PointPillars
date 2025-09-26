import numpy as np
from collections import defaultdict
from .metric import compute_ap, compute_ate_one, yaw_diff_deg, compute_ase_one

class PRAccumulator:
    def __init__(self, class_names, id2name, iou_fn, thr_map):
        self.cls = class_names      # ["jetski", ...]
        self.id2name = id2name      # {id:int -> name:str}
        self.iou_fn = iou_fn        # iou3d_fn_lidar 또는 iou_bev_fn_lidar
        self.thr_map = thr_map      # {name -> thr}
        self.scores = {c: [] for c in self.cls}
        self.tp = {c: [] for c in self.cls}
        self.n_gt = {c: 0 for c in self.cls}
        self.errs = []
    
    def add_frame(self, pred_boxes, pred_scores, pred_labels, 
                            gt_boxes, gt_labels, collect_errors = False):
        pred_names = np.array([self.id2name[int(i)] for i in pred_labels], dtype=object)
        gt_names = np.array([self.id2name[int(i)] for i in gt_labels], dtype=object)
        classes_in_frame = sorted(set(list(pred_names) + list(gt_names)))

        for cname in classes_in_frame:
            thr = self.thr_map.get(cname, 0.5)
            p_idx = np.where(pred_names == cname)[0]
            g_idx = np.where(gt_names == cname)[0]
            P = pred_boxes[p_idx]
            S = pred_scores[p_idx]
            G = gt_boxes[g_idx]

            # accumlate number of GT 
            self.n_gt[cname] += len(G)

            if len(P) == 0:
                continue

            order = np.argsort(-S)
            P = P[order]
            S = S[order]

            used = np.zeros(len(G), dtype=bool)
            IoU = self.iou_fn(P, G) if len(G) > 0 else None

            for i in range(len(P)):
                is_tp = 0
                j_best = -1
                if len(G) > 0:
                    j_best = int(np.argmax(IoU[i]))
                    iou = IoU[i, j_best]
                    if iou >= thr and not used[j_best]:
                        is_tp = 1
                        used[j_best] = True
                        if collect_errors:
                            ate = compute_ate_one(P[i], G[j_best])
                            aoe = yaw_diff_deg(P[i, -1], G[j_best, -1])
                            ase = compute_ase_one(P[i], G[j_best])
                            self.errs.append({"ate": ate, "aoe_deg": aoe, "ase": ase})
                self.scores[cname].append(float(S[i]))
                self.tp[cname].append(is_tp)
    
    def compute_map(self):
        per_class_ap = {}
        for c in self.cls:
            sc = np.asarray(self.scores[c], dtype=np.float32)
            tp = np.asarray(self.tp[c], dtype=np.int32)
            n_gt = int(self.n_gt[c])
            if sc.size == 0:
                per_class_ap[c] = 0.0
                continue
            order = np.argsort(-sc)
            tp_sorted = tp[order]
            fp_sorted = 1 - tp_sorted
            per_class_ap[c] = compute_ap(tp_sorted, fp_sorted, n_gt)
        return per_class_ap, self.errs

