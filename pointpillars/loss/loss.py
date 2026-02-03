import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1/9, cls_w=1.0, reg_w=2.0, dir_w=0.2):
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2.0
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none',
                                              beta=beta)
        self.dir_cls = nn.CrossEntropyLoss()
    
    def forward(self,
                bbox_cls_pred,
                bbox_pred,
                bbox_dir_cls_pred,
                batched_labels, 
                num_cls_pos, 
                batched_bbox_reg, 
                batched_dir_labels):
        '''
        bbox_cls_pred: (n, 3)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        '''
        # 1. bbox cls loss
        # focal loss: FL = - \alpha_t (1 - p_t)^\gamma * log(p_t)
        #             y == 1 -> p_t = p
        #             y == 0 -> p_t = 1 - p
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float() # (n, 3)

        # 2. focal loss -> cls loss for AMP
        logits = bbox_cls_pred
        targets = batched_labels.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)     # pt = p if y=1 else (1-p)
        alpha = self.alpha
        gamma = self.gamma
        focal_weight = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = focal_weight * (1 - pt).pow(gamma)
        cls_loss = focal_weight * bce_loss
        total_cls_loss = cls_loss.sum()
        avg_factor = max(num_cls_pos, 1.0)
        cls_loss = total_cls_loss / avg_factor
        
        # 2. regression loss
        if bbox_pred.size(0) > 0:
            reg_loss = self.smooth_l1_loss(bbox_pred[:, :-1], batched_bbox_reg[:, :-1]).sum() / bbox_pred.size(0)
            A = bbox_pred[:, -1]
            B = batched_bbox_reg[:, -1]
            sin_diff = torch.sin(A) * torch.cos(B) - torch.cos(A) * torch.sin(B)
            reg_loss += self.smooth_l1_loss(sin_diff, torch.zeros_like(sin_diff)).sum() / sin_diff.size(0)
        else:
            print("[Warning] No bbox regression targets → reg_loss = 0")
            reg_loss = bbox_pred.sum() * 0.0

        # 3. direction cls loss
        if bbox_dir_cls_pred.size(0) > 0:
            dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)
        else:
            print("[Warning] No dir classification targets → dir_cls_loss = 0")
            dir_cls_loss = bbox_dir_cls_pred.sum() * 0.0

        # 4. total loss
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        
        loss_dict={'cls_loss': cls_loss, 
                   'reg_loss': reg_loss,
                   'dir_cls_loss': dir_cls_loss,
                   'total_loss': total_loss}
        return loss_dict
    