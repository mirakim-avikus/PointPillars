import argparse
import cv2
import numpy as np
import os
import torch
import pdb
import yaml

from pointpillars.utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from pointpillars.model import PointPillars


def box3d_volume(box):
    """Calculate volume of 3D box : (x1, y1, z1, x2, y2, z2)"""
    x1, y1, z1, x2, y2, z2 = box
    return max(0, x2-x1) * max(0, y2-y1) * max(0, z2-z1)

def intersection_volume(boxA, boxB):
    """Intersection volume of two 3D boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])
    return box3d_volume((xA, yA, zA, xB, yB, zB))

def iou3d(boxA, boxB):
    """3D IoU"""
    inter_vol = intersection_volume(boxA, boxB)
    union_vol = box3d_volume(boxA) + box3d_volume(boxB) - inter_vol
    return inter_vol / union_vol if union_vol != 0 else 0 

def giou3d(boxA, boxB):
    """Generalized IoU for 3D"""
    iou_score = iou3d(boxA, boxB)

    xC1 = min(boxA[0], boxB[0])
    yC1 = min(boxA[1], boxB[1])
    zC1 = min(boxA[2], boxB[2])
    xC2 = max(boxA[3], boxB[3])
    yC2 = max(boxA[4], boxB[4])
    zC2 = max(boxA[5], boxB[5])

    enclosing_vol = box3d_volume((xC1, yC1, zC1, xC2, yC2, zC2))
    union_vol = box3d_volume(boxA) + box3d_volume(boxB) - intersection_volume(boxA, boxB)
    return iou_score - (enclosing_vol - union_vol) / enclosing_vol if enclosing_vol != 0 else iou_score #TODO 이거 맞는건가..?0

def diou3d(boxA, boxB):
    """Distance IoU for 3D"""
    iou_score = iou3d(boxA, boxB)

    # Center of each box 
    centerA = ((boxA[0] + boxA[3]) / 2, (boxA[1] + boxA[4]) / 2, (boxA[2] + boxA[5]) / 2)
    centerB = ((boxB[0] + boxB[3]) / 2, (boxB[1] + boxB[4]) / 2, (boxB[2] + boxB[5]) / 2)

    center_dist_sq = sum([(a - b) ** 2 for a, b in zip(centerA, centerB)])

    xC1 = min(boxA[0], boxB[0])
    yC1 = min(boxA[1], boxB[1])
    zC1 = min(boxA[2], boxB[2])
    xC2 = max(boxA[3], boxB[3])
    yC2 = max(boxA[4], boxB[4])
    zC2 = max(boxA[5], boxB[5])

    diag_sq = (xC2 - xC1) ** 2 + (yC2 - yC1) ** 2 + (zC2 - zC1) ** 2
    return iou_score - (center_dist_sq / diag_sq) if diag_sq != 0 else iou_score

def ciou3d(boxA, boxB):
    """Complete IoU for 3D (simplified : only aspect ratio + DIoU)"""
    iou_score = iou3d(boxA, boxB)
    diou_score = diou3d(boxA, boxB)

    # sizes
    wA = boxA[3] - boxA[0]
    hA = boxA[4] - boxA[1]
    dA = boxA[5] - boxA[2]
    wB = boxB[3] - boxB[0]
    hB = boxB[4] - boxB[1]
    dB = boxB[5] - boxB[2]

    # simplified 3D aspect ratio penalty 
    v = ((wA - wB) ** 2 + (hA - hB) ** 2 + (dA - dB) ** 2) / (wA ** 2 + hA ** 2+ dA ** 2 + 1e-6)
    alpha = v / (1 - iou_score + v) if iou_score < 1 else 0

    return diou_score - alpha * v 

def containment_score3d(pred_box, gt_box):
    """How much gt_box is contained in pred_box"""
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    zA = max(pred_box[2], gt_box[2])
    xB = min(pred_box[0], gt_box[0])
    yB = min(pred_box[1], gt_box[1])
    zB = min(pred_box[2], gt_box[2])
    inter_vol = box3d_volume(xA, yA, zA, xB, yB, zB)
    gt_vol = box3d_volume(gt_box)
    return inter_vol / gt_vol if gt_vol != 0 else 0

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 


def main(args):
    CLASSES = {
        'motorboat': 2
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -50., -10., 250., 50., 10.], dtype=np.float32)

    calib_path = args.calib_path
    calib_path_yaml = os.path.join(*calib_path.split('/')[:-1],"lidar.yaml")
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
    lidar2avikus = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    tr_velo_to_cam_4x4 = np.identity(4)
    tr_velo_to_cam_4x4[:3, :3] = R@lidar2avikus
    tr_velo_to_cam_4x4[:3, -1] = tvec

    point_cloud_range=[0, -50., -10., 250., 50., 10.]
    voxel_size=[0.16, 0.16, 4]

    if not args.no_cuda:
        model = PointPillars(nclasses=3, point_cloud_range=point_cloud_range, voxel_size = voxel_size, prefix='avikus').cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=3)
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError 
    pc = read_points(args.pc_path)
    pc[:, 3] = pc[:, 3] / 255.                  # intensity normalization
    pc = point_range_filter(pc, point_range=[0, -50., -10., 250., 50., 10.])
    pc_torch = torch.from_numpy(pc)
    if os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    if os.path.exists(args.img_path):
        img = cv2.imread(args.img_path, 1)
    else:
        img = None

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
        
        result_filter = model(batched_pts=[pc_torch], 
                              mode='test')[0]
    if calib_info is not None and img is not None:
        # tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        image_shape = img.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam_4x4, r0_rect, P2, image_shape, K=K, D=D, prefix='avikus')

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)

    if calib_info is not None and img is not None:
        bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
        bboxes_corners = bbox3d2corners_camera(camera_bboxes)
        points_normalized = bboxes_corners[:, :, :2] / bboxes_corners[:, :, 2:]
        points_distorted = cv2.fisheye.distortPoints(points_normalized.reshape(-1, 1, 2), K, D)
        image_points = points_distorted.reshape(bboxes_corners.shape[0], -1, 2)
        img = vis_img_3d(img, image_points, labels, rt=True)

    if calib_info is not None and gt_label is not None:
        # tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels != -1
        gt_labels = gt_labels[sel]
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam_4x4, r0_rect)
        bboxes_camera = bboxes_camera[sel]
        gt_lidar_bboxes = gt_lidar_bboxes[sel]

        gt_labels = [-1] * len(gt_label['name']) # to distinguish between the ground truth and the predictions
        
        pred_gt_lidar_bboxes = np.concatenate([lidar_bboxes, gt_lidar_bboxes], axis=0)
        pred_gt_labels = np.concatenate([labels, gt_labels])
        vis_pc(pc, gt_lidar_bboxes, labels=pred_gt_labels)

        if img is not None:
            bboxes_corners = bbox3d2corners_camera(bboxes_camera)
            points_normalized = bboxes_corners[:, :, :2] / bboxes_corners[:, :, 2:]
            points_distorted = cv2.fisheye.distortPoints(points_normalized.reshape(-1, 1, 2), K, D)
            image_points = points_distorted.reshape(bboxes_corners.shape[0], -1, 2)
            gt_labels = [-1] * len(gt_label['name'])
            img = vis_img_3d(img, image_points, gt_labels, rt=True)
    
    if calib_info is not None and img is not None:
        cv2.imwrite(f'{os.path.basename(args.img_path)}-3d_bbox.jpg', img)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', help='your point cloud path')
    parser.add_argument('--calib_path', default='', help='your calib file path')
    parser.add_argument('--gt_path', default='', help='your ground truth path')
    parser.add_argument('--img_path', default='', help='your image path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
