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


def point_range_filter(pts, point_range=[-300, -300.68, -100, 300.0, 300, 100]):
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
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([-100.0, -100.0, -100.0, 200.0, 100.0, 100.0], dtype=np.float32)

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

    R, _ = cv2.Rodrigues(rvec)                                  # avikus2camera
    lidar2avi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])   # lidar2avikus
    tr_velo_to_cam = R@lidar2avi                                # lidar2camera

    tr_velo_to_cam_4x4 = np.identity(4)
    tr_velo_to_cam_4x4[:3, :3] = tr_velo_to_cam
    tr_velo_to_cam_4x4[:3, -1] = tvec


    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError 
    pc = read_points(args.pc_path)              # avikus coord
    pc[:, 1:3] *= -1                            # avikus2lidar transformation [x, -y, -z]
    pc[:, 3] = pc[:, 3] / 255.                  # intensity normalization
    pc = point_range_filter(pc)
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
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam_4x4, r0_rect, P2, image_shape, K, D)

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)

    if calib_info is not None and img is not None:
        _, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']    # bboxes in camera coord
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
        # sel = gt_labels == -1
        gt_labels = gt_labels[:]
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam_4x4, r0_rect)
        bboxes_camera = bboxes_camera[:]
        gt_lidar_bboxes = gt_lidar_bboxes[:]

        gt_labels = [-1] * len(gt_label['name']) # to distinguish between the ground truth and the predictions
        
        pred_gt_lidar_bboxes = np.concatenate([lidar_bboxes, gt_lidar_bboxes], axis=0)
        pred_gt_labels = np.concatenate([labels, gt_labels])
        vis_pc(pc, pred_gt_lidar_bboxes, labels=pred_gt_labels)

        if img is not None:
            bboxes_corners = bbox3d2corners_camera(bboxes_camera)
            points_normalized = bboxes_corners[:, :, :2] / bboxes_corners[:, :, 2:]
            points_distorted = cv2.fisheye.distortPoints(points_normalized.reshape(-1, 1, 2), K, D)
            image_points = points_distorted.reshape(bboxes_corners.shape[0], -1, 2)

            # image_points = points_camera2image(bboxes_corners, P2)
            gt_labels = [-1] * len(gt_label['name'])
            img = vis_img_3d(img, image_points, gt_labels, rt=True)
            print(f"visualize GT 3d img")


    if calib_info is not None and img is not None:
        print(f"image write to {os.path.basename(args.img_path)}-3d_bbox_new.png!")
        cv2.imwrite(f'{os.path.basename(args.img_path)}-3d_bbox_new.png', img)
    print(f"process end!")            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', required=True, help='your point cloud path')
    parser.add_argument('--calib_path', required=True, default='', help='your calib file path')
    parser.add_argument('--gt_path', required=True, default='', help='your ground truth path')
    parser.add_argument('--img_path', required=True, default='', help='your image path')
    parser.add_argument('--point_range', default='', help='point range, [min_x, min_y, min_z, max_x, max_y, max_z]')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
