import argparse
import pdb
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys
import math
import yaml
from pathlib import Path

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR, '../')))

from pointpillars.utils import read_points, write_points, read_calib, read_label, \
    write_pickle, remove_outside_points, get_points_num_in_bbox, \
    points_in_bboxes_v2
from pointpillars.dataset import point_range_filter, POINT_CLOUD_RANGE

# TODO : indicate truncation
def judge_difficulty(annotation_dict, is_avikus):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        if is_avikus:
            difficultys.append(2)
        else:
            difficultys.append(difficulty)
    return np.array(difficultys, dtype=np.int32)

def find_closest_img(data_root, data_key, lidar_ts):
    image_dir = os.path.join(data_root, data_key, 'images')
    image_list = sorted([img for img in os.listdir(image_dir) if img.endswith('.jpg')])

    image_ts_list = [int(name.split('.')[0]) for name in image_list]
    min_diff = float('inf')
    closest_img = None

    for i, img_ts in enumerate(image_ts_list):
        diff = abs(img_ts - int(lidar_ts))
        if diff < min_diff:
            min_diff = diff
            closest_img = image_list[i]

    if closest_img is None:
        return None
    return int(closest_img.split('.')[0])

def euler_to_R(roll, pitch, yaw, order="ZYX"):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float32)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float32)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float32)

    if order == "ZYX":   # yaw->pitch->roll
        return Rz @ Ry @ Rx
    raise ValueError(order)

def compensate_points_and_bboxes_lidar_frame(
    pts_L, bboxes_L,             # pts: (N,4), bboxes: (M,7)
    R_IL, t_IL,                  # LiDAR -> IMU  (x_I = R_IL x_L + t_IL)
    roll, pitch, yaw,
    compensate_yaw=False
):
    # 1) IMU attitude rotation (IMU->World)
    if not compensate_yaw:
        yaw = 0.0
    roll, pitch, yaw = np.deg2rad(np.float32(roll)), np.deg2rad(np.float32(pitch)), np.deg2rad(np.float32(yaw))
    R_WI = euler_to_R(roll, pitch, yaw)  # IMU -> World
    R_IW = R_WI.T                        # World -> IMU (== inverse)  ✅ 자세 "제거"용

    # 2) Extrinsic inverse for going back to LiDAR
    R_LI = R_IL.T                        # IMU -> LiDAR
    t_IL = np.asarray(t_IL).reshape(3,)

    # --- points ---
    pts_Lc = pts_L
    if pts_L is not None and pts_L.shape[0] > 0:
        xyz_L = pts_L[:, :3]                               # (N,3)
        # L -> I
        xyz_I = (R_IL @ xyz_L.T).T + t_IL[None, :]         # (N,3)
        # attitude compensation in I (remove roll/pitch/yaw)
        xyz_Ic = (R_IW @ xyz_I.T).T                        # (N,3)
        # I -> L
        xyz_Lc = (R_LI @ (xyz_Ic - t_IL[None, :]).T).T     # (N,3)

        pts_Lc = pts_L.copy()
        pts_Lc[:, :3] = xyz_Lc

    # --- bboxes (center만 보정) ---
    bboxes_Lc = bboxes_L
    if bboxes_L is not None and bboxes_L.shape[0] > 0:
        centers_L = bboxes_L[:, :3]                        # (M,3)
        # L -> I
        centers_I = (R_IL @ centers_L.T).T + t_IL[None, :] # (M,3)
        # attitude compensation in I
        centers_Ic = (R_IW @ centers_I.T).T                # (M,3)
        # I -> L
        centers_Lc = (R_LI @ (centers_Ic - t_IL[None, :]).T).T  # (M,3)

        bboxes_Lc = bboxes_L.copy()
        bboxes_Lc[:, :3] = centers_Lc

    return pts_Lc, bboxes_Lc


def find_closest_imu(data_root, data_key, lidar_ts):
    """
    lidar_ts (int): lidar timestamp (ms)
    return: (imu_ts, roll, pitch, yaw) or None
    """
    imu_path = os.path.join(data_root, data_key, 'oru', 'attitude.csv')
    if not os.path.exists(imu_path):
        return np.array([lidar_ts, 0.0, 0.0, 0.0])

    # load csv: [N,4] -> ts, roll, pitch, yaw
    imu_data = np.loadtxt(imu_path, delimiter=',')
    if imu_data.ndim == 1:  # row 1개짜리 edge case
        imu_data = imu_data[None, :]

    if imu_data[0].shape[0] == 0:
        return np.array([lidar_ts, 0.0, 0.0, 0.0])
    else:
        imu_ts = imu_data[:, 0].astype(np.int64)
        lidar_ts = int(lidar_ts)
        idx = np.argmin(np.abs(imu_ts - lidar_ts))
        ts, roll, pitch, yaw = imu_data[idx]
        return np.array([ts, roll, pitch, yaw])

def create_data_info_pkl(data_root, data_type, prefix, label=True, db=False, args=None):
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    is_avikus = 'avikus' in prefix
    if is_avikus:
        ids_file = os.path.join(data_root, f"{data_type}.txt")
    else:
        ids_file = os.path.join(CUR, '..', 'pointpillars', 'dataset', 'ImageSets', f'{data_type}.txt')        
    
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]
    
    split = 'training' if label else 'testing'

    kitti_infos_dict = {}
    if db:
        kitti_dbinfos_train = {}
        db_points_saved_path = os.path.join(data_root, f'{prefix}_gt_database')
        os.makedirs(db_points_saved_path, exist_ok=True)
    num_invalid = 0
    for id in tqdm(ids):
        cur_info_dict={}
        if is_avikus:
            data_name = id.split()[-1].split('/')[0]  # '005'
            lidar_ts = Path(id).name.split('.')[0]
            image_id = find_closest_img(data_root, data_name, lidar_ts)
            imu = find_closest_imu(data_root, data_name, lidar_ts)
            img_path = os.path.join(data_root, data_name, 'images', f'{image_id}.jpg')
            lidar_path = os.path.join(data_root, data_name, 'pcd', f'{lidar_ts}.avikus.pcd')            
            calib_path = os.path.join(data_root, data_name, f'calib_{data_name}.txt')
        else:
            img_path = os.path.join(data_root, split, 'image_2', f'{id}.png')
            lidar_path = os.path.join(data_root, split, 'velodyne', f'{id}.bin')
            calib_path = os.path.join(data_root, split, 'calib', f'{id}.txt') 
        cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-3:])

        img = cv2.imread(img_path)
        image_shape = img.shape[:2]
        if is_avikus:
            img_idx = int(image_id)
        else:
            img_idx = int(id)

        cur_info_dict['image'] = {
            'image_shape': image_shape,
            'image_path': sep.join(img_path.split(sep)[-3:]), 
            'image_idx': img_idx,
            }

        cur_info_dict['imu'] = {'imu_ts': int(imu[0]), 'imu_rpy': imu[1:]}
        calib_dict = read_calib(calib_path)
        cur_info_dict['calib'] = calib_dict

        lidar_points = read_points(lidar_path)
        if is_avikus:
            calib_path_yaml = os.path.join(data_root, data_name, "new_lidar.yaml")
            with open(calib_path_yaml, 'rb') as f:
                calib_yaml_dict = yaml.safe_load(f)

            rvec = np.array([calib_yaml_dict['camera2lidar']['rvec_1'], calib_yaml_dict['camera2lidar']['rvec_2'], calib_yaml_dict['camera2lidar']['rvec_3']])
            tvec = np.array([calib_yaml_dict['camera2lidar']['tvec_1'], calib_yaml_dict['camera2lidar']['tvec_2'], calib_yaml_dict['camera2lidar']['tvec_3']])

            R, _ = cv2.Rodrigues(rvec)
            tr_velo_to_cam = np.identity(4)
            tr_velo_to_cam[:3, :3] = R
            tr_velo_to_cam[:3, -1] = tvec
        else:
            tr_velo_to_cam = calib_dict['Tr_velo_to_cam']

        if is_avikus:
            label_path = os.path.join(data_root, data_name, 'label', f'{lidar_ts}.txt')
        else:
            label_path = os.path.join(data_root, split, 'label_2', f'{id}.txt')
            tr_velo_to_cam = calib_dict['Tr_velo_to_cam']

        annotation_dict = read_label(label_path)
        annotation_dimensions = annotation_dict['dimensions']

        indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
            points_in_bboxes_v2(
                points=lidar_points,
                r0_rect=calib_dict['R0_rect'].astype(np.float32), 
                tr_velo_to_cam=tr_velo_to_cam.astype(np.float32),
                dimensions=annotation_dimensions.astype(np.float32),
                location=annotation_dict['location'].astype(np.float32),
                rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                name=annotation_dict['name']    
            )

        roll, pitch, yaw = imu[1], imu[2], imu[3]

        T_imu_to_lidar = calib_dict['Tr_imu_to_velo'].astype(np.float32)
        R_LI = T_imu_to_lidar[:3, :3]
        t_LI = T_imu_to_lidar[:3, 3]

        R_IL = R_LI.T
        t_IL = -R_IL @ t_LI

        if args.compensate_imu:
            lidar_points, bboxes_lidar = compensate_points_and_bboxes_lidar_frame(
                pts_L=lidar_points,
                bboxes_L=bboxes_lidar,
                R_IL=R_IL,
                t_IL=t_IL,
                roll=roll, pitch=pitch, yaw=yaw,
                compensate_yaw=False
            )

        if lidar_points is None or lidar_points.shape[0] == 0:
            continue
        tmp_dict = {'pts': lidar_points}
        tmp_dict = point_range_filter(tmp_dict, point_range=POINT_CLOUD_RANGE)
        filtered_pts = tmp_dict['pts']
        if filtered_pts.shape[0] <= int(args.min_pts_filter):
            num_invalid += 1
            continue

        # Avikus.__getitem__ reads velodyne_path directly (e.g. pcd-128/pcd/...)
        # rather than substituting in pts_prefix like Kitti does, so this reduced
        # copy is only ever consumed by the Kitti dataset class.
        if not is_avikus:
            saved_reduced_path = os.path.join(data_root, split, 'velodyne_reduced')
            os.makedirs(saved_reduced_path, exist_ok=True)
            saved_reduced_points_name = os.path.join(saved_reduced_path, f'{id}.bin')
            write_points(filtered_pts, saved_reduced_points_name)

        if label:            
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict, is_avikus)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=lidar_points,
                r0_rect=calib_dict['R0_rect'], 
                tr_velo_to_cam=tr_velo_to_cam,
                dimensions=annotation_dimensions,
                location=annotation_dict['location'],
                rotation_y=annotation_dict['rotation_y'],
                name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict

            if db:
                for j in range(n_valid_bbox):
                    db_points = lidar_points[indices[:, j]]
                    db_points[:, :3] -= bboxes_lidar[j, :3]
                    db_points_saved_name = os.path.join(db_points_saved_path, f'{int(lidar_ts)}_{name[j]}_{j}.bin')
                    write_points(db_points, db_points_saved_name)
                    db_info={
                        'name': name[j],
                        'path': os.path.join(os.path.basename(db_points_saved_path), f'{int(lidar_ts)}_{name[j]}_{j}.bin'),
                        'box3d_lidar': bboxes_lidar[j],
                        'difficulty': annotation_dict['difficulty'][j], 
                        'num_points_in_gt': len(db_points), 
                    }
                    if name[j] not in kitti_dbinfos_train:
                        kitti_dbinfos_train[name[j]] = [db_info]
                    else:
                        kitti_dbinfos_train[name[j]].append(db_info)
        
        kitti_infos_dict[int(lidar_ts)] = cur_info_dict

    if data_type == 'train':
        print(f'num_invalid ( num_pts <= {args.min_pts_filter} ): {num_invalid}')
    saved_path = os.path.join(data_root, f'{prefix}_infos_{data_type}.pkl')
    write_pickle(kitti_infos_dict, saved_path)
    if db:
        saved_db_path = os.path.join(data_root, f'{prefix}_dbinfos_train.pkl')
        write_pickle(kitti_dbinfos_train, saved_db_path)
    return kitti_infos_dict


def main(args):
    args.data_root = os.path.normpath(args.data_root)
    data_root = args.data_root
    prefix = args.prefix

    ## 1. train: create data infomation pkl file && create reduced point clouds 
    ##           && create database(points in gt bbox) for data aumentation
    kitti_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix, db=True, args=args)

    ## 2. val: create data infomation pkl file && create reduced point clouds
    kitti_val_infos_dict = create_data_info_pkl(data_root, 'val', prefix, args=args)
    
    ## 3. trainval: create data infomation pkl file
    kitti_trainval_infos_dict = {**kitti_train_infos_dict, **kitti_val_infos_dict}
    saved_path = os.path.join(data_root, f'{prefix}_infos_trainval.pkl')
    write_pickle(kitti_trainval_infos_dict, saved_path)

    ## 4. test: create data infomation pkl file && create reduced point clouds
    kitti_test_infos_dict = create_data_info_pkl(data_root, 'test', prefix, label=False, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', required=True, default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--prefix', required=True, default='kitti', 
                        help='the prefix name for the saved .pkl file')
    parser.add_argument('--min_pts_filter', type=int, default=10, 
                        help='compansate IMU or Not to points and bbox')
    parser.add_argument('--compensate_imu', action='store_true',
                        help='compansate IMU or Not to points and bbox')  # TODO compare performance
    args = parser.parse_args()

    main(args)