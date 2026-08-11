import yaml
import cv2
import os
import sys
import numpy as np
import argparse

def matrix_to_kitti_format(mat):
    if mat.shape == (4,4):
        mat = mat[:3]
    return ' '.join(f'{val:.12e}' for val in mat.flatten())

def calib_dict_to_kitti(calib):
    lines = []
    for key in ['P0', 'P1', 'P2', 'P3']:
        lines.append(f"{key}: {matrix_to_kitti_format(calib[key])}")
    R0_rect = calib['R0_rect'][:3, :3]
    lines.append(f"'R0_rect': {matrix_to_kitti_format(calib['R0_rect'])}")
    lines.append(f"'Tr_velo_to_cam': {matrix_to_kitti_format(calib['Tr_velo_to_cam'])}")
    lines.append(f"'Tr_imu_to_velo': {matrix_to_kitti_format(calib['Tr_imu_to_velo'])}")
    return '\n'.join(lines)

def main(args):
    args.data_root = os.path.normpath(args.data_root)
    data_root = args.data_root
    data_key_excluded = ['avikus_gt_database', 'training', 'testing']
    for data_key in os.listdir(data_root):
        if data_key in data_key_excluded:
            continue
        full_path = os.path.join(data_root, data_key)
        if not os.path.isdir(full_path):
            continue
        if data_key in ["labels", "meta", "tmp"]:
            continue

        calib_dir = os.path.join(data_root, data_key)

        calib_path = os.path.join(calib_dir, "lidar.yaml")
        if not os.path.exists(calib_path):
            print(f"[SKIP] {data_key}: no lidar.yaml")
            continue

        with open(calib_path, 'rb') as f:
            calib = yaml.safe_load(f)

        cam = calib['camera']
        K = np.array([
            [cam['fx'], cam['skew'], cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ], dtype = np.float32)

        D = np.array([cam['k1'], cam['k2'], cam['k3'], cam['k4']], dtype=np.float32)

        # Camera2LiDAR
        rvec = np.array([
            calib['camera2lidar']['rvec_1'],
            calib['camera2lidar']['rvec_2'],
            calib['camera2lidar']['rvec_3'],
        ], dtype = np.float32)

        tvec = np.array([
            calib['camera2lidar']['tvec_1'],
            calib['camera2lidar']['tvec_2'],
            calib['camera2lidar']['tvec_3'],
        ], dtype = np.float32)

        # IMU2LiDAR 
        ref2lidar_rvec = np.array([
            calib['reference2lidar']['rvec_1'],
            calib['reference2lidar']['rvec_2'],
            calib['reference2lidar']['rvec_3'],
        ], dtype = np.float32)
        ref2lidar_R, _ = cv2.Rodrigues(ref2lidar_rvec)
        ref2lidar_tvec = np.array([
            calib['reference2lidar']['tvec_1'],
            calib['reference2lidar']['tvec_2'],
            calib['reference2lidar']['tvec_3'],
        ], dtype = np.float32)

        ref2lidar_R_inv = ref2lidar_R.T
        ref2lidar_t_inv = -ref2lidar_R_inv @ ref2lidar_tvec.reshape(3, 1)
        ref2lidar_Rt_inv = np.hstack((ref2lidar_R_inv, ref2lidar_t_inv))

        R, _ = cv2.Rodrigues(rvec)                                  # avikus2camera
        tr_velo_to_cam = np.identity(4)
        tr_velo_to_cam[:3, :3] = R
        tr_velo_to_cam[:3, -1] = tvec

        calib = {
            'P0': K@tr_velo_to_cam[:3, :],
            'P1': K@tr_velo_to_cam[:3, :],
            'P2': K@tr_velo_to_cam[:3, :], # bbox projection to image plane. P2 is used data. copy P2 to P0~P3
            'P3': K@tr_velo_to_cam[:3, :], 
            'R0_rect': np.identity(3), # for rectification (given multiple images)
            'Tr_velo_to_cam': tr_velo_to_cam[:3, :],
            'Tr_imu_to_velo': ref2lidar_Rt_inv,
        }

        kitti_format_str = calib_dict_to_kitti(calib)
        calib_index = os.path.normpath(calib_path).split('/')[-2]
        with open(os.path.join(data_root, data_key, f'calib_{calib_index}.txt'), 'w') as f:
            print(f'data path : {os.path.join(data_root, data_key, f"calib_{calib_index}.txt")}')
            f.write(kitti_format_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='', required=True, help='your image path')
    args = parser.parse_args()
    main(args)