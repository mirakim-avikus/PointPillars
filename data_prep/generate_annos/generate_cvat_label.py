import cv2 
import os 
import xml.etree.ElementTree as ET
import yaml
import pickle
import pdb
import argparse
import numpy as np
import open3d as o3d

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE, '../')))

from pointpillars.utils import bbox3d2corners

EXCLUDE_PATH=['avikus_gt_database', 'testing', 'training']

def load_frame_ids(frame_list_path):
    with open(frame_list_path, 'r') as f:
        lines = f.readlines()
    frame_ids = [int(line.strip().split()[0]) for line in lines]
    pcd_filenames = [int(line.strip().split()[1].split('.')[0]) for line in lines]
    return frame_ids, pcd_filenames

def load_pkl_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_annos_for_frames(frame_ids, pkl_data):
    results = {}
    for frame_id in frame_ids:
        if frame_id in pkl_data:
            results[frame_id] = []
            for data in pkl_data[frame_id]:
                results[frame_id].append(data.get('annos', {}))
            print(f"Good: Frame ID {frame_id} found in PKL data.")
        else:
            print(f"Warning: Frame ID {frame_id} not found in PKL data.")
    return results

def save_annos_dict_as_txt(data_root, data_name, annos_dict, output_dir, pcd_filenames):
    os.makedirs(os.path.join(data_root, data_name, output_dir), exist_ok=True)
    
    for (frame_id, annos), pcd_filename in zip(annos_dict.items(), pcd_filenames):
        file_name = f"{pcd_filename}.txt"
        file_path = os.path.join(data_root, data_name, output_dir, file_name)

        with open(file_path, 'w') as f:
            for i in range(len(annos)):
                name = annos[i]['name'][0].item()
                truncated = annos[i]['truncated']
                occluded = annos[i]['occluded']
                alpha = annos[i]['alpha']
                bbox = annos[i]['bbox']
                dimensions = annos[i]['dimensions']
                location = annos[i]['location']
                rotation_z = float(annos[i]['rotation_z'])
                line =f"{name} {truncated:.2f} {occluded} {alpha:.2f} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} {dimensions[0]:.2f} {dimensions[1]:.2f} {dimensions[2]:.2f} {location[0]:.2f} {location[1]:.2f} {location[2]:.2f} {rotation_z:.2f}\n"
                f.write(line)


# find closest image filename
def find_closest_imagefile(lidar_filename, image_dir_path, max_diff=200):
    lidar_ts = int(lidar_filename.split('.')[0])
    img_list = sorted(os.listdir(image_dir_path))
    differences = np.array([int(ts.split('.')[0]) - lidar_ts for ts in img_list])
    closest_idx = np.argmin(np.abs(differences))
    if differences[closest_idx] >= max_diff:
        return None
    return img_list[closest_idx]

def count_num_points_in_gt(lidar_points, dimensions, location, rotation_x, rotation_y, rotation_z):
    # avikus frame
    bboxes = np.concatenate([location, dimensions, rotation_z], axis = -1).reshape(1, -1)
    bboxes_corners = bbox3d2corners(bboxes)
    num_points_in_gt = []
    for bbox in bboxes_corners:
        num_points = 0
        for point in lidar_points:
            if (point[0] >= min(bbox[:, 0]) and point[0] <= max(bbox[:, 0]) and \
                point[1] >= min(bbox[:, 1]) and point[1] <= max(bbox[:, 1]) and \
                point[2] >= min(bbox[:, 2]) and point[2] <= max(bbox[:, 2])):
                num_points += 1
        num_points_in_gt.append(num_points)
        print(F"num points : {num_points}")
    return num_points_in_gt

def project_lidar_to_image(lidar_points, img_path, Rt, K, D):
    # read image
    img = cv2.imread(img_path)
    lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    cam_points = (Rt @ lidar_hom.T).T
    cam_points = cam_points[cam_points[:, 2] > 0]   # z > 0

    # calculate distance for coloring
    distance = np.linalg.norm(cam_points, axis = 1)
    hsv = np.zeros((len(distance), 3), dtype=np.uint8)
    hsv[:, 0] = np.clip((distance / 500.0) * 180.0, 0, 179)
    hsv[:, 1] = 255 # Saturation
    hsv[:, 2] = 255 # Value    

    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)

    # project valid points using fisheye model
    points_normalized = cam_points[:, :2] / cam_points[:, 2:]
    points_distorted = cv2.fisheye.distortPoints(points_normalized.reshape(-1, 1, 2), K, D)
    points_distorted = points_distorted.reshape(-1, 2)
    
    mask = (points_distorted[:, 0] >= 0) & (points_distorted[:, 0] < img.shape[1]) & \
            (points_distorted[:, 1] >= 0) & (points_distorted[:, 1] < img.shape[0])
    
    for pt, color in zip(points_distorted[mask], bgr[mask]):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (int(color[0]), int(color[1]), int(color[2])), -1)

    if not os.path.exists("projection"):
        os.makedirs("projection")

    cv2.imwrite(f"projection/new_{os.path.basename(img_path)}", img)

    return points_distorted, K, Rt

def main(args):
    args.data_root = os.path.normpath(args.data_root)
    data_root = args.data_root
    for data_name in os.listdir(data_root):
        if data_name in EXCLUDE_PATH or not os.path.isdir(os.path.join(data_root, data_name)):
            continue
        xml_file_path = os.path.join(data_root, data_name, 'tracklet_labels.xml')
        if not os.path.exists(xml_file_path):
            continue
        frame_list_path = os.path.join(data_root, data_name, 'frame_list.txt') 
        image_dir_path = os.path.join(data_root, data_name, 'images')
        lidar_dir_path = os.path.join(data_root, data_name, 'pcd')
        calib_path = os.path.join(data_root, data_name, 'new_lidar.yaml')

        frame_dict = {}
        with open(frame_list_path, 'r') as f:
            for line in f:
                index, filename = line.strip().split()
                frame_dict[int(index)] = filename

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

        R, _ = cv2.Rodrigues(rvec)
        tr_velo_to_cam = np.identity(4)
        tr_velo_to_cam[:3, :3] = R
        tr_velo_to_cam[:3, -1] = tvec
        Rt = tr_velo_to_cam[:3, :]

        kitti_like_dict = {}

        # xml parsing
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        kitti_like_dict = {}
        # iterate every track item 
        for idx, item in enumerate(root.findall(".//item")):
            try:
                object_type = item.findtext('objectType')
                h = float(item.findtext('h'))
                w = float(item.findtext('w'))
                l = float(item.findtext('l'))
                first_frame = int(item.findtext('first_frame'))
            except:
                continue

            pose_item = item.find('.//poses/item')
            if pose_item is None:
                continue
            
            tx = float(pose_item.findtext('tx'))
            ty = float(pose_item.findtext('ty'))
            tz = float(pose_item.findtext('tz'))
            rx = float(pose_item.findtext('rx'))
            ry = float(pose_item.findtext('ry'))
            rz = float(pose_item.findtext('rz'))

            lidar_filename = frame_dict.get(first_frame)
            if lidar_filename is None:
                print(f"Frame {first_frame} not found in frame list.")
                continue
            
            image_filename = find_closest_imagefile(lidar_filename, image_dir_path)
            if (image_filename == None):
                print(f'closest img not valid at lidar : {lidar_filename}')
                continue

            image_path = os.path.join(image_dir_path, image_filename)
            image = cv2.imread(image_path)

            lidar_points = np.asarray(o3d.io.read_point_cloud(os.path.join(lidar_dir_path, lidar_filename + ".avikus.pcd")).points)
            _, K, Rt = project_lidar_to_image(lidar_points, image_path, Rt, K, D)

            velodyne_path = lidar_filename + ".avikus.pcd"

            num_points_in_gt = count_num_points_in_gt(lidar_points, np.array([w, h, l]), np.array([tx, ty, tz]), np.array([rx]), np.array([ry]), np.array([rz]))

            image = {
                'image_shape': tuple(image.shape[:2]), # H, W
                'image_path': image_path,
                'image_idx': 0 # TODO tmp
            }
            calib = {
                'P0': K@Rt,
                'P1': K@Rt,
                'P2': K@Rt, # bbox projection to image plane. P2 is used data. copy P2 to P0~P3
                'P3': K@Rt, 
                'R0_rect': np.identity(4), # for rectification (given multiple images)
                'Tr_velo_to_cam': Rt,
                'Tr_imu_to_velo': ref2lidar_Rt_inv,
            }

            # annos에서 저장할 때 avikus2camera로 저장!
            annos = {
                'name': np.array([object_type]),
                'truncated': 0,
                'occluded': 0,
                'alpha': 0, # TODO : 확인해보기 (객체와의 각도) -> 사용하면 그때 값 작성하기
                'bbox': [0.0, 0.0, 0.0, 0.0] ,
                'dimensions': np.array([w, h, l]),
                'location': np.array([tx, ty, tz - h/2]),
                'rotation_x': np.array([rx]),
                'rotation_y': np.array([ry]),
                'rotation_z': np.array([rz]),
                'difficulty': 0,
                'num_points_in_gt': num_points_in_gt   # TODO
            }

            # image와 기타 정보들 들어가있음
            if first_frame in kitti_like_dict:
                kitti_like_dict[first_frame].append({
                    'velodyne_path': velodyne_path,
                    'image': image,
                    'calib': calib,
                    'annos': annos, 
                })
            else:
                kitti_like_dict[first_frame] = [{
                    'velodyne_path': velodyne_path,
                    'image': image,
                    'calib': calib,
                    'annos': annos, 
                }]

        frame_ids, pcd_filenames = load_frame_ids(frame_list_path)
        annos_dict = extract_annos_for_frames(frame_ids, kitti_like_dict)
        output_dir = "label"
        save_annos_dict_as_txt(data_root, data_name, annos_dict, output_dir, pcd_filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='', required=True, help='your image path')
    args = parser.parse_args()

    main(args)