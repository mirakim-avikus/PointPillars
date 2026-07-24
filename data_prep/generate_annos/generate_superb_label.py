# !pip install --upgrade superb-ai-label

import os
import copy
import json
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import pandas as pd
import yaml
import shutil
import numpy as np
import time
import glob
import cv2
import argparse
import random
import pdb

def project_bbox_2d(frame_label_info, Rt, K, D, img_shape):
    """
    3D 바운딩박스를 이미지로 투영하여 2D bbox (left, top, right, bottom) 반환.

    Args:
        frame_label_info: dict, frame 단위 라벨 메타 정보
        Rt: np.ndarray, (3x4) 또는 (4x4) 카메라 외부 파라미터 행렬
        K: np.ndarray, (3x3) 카메라 내부 파라미터 행렬
        D: np.ndarray, (4,) 어안(fisheye) 왜곡 계수
        img_shape: tuple, (width, height) 이미지 해상도

    Returns:
        left, top, right, bottom: float, 2D bbox 좌표
    """
    # 1) 3D 박스 크기, 위치, 회전 쿼터니언 추출
    size = frame_label_info["annotation"]["coord"]["size"]
    w, h, l = size["x"], size["y"], size["z"]

    pos = frame_label_info["annotation"]["coord"]["position"]
    x, y, z = pos["x"], pos["y"], pos["z"]

    q = frame_label_info["annotation"]["coord"]["rotation_quaternion"]
    r = R.from_quat([q["x"], q["y"], q["z"], q["w"]])
    rot_mat = r.as_matrix()  # 3x3

    # 2) 로컬 좌표계에서 8개 코너 생성 (l: 길이, w: 너비, h: 높이)
    corners = np.array([
        [ l/2,  h/2,  w/2],
        [ l/2,  h/2, -w/2],
        [-l/2,  h/2, -w/2],
        [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2],
        [ l/2, -h/2, -w/2],
        [-l/2, -h/2, -w/2],
        [-l/2, -h/2,  w/2],
    ])

    # 3) 회전 및 병진 적용 → 월드 좌표
    corners_world = (rot_mat @ corners.T).T + np.array([x, y, z])

    # 4) 동차 좌표 변환 및 카메라 좌표계로 투영
    corners_hom = np.hstack([corners_world, np.ones((8,1))])  # (8,4)
    cam_corners = (Rt @ corners_hom.T).T                    # (8,4) 또는 (8,3)
    # z > 0 인 점만
    cam_corners = cam_corners[cam_corners[:,2] > 0]

    if cam_corners.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 5) 정규화 및 왜곡 적용 (fisheye)
    pts_norm = cam_corners[:, :2] / cam_corners[:, 2:3]
    pts_dist = cv2.fisheye.distortPoints(
        pts_norm.reshape(-1,1,2), K, D
    ).reshape(-1,2)

    # 6) 이미지 경계로 클리핑
    img_w, img_h = img_shape
    pts_dist[:,0] = np.clip(pts_dist[:,0], 0, img_w-1)
    pts_dist[:,1] = np.clip(pts_dist[:,1], 0, img_h-1)

    # 7) 2D bbox 계산
    left   = float(np.min(pts_dist[:,0]))
    top    = float(np.min(pts_dist[:,1]))
    right  = float(np.max(pts_dist[:,0]))
    bottom = float(np.max(pts_dist[:,1]))

    return left, top, right, bottom


def load_all_json(root_dir):
    """
    root_dir 하위의 모든 .json 파일을 찾아
    {파일경로: 로드된파이썬객체} 형태의 dict를 반환합니다.
    """
    all_data = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.json'):
                fullpath = os.path.join(dirpath, fname)
                with open(fullpath, 'r', encoding='utf-8') as f:
                    all_data[fullpath] = json.load(f)
    return all_data

def read_calib(calib_path):
    with open(calib_path, 'r') as file:
        config = yaml.safe_load(file)
        camera2lidar = config['camera2lidar']
        rvec = np.array([camera2lidar['rvec_1'],
                        camera2lidar['rvec_2'],
                        camera2lidar['rvec_3']], dtype=np.float32)
        tvec = np.array([camera2lidar['tvec_1'],
                        camera2lidar['tvec_2'],
                        camera2lidar['tvec_3']], dtype=np.float32)
        R, _ = cv2.Rodrigues(rvec)
        Rt = np.hstack([R, tvec.reshape(3, 1)])
        camera = config['camera']
        K = np.array([[camera['fx'], camera['skew'], camera['cx']], 
                      [0, camera['fy'], camera['cy']], 
                       [0, 0, 1]])
        D = np.array([camera['k1'], camera['k2'], camera['k3'], camera['k4']], dtype=np.float32)
    return Rt, K, D

def process_label_original(root_prefix, dir_name):
    """label_original/*.txt (raw KITTI export, z=bbox center) -> label/*.txt (z=bbox bottom center)"""
    full_path = os.path.join(root_prefix, dir_name, "label_original")
    OUT_DIR = os.path.join(root_prefix, dir_name, "label")
    os.makedirs(OUT_DIR, exist_ok=True)
    label_files = glob.glob(os.path.join(full_path, "*.txt"))
    for file_path in label_files:
        modified_lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                height = float(parts[9])
                loc_z = float(parts[13])
                new_loc_z = loc_z - (height / 2.0)
                parts[13] = f"{new_loc_z:.2f}"
                new_line = " ".join(parts) + '\n'
                modified_lines.append(new_line)
            except (ValueError, IndexError) as e:
                print(f"Error processing line in {file_path}: {e}")
                modified_lines.append(line)
        file_name = os.path.basename(file_path)
        with open(os.path.join(OUT_DIR, file_name), 'w') as out_f:
            out_f.writelines(modified_lines)
    print(f"[{dir_name}] label_original -> label converted.")


def process_label_meta(ROOT_PREFIX, dir, label_meta):
        print("start meta..")
        label_path = os.path.join(ROOT_PREFIX, label_meta["label_path"][0])

        CALIB_DIR = os.path.join(ROOT_PREFIX, dir)  # 공통 경로
        new_yaml = os.path.join(CALIB_DIR, "new_lidar.yaml")
        old_yaml = os.path.join(CALIB_DIR, "lidar.yaml")

        if os.path.exists(new_yaml):
            CALIB_PATH = new_yaml
        else:
            print(f"[WARNING] new_lidar.yaml not found. Using fallback: {old_yaml}")
            CALIB_PATH = old_yaml

        IMAGE_DIR = os.path.join(ROOT_PREFIX, dir, "images")
        PCD_DIR = os.path.join(ROOT_PREFIX, dir, "pcd")

        LABEL_OUTPUT_PATH = os.path.join(ROOT_PREFIX, dir, "label")

        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(PCD_DIR, exist_ok=True)
        os.makedirs(LABEL_OUTPUT_PATH, exist_ok=True)

        jpg_list = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        if not jpg_list:
            print(f"No JPG files found in {IMAGE_DIR}")

        pcd_path_list = glob.glob(os.path.join(PCD_DIR, "*.pcd"))
        pcd_list = []
        for path in pcd_path_list:
            filename = os.path.basename(path)  # 1736093112278.avikus.pcd
            name_without_ext = os.path.splitext(filename)[0]  # 1736093112278.avikus
            pcd_list.append(name_without_ext)

        img_path = random.choice(jpg_list)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        len(jpg_list)
        len(pcd_list)

        Rt, K, D = read_calib(CALIB_PATH)

        with open(label_path, 'r', encoding='utf-8') as f:
            label_infos = json.load(f)

        for idx, frame_label_meta in enumerate(label_meta["frame_infos"]):
            pcd_path_name = frame_label_meta["frame_file_name"]

            pcd_name = os.path.basename(pcd_path_name)
            pcd_name = os.path.splitext(pcd_name)[0]
            frame_name = pcd_name.split(".")[0]
            
            frame_number = frame_label_meta["frame_number"]
            # 파일명에 frame_number 또는 timestamp를 사용
            # timestamp = frame_label_meta.get("timestamp", frame_number)
            # timestamp = str(frame_number).zfill(6)
            output_file = os.path.join(LABEL_OUTPUT_PATH, f"{frame_name}.txt")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as out_f:
                for label_info in copy.deepcopy(label_infos["objects"]):
                    tracking_id = label_info["tracking_id"]
                    for frame_label_info in label_info["frames"]:
                        # 해당 프레임의 정보만 처리
                        if frame_label_info["num"] + 1 != frame_number:
                            continue

                        # 0: class 이름
                        class_name = label_info["class_name"]

                        # 1: truncation (float 0.0~1.0)
                        # 2: occlusion (int 0~4)
                        # 3: alpha (float)
                        truncation, occlusion, alpha = 0.0, 0, 0.0
                        # print(frame_label_info["properties"])-------------------------------
                        for prop in frame_label_info["properties"]:
                            if prop["property_name"] == "truncated":
                                truncation = float(prop["option_name"])
                            if prop["property_name"] == "occluded":
                                occlusion = int(prop["option_name"])
                            if prop["property_name"] == "alpha":
                                alpha = float(prop["option_name"])

                        # 4~7: 2D bbox (left, top, right, bottom)
                        left, top, right, bottom = project_bbox_2d(frame_label_info,Rt, K, D, img_shape=[width,height])

                        # 8~10: 3D 치수 (width, height, length)
                        size = frame_label_info["annotation"]["coord"]["size"]
                        width, height, length = size["x"], size["y"], size["z"]

                        # 11~13: 3D 위치 (bbox 밑면 중심)
                        pos = frame_label_info["annotation"]["coord"]["position"]
                        loc_x, loc_y, loc_z = pos["x"], pos["y"], pos["z"]
                        loc_z -= length / 2     # l, w, h

                        # 14: z축 기준 회전 각도 (radian)
                        q = frame_label_info["annotation"]["coord"]["rotation_quaternion"]
                        # yaw = quaternion_to_yaw(q["x"], q["y"], q["z"], q["w"])
                        r = R.from_quat([q["x"], q["y"], q["z"], q["w"]])
                        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

                        # 15: tracking ID
                        track_id = tracking_id
                        # 한 줄 작성
                        out_f.write(
                            f"{class_name} "
                            f"{truncation:.2f} {occlusion} {alpha:.2f} "
                            f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
                            f"{width:.2f} {height:.2f} {length:.2f} "
                            f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
                            f"{yaw:.2f} {track_id}\n"
                        )

            print(f"Saved {output_file}")


def main(args):
    args.data_root = os.path.normpath(args.data_root)
    ROOT_PREFIX = args.data_root

    LABEL_META_PATH = os.path.join(ROOT_PREFIX, "meta/")
    all_label_meta = load_all_json(LABEL_META_PATH) if os.path.isdir(LABEL_META_PATH) else {}
    meta_by_dir = {
        os.path.basename(os.path.normpath(path)).split('.')[0]: label_meta
        for path, label_meta in all_label_meta.items()
    }

    for dir_name in os.listdir(ROOT_PREFIX):
        if not os.path.isdir(os.path.join(ROOT_PREFIX, dir_name)):
            continue

        if os.path.isdir(os.path.join(ROOT_PREFIX, dir_name, "label_original")):
            process_label_original(ROOT_PREFIX, dir_name)
        elif dir_name in meta_by_dir:
            process_label_meta(ROOT_PREFIX, dir_name, meta_by_dir[dir_name])
        else:
            print(f"[SKIP] {dir_name}: no label_original/ and no matching meta/*.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    args = parser.parse_args()
    main(args)
