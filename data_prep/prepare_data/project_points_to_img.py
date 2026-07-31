import argparse
import glob
import os
import random

import cv2
import numpy as np
import open3d as o3d
import yaml

'''
Sanity check for prepare_data/convert_data.sh: after pcd_flipper.py flips the
point cloud and convert_lidar_rvec.py produces lidar.yaml, project one sampled
pcd's points onto its nearest-timestamp image using that calibration. If the
flip and the calibration agree, the projected points should visibly line up
with the real scene in the image.
'''


def pick_random_sample(root):
    data_names = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, 'pcd'))
        and os.path.isdir(os.path.join(root, d, 'images'))
        and os.path.isfile(os.path.join(root, d, 'lidar.yaml'))
    ]
    if not data_names:
        raise FileNotFoundError(f'no session under {root} has pcd/, images/, and lidar.yaml')
    data_name = random.choice(data_names)

    pcd_paths = glob.glob(os.path.join(root, data_name, 'pcd', '*.avikus.pcd'))
    if not pcd_paths:
        raise FileNotFoundError(f'no pcd files under {root}/{data_name}/pcd')
    pcd_id = os.path.basename(random.choice(pcd_paths)).split('.')[0]

    return data_name, pcd_id


def find_closest_img(images_dir, lidar_ts, max_diff=200):
    image_list = sorted(img for img in os.listdir(images_dir) if img.endswith('.jpg'))
    if not image_list:
        return None
    image_ts_list = [int(name.split('.')[0]) for name in image_list]
    diffs = [abs(ts - lidar_ts) for ts in image_ts_list]
    min_idx = min(range(len(diffs)), key=diffs.__getitem__)
    if diffs[min_idx] > max_diff:
        return None
    return os.path.join(images_dir, image_list[min_idx])


def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        config = yaml.safe_load(f)
    camera2lidar = config['camera2lidar']
    rvec = np.array([camera2lidar['rvec_1'], camera2lidar['rvec_2'], camera2lidar['rvec_3']], dtype=np.float32)
    tvec = np.array([camera2lidar['tvec_1'], camera2lidar['tvec_2'], camera2lidar['tvec_3']], dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    # Despite the yaml key name, this rvec/tvec is lidar -> camera, not
    # camera -> lidar (matches generate_annos/generate_calib.py's identical
    # usage, commented "avikus2camera" there).
    Rt = np.hstack([R, tvec.reshape(3, 1)])

    camera = config['camera']
    K = np.array([[camera['fx'], camera['skew'], camera['cx']],
                  [0, camera['fy'], camera['cy']],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([camera['k1'], camera['k2'], camera['k3'], camera['k4']], dtype=np.float32)
    return Rt, K, D


def project_points_to_img(lidar_points, img, Rt, K, D):
    lidar_hom = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
    cam_points = (Rt @ lidar_hom.T).T
    cam_points = cam_points[cam_points[:, 2] > 0]

    # color by distance, matching generate_annos/generate_superb_label.py's
    # project_lidar_to_image
    distance = np.linalg.norm(cam_points, axis=1)
    hsv = np.zeros((len(distance), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.clip((distance / 500.0) * 180.0, 0, 179)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)

    points_normalized = cam_points[:, :2] / cam_points[:, 2:]
    points_distorted = cv2.fisheye.distortPoints(points_normalized.reshape(-1, 1, 2), K, D).reshape(-1, 2)

    mask = (points_distorted[:, 0] >= 0) & (points_distorted[:, 0] < img.shape[1]) & \
           (points_distorted[:, 1] >= 0) & (points_distorted[:, 1] < img.shape[0])

    for pt, color in zip(points_distorted[mask], bgr[mask]):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (int(color[0]), int(color[1]), int(color[2])), -1)

    return img


def sanity_check(root, data_name, pcd_id):
    dataset_dir = os.path.join(root, data_name)
    pcd_path = os.path.join(dataset_dir, 'pcd', f'{pcd_id}.avikus.pcd')
    if not os.path.exists(pcd_path):
        print(f'[project_points_to_img.py] {pcd_path} not found, skip')
        return

    img_path = find_closest_img(os.path.join(dataset_dir, 'images'), int(pcd_id))
    if img_path is None:
        print(f'[project_points_to_img.py] no image within 200ms of {pcd_id}, skip')
        return

    calib_path = os.path.join(dataset_dir, 'lidar.yaml')
    if not os.path.exists(calib_path):
        print(f'[project_points_to_img.py] no lidar.yaml under {dataset_dir}, skip')
        return
    Rt, K, D = read_calib(calib_path)

    lidar_points = np.asarray(o3d.io.read_point_cloud(pcd_path).points)
    img = cv2.imread(img_path)
    img = project_points_to_img(lidar_points, img, Rt, K, D)

    print(f'[project_points_to_img.py] {data_name}/{os.path.basename(pcd_path)} -> {os.path.basename(img_path)}')
    cv2.imshow('lidar -> image projection sanity check (press any key to close)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sanity-check flipped pcd/ + converted lidar.yaml by projecting one sampled point cloud onto its nearest image')
    parser.add_argument('--root', required=True, help='data_root containing <data_name>/ session folders')
    parser.add_argument('--data_name', default=None, help='session name (random if omitted)')
    parser.add_argument('--id', default=None, help='pcd timestamp id (random if omitted)')
    args = parser.parse_args()

    data_name, pcd_id = args.data_name, args.id
    if data_name is None or pcd_id is None:
        picked_data_name, picked_id = pick_random_sample(args.root)
        data_name = data_name or picked_data_name
        pcd_id = pcd_id or picked_id
        print(f'[project_points_to_img.py] no --data_name/--id given, randomly picked: {data_name} / {pcd_id}')

    sanity_check(args.root, data_name, pcd_id)
