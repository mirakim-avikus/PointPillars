import argparse
import numpy as np
import os

from pointpillars.dataset import Avikus
from pointpillars.utils import read_points, read_calib, read_label, bbox_avikus2lidar, vis_pc


def vis_gt_pc(root, data_name, id):
    lidar_path = os.path.join(root, data_name, 'pcd', f'{id}.avikus.pcd')
    calib_path = os.path.join(root, data_name, f'calib_{data_name}.txt')
    label_path = os.path.join(root, data_name, 'label', f'{id}.txt')

    lidar_points = read_points(lidar_path)
    calib_dict = read_calib(calib_path)
    annotation_dict = read_label(label_path)

    names = annotation_dict['name']
    dimensions = annotation_dict['dimensions']
    location = annotation_dict['location']
    rotation_y = annotation_dict['rotation_y']

    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
    tr_velo_to_cam = calib_dict['Tr_velo_to_cam']
    r0_rect = calib_dict['R0_rect']
    bboxes_lidar = bbox_avikus2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)

    labels = [Avikus.CLASSES.get(name, -1) for name in names]
    print(f'{len(labels)} labels: {list(zip(names, labels))}')

    vis_pc(lidar_points, bboxes_lidar, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive 3D point cloud + GT bbox viewer for one avikus frame (Open3D only, no cv2 GUI)')
    parser.add_argument('--root', required=True, help='data_root containing <data_name>/ session folders')
    parser.add_argument('--data_name', required=True, help='session name, e.g. batch04')
    parser.add_argument('--id', required=True, help='frame timestamp id, e.g. 1737318282526')
    args = parser.parse_args()

    vis_gt_pc(args.root, args.data_name, args.id)
