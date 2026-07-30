import argparse
import numpy as np
import os

from pointpillars.dataset import Avikus
from pointpillars.utils import read_points, read_label, vis_pc


def vis_gt_pc(root, data_name, id):
    lidar_path = os.path.join(root, data_name, 'pcd', f'{id}.avikus.pcd')
    label_path = os.path.join(root, data_name, 'label', f'{id}.txt')

    lidar_points = read_points(lidar_path)
    annotation_dict = read_label(label_path)

    names = annotation_dict['name']
    dimensions = annotation_dict['dimensions']
    location = annotation_dict['location']
    rotation_y = annotation_dict['rotation_y']

    # location/dimensions/rotation_y are already in lidar coordinates for this
    # label format - same as Avikus.__getitem__ builds gt_bboxes_3d for
    # training, with no camera->lidar transform applied.
    bboxes_lidar = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1).astype(np.float32)

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
