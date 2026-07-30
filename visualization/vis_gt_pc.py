import argparse
import numpy as np
import os
import random

from pointpillars.dataset import Avikus
from pointpillars.utils import read_points, read_label, vis_pc

EXCLUDE_DIRS = ['avikus_gt_database', 'testing', 'training', 'labels', 'meta', 'tmp']


def pick_random_sample(root):
    data_names = [
        d for d in os.listdir(root)
        if d not in EXCLUDE_DIRS
        and os.path.isdir(os.path.join(root, d, 'pcd'))
        and os.path.isdir(os.path.join(root, d, 'label'))
    ]
    if not data_names:
        raise FileNotFoundError(f'no session under {root} has both pcd/ and label/')
    data_name = random.choice(data_names)

    label_dir = os.path.join(root, data_name, 'label')
    ids = [f[:-len('.txt')] for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not ids:
        raise FileNotFoundError(f'no label files under {label_dir}')

    return data_name, random.choice(ids)


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
    parser.add_argument('--data_name', default=None, help='session name, e.g. batch04 (random if omitted)')
    parser.add_argument('--id', default=None, help='frame timestamp id, e.g. 1737318282526 (random if omitted)')
    args = parser.parse_args()

    data_name, id = args.data_name, args.id
    if data_name is None or id is None:
        picked_data_name, picked_id = pick_random_sample(args.root)
        data_name = data_name or picked_data_name
        id = id or picked_id
        print(f'[vis_gt_pc.py] no --data_name/--id given, randomly picked: {data_name} / {id}')

    vis_gt_pc(args.root, data_name, id)
