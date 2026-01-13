import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from pointpillars.utils import read_pickle, read_points, bbox_camera2lidar, bbox_avikus2lidar
from pointpillars.dataset import point_range_filter, object_range_filter, data_augment

class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Avikus(Dataset):

    CLASSES = {
        'jetski': 0,
        'smallboat': 1,
        'mediumboat': 2,
        'c-marker': 3,
        'yacht': 4,
        'pole': 5,
        'dinghyboat': 6,
        'bigboat': 7,
        'bridgepillar' : 8,
        'buoy' : 9
        }

    # TODO : check IoU in 2D image detection domain 
    IOU_THRESHOLDS = {
        'jetski': 0.5,
        'smallboat': 0.5,
        'mediumboat': 0.6,
        'c-marker': 0.5,
        'yacht': 0.6,
        'pole': 0.4,
        'dinghyboat': 0.5,
        'bigboat': 0.6,
        'bridgepillar' : 0.5,
        'buoy' : 0.4,
    }

    def __init__(self, data_root, split, point_cloud_range, pts_prefix='velodyne_reduced', min_pts_after_filter=10):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        self.min_pts_after_filter = min_pts_after_filter
        self.data_infos = read_pickle(os.path.join(data_root, f'avikus_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())
        self.valid_ids = self._build_valid_ids(point_cloud_range)

        db_infos = read_pickle(os.path.join(data_root, 'avikus_dbinfos_train.pkl'))
        db_infos = self.filter_db(db_infos)

        print(
            f"[Avikus] split={split} "
            f"total={len(self.sorted_ids)} → valid={len(self.valid_ids)} "
            f"(min_pts_after_filter={self.min_pts_after_filter})"
        )

        db_sampler = {}
        for cat_name in self.CLASSES:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config=dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(jetski=3, smallboat=2, mediumboat=2, **{"c-marker": 2}, yacht=2, pole=3, dinghyboat=3, bigboat=3, bridgepillar=2, buoy=3)
                ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            point_range_filter=point_cloud_range,
            object_range_filter=point_cloud_range         
        )

    def _build_valid_ids(self, point_cloud_range):
        valid_ids = []

        for k in tqdm(self.sorted_ids, desc=f"[{self.split}] filtering valid ids"):
            info = self.data_infos[k]

            velodyne_path = info['velodyne_path'].replace('velodyne', self.pts_prefix)
            pts_path = os.path.join(self.data_root, velodyne_path)

            try:
                pts = read_points(pts_path)
            except Exception:
                continue

            if pts is None or pts.shape[0] == 0:
                continue

            pts[:, -1] = pts[:, -1] / 255.0

            tmp_dict = {'pts': pts}
            tmp_dict = point_range_filter(tmp_dict, point_range=point_cloud_range)

            if tmp_dict['pts'].shape[0] >= self.min_pts_after_filter:
                valid_ids.append(k)

        return valid_ids

    def filter_by_occ_trunc(self, annos_info, occ_thr=2, trunc_thr=1):
        occluded = annos_info['occluded']
        truncated = annos_info['truncated']
        keep_ids = (occluded <= occ_thr) & (truncated <= trunc_thr)
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]

        filter_thrs = dict(jetski=10, smallboat=10, mediumboat=20, **{"c-marker": 10}, yacht=20, pole=8, dinghyboat=10, bigboat=40, bridgepillar=20, buoy=5)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt']>= filter_thr]
        
        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.valid_ids[index]]
        image_info, calib_info, annos_info = \
            data_info['image'], data_info['calib'], data_info['annos']
    
        # point cloud input
        velodyne_path = data_info['velodyne_path'].replace('velodyne', self.pts_prefix)
        pts_path = os.path.join(self.data_root, velodyne_path)
        pts = read_points(pts_path)
        pts[:, -1] = pts[:, -1] / 255.0

        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        self.r0_rect = calib_info['R0_rect'].astype(np.float32)

        # annotations input
        annos_info = self.filter_by_occ_trunc(annos_info, occ_thr=1, trunc_thr=1)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']

        rotation_y = annos_info['rotation_y']
        gt_bboxes_3d = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels), 
            'gt_names': annos_name,
            'difficulty': np.atleast_1d(annos_info['difficulty']),
            'image_info': image_info,
            'calib_info': calib_info,
            'location': annos_location,
            'dimension': annos_dimension,
            'rotation_y': rotation_y
        }
        # if self.split in ['train', 'trainval']:
        #     data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
        # else:
        data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])

        return data_dict

    def __len__(self):
        return len(self.valid_ids)
 

if __name__ == '__main__':
    
    kitti_data = Kitti(data_root='/mnt/ssd1/lifa_rdata/det/kitti', 
                       split='train')
    kitti_data.__getitem__(9)
