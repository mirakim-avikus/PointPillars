import copy
import numba
import numpy as np
import os
import pdb
from pointpillars.utils import bbox3d2bevcorners, box_collision_test, read_points, \
    remove_pts_in_bboxes, limit_period

@numba.jit(nopython=True)
def get_occupancy_map(points, resolution, min_angle=-np.pi, max_angle=np.pi):
    """
    각도별 최소 거리(r_min)를 계산하여 Occupancy Map 생성
    resolution: Lidar 수평 해상도 (radian 단위, 예: 0.1도 -> np.deg2rad(0.1))
    """
    num_bins = int(np.ceil((max_angle - min_angle) / resolution))
    # 초기값은 아주 먼 거리로 설정
    occ_map = np.full(num_bins, 1e6, dtype=np.float32)
    
    for i in range(len(points)):
        x, y = points[i, 0], points[i, 1]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        
        # 각도를 bin 인덱스로 변환
        bin_idx = int((phi - min_angle) / resolution)
        if 0 <= bin_idx < num_bins:
            if r < occ_map[bin_idx]:
                occ_map[bin_idx] = r
                
    return occ_map

@numba.jit(nopython=True)
def is_sample_visible_strict(box_3d, occ_map, resolution, fov_deg=120.0):
    x, y, z, l, w, h, r = box_3d
    cos_r, sin_r = np.cos(r), np.sin(r)
    half_fov_rad = np.deg2rad(fov_deg / 2.0)

    # 1. 테두리를 따라 샘플링할 지점 개수 결정 (1m 간격 권장)
    n_l = int(np.ceil(l)) + 1 # 길이 방향 샘플 수
    n_w = int(np.ceil(w)) + 1 # 폭 방향 샘플 수
    total_points = (n_l * 2) + (n_w * 2)
    
    check_points = np.empty((total_points, 2), dtype=np.float32)
    idx = 0

    # 2. 테두리 점 생성 (박스 로컬 좌표 기준)
    # 좌/우 긴 변
    for i in range(n_l):
        cur_l = -l/2 + (l / (n_l - 1)) * i
        for cur_w in [-w/2, w/2]:
            check_points[idx, 0] = cur_l * cos_r - cur_w * sin_r + x
            check_points[idx, 1] = cur_l * sin_r + cur_w * cos_r + y
            idx += 1
    # 앞/뒤 짧은 변
    for i in range(1, n_w - 1): # 모서리 중복 제외
        cur_w = -w/2 + (w / (n_w - 1)) * i
        for cur_l in [-l/2, l/2]:
            check_points[idx, 0] = cur_l * cos_r - cur_w * sin_r + x
            check_points[idx, 1] = cur_l * sin_r + cur_w * cos_r + y
            idx += 1

    # 3. 모든 테두리 점에 대해 가시성 검사
    for i in range(idx):
        px, py = check_points[i, 0], check_points[i, 1]
        r_p = np.sqrt(px**2 + py**2)
        phi_p = np.arctan2(py, px)

        if phi_p < -half_fov_rad or phi_p > half_fov_rad:
            continue
            
        bin_idx = int((phi_p - (-half_fov_rad)) / resolution)
        if bin_idx < 0 or bin_idx >= len(occ_map):
            continue
            
        # [핵심] 마진을 0.5m로 줄여서 장애물과 조금만 겹쳐도 바로 탈락시킴
        if occ_map[bin_idx] < 1000 and r_p > (occ_map[bin_idx] - 1.0):
            # r_p가 장애물 거리(occ_map)보다 뒤에 있으면(가려지면) 탈락
            # 안전을 위해 장애물보다 1m 앞까지만 허용
            return False
            
    return True

def filter_samples_by_visibility(sampled_list, pts, res_deg=0.1):
    """
    dbsample 내부에서 호출할 래퍼 함수
    """
    res_rad = np.deg2rad(res_deg)
    half_fov = np.deg2rad(60) 
    occ_map = get_occupancy_map(pts, res_rad, min_angle=-half_fov, max_angle=half_fov)
    
    keep_list = []
    for sample in sampled_list:
        if is_sample_visible_strict(sample['box3d_lidar'], occ_map, res_rad):
            keep_list.append(sample)
    return keep_list

def dbsample(CLASSES, data_root, data_dict, db_sampler, sample_groups):
    '''
    CLASSES: dict(Pedestrian=0, Cyclist=1, Car=2)
    data_root: str, data root
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    db_infos: dict(Pedestrian, Cyclist, Car, ...)
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
    gt_difficulty = data_dict['difficulty']
    image_info, calib_info, pcd_info = data_dict['image_info'], data_dict['calib_info'], data_dict['pcd_info']
    location, dimension, rotation_y = data_dict['location'], data_dict['dimension'], data_dict['rotation_y']

    sampled_pts, sampled_names, sampled_labels = [], [], []
    sampled_bboxes, sampled_difficulty = [], []

    avoid_coll_boxes = copy.deepcopy(gt_bboxes_3d)
    for name, v in sample_groups.items():
        # 0. skip class not in GT db
        if name not in db_sampler:
            continue
        # 1. calculate sample numbers
        sampled_num = v - np.sum(gt_names == name)
        if sampled_num <= 0:
            continue

        # 2. sample databases bboxes
        sampled_cls_list = db_sampler[name].sample(sampled_num)

        # 2.5 Space Filtering (극좌표계/Alpha Sample 활용)
        sampled_cls_list = filter_samples_by_visibility(sampled_cls_list, pts, res_deg=0.2)
        if len(sampled_cls_list) == 0:
            continue
        sampled_cls_bboxes = np.array([item['box3d_lidar'] for item in sampled_cls_list], dtype=np.float32)

        # 3. box_collision_test
        avoid_coll_boxes_bv_corners = bbox3d2bevcorners(avoid_coll_boxes)
        sampled_cls_bboxes_bv_corners = bbox3d2bevcorners(sampled_cls_bboxes)
        coll_query_matrix = np.concatenate([avoid_coll_boxes_bv_corners, sampled_cls_bboxes_bv_corners], axis=0)
        coll_mat = box_collision_test(coll_query_matrix, coll_query_matrix)
        n_gt, tmp_bboxes = len(avoid_coll_boxes_bv_corners), []
        for i in range(n_gt, len(coll_mat)):
            if any(coll_mat[i]):
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                cur_sample = sampled_cls_list[i - n_gt]
                pt_path = os.path.join(data_root, cur_sample['path'])
                sampled_pts_cur = read_points(pt_path)
                sampled_pts_cur[:, :3] += cur_sample['box3d_lidar'][:3]
                sampled_pts.append(sampled_pts_cur)
                sampled_names.append(cur_sample['name'])
                sampled_labels.append(CLASSES[cur_sample['name']])
                sampled_bboxes.append(cur_sample['box3d_lidar'])
                tmp_bboxes.append(cur_sample['box3d_lidar'])
                sampled_difficulty.append(cur_sample['difficulty'])
        if len(tmp_bboxes) == 0:
            tmp_bboxes = np.array(tmp_bboxes).reshape(-1, 7)
        else:
            tmp_bboxes = np.array(tmp_bboxes)
        avoid_coll_boxes = np.concatenate([avoid_coll_boxes, tmp_bboxes], axis=0)
        
    # merge sampled database
    # remove raw points in sampled_bboxes firstly
    if len(sampled_bboxes) > 0:
        pts = remove_pts_in_bboxes(pts, np.stack(sampled_bboxes, axis=0))
        # pts = np.concatenate([pts, np.concatenate(sampled_pts, axis=0)], axis=0)
        pts = np.concatenate([np.concatenate(sampled_pts, axis=0), pts], axis=0)
        gt_labels = np.concatenate([gt_labels, np.array(sampled_labels)], axis=0)
        gt_names = np.concatenate([gt_names, np.array(sampled_names)], axis=0)
        difficulty = np.concatenate([gt_difficulty, np.array(sampled_difficulty)], axis=0)
    else:
        difficulty = gt_difficulty
    gt_bboxes_3d = avoid_coll_boxes.astype(np.float32)
    data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels, 
            'gt_names': gt_names,
            'difficulty': difficulty,
            'image_info': image_info,
            'calib_info': calib_info,
            'pcd_info': pcd_info,
            'location': location,
            'dimension': dimension,
            'rotation_y': rotation_y
        }
    return data_dict


@numba.jit(nopython=True, parallel=True)
def object_noise_core(pts, gt_bboxes_3d, bev_corners, trans_vec, rot_angle, rot_mat, masks):
    """
    최적화 포인트:
    1. Python 딕셔너리(visit) 제거 -> Numba 최적화 가속
    2. 중첩 루프 구조 개선 -> 포인트 업데이트를 행렬 연산화
    3. prange를 활용한 병렬 처리 적용
    """
    n_bbox, num_try = trans_vec.shape[:2]
    num_pts = pts.shape[0]
    
    # 1. 충돌 테스트 및 성공 마스크 계산 (기존 로직 유지하되 최적화)
    succ_mask = -np.ones(n_bbox, dtype=np.int32)
    for i in range(n_bbox):
        for j in range(num_try):
            # 로컬 좌표계 변환 및 회전/이동
            rel_corners = bev_corners[i] - gt_bboxes_3d[i, :2]
            rot = rot_mat[i, j].copy()
            trans = trans_vec[i, j, :2]
            
            # (4, 2) @ (2, 2) + (2,)
            cur_bbox = (rel_corners @ rot) + gt_bboxes_3d[i, :2] + trans
            
            coll_mat = box_collision_test(np.expand_dims(cur_bbox, 0), bev_corners)
            coll_mat[0, i] = False
            
            if not coll_mat.any():
                bev_corners[i] = cur_bbox
                succ_mask[i] = j
                break

    # 2. 포인트 업데이트 (핵심 최적화 구간)
    # 각 포인트가 어떤 박스에 속해 있고, 노이즈 적용이 성공했는지 미리 계산
    # point_to_bbox: 각 포인트가 노이즈가 적용될 박스 인덱스를 저장 (-1이면 적용 안 함)
    point_to_bbox = -np.ones(num_pts, dtype=np.int32)
    for i in range(n_bbox):
        if succ_mask[i] != -1:
            for k in range(num_pts):
                if masks[k, i]:
                    # 중복 방지를 위해 첫 번째 발견된 박스만 적용 (기존 visit 로직과 동일)
                    if point_to_bbox[k] == -1:
                        point_to_bbox[k] = i

    # prange를 사용하여 포인트 변환 병렬화
    for k in numba.prange(num_pts):
        bbox_idx = point_to_bbox[k]
        if bbox_idx != -1:
            i = bbox_idx
            jj = succ_mask[i]
            
            # 변환 값 추출
            c_trans = trans_vec[i, jj]
            c_rot = rot_mat[i, jj].copy()
            center = gt_bboxes_3d[i, :3]
            
            # 포인트 변환: 중심 기준 회전 후 이동
            # pts[k, :2] 회전
            rel_pt = pts[k, :2] - center[:2]
            new_xy = (rel_pt @ c_rot) + center[:2] + c_trans[:2]
            
            pts[k, 0] = new_xy[0]
            pts[k, 1] = new_xy[1]
            pts[k, 2] = pts[k, 2] + c_trans[2] # Z축 이동

    # 3. BBox 정보 업데이트
    for i in range(n_bbox):
        jj = succ_mask[i]
        if jj != -1:
            gt_bboxes_3d[i, :3] += trans_vec[i, jj]
            gt_bboxes_3d[i, 6] += rot_angle[i, jj]

    return gt_bboxes_3d, pts


def object_noise(data_dict, num_try, translation_std, rot_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    num_try: int, 100
    translation_std: shape=[3, ]
    rot_range: shape=[2, ]
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    n_bbox = len(gt_bboxes_3d)
    
    # 1. generate rotation vectors and rotation matrices
    trans_vec = np.random.normal(scale=translation_std, size=(n_bbox, num_try, 3)).astype(np.float32)
    rot_angle = np.random.uniform(rot_range[0], rot_range[1], size=(n_bbox, num_try)).astype(np.float32)
    rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
    # in fact, - rot_angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2, n_bbox, num_try)
    rot_mat = np.transpose(rot_mat, (2, 3, 1, 0)) # (n_bbox, num_try, 2, 2)
    
    # 2. generate noise for each bbox and the points inside the bbox.
    bev_corners = bbox3d2bevcorners(gt_bboxes_3d) # (n_bbox, 4, 2) # for collision test
    masks = remove_pts_in_bboxes(pts, gt_bboxes_3d, rm=False) # identify which point should be added noise
    gt_bboxes_3d, pts = object_noise_core(pts=pts, 
                                          gt_bboxes_3d=gt_bboxes_3d, 
                                          bev_corners=bev_corners, 
                                          trans_vec=trans_vec, 
                                          rot_angle=rot_angle, 
                                          rot_mat=rot_mat, 
                                          masks=masks)
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'pts': pts})

    return data_dict


def random_flip(data_dict, random_flip_ratio):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    random_flip_ratio: float, 0-1
    return: data_dict
    '''
    random_flip_state = np.random.choice([True, False], p=[random_flip_ratio, 1-random_flip_ratio])
    if random_flip_state:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        pts[:, 1] = -pts[:, 1] 
        gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
        gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] + np.pi
        data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
        data_dict.update({'pts': pts})
    return data_dict


def global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    rot_range: [a, b]
    scale_ratio_range: [c, d] 
    translation_std:  [e, f, g]
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    
    # 1. rotation
    rot_angle = np.random.uniform(rot_range[0], rot_range[1])
    rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
    # in fact, - rot_angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2)
    # 1.1 bbox rotation
    gt_bboxes_3d[:, :2] = gt_bboxes_3d[:, :2] @ rot_mat.T
    gt_bboxes_3d[:, 6] += rot_angle
    # 1.2 point rotation
    pts[:, :2] = pts[:, :2] @ rot_mat.T

    # 2. scaling
    scale_fator = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])
    gt_bboxes_3d[:, :6] *= scale_fator
    pts[:, :3] *= scale_fator

    # 3. translation
    trans_factor = np.random.normal(scale=translation_std, size=(1, 3))
    gt_bboxes_3d[:, :3] += trans_factor
    pts[:, :3] += trans_factor
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'pts': pts})
    return data_dict


# shared lidar point-cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
POINT_CLOUD_RANGE = [5, -72., -10., 180., 72., 30.]


def point_range_filter(data_dict, point_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    pts = data_dict['pts']
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    data_dict.update({'pts': pts})
    return data_dict 


def object_range_filter(data_dict, object_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']

    # bev filter
    flag_x_low = gt_bboxes_3d[:, 0] > object_range[0]
    flag_y_low = gt_bboxes_3d[:, 1] > object_range[1]
    flag_x_high = gt_bboxes_3d[:, 0] < object_range[3]
    flag_y_high = gt_bboxes_3d[:, 1] < object_range[4]
    keep_mask = flag_x_low & flag_y_low & flag_x_high & flag_y_high

    gt_bboxes_3d, gt_labels = gt_bboxes_3d[keep_mask], gt_labels[keep_mask]
    gt_names, difficulty = gt_names[keep_mask], difficulty[keep_mask]
    gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], 0.5, 2 * np.pi)
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'gt_labels': gt_labels})
    data_dict.update({'gt_names': gt_names})
    data_dict.update({'difficulty': difficulty})
    return data_dict


def points_shuffle(data_dict):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    '''
    pts = data_dict['pts']
    indices = np.arange(0, len(pts))
    np.random.shuffle(indices)
    pts = pts[indices]
    data_dict.update({'pts': pts})
    return data_dict


def filter_bboxes_with_labels(data_dict, label=-1):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    label: int
    '''
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']
    idx = gt_labels != label
    gt_bboxes_3d = gt_bboxes_3d[idx]
    gt_labels = gt_labels[idx]
    gt_names = gt_names[idx]
    difficulty = difficulty[idx]
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'gt_labels': gt_labels})
    data_dict.update({'gt_names': gt_names})
    data_dict.update({'difficulty': difficulty})
    return data_dict


def data_augment(CLASSES, data_root, data_dict, data_aug_config):
    '''
    CLASSES: dict(Pedestrian=0, Cyclist=1, Car=2)
    data_root: str, data root
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    data_aug_config: dict()
    return: data_dict
    '''

    # 1. sample databases and merge into the data 
    db_sampler_config = data_aug_config['db_sampler']
    data_dict = dbsample(CLASSES,
                         data_root,
                         data_dict, 
                         db_sampler=db_sampler_config['db_sampler'],
                         sample_groups=db_sampler_config['sample_groups'])
    # 2. object noise
    object_noise_config = data_aug_config['object_noise']
    data_dict = object_noise(data_dict, 
                             num_try=object_noise_config['num_try'],
                             translation_std=object_noise_config['translation_std'],
                             rot_range=object_noise_config['rot_range'])
    
    # 3. random flip
    random_flip_ratio = data_aug_config['random_flip_ratio']
    data_dict = random_flip(data_dict, random_flip_ratio)

    # 4. global rotation, scaling and translation
    global_rot_scale_trans_config = data_aug_config['global_rot_scale_trans']
    rot_range = global_rot_scale_trans_config['rot_range']
    scale_ratio_range = global_rot_scale_trans_config['scale_ratio_range']
    translation_std = global_rot_scale_trans_config['translation_std']
    data_dict = global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std)

    # 5. points range filter
    point_range = data_aug_config['point_range_filter']
    data_dict = point_range_filter(data_dict, point_range)

    # 6. object range filter
    object_range = data_aug_config['object_range_filter']
    data_dict = object_range_filter(data_dict, object_range)

    # 7. points shuffle
    data_dict = points_shuffle(data_dict)

    # 8. filter bboxes with label=-1
    data_dict = filter_bboxes_with_labels(data_dict)
    
    return data_dict
