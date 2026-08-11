import argparse
import cv2
import numpy as np
import os
import sys
import onnx
import onnxruntime
import time
import torch
from pytorch2onnx import CLASSES, POINT_CLOUD_RANGE, VOXEL_SIZE

SAVE_BIN = True
DURATION = False
RAW_POINTS = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE, '../')))

from pointpillars.utils import read_points, keep_bbox_from_lidar_range, vis_pc
from pointpillars.model import PointPillarsPre, PointPillarsPos

def pillars_to_bev_rgb(
    pillars: torch.Tensor,              # (P, 32, 4)
    coors: torch.Tensor,                # (P, 3) or (P, 2)
    npoints: torch.Tensor,              # (P,)
    pc_range,
    voxel_size,
    max_density=32
):
    """
    coord:
      x+ forward, y+ right, z+ down
    BEV:
      up = forward, right = right
    """

    pillars = pillars.cpu().numpy()
    coors = coors.cpu().numpy()
    npoints = npoints.cpu().numpy()

    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    vx, vy, vz = voxel_size

    H = int((x_max - x_min) / vx)
    W = int((y_max - y_min) / vy)

    bev_h = np.zeros((H, W), dtype=np.float32)
    bev_d = np.zeros((H, W), dtype=np.float32)
    bev_i = np.zeros((H, W), dtype=np.float32)

    for i in range(pillars.shape[0]):
        x_idx = coors[i][0]
        y_idx = coors[i][1]

        # 시각화 좌표계
        row = H - 1 - x_idx
        col = y_idx

        pts = pillars[i, :npoints[i]]   # (Ni, 4)

        if pts.shape[0] == 0:
            continue

        # height (z down → -z up)
        height = (-pts[:, 2]).max()

        # density
        density = min(npoints[i], max_density) / max_density

        # intensity
        intensity = pts[:, 3].max()

        bev_h[row, col] = max(bev_h[row, col], height)
        bev_d[row, col] = max(bev_d[row, col], density)
        bev_i[row, col] = max(bev_i[row, col], intensity)

    # bev_h = np.clip((bev_h - h_min) / (h_max - h_min + 1e-6), 0, 1)
    v = bev_h[np.isfinite(bev_h)]
    if v.size > 0:
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        bev_h = np.clip((bev_h - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        bev_h[:] = 0
        
    # normalize intensity (robust)
    v = bev_i[bev_i > 0]
    if v.size > 0:
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        bev_i = np.clip((bev_i - lo) / (hi - lo + 1e-6), 0, 1)

    bev_rgb = np.stack([
        (bev_h * 255).astype(np.uint8),   # R
        (bev_d * 255).astype(np.uint8),   # G
        (bev_i * 255).astype(np.uint8)    # B
    ], axis=2)

    return bev_rgb


def make_bev(points: np.ndarray,
             pc_range=POINT_CLOUD_RANGE,
             resolution=0.2,
             mode="density",
             max_density=64):
    """
    points: (N,3) or (N,4) [x,y,z,(intensity)]
      coord: x forward(+), y right(+), z down(+)

    pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    resolution: meters per pixel
    mode: density | height | intensity | rgb
    """
    assert points.ndim == 2 and points.shape[1] in (3, 4)

    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    z = points[:, 2].astype(np.float32)

    # (이미 필터되어있어도) 안전 마스크
    m = (
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max) &
        (z >= z_min) & (z < z_max)
    )
    pts = points[m]
    if pts.shape[0] == 0:
        H = int(np.ceil((x_max - x_min) / resolution))
        W = int(np.ceil((y_max - y_min) / resolution))
        return np.zeros((H, W), dtype=np.uint8)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # grid size: row=x, col=y
    H = int(np.ceil((x_max - x_min) / resolution))
    W = int(np.ceil((y_max - y_min) / resolution))

    gx = np.floor((x - x_min) / resolution).astype(np.int32)  # 0..H-1
    gy = np.floor((y - y_min) / resolution).astype(np.int32)  # 0..W-1
    gx = np.clip(gx, 0, H - 1)
    gy = np.clip(gy, 0, W - 1)

    # 화면 좌표:
    # 위쪽=전방(x+) => row = H-1-gx
    # 오른쪽=우현(y+) => col = gy (뒤집지 않음!)
    row = (H - 1) - gx
    col = gy

    lin = row * W + col  # linear index

    def _density():
        counts = np.bincount(lin, minlength=H * W).astype(np.float32).reshape(H, W)
        counts = np.minimum(counts, float(max_density))
        img = counts / float(max_density)
        return (img * 255.0).astype(np.uint8)

    def _height():
        # z down(+) => "위로"는 -z
        height = (-z).astype(np.float32)  # larger = higher
        hmap = np.full((H, W), -np.inf, dtype=np.float32)
        np.maximum.at(hmap.reshape(-1), lin, height)

        # normalize using -z range
        h_min = -z_max
        h_max = -z_min
        denom = max(h_max - h_min, 1e-6)
        hmap = np.where(np.isfinite(hmap), hmap, h_min)
        hmap = np.clip((hmap - h_min) / denom, 0.0, 1.0)
        return (hmap * 255.0).astype(np.uint8)

    def _intensity():
        if pts.shape[1] != 4:
            raise ValueError("intensity mode requires points shape (N,4).")
        inten = pts[:, 3].astype(np.float32)
        imap = np.zeros((H, W), dtype=np.float32)
        np.maximum.at(imap.reshape(-1), lin, inten)

        v = imap[imap > 0]
        if v.size > 0:
            lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            denom = max(hi - lo, 1e-6)
            imap = np.clip((imap - lo) / denom, 0.0, 1.0)
        else:
            imap[:] = 0
        return (imap * 255.0).astype(np.uint8)

    if mode == "density":
        return _density()
    elif mode == "height":
        return _height()
    elif mode == "intensity":
        return _intensity()
    elif mode == "rgb":
        d = _density()
        h = _height()
        it = _intensity() if pts.shape[1] == 4 else np.zeros((H, W), dtype=np.uint8)
        # R=height, G=density, B=intensity
        return np.stack([h, d, it], axis=2)
    else:
        raise ValueError("mode must be one of: density, height, intensity, rgb")

def _save_shape_txt(path, shape_tuple):
    with open(path, 'w') as f:
        f.write(' '.join(str(int(s)) for s in shape_tuple))

def save_bins(pillars, coors_batch, npoints_per_pillar, dump_dir):
    # 1) pillars -> float32 (flatten)
    pillars_np = to_numpy(pillars).astype(np.float32, copy=False).ravel(order='C')
    pillars_bin = os.path.join(dump_dir, 'pillars.bin')
    pillars_shape = os.path.join(dump_dir, 'pillars.shape.txt')
    pillars_np.tofile(pillars_bin)
    _save_shape_txt(pillars_shape, to_numpy(pillars).shape)

    # 2) coors_batch -> int64 (flatten)  [ex: (num_pillars, 4) = [batch, z, y, x] or [b,x,y,z]]
    coors_np = to_numpy(coors_batch).astype(np.int64, copy=False).ravel(order='C')
    coors_bin = os.path.join(dump_dir, 'voxel_indices.bin')
    coors_shape = os.path.join(dump_dir, 'voxel_indices.shape.txt')
    coors_np.tofile(coors_bin)
    _save_shape_txt(coors_shape, to_numpy(coors_batch).shape)

    # 3) npoints_per_pillar -> int32 (flatten) [ex: (num_pillars,)]
    npoints_np = to_numpy(npoints_per_pillar).astype(np.int32, copy=False).ravel(order='C')
    npoints_bin = os.path.join(dump_dir, 'point_counts.bin')
    npoints_shape = os.path.join(dump_dir, 'point_counts.shape.txt')
    npoints_np.tofile(npoints_bin)
    _save_shape_txt(npoints_shape, to_numpy(npoints_per_pillar).shape)

    print(f"[Dump] saved to: {dump_dir}")
    print(f"  - pillars: {pillars_np.shape} -> {pillars_bin}")
    print(f"  - voxel_indices: {coors_np.shape} -> {coors_bin}")
    print(f"  - point_counts: {npoints_np.shape} -> {npoints_bin}")
    return

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main(args):
    prefix = args.prefix
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

    dump_dir = getattr(args, 'dump_dir', 'dump_pre_io')
    os.makedirs(dump_dir, exist_ok=True)

    ## 1.1 onnx check and onnx load
    try:
        # 当我们的模型不可用时，将会报出异常
        onnx.checker.check_model(args.onnx_path)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        # 模型可用时，将不会报出异常，并会输出“The model is valid!”
        print("The model is valid!")

    # sess = onnxruntime.InferenceSession('checkpoint/model.onnx')
    print('onnx version: ', onnxruntime.get_device())
    sess = onnxruntime.InferenceSession(args.onnx_path, providers=['CUDAExecutionProvider'])
    print(sess.get_providers())
    input_pillars_name = sess.get_inputs()[0].name
    input_coors_batch_name = sess.get_inputs()[1].name
    input_npoints_per_pillar_name = sess.get_inputs()[2].name
    output_name = sess.get_inputs()[0].name

    if not args.no_cuda:
        model_pre = PointPillarsPre(voxel_size=VOXEL_SIZE, point_cloud_range=POINT_CLOUD_RANGE).cuda()
        model_post = PointPillarsPos(nclasses=len(CLASSES)).cuda()
    else:
        model_pre = PointPillarsPre(voxel_size=VOXEL_SIZE, point_cloud_range=POINT_CLOUD_RANGE)
        model_post = PointPillarsPos(nclasses=len(CLASSES))
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError 
    pc = read_points(args.pc_path)
    pc = point_range_filter(pc, POINT_CLOUD_RANGE)
    if 'avikus' in prefix:
        pc[:, -1] /= 255.0

    if RAW_POINTS:
        pc[:, 1] *= -1
        pc[:, 2] *= -1

    # Generate BEV from raw point cloud
    # bev = make_bev(pc, resolution=0.2, mode="rgb")
    # cv2.imwrite("bev_rgb_raw.png", bev)
    # print("saved bev_rgb_raw.png")

    pc_torch = torch.from_numpy(pc)

    model_pre.eval()
    model_post.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
        pillars, coors_batch, npoints_per_pillar = model_pre(batched_pts=[pc_torch])
        if args.bev:
            # Generate BEV from pillarized point cloud
            bev_rgb = pillars_to_bev_rgb(
                pillars,
                coors_batch,
                npoints_per_pillar,
                POINT_CLOUD_RANGE,
                VOXEL_SIZE
                )
            cv2.imwrite("bev_rgb_pillars.png", bev_rgb)
            print("saved bev_rgb_pillars.png")

        if SAVE_BIN:
            save_bins(pillars, coors_batch, npoints_per_pillar, dump_dir)

        input_data = {input_pillars_name: to_numpy(pillars),
                      input_coors_batch_name: to_numpy(coors_batch),
                      input_npoints_per_pillar_name: to_numpy(npoints_per_pillar)}
        result = sess.run(None, input_data)
        result = [torch.from_numpy(item).cuda() for item in result]
        result_filter = model_post(result)[0]
    result_filter = keep_bbox_from_lidar_range(result_filter, np.array(POINT_CLOUD_RANGE))
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']
    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)
    np.set_printoptions(precision=3, suppress=True)
    lidar_bboxes = lidar_bboxes[np.argsort(-lidar_bboxes[:, 0])]
    print(f'lidar_bboxes : {lidar_bboxes}')
    print(f'labels : {labels}')
    print(f'scores : {scores}')
    result_array = np.concatenate([lidar_bboxes, scores[:, None], labels[:, None]], axis=-1)
    os.makedirs(os.path.dirname(args.saved_path), exist_ok=True)
    np.savetxt(args.saved_path, result_array, fmt='%.4f')

    if DURATION:
        time_total, time_pre, time_model, time_post = 0.0, 0.0, 0.0, 0.0
        test_samples = 100
        start_total_time = time.time()
        for i in range(test_samples):
            with torch.no_grad():
                if not args.no_cuda:
                    pc_torch = pc_torch.cuda()
                start_pre_time = time.time()
                pillars, coors_batch, npoints_per_pillar = model_pre(batched_pts=[pc_torch])
                end_pre_time = time.time()
                time_pre += (end_pre_time - start_pre_time)

                start_model_time = time.time()
                input_data = {input_pillars_name: to_numpy(pillars),
                            input_coors_batch_name: to_numpy(coors_batch),
                            input_npoints_per_pillar_name: to_numpy(npoints_per_pillar)}
                result = sess.run(None, input_data)
                result = [torch.from_numpy(item).cuda() for item in result]
                end_model_time = time.time()
                time_model += (end_model_time - start_model_time)

                start_post_time = time.time()
                result_filter = model_post(result)[0]
                result_filter = keep_bbox_from_lidar_range(result_filter, POINT_CLOUD_RANGE)
                end_post_time = time.time()
                time_post += (end_post_time - start_post_time)
        end_total_time = time.time()
        time_total = end_total_time - start_total_time

        avg_total_time = time_total * 1.0 / test_samples * 1000.0
        avg_pre_time = time_pre * 1.0 / test_samples * 1000.0
        avg_model_time = time_model * 1.0 / test_samples * 1000.0
        avg_post_time = time_post * 1.0 / test_samples * 1000.0
        print('ONNX total: {:.2f}ms, pre: {:.2f}ms, model: {:.2f}ms, post: {:.2f}ms'
            .format(avg_total_time, avg_pre_time, avg_model_time, avg_post_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--pc_path', help='your point cloud path')
    parser.add_argument('--saved_path', default='infer_results/onnx.txt',
                        help='your saved path for comparision bewteen PyTorch, ONNX and TRT')
    parser.add_argument('--onnx_path', default='../pretrained/model.onnx',
                        help='your saved onnx path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--prefix', default='avikus')
    parser.add_argument('--bev', action='store_true', help='Draw BEV from pillars')
    parser.add_argument('--dump_dir', required=True, default='dump_pre_io', help='directory to dump precomputed tensors')
    args = parser.parse_args()

    main(args)
