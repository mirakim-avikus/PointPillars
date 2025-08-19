import argparse
import os
import torch
from tqdm import tqdm
import pdb
import cv2
import yaml
import numpy as np

from pointpillars.utils import setup_seed, vis_pc, keep_bbox_from_image_range, bbox3d2corners_camera, vis_img_3d, read_calib, bbox3d2corners, keep_bbox_from_lidar_range, read_label, points_camera2image, bbox_camera2lidar
from pointpillars.dataset import Avikus, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter

CLASSES = {
    # 'human': 0,
    # 'jetski': 1,
    'smallboat': 2,
    'mediumboat': 3,
    'c-marker': 4,
    'yacht': 5
    }

def find_closest_lidar(lidar_dir, data_name):
    lidar_list = sorted(os.listdir(lidar_dir))
    lidar_ts_list = [int(f.split('.')[0]) for f in lidar_list]
    img_ts = int(data_name)

    if not lidar_ts_list:
        raise ValueError("LiDAR Directory is Empty.")

    if img_ts <= lidar_ts_list[0]:
        return str(lidar_ts_list[0])
    if img_ts >= lidar_ts_list[-1]:
        return str(lidar_ts_list[-1])

    closest_ts = min(lidar_ts_list, key=lambda x: abs(x - img_ts))
    return str(closest_ts)
    
def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


def get_parameters(calib_path_yaml, calib_info):
    with open(calib_path_yaml, 'rb') as f:
        calib = yaml.safe_load(f)
    cam = calib['camera']
    K = np.array([
        [cam['fx'], cam['skew'], cam['cx']],
        [0, cam['fy'], cam['cy']],
        [0, 0, 1]
    ], dtype = np.float32)
    D = np.array([cam['k1'], cam['k2'], cam['k3'], cam['k4']], dtype=np.float32)

    rvec = np.array([calib['camera2lidar']['rvec_1'], calib['camera2lidar']['rvec_2'], calib['camera2lidar']['rvec_3']])
    tvec = np.array([calib['camera2lidar']['tvec_1'], calib['camera2lidar']['tvec_2'], calib['camera2lidar']['tvec_3']])

    R, _ = cv2.Rodrigues(rvec)
    tr_velo_to_cam = np.identity(4)
    lidar2avikus = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    tr_velo_to_cam[:3, :3] = R@lidar2avikus
    tr_velo_to_cam[:3, -1] = tvec

    r0_rect = calib_info['R0_rect'].astype(np.float32)
    P2 = calib_info['P2'].astype(np.float32)
    return tr_velo_to_cam, r0_rect, P2, K, D


def main(args):
    setup_seed()
    train_dataset = Avikus(data_root=args.data_root,
                        split='train')
    val_dataset = Avikus(data_root=args.data_root,
                        split='val')

    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    point_cloud_range=[4, -72., -10., 180., 72., 30.]
    pcd_limit_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size=[0.25, 0.25, 4]

    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses, point_cloud_range=point_cloud_range, voxel_size=voxel_size, prefix='avikus').cuda()
    else:
        pointpillars = PointPillars(nclasses=args.nclasses, point_cloud_range=point_cloud_range, voxel_size = voxel_size,  prefix='avikus')
    loss_func = Loss()

    # load pretrained weight 
    checkpoint = torch.load("pretrained/epoch_160.pth")
    model_dict = pointpillars.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                pretrained_dict[k] = v
            else:
                pretrained_dict[k] = model_dict[k]
                pretrained_dict[k][:v.shape[0]] = v

    model_dict.update(pretrained_dict)
    pointpillars.load_state_dict(model_dict)

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    for epoch in range(args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            if len(batched_gt_bboxes[0]) == 0:
                print(f"data {data_dict['batched_img_info'][0]['image_path']} has no GT!")
                continue
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']

            for k, v in pointpillars.state_dict().items():
                if torch.isnan(v).any():
                    import pdb 
                    pdb.set_trace()

            for box in batched_gt_bboxes:
                if len(box) == 0:
                    print(f'batched gt bboxes : 0!')
                    import pdb
                    pdb.set_trace()

            # visualize GT bbox
            vis_pc(batched_pts[0].cpu().numpy(), bboxes=batched_gt_bboxes[0].cpu().numpy(), labels=batched_labels[0].cpu().numpy())
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                pointpillars(batched_pts=batched_pts, 
                             mode='train',
                             batched_gt_bboxes=batched_gt_bboxes, 
                             batched_gt_labels=batched_labels)

            device = bbox_cls_pred.device
            feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
            anchors = pointpillars.anchors_generator.get_multi_anchors(feature_map_size)
            batch_size = len(batched_pts)
            batched_anchors = [anchors for _ in range(batch_size)]
            result_filter = pointpillars.get_predicted_bboxes(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors)[0]
            data_name = os.path.basename(os.path.normpath(data_dict['batched_img_info'][0]['image_path'])).split('.')[0]

            if result_filter == None:
                print(f'prediction is invalid in {data_name}.png')
                continue
            print(f'start analyzing data : {data_name}')
            calib_info = read_calib(f"{os.path.normpath(args.data_root)}/calib_{os.path.basename(os.path.normpath(args.data_root))}.txt")
            calib_path_yaml = os.path.join(*os.path.normpath(args.data_root).split('/'),"lidar.yaml")

            tr_velo_to_cam, r0_rect, P2, K, D = get_parameters(calib_path_yaml, calib_info)

            parent_path = os.path.dirname(os.path.normpath(args.data_root))
            img_path = os.path.join(*parent_path.split('/'), data_dict['batched_img_info'][0]['image_path'])
            img = cv2.imread(img_path)
            image_shape = img.shape[:2]

            result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape, K=K, D=D, prefix='avikus')
            result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']
            vis_pc(batched_pts[0].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)
            
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
            
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            bbox_pred = bbox_pred[pos_idx]
            bbox_pred_vis = bbox_pred.detach().clone()
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
            batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
            
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
            print(f"train loss: {loss_str}")

            loss = loss_dict['total_loss']
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        if epoch % 2 == 0:
            continue

        pointpillars.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']

                orig_bbox_cls_pred, orig_bbox_pred, orig_bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)

                device = orig_bbox_cls_pred.device
                feature_map_size = torch.tensor(list(orig_bbox_cls_pred.size()[-2:]), device=device)
                anchors = pointpillars.anchors_generator.get_multi_anchors(feature_map_size)
                batch_size = len(batched_pts)
                batched_anchors = [anchors for _ in range(batch_size)]
                result_filter = pointpillars.get_predicted_bboxes(orig_bbox_cls_pred, orig_bbox_pred, orig_bbox_dir_cls_pred, batched_anchors)[0]
                data_name = os.path.basename(os.path.normpath(data_dict['batched_img_info'][0]['image_path'])).split('.')[0]

                if result_filter == None:
                    print(f'prediction is invalid in {data_name}.png')
                    continue

                bbox_cls_pred = orig_bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = orig_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = orig_bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                bbox_pred_vis = bbox_pred.detach().clone()
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)

                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                print(f"val loss: {loss_str}")

                # visualize image
                # vis_pc(batched_pts[0].cpu().numpy(), bboxes=batched_gt_bboxes[0].cpu().numpy(), labels=batched_labels[0].cpu().numpy())

                # visualize image
                device = orig_bbox_cls_pred.device
                feature_map_size = torch.tensor(list(orig_bbox_cls_pred.size()[-2:]), device=device)
                anchors = pointpillars.anchors_generator.get_multi_anchors(feature_map_size)
                batch_size = len(batched_pts)
                batched_anchors = [anchors for _ in range(batch_size)]
                result_filter = pointpillars.get_predicted_bboxes(orig_bbox_cls_pred, orig_bbox_pred, orig_bbox_dir_cls_pred, batched_anchors)[0]
                data_name = os.path.basename(os.path.normpath(data_dict['batched_img_info'][0]['image_path'])).split('.')[0]

                if result_filter == None:
                    print(f'prediction is invalid in {data_name}.png')
                    continue

                lidar_dir = os.path.join(args.data_root, 'lidar', 'flippedData')
                lidar_name = find_closest_lidar(lidar_dir, data_name)
                gt_path = os.path.join(args.data_root,  'annos_dir', f'{lidar_name}.txt')
                calib_path = os.path.join(args.data_root, f'calib_{os.path.basename(os.path.normpath(args.data_root))}.txt')
                parent_path = os.path.dirname(os.path.normpath(args.data_root))
                img_path = os.path.join(*parent_path.split('/'), data_dict['batched_img_info'][0]['image_path'])

                data_root = args.data_root
                calib_path_yaml = os.path.join(*os.path.normpath(data_root).split('/'), "lidar.yaml")
                calib_info = read_calib(calib_path)
                gt_label = read_label(gt_path)

                tr_velo_to_cam, r0_rect, P2, K, D = get_parameters(calib_path_yaml, calib_info)

                img = cv2.imread(img_path, 1)
                image_shape = img.shape[:2]

                result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape, K=K, D=D, prefix='avikus')
                result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']

                dimensions = gt_label['dimensions']
                location = gt_label['location']
                rotation_y = gt_label['rotation_y']
                gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
                gt_lidar_bboxes = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)

                gt_labels = [-1] * len(gt_label['name']) # to distinguish between the ground truth and the predictions
                    
                pred_gt_lidar_bboxes = np.concatenate([lidar_bboxes, gt_lidar_bboxes], axis=0)
                pred_gt_labels = np.concatenate([labels, gt_labels])

                # vis_pc(batched_pts[0].cpu().numpy(), bboxes=lidar_bboxes, labels=labels)
                # pdb.set_trace()

                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
                bboxes_corners = bbox3d2corners_camera(camera_bboxes)

                points_normalized = bboxes_corners[:, :, :2] / bboxes_corners[:, :, 2:]
                points_distorted = cv2.fisheye.distortPoints(points_normalized.reshape(-1, 1, 2), K, D)
                image_points = points_distorted.reshape(bboxes_corners.shape[0], -1, 2)

                img = vis_img_3d(img, image_points, labels, rt=True)
                os.makedirs('result_imgs', exist_ok=True)
                cv2.imwrite(f'result_imgs/{data_name}-3d-bbox_{epoch}.png', img)

                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                    # add projected image w/ bbox
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    writer.add_image('3D_BBox_Prediction', img_tensor[0], global_step)

                val_step += 1
        pointpillars.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', required=True, default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--nclasses', required=True, type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
