#!/bin/bash

# 1. avikus vehicle - 3D points checked. 2D projection not due to camera delay 
# image_path="data/avikus/motorboat/cars/camera/1747113334782.jpg"
# pcd_path="data/avikus/motorboat/cars/lidar/Data/1747113334772.avikus.pcd"
# pretrained_weight="pretrained/epoch_160.pth"
# gt_path="pointpillars/dataset/annos_dir/1736092769837.txt"
# calib_path="data/avikus/motorboat/cars02/calib_cars.txt"

# # 2. avikus vessel - 3D points checked. 2D projection checked
# image_path="data/avikus/motorboat/005/camera/1736092769839.jpg"
# pcd_path="data/avikus/motorboat/005/lidar/Data/1736092769837.avikus.pcd"
# pretrained_weight="pretrained/epoch_160.pth"
# gt_path="data/avikus/motorboat/005/annos_dir/1736092769837.txt"
# calib_path="data/avikus/motorboat/005/calib_005.txt"

# # 2. avikus vessel - 3D points checked. 2D projection checked
# image_path="data/avikus/motorboat/007/camera/1736093515074.jpg"
# pcd_path="data/avikus/motorboat/007/lidar/flippedData/1736093515111.avikus.pcd"
# pretrained_weight="pretrained/epoch_160.pth"
# gt_path="data/avikus/motorboat/007/annos_dir/1736093515111.txt"
# calib_path="data/avikus/motorboat/007/calib_007.txt"

# 2. avikus vessel - 3D points checked. 2D projection checked
image_path="data/avikus/motorboat/007/camera/1736093508474.jpg"
pcd_path="data/avikus/motorboat/007/lidar/flippedData/1736093508412.avikus.pcd"
pretrained_weight="pretrained/epoch_160.pth"
gt_path="data/avikus/motorboat/007/annos_dir/1736093508412.txt"
calib_path="data/avikus/motorboat/007/calib_007.txt"

python3 test.py --ckpt $pretrained_weight --pc_path $pcd_path --calib_path $calib_path  --gt_path $gt_path --img_path $image_path

# image_path="data/avikus/motorboat/007/camera/1736093514474.jpg"
# pcd_path="data/avikus/motorboat/007/lidar/flippedData/1736093514505.avikus.pcd"
# pretrained_weight="pretrained/epoch_160.pth"
# gt_path="data/avikus/motorboat/007/annos_dir/1736093514505.txt"
# calib_path="data/avikus/motorboat/007/calib_007.txt"

# python3 test.py --ckpt $pretrained_weight --pc_path $pcd_path --calib_path $calib_path  --gt_path $gt_path --img_path $image_path