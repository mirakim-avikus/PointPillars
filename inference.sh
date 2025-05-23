#!/bin/bash

# 1. avikus vehicle - 3D points checked. 2D projection not due to camera delay 
# image_path="data/avikus/motorboat/cars/camera/1747113334782.jpg"
# pcd_path="data/avikus/motorboat/cars/lidar/Data/1747113334772.avikus.pcd"
# pretrained_weight="pretrained/epoch_160.pth"
# gt_path="pointpillars/dataset/annos_dir/1736092769837.txt"
# calib_path="data/avikus/motorboat/cars02/calib_cars.txt"

# 2. avikus vessel - 3D points checked. 2D projection checked
image_path="data/avikus/motorboat/005/camera/1736092769839.jpg"
pcd_path="data/avikus/motorboat/005/lidar/Data/1736092769837.avikus.pcd"
pretrained_weight="pretrained/epoch_160.pth"
gt_path="data/avikus/motorboat/005/annos_dir/1736092769837.txt"
calib_path="data/avikus/motorboat/005/calib_005.txt"

python3 test_avikus.py --ckpt $pretrained_weight --pc_path $pcd_path --calib_path $calib_path  --gt_path $gt_path --img_path $image_path