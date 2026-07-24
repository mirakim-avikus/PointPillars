#!/bin/bash

DATA_ROOT="/workspace/data_to_superbai_1114"

for subdir in "$DATA_ROOT"/*; do
    # 디렉토리만 처리
    if [ ! -d "$subdir" ]; then
        continue
    fi

    YAML_PATH="$subdir/new_lidar.yaml"

    # yaml 파일이 있을 때만 실행
    if [ -f "$YAML_PATH" ]; then
        echo "▶ Processing: $subdir"
        python3 pcd_flipper.py --dataset_dir="$subdir"
        python3 convert_lidar_rvec.py --lidar_yaml_path="$YAML_PATH"
    else
        echo "⏭  Skip (no new_lidar.yaml): $subdir"
    fi
done
