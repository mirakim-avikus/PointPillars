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

        # Rename raw capture folders into the layout data_prep/generate_annos/
        # expects: camera/ -> images/, lidar/flippedData/ -> pcd/. lidar/Data/
        # (pre-flip) and lidar/Status are left behind, unused by the pipeline.
        if [ -d "$subdir/camera" ] && [ ! -d "$subdir/images" ]; then
            mv "$subdir/camera" "$subdir/images"
            echo "  camera/ -> images/"
        fi
        if [ -d "$subdir/lidar/flippedData" ] && [ ! -d "$subdir/pcd" ]; then
            mv "$subdir/lidar/flippedData" "$subdir/pcd"
            echo "  lidar/flippedData/ -> pcd/"
        fi
    else
        echo "⏭  Skip (no new_lidar.yaml): $subdir"
    fi
done
