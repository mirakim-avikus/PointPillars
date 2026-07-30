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
        # Gated on camera/ still existing (i.e. this session hasn't been
        # through this conversion yet), so it and the z conversion below only
        # ever run once per session.
        if [ -d "$subdir/camera" ] && [ ! -d "$subdir/images" ]; then
            # Raw captures that arrive with label/*.txt already populated skip
            # generate_superb_label.py/generate_cvat_label.py (nothing to
            # trigger them), so their z never gets converted from center to
            # bottom-center like those scripts do for SuperbAI/CVAT sources.
            # bbox3d2corners (and therefore Avikus.__getitem__'s
            # gt_bboxes_3d for training) expects bottom-center.
            if [ -d "$subdir/label" ]; then
                python3 convert_label_z_center_to_bottom.py --dataset_dir="$subdir"
            fi

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
