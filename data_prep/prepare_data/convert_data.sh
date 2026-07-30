#!/bin/bash

DATA_ROOT="/workspace/data_to_superbai_1114"

for subdir in "$DATA_ROOT"/*; do
    # 디렉토리만 처리
    if [ ! -d "$subdir" ]; then
        continue
    fi

    LIDAR_YAML="$subdir/lidar.yaml"
    LIDAR_YAML_BAK="$subdir/lidar.yaml.bak"
    NEW_LIDAR_YAML="$subdir/new_lidar.yaml"

    if [ -f "$LIDAR_YAML_BAK" ]; then
        echo "⏭  Skip (already converted): $subdir"
        continue
    fi

    if [ -f "$NEW_LIDAR_YAML" ] && [ ! -f "$LIDAR_YAML" ]; then
        # Session arrived with the yz-flip already applied upstream (no
        # separate original lidar.yaml to preserve) - just promote it.
        mv "$NEW_LIDAR_YAML" "$LIDAR_YAML"
        touch "$LIDAR_YAML_BAK"  # marker: nothing to back up, already flipped on arrival
        echo "▶ Processing: $subdir (new_lidar.yaml already provided, promoted -> lidar.yaml)"
    elif [ -f "$LIDAR_YAML" ]; then
        echo "▶ Processing: $subdir"
        mv "$LIDAR_YAML" "$LIDAR_YAML_BAK"
        python3 convert_lidar_rvec.py --lidar_yaml_path="$LIDAR_YAML_BAK" --output_path="$LIDAR_YAML"
    else
        echo "⏭  Skip (no lidar.yaml or new_lidar.yaml): $subdir"
        continue
    fi

    python3 pcd_flipper.py --dataset_dir="$subdir"

    # Rename raw capture folders into the layout data_prep/generate_annos/
    # expects: camera/ -> images/, lidar/flippedData/ -> pcd/. lidar/Data/
    # (pre-flip) and lidar/Status are left behind, unused by the pipeline.
    # Gated on camera/ still existing (i.e. this session hasn't been
    # through this conversion yet), so it only ever runs once per session.
    if [ -d "$subdir/camera" ] && [ ! -d "$subdir/images" ]; then
        mv "$subdir/camera" "$subdir/images"
        echo "  camera/ -> images/"
    fi
    if [ -d "$subdir/lidar/flippedData" ] && [ ! -d "$subdir/pcd" ]; then
        mv "$subdir/lidar/flippedData" "$subdir/pcd"
        echo "  lidar/flippedData/ -> pcd/"
    fi
done
