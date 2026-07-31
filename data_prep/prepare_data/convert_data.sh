#!/bin/bash

# Stop the whole script as soon as one step fails, instead of silently
# continuing on to the next step with bad/partial data.
run_step() {
    local desc="$1"
    shift
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "[convert_data.sh] ${desc} script fails. solve this first" >&2
        exit 1
    fi
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <DATA_ROOT>"
    echo "Example:"
    echo "  $0 /workspace/data_to_superbai_1114"
    exit 1
fi

DATA_ROOT="$1"

for subdir in "$DATA_ROOT"/*; do
    # 디렉토리만 처리
    if [ ! -d "$subdir" ]; then
        continue
    fi

    LIDAR_YAML="$subdir/lidar.yaml"
    LIDAR_YAML_BAK="$subdir/lidar.yaml.bak"
    NEW_LIDAR_YAML="$subdir/new_lidar.yaml"
    YAML_CONVERTED_MARKER="$subdir/.yaml_converted"

    if [ -f "$YAML_CONVERTED_MARKER" ]; then
        echo "⏭  Skip (already converted): $subdir"
        continue
    fi

    if [ -f "$NEW_LIDAR_YAML" ] && [ ! -f "$LIDAR_YAML" ]; then
        # Session arrived with the yz-flip already applied upstream (no
        # separate original lidar.yaml to preserve) - just promote it.
        mv "$NEW_LIDAR_YAML" "$LIDAR_YAML"
        touch "$YAML_CONVERTED_MARKER"
        echo "▶ Processing: $subdir (new_lidar.yaml already provided, promoted -> lidar.yaml)"
    elif [ -f "$LIDAR_YAML" ]; then
        echo "▶ Processing: $subdir"
        mv "$LIDAR_YAML" "$LIDAR_YAML_BAK"
        run_step "convert_lidar_rvec.py" python3 convert_lidar_rvec.py --lidar_yaml_path="$LIDAR_YAML_BAK" --output_path="$LIDAR_YAML"
        touch "$YAML_CONVERTED_MARKER"
    else
        echo "⏭  Skip (no lidar.yaml or new_lidar.yaml): $subdir"
        continue
    fi

    run_step "pcd_flipper.py" python3 pcd_flipper.py --dataset_dir="$subdir"

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

    run_step "matching_jpg_pcd.py" python3 matching_jpg_pcd.py --dataset_dir="$subdir"
done

# ===============================
# 결과 확인 (flip + calib 정합성 검증)
# ===============================
# Not run_step: by this point every session's pcd/calib data is already
# converted and correct - a visualization failure (e.g. no DISPLAY)
# shouldn't fail the script, just warn so it can be checked separately.
python3 project_points_to_img.py --root "$DATA_ROOT"
if [ $? -ne 0 ]; then
    echo "[convert_data.sh] WARNING: prepare_data conversion is READY, but project_points_to_img.py failed to show the visualization. Check what's wrong (e.g. DISPLAY, xhost) and rerun project_points_to_img.py manually." >&2
fi
