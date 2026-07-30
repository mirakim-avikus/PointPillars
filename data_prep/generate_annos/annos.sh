SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Stop the whole pipeline as soon as one step fails, instead of silently
# continuing on to the next script with bad/partial data.
run_step() {
    local desc="$1"
    shift
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "[annos.sh] ${desc} script fails. solve this first" >&2
        exit 1
    fi
}

# ===============================
# 입력 인자 체크
# ===============================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <DATA_ROOT>"
    echo "Example:"
    echo "  $0 /workspace/data_0115"
    exit 1
fi

DATA_ROOT="$1"

# 환경 변수 전달
export DATA_ROOT

# ===============================
# 사전 체크
# ===============================
chmod +x "$SCRIPT_DIR/check_same_timestamp.sh"
run_step "check_same_timestamp.sh" "$SCRIPT_DIR/check_same_timestamp.sh"

# ===============================
# data 가공 파라미터
# ===============================
MIN_PTS_AFTER_FILTER=-1
COMPENSATE_IMU=false

COMPENSATE_IMU_FLAG=""
if [ "$COMPENSATE_IMU" = true ]; then
    echo "Compensate IMU"
    COMPENSATE_IMU_FLAG="--compensate_imu"
fi

# ===============================
# pipeline
# ===============================
run_step "generate_superb_label.py" python3 "$SCRIPT_DIR/generate_superb_label.py" --data_root "$DATA_ROOT"
run_step "generate_cvat_label.py" python3 "$SCRIPT_DIR/generate_cvat_label.py" --data_root "$DATA_ROOT"
run_step "convert_label_z_center_to_bottom.py" python3 "$SCRIPT_DIR/convert_label_z_center_to_bottom.py" --data_root "$DATA_ROOT"
run_step "split_train_val.py" python3 "$SCRIPT_DIR/split_train_val.py" --data_root "$DATA_ROOT"
run_step "generate_calib.py" python3 "$SCRIPT_DIR/generate_calib.py" --data_root "$DATA_ROOT"
run_step "pre_process_kitti.py" python3 "$SCRIPT_DIR/pre_process_kitti.py" --data_root "$DATA_ROOT" --prefix avikus --min_pts_filter "$MIN_PTS_AFTER_FILTER" $COMPENSATE_IMU_FLAG
run_step "set_anchor.py" python3 "$SCRIPT_DIR/set_anchor.py" --data_root "$DATA_ROOT"

# ===============================
# 결과 확인 (임의 프레임 3D 시각화)
# ===============================
# Not run_step: data preprocessing is already done and correct at this point,
# so a visualization failure (e.g. no DISPLAY) shouldn't fail the pipeline -
# just warn so the user knows to check it separately.
python3 "$SCRIPT_DIR/../../visualization/vis_gt_pc.py" --root "$DATA_ROOT"
if [ $? -ne 0 ]; then
    echo "[annos.sh] WARNING: data preprocessing is READY, but vis_gt_pc.py failed to show the visualization. Check what's wrong (e.g. DISPLAY, xhost) and rerun visualization/vis_gt_pc.py manually." >&2
fi