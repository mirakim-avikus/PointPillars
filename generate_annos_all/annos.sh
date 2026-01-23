SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
"$SCRIPT_DIR/check_same_timestamp.sh"

# ===============================
# data 가공 파라미터
# ===============================
MIN_PTS_AFTER_FILTER=10
COMPENSATE_IMU=false

COMPENSATE_IMU_FLAG=""
if [ "$COMPENSATE_IMU" = true ]; then
    echo "Compensate IMU"
    COMPENSATE_IMU_FLAG="--compensate_imu"
fi

# ===============================
# pipeline
# ===============================
python3 "$SCRIPT_DIR/generate_superb_label.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/generate_cvat_label.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/split_train_val.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/generate_calib.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/pre_process_kitti.py" --data_root "$DATA_ROOT" --prefix avikus --min_pts_filter "$MIN_PTS_AFTER_FILTER" $COMPENSATE_IMU_FLAG
python3 "$SCRIPT_DIR/set_anchor.py" --data_root "$DATA_ROOT"