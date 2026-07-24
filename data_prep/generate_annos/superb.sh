#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# legacy superb-label pipeline, superseded by annos.sh (which now generates both
# cvat and superb labels via generate_cvat_label.py / generate_superb_label.py).
# Kept for reference / reprocessing old data_to_superbai_1114 data.
DATA_ROOT="$SCRIPT_DIR/../../data_to_superbai_1114"

# 환경 변수 전달
export DATA_ROOT

chmod +x "$SCRIPT_DIR/check_same_timestamp.sh"
"$SCRIPT_DIR/check_same_timestamp.sh"

python3 "$SCRIPT_DIR/generate_superb_label.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/split_train_val_superb.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/generate_calib_superb.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/pre_process_kitti_superb.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/set_anchor.py" --data_root "$DATA_ROOT"