#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# legacy cvat-label pipeline, superseded by annos.sh. Kept for reference /
# reprocessing old cvat_test data.
DATA_ROOT="$SCRIPT_DIR/../../cvat_test"

# 환경 변수 전달
export DATA_ROOT

python3 "$SCRIPT_DIR/generate_cvat_label.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/split_train_val_cvat.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/generate_calib_cvat.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/pre_process_kitti_cvat.py" --data_root "$DATA_ROOT" --prefix avikus
python3 "$SCRIPT_DIR/set_anchor.py" --data_root "$DATA_ROOT"