#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATA_ROOT="$SCRIPT_DIR/../cvat_test"

# 환경 변수 전달
export DATA_ROOT

python3 "$SCRIPT_DIR/generate_label.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/split_train_val.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/generate_calib.py" --data_root "$DATA_ROOT"
python3 "$SCRIPT_DIR/pre_process_kitti.py" --data_root "$DATA_ROOT" --prefix avikus
python3 "$SCRIPT_DIR/set_anchor.py" --data_root "$DATA_ROOT"