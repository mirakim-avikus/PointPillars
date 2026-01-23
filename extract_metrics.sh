#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="/workspace/data_to_superbai_1114"
TMP_ROOT="/workspace/data_to_eval"
SCRIPT="BEV_with_metric.py"

mkdir -p "${TMP_ROOT}"

echo "[CHECK] Place trained model weight in pretrained directory!"

for SUBDIR in "${BASE_DIR}"/*; do
    # 디렉토리만 대상
    [ -d "${SUBDIR}" ] || continue

    # label 폴더가 없으면 skip
    if [ ! -d "${SUBDIR}/label" ]; then
        echo "[SKIP] No label dir in $(basename "${SUBDIR}")"
        continue
    fi

    NAME=$(basename "${SUBDIR}")
    TMP_DIR="${TMP_ROOT}"

    echo "===================================================="
    echo "[START] Processing ${NAME}"
    echo "----------------------------------------------------"

    # 이전 잔여물 있으면 제거
    echo "[0/4] Removing Old stuffs..."
    rm -rf "${TMP_DIR}"
    mkdir "${TMP_DIR}"

    echo "[1/4] Copying data to data_to_eval..."
    cp -r "${SUBDIR}" "${TMP_DIR}"

    echo "[2/4] Make data from data_to_eval..."
    cd generate_annos_all && ./annos.sh "${TMP_ROOT}" && cd ..

    echo "[3/4] Running BEV_with_metric.py..."
    python3 "${SCRIPT}" \
        --data_root="${TMP_DIR}" \
        --out_dir="${BASE_DIR}" \
        --batch_size=1 || {
            echo "[ERROR] Script failed for ${NAME}"
        }

    echo "[4/4] Cleaning up data_to_eval..."
    rm -rf "${TMP_DIR}"

    echo "[DONE] ${NAME}"
    echo
done


echo "===================================================="
echo "[1/2] Make Video for all scores (01, 02, 03)..."
./make_video.sh "${BASE_DIR}" 8 "gt_pred_noimu.mp4" || {
        echo "[ERROR] Script failed for Making Video"
    }

echo "[2/2] BEV sorting..."
python3 bev_sorting.py || {
        echo "[ERROR] Script failed for BEV Sorting"
    }

echo "✅ All subdirectories processed."
