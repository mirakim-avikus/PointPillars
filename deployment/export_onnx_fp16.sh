#!/usr/bin/env bash
set -e

# 사용법 체크
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 CKPT_PATH OUTPUT_ONNX_PATH [--no_cuda]"
    echo "  CKPT_PATH         : PyTorch checkpoint (.pth)"
    echo "  OUTPUT_ONNX_PATH  : FP32 ONNX 저장 경로"
    echo "  --no_cuda         : (옵션) CUDA 사용 안 함"
    exit 1
fi

CKPT_PATH="$1"
OUTPUT_ONNX_PATH="$2"
shift 2

# 스크립트 위치 기준으로 python 파일 찾기
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH2ONNX="${SCRIPT_DIR}/pytorch2onnx.py"
ONNX_FP16_MIXED="${SCRIPT_DIR}/onnx_fp16_mixed.py"

echo "[1/2] Export PyTorch -> ONNX (FP32)"
python3 "${PYTORCH2ONNX}" \
    --ckpt "${CKPT_PATH}" \
    --saved_onnx_path "${OUTPUT_ONNX_PATH}" \
    "$@"

echo "[2/2] Convert ONNX FP32 -> FP16 + mixed-precision fix"
python3 "${ONNX_FP16_MIXED}" \
    --model "${OUTPUT_ONNX_PATH}"

echo "Done!"
