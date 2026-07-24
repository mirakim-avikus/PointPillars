#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/data_to_superbai_1114}"
FPS="${2:-8}"
FILE_NAME="${3:?에러: 3번째 인자로 출력 파일명(예: result.mp4)을 입력해야 합니다.}"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found"; exit 1; }

shopt -s nullglob

for d in "$ROOT"/*; do
  [[ -d "$d" ]] || continue

  # d/bev/01, d/bev/02 등 모든 하위 폴더를 찾기 위한 루프 추가
  for score_dir in "$d/bev"/[0-9][0-9]; do
    [[ -d "$score_dir" ]] || continue

    GT_PRED_DIR="$score_dir/gt_pred"
    OUT_MP4="$score_dir/$FILE_NAME"

    # png가 있는지 확인
    pngs=( "$GT_PRED_DIR"/*.png )
    if (( ${#pngs[@]} > 0 )); then
      echo "Making video: $OUT_MP4 (png=${#pngs[@]})"

      TMP_LIST="$(mktemp)"
      # 자연 정렬(sort -V)로 프레임 순서 보장
      ls -1 "$GT_PRED_DIR"/*.png | sort -V | sed "s|^|file '|;s|$|'|" > "$TMP_LIST"

      ffmpeg -y -hide_banner -loglevel error \
        -r "$FPS" -f concat -safe 0 -i "$TMP_LIST" \
        -c:v libx264 -pix_fmt yuv420p \
        "$OUT_MP4"

      rm -f "$TMP_LIST"
    fi
  done
done

echo "All videos have been generated."