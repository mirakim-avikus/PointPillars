#!/bin/bash

# 데이터 루트가 설정되어 있는지 확인
if [ -z "$DATA_ROOT" ]; then
    echo "❌ DATA_ROOT 환경 변수가 설정되어 있지 않습니다."
    echo "예: DATA_ROOT=/path/to/data ./check_same_timestamp.sh"
    exit 1
fi

# 중복 여부를 기록하기 위한 해시맵 역할
declare -A seen

# 모든 pcd 파일 순회
while IFS= read -r -d '' file; do
    fname=$(basename "$file")  # 파일명만 추출

    if [[ -n "${seen[$fname]}" ]]; then
        echo "중복 발견!"
        echo "이미 존재: ${seen[$fname]}"
        echo "중복 파일: $file"
        exit 1
    else
        seen[$fname]="$file"
    fi
done < <(find "$DATA_ROOT" -type d -name sample -prune -o -type f -name "*.avikus.pcd" -print0)

echo "중복 없음 ✅"
exit 0
