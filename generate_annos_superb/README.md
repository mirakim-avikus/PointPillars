# superbAI Labeling Repo

superbAI에서 받은 데이터를 라벨링하기 위해 사용하는 레포입니다.

## 전제조건
- superbAI에 보낸 데이터와 받은 label, meta 등을 `data_to_superbAI` 디렉토리에 둘 것.

디렉토리 구조 예시 (depth 1):

```bash
mirakim@DESKTOP-0AVQA8J:~/workspace/PointPillars/data_to_superbai$ tree -L 1
.
├── boat_1
├── boat_2_1
├── boat_buoy_1
├── boat_buoy_2
├── bridge_pillar
├── buoy_boat_1
├── buoy_bridgepillar_1
├── buoy_cmarker_1
├── buoy_cmarker_2
├── buoy_cmarker_pole_boats_1
...
├── training
├── val.txt
````

## 사용법

1. 먼저 권한을 줍니다:

   ```bash
   chmod +x check_same_timestamp.sh
   ```

2. 중복된 timestamp `.pcd` 파일이 있는지 확인하려면 아래 명령어를 실행하세요:

   ```bash
   ./check_same_timestamp.sh
   ```

   * 중복이 발견되면, 해당 파일 경로가 출력되고 스크립트가 종료됩니다.
   * `sample/` 디렉토리는 검사에서 제외됩니다.

```
