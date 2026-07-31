# [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784) 

Avikus fork for maritime 3D detection (10 classes: boats, jetskis, markers, buoys,
bridge pillars) from LiDAR + camera, adapted from
[zhulf0804/PointPillars](https://github.com/zhulf0804/PointPillars/tree/main) — a
from-scratch PyTorch PointPillars implementation kept dependency-light (no
Spconv/mmdet/mmdet3d install required). [[Original author's writeup](https://zhuanlan.zhihu.com/p/521277176)]

## Installation

### 1) Docker container

Build the docker image and start a container. This allows others to develop in exactly the same environment with a single command:

```
cd PointPillars/
docker build -t custom-open3d-python-cu111 .
docker run --name pointpillars --gpus all --runtime=nvidia --privileged --network=host \
  --security-opt label=disable -e DISPLAY=:11.0 -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace -it -d custom-open3d-python-cu111 bash

// for Avikus server with four A100
docker run -d --runtime=nvidia --gpus all  -it -d -v $(pwd):/workspace   --device=/dev/nvidia-uvm     --device=/dev/nvidia-uvm-tools     --device=/dev/nvidia-modeset     --device=/dev/nvidiactl     --device=/dev/nvidia0   --device=/dev/nvidia1  --device=/dev/nvidia2  --device=/dev/nvidia3  custom-open3d-python-cu111  bash -c "while [ true ]; do nvidia-smi -L; sleep 5; done"
```

`DISPLAY` must match the X display the host is actually using (check with `echo $DISPLAY`), and the host must allow the container to connect for the open3d GUI viewer (e.g. `pointpillars/utils/vis_o3d.py`) to render. If a GUI script (`vis_gt_pc.py`, `project_points_to_img.py`, etc.) fails with:
- `Authorization required, but no authorization protocol specified` - the host hasn't authorized the container's user. Containers here run as **root**, so on the **host** run `xhost +si:localuser:root` (NOT `xhost +local:docker` - `docker` isn't a real xhost family and silently does nothing), then retry. This doesn't persist across host reboots/X server restarts.
- `GLFW Error: X11: Failed to open display :N.N` - `DISPLAY` has drifted from what the host is currently using (common after reconnecting to the host); re-check `echo $DISPLAY` on the host and pass it explicitly, e.g. `docker exec -e DISPLAY=:<N>.0 <container> ...`.

If `docker build` fails with a TLS/certificate error (`SSLError`, `self signed certificate in certificate chain`, etc.), your network is TLS-inspecting outbound HTTPS traffic (e.g. a corporate SASE gateway) - see [`certs/README.md`](certs/README.md) for how to fix it.

Once built, enter the running container with:

```
docker exec -it pointpillars bash
```

### 2) Install PointPillars package

Install PointPillars as a python package and all its dependencies as follows (run this inside the container - `/workspace` there is already the repo root, bind-mounted from the host's `PointPillars/`, so there's no extra `PointPillars/` to `cd` into):

```
cd /workspace
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install .
```

## Data Setting

데이터는 아래 2개의 소스로부터 확보한다.

### 1) SuperbAI

- **(1) 데이터 Export**
  - SuperbAI 플랫폼에서 **Kitti data format**으로 export  
  - 경로: `프로젝트 → 라벨 내보내기 → 다운로드 → Superb AI 포맷 다운로드`

- **(2) 데이터 폴더 구조**
  - `data_root` 안의 각 `data_name` 폴더 구조는 아래와 같다.

    ```text
    data_root
    ├─ data_name_A
    │  └─ images/, pcd/, yaml, label/
    ├─ data_name_B
    │  └─ images/, pcd/, yaml, label/
    ├─ data_name_C
    │  └─ ...
    └─ zip
    ```

  - SuperbAI export는 `label/`에 원본 라벨(.txt, KITTI 포맷, z 기준 = bbox 중심)이 바로 담겨 온다 - 별도 `label_original/`이나 `meta/*.json`은 쓰지 않는다.
  - `annos.sh` 실행 시 `tracklet_labels.xml`이 없는 세션은 모두 SuperbAI로 간주되어 `convert_label_z_center_to_bottom.py`가 `label/`의 z 좌표 기준을 (bbox 중심 → bbox 바닥 중심으로) 자동 변환한다 (재실행해도 중복 변환되지 않도록 `.z_converted` 마커로 추적).


---

### 2) CVAT

- **(1) 데이터 Export**
  - CVAT에서 **Kitti data format**으로 export  
  - 경로:  
    `프로젝트 → Menu → Export → Kitti Raw Format 1.0`

- **(2) 데이터 폴더 구조**
  - `data_root` 안의 각 `data_name` 폴더 구조는 아래와 같다.

    ```text
    data_root
    ├─ data_name_A
    │  └─ 이미지, pcd, yaml, zip
    ├─ data_name_B
    │  └─ 이미지, pcd, yaml, zip
    ├─ data_name_C
    │  └─ ...
    ```
---
### 3) 학습을 위한 Annotation 생성
  - CVAT과 SuperbAI에서 다운로드한 zip들을 unzip 후 아래 스크립트 실행
    ```bash
    cd data_prep/generate_annos
    ./annos.sh
    ```
  - `DATA_ROOT`를 `data_root`에 맞게 수정

---
### 4) Annotation 확인 (3D 시각화)
  - `annos.sh` 실행 후 생성된 pcd + label을 3D로 확인 (Open3D 창, cv2 불필요)
    ```bash
    cd visualization
    python3 vis_gt_pc.py --root <data_root> --data_name <data_name> --id <frame_id>
    ```
  - 예시
    ```bash
    python3 vis_gt_pc.py --root /workspace/test_data --data_name batch04 --id 1737318282526
    ```
  - `--root`: `data_root` (session 폴더들을 담고 있는 상위 폴더)
  - `--data_name`: session 폴더명 (예: `batch04`)
  - `--id`: 확인할 frame의 timestamp (`<data_name>/pcd/<id>.avikus.pcd`, `<data_name>/label/<id>.txt` 기준)

---

## Repository Layout

```
training/       train.py, train.sh
evaluation/     evaluate.py, test.py, metric.py, AP_calculator.py, BEV_with_metric.py,
                extract_metrics.sh, find_most_empty_labels.sh, inference.sh
visualization/  bev_sorting.py, make_video.sh, test_o3d.py, vis_data_gt.py, vis_gt_pc.py
data_prep/
  generate_annos/  annotation-generation pipeline (see data_prep/generate_annos/README.md)
  prepare_data/    convert_data.sh <DATA_ROOT>, convert_lidar_rvec.py, pcd_flipper.py,
                   matching_jpg_pcd.py, project_points_to_img.py
pointpillars/   core package (dataset, model, ops, loss, utils) - installed via setup.py
deployment/     ONNX/TensorRT export assets
docs/           log.md - original implementation notes
```

## [Training]

```
cd /workspace
python training/train.py --data_root <data_root> --ckpt_name <run_name>
```

`--data_root` points at the same `data_root` `annos.sh` was run against (needs `avikus_infos_{train,val,test,trainval}.pkl` and `avikus_dbinfos_train.pkl` already generated). `--ckpt_name` is required (names the run under `--saved_path`, default `pillar_logs/`). Other useful flags (see `training/train.py`'s argparse for the full list): `--batch_size` (default 6), `--max_epoch` (default 280), `--val_freq_epoch` / `--ckpt_freq_epoch` (default 10 / 20), `--pretrained --pretrained_weight <path>` to fine-tune from an existing checkpoint instead of training from scratch.

## [Validation]

```
cd /workspace/evaluation
python BEV_with_metric.py --data_root <data_root> --weight <ckpt_path> --out_dir <out_dir>
```

Reports both per-class (`Avikus.CLASSES`, 10 classes, each matched against its own IOU threshold in `Avikus.IOU_THRESHOLDS` - 0.4 for pole/buoy, 0.5 for jetski/smallboat/c-marker/dinghyboat/bridgepillar, 0.6 for mediumboat/yacht/bigboat) and grouped into 4 broader categories (`motorboat`, `jetski`, `cmarker`, `bridgepillar`) for a coarser read. Metrics per class/group:

- `mAP_3D` / `mAP_BEV` - mean average precision, 3D box and bird's-eye-view
- `ATE` - average translation error
- `AOE_deg` - average orientation error, in degrees
- `ASE` - average scale error

## [Model Conversion]

The model is split for export purposes into `PointPillarsPre` (voxelization), `PointPillarsCore` (encoder → backbone → neck → head, the ONNX-exportable part), and `PointPillarsPos` (NMS post-processing) — see `pointpillars/model/pointpillars_export.py`. Only `PointPillarsCore` supports batch_size > 1 during training (`PointPillars` in `pointpillars/model/pointpillars.py`); the export path only supports batch_size=1.

1. PyTorch → ONNX

    ```
    cd /workspace/deployment
    python pytorch2onnx.py --ckpt ../pretrained/epoch_160.pth --saved_onnx_path ../pretrained/model.onnx
    ```

2. (Optional) FP32 → FP16 mixed-precision ONNX, in one step

    ```
    cd /workspace/deployment
    ./export_onnx_fp16.sh ../pretrained/epoch_160.pth ../pretrained/model.onnx
    ```

3. (Optional) ONNX inference / sanity check against the PyTorch output

    ```
    cd /workspace/deployment
    python onnx_infer.py --pc_path ../pointpillars/dataset/demo_data/val/000134.bin --onnx_path ../pretrained/model.onnx
    python pytorch_infer.py --ckpt ../pretrained/epoch_160.pth --pc_path ../pointpillars/dataset/demo_data/val/000134.bin
    ```

4. (Optional) ONNX → TensorRT + C++ inference

    A TensorRT/C++ inference pipeline (voxelization + NMS reimplemented in CUDA/C++) lives in `deployment/trt_infer/`; see `docs/deployment_log.md` for the ONNX2TRT conversion notes and troubleshooting (ScatterND plugin support, dynamic shapes, etc).

## Acknowledements

Thanks for the open source code [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d).
