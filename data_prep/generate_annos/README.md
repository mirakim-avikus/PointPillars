# generate_annos

Generates KITTI-style annotation pkl/txt files from raw CVAT + SuperbAI exports. Both
sources are expected to live under one shared `DATA_ROOT`, one subfolder per recording
session:

```
DATA_ROOT/
├─ <data_name>/            # e.g. "boat_1" - one recording session
│  ├─ images/               # .jpg, camera frames
│  ├─ pcd/                  # .avikus.pcd, lidar frames (same timestamp convention as images/)
│  ├─ new_lidar.yaml        # camera intrinsics + camera2lidar / reference2lidar(IMU) extrinsics
│  ├─ frame_list.txt        # frame index -> pcd filename mapping
│  │
│  │  # exactly one of the following two identifies the session's label source:
│  ├─ label_original/       # SuperbAI: raw KITTI-format label export, as-is
│  ├─ meta/*.json           # SuperbAI: labeling-tool JSON export (used if label_original/ absent)
│  └─ tracklet_labels.xml   # CVAT: raw tracklet export
```

## Pipeline (`annos.sh <DATA_ROOT>`)

| # | Script | In | Out | Applies to |
|---|---|---|---|---|
| 0 | `check_same_timestamp.sh` | `<data_name>/pcd/*.avikus.pcd` (all sessions) | pass/fail only | both — guards against two sessions from different sources landing on the same DATA_ROOT with a colliding timestamp |
| 1a | `generate_superb_label.py` | `label_original/` or `meta/*.json` + `images/`, `pcd/`, `new_lidar.yaml` | `<data_name>/label/*.txt` | SuperbAI sessions only (per-session: uses `label_original/` if present, else `meta/*.json`, else skips the session) |
| 1b | `generate_cvat_label.py` | `tracklet_labels.xml` + `frame_list.txt`, `pcd/`, `images/`, `new_lidar.yaml` | `<data_name>/label/*.txt` | CVAT sessions only (skips any session without `tracklet_labels.xml`) |
| 2 | `split_train_val.py` | every `<data_name>/label/*.txt` produced above | `DATA_ROOT/{train,val,test}.txt` | both (walks the whole merged root) |
| 3 | `generate_calib.py` | `<data_name>/new_lidar.yaml` | `<data_name>/calib_<data_name>.txt` | both |
| 4 | `pre_process_kitti.py --prefix avikus [--compensate_imu] [--min_pts_filter N]` | `{train,val,test}.txt`, `pcd/`, `images/`, `calib_*.txt`, `label/*.txt`, optional `oru/attitude.csv` | `velodyne_reduced/*.avikus.pcd`, `avikus_infos_{train,val,test,trainval}.pkl`, `avikus_dbinfos_train.pkl` + `avikus_gt_database/*.bin` | both |
| 5 | `set_anchor.py` | `avikus_dbinfos_train.pkl` | prints suggested per-class anchor `w l h` (no file) | both |

Steps 1a/1b run unconditionally and both **self-filter**: each session folder is only
processed by the generator matching its own label source, so a mixed `DATA_ROOT` works
without picking a source upfront. Steps 2–5 are source-agnostic and run once over the
combined dataset.

## Legacy scripts

`cvat.sh` and `superb.sh` are earlier, narrower pipelines that only handled one label
source each, kept for reference / reprocessing old `cvat_test` or `data_to_superbai_1114`
exports with their original logic (no IMU compensation, different calib-fallback rules,
and — for `pre_process_kitti_superb.py` specifically — the older combined `sample/`
folder instead of `images/`+`pcd/`):

- `cvat.sh` -> `generate_cvat_label.py`, `split_train_val_cvat.py`, `generate_calib_cvat.py`,
  `pre_process_kitti_cvat.py`, `set_anchor.py`
- `superb.sh` -> `generate_superb_label.py`, `split_train_val_superb.py`,
  `generate_calib_superb.py`, `pre_process_kitti_superb.py`, `set_anchor.py`

`README_superb_legacy.md` documents the original SuperbAI-only labeling workflow.
