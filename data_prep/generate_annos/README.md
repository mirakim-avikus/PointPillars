# generate_annos

Generates KITTI-style annotation pkl/txt files from raw CVAT + SuperbAI exports. Both
sources are expected to live under one shared `DATA_ROOT`, one subfolder per recording
session:

```
DATA_ROOT/
├─ <data_name>/            # e.g. "boat_1" - one recording session
│  ├─ images/               # .jpg, camera frames
│  ├─ pcd/                  # .avikus.pcd, lidar frames (same timestamp convention as images/)
│  ├─ lidar.yaml            # camera intrinsics + camera2lidar / reference2lidar(IMU) extrinsics (yz-flipped, converted by prepare_data/convert_data.sh; original backed up to lidar.yaml.bak if one existed, marked done via .yaml_converted)
│  ├─ frame_list.txt        # frame index -> pcd filename mapping
│  │
│  ├─ label/                # SuperbAI sessions: raw KITTI-format export, z=bbox center (converted in place, see step 1a)
│  └─ tracklet_labels.xml   # CVAT sessions: raw tracklet export (identifies the session as CVAT)
```

SuperbAI and CVAT are told apart by `tracklet_labels.xml`: if it's present, the session is
CVAT; otherwise it's SuperbAI. SuperbAI exports always arrive as plain `label/` (never a
separate `label_original/` or `meta/*.json` — verified against every session in
production data).

## Pipeline (`annos.sh <DATA_ROOT>`)

| # | Script | In | Out | Applies to |
|---|---|---|---|---|
| 0 | `check_same_timestamp.sh` | `<data_name>/pcd/*.avikus.pcd` (all sessions) | pass/fail only | both — guards against two sessions from different sources landing on the same DATA_ROOT with a colliding timestamp |
| 1a | `generate_cvat_label.py` | `tracklet_labels.xml` + `frame_list.txt`, `pcd/`, `images/`, `lidar.yaml` | `<data_name>/label/*.txt` | CVAT sessions only (skips any session without `tracklet_labels.xml`) |
| 1b | `convert_label_z_center_to_bottom.py` | `<data_name>/label/*.txt` for sessions without `tracklet_labels.xml` (i.e. SuperbAI) | `<data_name>/label/*.txt`, z shifted from bbox-center to bbox-bottom-center in place | SuperbAI sessions only (marker file `.z_converted` makes this idempotent across re-runs; height read from field 10, empirically verified against real point-cloud extents) |
| 2 | `split_train_val.py` | every `<data_name>/label/*.txt` produced above | `DATA_ROOT/{train,val,test}.txt` | both (walks the whole merged root) |
| 3 | `generate_calib.py` | `<data_name>/lidar.yaml` | `<data_name>/calib_<data_name>.txt` | both |
| 4 | `pre_process_kitti.py --prefix avikus [--compensate_imu] [--min_pts_filter N]` | `{train,val,test}.txt`, `pcd/`, `images/`, `calib_*.txt`, `label/*.txt`, optional `oru/attitude.csv` | `velodyne_reduced/*.avikus.pcd`, `avikus_infos_{train,val,test,trainval}.pkl`, `avikus_dbinfos_train.pkl` + `avikus_gt_database/*.bin` | both |
| 5 | `set_anchor.py` | `avikus_dbinfos_train.pkl` | prints suggested per-class anchor `w l h`, and auto-patches the matching rows into `pointpillars/model/pointpillars.py`'s `sizes` list | both |
| 6 | `visualization/vis_gt_pc.py --root DATA_ROOT` | a random processed session/frame | pops up a 3D Open3D view of it | both (not fatal to the pipeline if it fails, e.g. no DISPLAY — just warns) |

Step 1a only touches CVAT sessions (has `tracklet_labels.xml`). Step 1b picks up
everything else that has a `label/` (SuperbAI sessions) — since the two are told apart by
`tracklet_labels.xml`'s presence, a mixed `DATA_ROOT` works without picking a source
upfront. Steps 2–6 are source-agnostic and run once over the combined dataset.

## Legacy scripts

`cvat.sh` is an earlier, narrower pipeline that only handled CVAT sessions, kept for
reference / reprocessing old `cvat_test` exports with its original logic (no IMU
compensation):

- `cvat.sh` -> `generate_cvat_label.py`, `split_train_val_cvat.py`, `generate_calib_cvat.py`,
  `pre_process_kitti_cvat.py`, `set_anchor.py`
