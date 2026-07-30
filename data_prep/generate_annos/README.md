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
│  │  # exactly one of the following two identifies the session's label source:
│  ├─ label_original/       # SuperbAI: raw KITTI-format label export, as-is
│  ├─ meta/*.json           # SuperbAI: labeling-tool JSON export (used if label_original/ absent)
│  └─ tracklet_labels.xml   # CVAT: raw tracklet export
```

## Pipeline (`annos.sh <DATA_ROOT>`)

| # | Script | In | Out | Applies to |
|---|---|---|---|---|
| 0 | `check_same_timestamp.sh` | `<data_name>/pcd/*.avikus.pcd` (all sessions) | pass/fail only | both — guards against two sessions from different sources landing on the same DATA_ROOT with a colliding timestamp |
| 1a | `generate_superb_label.py` | `label_original/` or `meta/*.json` + `images/`, `pcd/`, `lidar.yaml` | `<data_name>/label/*.txt` | SuperbAI sessions only (per-session: uses `label_original/` if present, else `meta/*.json`, else skips the session) |
| 1b | `generate_cvat_label.py` | `tracklet_labels.xml` + `frame_list.txt`, `pcd/`, `images/`, `lidar.yaml` | `<data_name>/label/*.txt` | CVAT sessions only (skips any session without `tracklet_labels.xml`) |
| 1c | `convert_label_z_center_to_bottom.py` | `<data_name>/label/*.txt` for sessions with none of `label_original/`, matching `meta/*.json`, or `tracklet_labels.xml` (raw-capture sessions that arrive with `label/` already populated) | `<data_name>/label/*.txt`, z shifted from bbox-center to bbox-bottom-center in place | raw-capture sessions only (marker file `.z_converted` makes this idempotent across re-runs) |
| 2 | `split_train_val.py` | every `<data_name>/label/*.txt` produced above | `DATA_ROOT/{train,val,test}.txt` | both (walks the whole merged root) |
| 3 | `generate_calib.py` | `<data_name>/lidar.yaml` | `<data_name>/calib_<data_name>.txt` | both |
| 4 | `pre_process_kitti.py --prefix avikus [--compensate_imu] [--min_pts_filter N]` | `{train,val,test}.txt`, `pcd/`, `images/`, `calib_*.txt`, `label/*.txt`, optional `oru/attitude.csv` | `velodyne_reduced/*.avikus.pcd`, `avikus_infos_{train,val,test,trainval}.pkl`, `avikus_dbinfos_train.pkl` + `avikus_gt_database/*.bin` | both |
| 5 | `set_anchor.py` | `avikus_dbinfos_train.pkl` | prints suggested per-class anchor `w l h`, and auto-patches the matching rows into `pointpillars/model/pointpillars.py`'s `sizes` list | both |
| 6 | `visualization/vis_gt_pc.py --root DATA_ROOT` | a random processed session/frame | pops up a 3D Open3D view of it | both (not fatal to the pipeline if it fails, e.g. no DISPLAY — just warns) |

Steps 1a/1b run unconditionally and both **self-filter**: each session folder is only
processed by the generator matching its own label source, so a mixed `DATA_ROOT` works
without picking a source upfront. Step 1c picks up whatever's left over (raw-capture
sessions handed off with `label/` already populated). Steps 2–6 are source-agnostic and
run once over the combined dataset.

### SuperbAI raw export: `label/` -> `label_original/` without a manual rename

`generate_superb_label.py`'s `label_original/` path expects the *raw* SuperbAI export
(bbox-center z). If your export tool instead hands off that same raw data already named
`label/`, don't rename it by hand — just touch an empty flag file in the session dir:

```bash
touch DATA_ROOT/<data_name>/.needs_label_conversion
```

On the next `./annos.sh` run, `generate_superb_label.py` will rename that session's
`label/` -> `label.bak/`, regenerate `label/` from it via the same conversion
`label_original/` gets, and consume the flag (so re-running `annos.sh` doesn't redo it).
Sessions without the flag are untouched — e.g. raw-capture data whose `label/` is already
in final format only goes through step 1c above.

## Legacy scripts

`cvat.sh` and `superb.sh` are earlier, narrower pipelines that only handled one label
source each, kept for reference / reprocessing old `cvat_test` or `data_to_superbai_1114`
exports with their original logic (no IMU compensation, and — for
`pre_process_kitti_superb.py` specifically — the older combined `sample/` folder instead
of `images/`+`pcd/`):

- `cvat.sh` -> `generate_cvat_label.py`, `split_train_val_cvat.py`, `generate_calib_cvat.py`,
  `pre_process_kitti_cvat.py`, `set_anchor.py`
- `superb.sh` -> `generate_superb_label.py`, `split_train_val_superb.py`,
  `generate_calib_superb.py`, `pre_process_kitti_superb.py`, `set_anchor.py`

`README_superb_legacy.md` documents the original SuperbAI-only labeling workflow.
