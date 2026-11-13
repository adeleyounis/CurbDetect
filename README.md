# CurbDetect

## Requirements

- **Python 3.12+** (required by `numpy==2.5.4` and `scipy==1.18.1`)

### Installation

From the project root (`CurbDetect/`), install the required dependencies in your Python 3.12+ environment:

```bash
pip install -r requirements.txt
```

## Dataset Setup (KITTI-360)

This project uses the **KITTI-360** dataset.

1. Register and log in on the [KITTI-360 Download Page](https://www.cvlibs.net/datasets/kitti-360/download.php).
2. Create a `KITTI-360` directory inside `CurbDetect/` and download/extract the following items into it:
   - **Calibrations** (`calibration/`)
   - **Perspective Images / 2D Raw** (`data_2d_raw/`, via `download_2d_perspective.sh`)
   - **2D Semantic Labels** (`data_2d_semantics/`)
   - **3D Bounding Boxes** (`data_3d_bboxes/`)
   - **3D Semantics** (`data_3d_semantics/`) — labeled on the website as **"Accumulated Point Clouds for Train & Val (12G)"**
   - **Vehicle Poses** (`data_poses/`)

### Expected Directory Structure

```text
CurbDetect/
├── KITTI-360/
│   ├── calibration/
│   ├── data_2d_raw/
│   ├── data_2d_semantics/
│   ├── data_3d_bboxes/
│   ├── data_3d_semantics/
│   ├── data_poses/
│   ├── download_2d_perspective/
│   ├── download_2d_perspective.sh
│   └── download_3d_velodyne.sh
├── src/
├── README.md
└── requirements.txt
```

## Curb slices: compute, check, and view

The notebook `heights_per_slice.ipynb` is the exploratory version of the pipeline and is unchanged.
The scripts below run the same steps outside the notebook, save the results per frame, and let you
inspect them in an interactive 3D viewer.

### Install

On top of the existing dependencies (`numpy`, `scipy`, `matplotlib`, `pillow`, `plyfile`):

```
pip install viser plotly
```

### Run everything (from `src/`)

```bash
# 1. Compute curb slices for some frames -> one .npz per frame in ../outputs/slices/
python compute_slices.py --frames 250 325
python compute_slices.py --range 200 400 25          # frames 200, 225, ..., 375

# 2. (optional) sanity-check the pipeline on those frames -> ../outputs/checks/
python check_pipeline.py --frames 250 325

# 3. Open the viewer, then go to http://localhost:8080
python curb_viewer.py ../outputs/slices               # all frames in the folder (Frame dropdown)
python curb_viewer.py ../outputs/slices/<file>.npz    # a single frame
```

Default paths assume this layout (all overridable with `--root` / `--out`):

```
CurbDetect/
├── KITTI-360/            # dataset (calibration/, data_poses/, data_3d_semantics/, data_2d_raw/)
├── outputs/
│   ├── slices/           # compute_slices.py output: <seq>_cam<id>_frame<NNNNNN>_slices.npz
│   └── checks/           # check_pipeline.py output: .npz + check_frame<N>.png overlays
└── src/                  # run the scripts from here
```

Add `outputs/` to `.gitignore`.

Useful `compute_slices.py` options: `--n` (number of depth bands, 10), `--candidates` (boundary
candidates tried per band, 5), `--seed` (0; `-1` = random every run, like the notebook),
`--seq`, `--split`, `--cam`, `--road-id 7`, `--sidewalk-id 8`, `--context-max 40000`
(LiDAR context points stored for the viewer).

### What the pipeline does (per frame)

1. Load the static point cloud chunk(s), pose, calibration and image; project the points into the
   camera and keep the visible ones (z-buffer).
2. Pick `n` target camera depths between the 10th and 90th percentile of visible road/sidewalk
   depths. Each depth band keeps points within ±2 m of its target.
3. In each band, find road/sidewalk boundary candidates and fit the first one that works with
   `analyze_boundary.boundary_estimator` (quadratic road + linear ramp + linear sidewalk).
4. For each fit, record which curb it is on (`left`/`right`) and which LiDAR points it used.
5. Save the fits, their points, the camera, and a LiDAR context sample to one `.npz` per frame.

### `slice_pipeline.py`: use the pipeline from your own code

```python
import slice_pipeline as sp

fr = sp.load_frame(root, seq=sp.DEFAULT_SEQ, split="train", cam_id=0, frame=325)
results = sp.run_slices(fr, n=10, candidate_num=5, seed=0)
path = sp.save_frame(fr, results, "../outputs/slices")
```

| Function | Returns |
|---|---|
| `load_frame(root, seq, split, cam_id, frame)` | `fr` dict: `cloud` (xyz, semantic, …), visible points `u, v, z, idx` (`idx` = indices into `cloud`), `cam_to_world`, `proj_matrix`, `W`, `H`, `img_path`, `frame`, `seq`, `cam_id` |
| `run_slices(fr, road_id=7, sidewalk_id=8, n=10, candidate_num=5, seed=0, band_halfwidth=2.0)` | list with one dict per depth band: `depth`, `height_cm`, `sidewalk_slope`, `fit` (`c`, `width`, `beta`, `mse`), `result` (full estimator output, or `None` if the band failed), `anchor`, `side`, `cloud_idx` (indices into `fr["cloud"]` of the points used), `band` (mask over visible points) |
| `save_frame(fr, results, out_dir, ...)` | path of the written `.npz` |
| `visible_mask(u, v, z, W, H)` | boolean visibility mask (same as the notebook) |
| `slices_filename(seq, cam_id, frame)` | the standard file name |

`seed=None` reproduces the notebook exactly (random candidates every run); an integer makes runs
reproducible (band *i* uses `seed + i`).

### Reading the `.npz`

```python
import results_io
meta, slices = results_io.load_frame_results("../outputs/slices/<file>.npz")
s = slices[0]
s["step_cm"], s["side"], s["fit"]["beta"]   # height, curb side, [b, m_r, h, m_s, q]
s["xyz"], s["s"], s["z"], s["lab"]          # the LiDAR points the fit used (world xyz, across-curb s, z, label)
s["curve_xyz"]                              # fitted profile sampled in world XYZ
```

Only successful slices are stored; `meta["all_depths"]` / `meta["all_heights_cm"]` keep every band
(failed = NaN). Units are metres, except keys ending in `_cm`. The full key list is in the
`results_io.py` docstring and in the PR description.

### Viewer controls

| | |
|---|---|
| **← / →** | previous / next slice (click the 3D view first so the page has focus) |
| **Frame** dropdown | switch frame (when you open a folder or several files) |
| **Curb side** | show both curbs, or only left / right slices |
| click an orb or ribbon | select that slice |
| **shift + click** | pick any LiDAR point in 3D → green ball in 3D, green cross in the photo, height above road in the panel |
| mouse, WASD, ↑/↓, Q/E | move the 3D camera |
| **Focus on selected slice** / **View from camera** | camera shortcuts |

Panels: **Photo** (camera image with all slices; selected slice in magenta, its points coloured
road blue / sidewalk red; plus a zoom on it), **Cross-section** (raw points, bin medians, fitted
profile, ramp region), **Heights by depth**, **3D display** (colour mode, vertical exaggeration,
point size, layers), **Curb ribbons** (length, margin, or "length = data extent").

The side panel is docked on the right; drag its edge to resize it (`--panel-width` sets the start
width). Other options: `--port 8080`; arrow keys use a second local port, `--key-port`
(default port + 1). If you run the viewer on a remote machine, forward both ports.