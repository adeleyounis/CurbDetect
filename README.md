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