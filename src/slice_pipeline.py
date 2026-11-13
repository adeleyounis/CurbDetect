"""
Per-frame curb slice pipeline, usable outside the notebook.

Same steps as heights_per_slice.ipynb (load + project + visibility filter, then one boundary fit
per camera-depth band), plus what the viewer and the interpolation need for each slice:
the curb side and the indices of the LiDAR points used by the fit.

    fr = load_frame(root, seq, split, cam_id, frame)
    results = run_slices(fr)                 # list, one dict per depth band
    save_frame(fr, results, out_dir)         # -> <seq>_cam<id>_frame<NNNNNN>_slices.npz

The notebook itself is not changed; this module mirrors its logic.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter

import analyze_boundary
import data_loader
import results_io

# defaults (same values as the notebook)
DEFAULT_SEQ = "2013_05_28_drive_0000_sync"
DEFAULT_SPLIT = "train"
ROAD_ID = 7
SIDEWALK_ID = 8


def visible_mask(u, v, z, W, H, k=5, tol=0.7):
    """Keep points that are not hidden behind a closer point (z-buffer with a k x k min filter)."""
    depth = np.full((H, W), np.inf)
    u_i, v_i = u.astype(int), v.astype(int)
    order = np.argsort(-z)
    depth[v_i[order], u_i[order]] = z[order]
    local_min = minimum_filter(depth, size=k)
    return z <= local_min[v_i, u_i] + tol


def load_frame(root, seq=DEFAULT_SEQ, split=DEFAULT_SPLIT, cam_id=0, frame=250):
    """Load the static cloud, pose, calibration and image path for one frame, and project.

    Returns a dict with the cloud, the visible points (u, v, z, idx), the camera and the image path.
    """
    root = Path(root)
    proj_matrix, W, H = data_loader.load_perspective_calibration(root, cam_id)
    poses = data_loader.load_cam0_to_world(root, seq)
    if frame not in poses:
        raise KeyError(f"frame {frame} has no pose in cam0_to_world.txt")
    cam_to_world = poses[frame]
    chunks = data_loader.list_static_chunks(root, split, seq)
    cloud = data_loader.load_cloud_for_frame(chunks, frame)
    u, v, z, idx = data_loader.project(cloud["xyz"], cam_to_world, proj_matrix, W, H)
    vis = visible_mask(u, v, z, W, H)
    return dict(cloud=cloud, u=u[vis], v=v[vis], z=z[vis], idx=idx[vis],
                cam_to_world=cam_to_world, proj_matrix=proj_matrix.astype(float), W=W, H=H,
                img_path=data_loader.get_img_path(root, seq, cam_id, frame),
                frame=frame, seq=seq, cam_id=cam_id)


def run_slices(fr, road_id=ROAD_ID, sidewalk_id=SIDEWALK_ID, n=10, candidate_num=5, seed=0,
               band_halfwidth=2.0):
    """One boundary fit per camera-depth band (the notebook's run_experiment, without plotting).

    seed=None reproduces the notebook exactly (random candidates every run);
    an int makes the run reproducible (band i uses seed + i).

    Each entry: depth, height_cm, sidewalk_slope, fit, result (estimator output or None),
    anchor, side ("left"/"right"/None), cloud_idx (indices into fr["cloud"] of the slice's points),
    band (boolean mask over the visible points).
    """
    cloud, z, idx = fr["cloud"], fr["z"], fr["idx"]
    ground_visible = np.isin(cloud["semantic"][idx], [road_id, sidewalk_id])
    if not ground_visible.any():
        return []
    target_depths = np.linspace(*np.percentile(z[ground_visible], [10, 90]), n)

    out = []
    for i, target in enumerate(target_depths, start=1):
        band = ground_visible & (np.abs(z - target) <= band_halfwidth)
        depth_cloud = {k: val[idx[band]] for k, val in cloud.items()}
        band_seed = None if seed is None else seed + i
        anchors = analyze_boundary.boundary_candidates(depth_cloud, road_id, sidewalk_id,
                                                       n=candidate_num, seed=band_seed)
        result, anchor_used = None, None
        for anchor in anchors:  # first candidate that gives a fit, as in the notebook
            cand = analyze_boundary.boundary_estimator(depth_cloud, anchor, road_id, sidewalk_id)
            if cand is not None:
                result, anchor_used = cand, anchor
                break

        cloud_idx = side = None
        if result is not None:
            cloud_idx = idx[np.flatnonzero(band)[result["point_idx"]]]
            side = analyze_boundary.get_curb_side(result, fr["cam_to_world"], fr["proj_matrix"])

        out.append(dict(
            depth=target,
            height_cm=result["step_cm"] if result else np.nan,
            sidewalk_slope=result["sidewalk_slope"] if result else np.nan,
            fit=result["fit"] if result else None,
            result=result, anchor=anchor_used, side=side, cloud_idx=cloud_idx, band=band,
        ))
    return out


def slices_filename(seq, cam_id, frame):
    return f"{seq}_cam{cam_id}_frame{frame:06d}_slices.npz"


def save_frame(fr, results, out_dir, road_id=ROAD_ID, sidewalk_id=SIDEWALK_ID, **kwargs):
    """Write the frame's slices to <out_dir>/<seq>_cam<id>_frame<NNNNNN>_slices.npz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / slices_filename(fr["seq"], fr["cam_id"], fr["frame"])
    return results_io.save_frame_results(
        path, results, fr["cloud"], fr["idx"], fr["cam_to_world"], fr["proj_matrix"], fr["img_path"],
        frame=fr["frame"], seq=fr["seq"], cam_id=fr["cam_id"],
        road_id=road_id, sidewalk_id=sidewalk_id, **kwargs)