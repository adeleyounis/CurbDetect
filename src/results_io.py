"""
Save / load per-frame curb-fit results to a single .npz file.

Global keys hold one row per successful slice (distances, anchors_3d, betas, ...).
Per-slice point data has a variable length, so it is stored as slice_{k}_<name>.

Everything is stored as plain arrays, so the file loads with allow_pickle=False.
Units: metres everywhere, except keys ending in _cm.
"""
from pathlib import Path
import numpy as np

import analyze_boundary

FORMAT_VERSION = 1


def _to_image(points_xyz, cam_to_world, proj_matrix):
    """World XYZ -> pixel uv (NaN where behind the camera)."""
    world_to_cam = np.linalg.inv(np.asarray(cam_to_world, float))
    P = np.asarray(proj_matrix, float)
    pc = points_xyz @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    uvw = pc @ P[:, :3].T + P[:, 3]
    uv = np.full((len(points_xyz), 2), np.nan)
    ok = uvw[:, 2] > 0
    uv[ok] = uvw[ok, :2] / uvw[ok, 2:3]
    return uv


def sample_profile(res, n_samples=200):
    """Evaluate the fitted profile over the observed across-curb range.

    Returns s (n,), xyz (n, 3) in world coordinates.
    xy = mean + s * normal  (the line through the PCA origin, perpendicular to the curb).
    """
    fit = res["fit"]
    centers = res["bins"][0]
    s = np.linspace(centers.min(), centers.max(), n_samples)
    z = analyze_boundary.design_matrix(s, fit["c"], fit["width"]) @ fit["beta"]
    xy = np.asarray(res["mean"]) + s[:, None] * np.asarray(res["normal"])
    return s, np.column_stack((xy, z))


def save_frame_results(path, results, cloud, vis_idx, cam_to_world, proj_matrix,
                       image_path, *, frame, seq, cam_id, road_id, sidewalk_id,
                       n_curve=200, context_margin=3.0, context_max=40_000, seed=0):
    """
    results      : list returned by run_experiment (one entry per depth band; failed bands
                   have result=None). Each successful entry needs "result" with "fit" and
                   "point_idx", plus "cloud_idx" and "side" added in run_experiment.
    cloud        : the full frame cloud dict (xyz, semantic, ...)
    vis_idx      : `idx` from the notebook after the visibility filter (full-cloud indices)
    """
    path = Path(path)
    cam_to_world = np.asarray(cam_to_world, float)
    proj_matrix = np.asarray(proj_matrix, float)

    ok = [r for r in results if r["result"] is not None]
    K = len(ok)
    if K == 0:
        raise ValueError("No successful slices to save.")

    out = dict(
        format_version=np.int64(FORMAT_VERSION),
        image_path=np.array(str(image_path)),
        frame=np.int64(frame), seq=np.array(seq), cam_id=np.int64(cam_id),
        road_id=np.int64(road_id), sidewalk_id=np.int64(sidewalk_id),
        cam_to_world=cam_to_world, proj_matrix=proj_matrix,
        # record of every band, including failures (NaN)
        all_depths=np.array([r["depth"] for r in results], float),
        all_heights_cm=np.array([r["height_cm"] for r in results], float),
        n_slices=np.int64(K),
    )

    distances, anchors_3d, normals, means, curves = [], [], [], [], []
    s_steps, steps_cm, ramp_cm, side_slope = [], [], [], []
    betas, cs, widths, mses, sides, band_ids = [], [], [], [], [], []

    for k, item in enumerate(ok):
        res = item["result"]
        fit = res["fit"]

        s, xyz = sample_profile(res, n_curve)
        c = fit["c"]
        z_c = analyze_boundary.design_matrix(np.array([c]), c, fit["width"]) @ fit["beta"]
        anchor = np.r_[np.asarray(res["mean"]) + c * np.asarray(res["normal"]), z_c[0]]

        distances.append(item["depth"]); anchors_3d.append(anchor)
        normals.append(res["normal"]); means.append(res["mean"]); curves.append(xyz)
        s_steps.append(res["s_step"]); steps_cm.append(res["step_cm"])
        ramp_cm.append(res["ramp_width_cm"]); side_slope.append(res["sidewalk_slope"])
        betas.append(fit["beta"]); cs.append(c); widths.append(fit["width"]); mses.append(fit["mse"])
        sides.append(item.get("side") or "unknown")
        band_ids.append(results.index(item))

        # per-slice ragged data
        cloud_idx = np.asarray(item["cloud_idx"])
        out[f"slice_{k}_xyz"] = cloud["xyz"][cloud_idx]
        out[f"slice_{k}_s"] = np.asarray(res["pts"], float)
        out[f"slice_{k}_z"] = np.asarray(res["z"], float)
        out[f"slice_{k}_lab"] = np.asarray(res["lab"])
        out[f"slice_{k}_bins"] = np.column_stack(res["bins"])
        out[f"slice_{k}_fit_sz"] = np.column_stack((s, xyz[:, 2]))
        out[f"slice_{k}_cloud_idx"] = cloud_idx  # indices into the full frame cloud

    anchors_3d = np.array(anchors_3d)
    out.update(
        distances=np.array(distances), anchors_3d=anchors_3d,
        anchors_uv=_to_image(anchors_3d, cam_to_world, proj_matrix),
        normals_xy=np.array(normals), means_xy=np.array(means),
        curves_xyz=np.array(curves),                     # (K, n_curve, 3)
        s_steps=np.array(s_steps), steps_cm=np.array(steps_cm),
        ramp_widths_cm=np.array(ramp_cm), sidewalk_slopes=np.array(side_slope),
        betas=np.array(betas), fit_c=np.array(cs), fit_width=np.array(widths),
        fit_mse=np.array(mses), sides=np.array(sides), band_index=np.array(band_ids),
    )

    # context cloud: visible road/sidewalk points near the slices, subsampled
    vis_idx = np.asarray(vis_idx)
    sem = cloud["semantic"][vis_idx]
    ground = vis_idx[np.isin(sem, [road_id, sidewalk_id])]
    xyz_g = cloud["xyz"][ground]
    all_slice_xy = np.concatenate([out[f"slice_{k}_xyz"][:, :2] for k in range(K)])
    lo = all_slice_xy.min(0) - context_margin
    hi = all_slice_xy.max(0) + context_margin
    near = np.all((xyz_g[:, :2] >= lo) & (xyz_g[:, :2] <= hi), axis=1)
    ctx = ground[near]
    if len(ctx) > context_max:
        ctx = np.random.default_rng(seed).choice(ctx, context_max, replace=False)
    out["context_xyz"] = cloud["xyz"][ctx]
    out["context_lab"] = cloud["semantic"][ctx]
    out["context_cloud_idx"] = ctx

    np.savez_compressed(path, **out)
    print(f"saved {K} slices ({len(results) - K} failed bands) -> {path}")
    return path


def load_frame_results(path):
    """Load the .npz back into (meta dict, list of per-slice dicts)."""
    d = np.load(path, allow_pickle=False)
    K = int(d["n_slices"])
    per_slice_keys = ("xyz", "s", "z", "lab", "bins", "fit_sz", "cloud_idx")
    global_keys = [k for k in d.files if not k.startswith("slice_")]
    meta = {k: d[k] for k in global_keys}
    slices = []
    for k in range(K):
        sl = {name: d[f"slice_{k}_{name}"] for name in per_slice_keys}
        sl.update(
            depth=float(d["distances"][k]), step_cm=float(d["steps_cm"][k]),
            side=str(d["sides"][k]), mean=d["means_xy"][k], normal=d["normals_xy"][k],
            fit=dict(c=float(d["fit_c"][k]), width=float(d["fit_width"][k]),
                     beta=d["betas"][k], mse=float(d["fit_mse"][k])),
            curve_xyz=d["curves_xyz"][k], anchor_3d=d["anchors_3d"][k],
        )
        slices.append(sl)
    return meta, slices