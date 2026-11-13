"""
Sanity checks for the updated analyze_boundary.py and results_io.py on real KITTI-360 frames.

Run from src/ (next to data_loader.py, analyze_boundary.py, results_io.py):

    python check_pipeline.py --frames 250 325 1000

If you keep a copy of the ORIGINAL analyze_boundary.py as analyze_boundary_old.py next to it,
the script also checks that the new estimator gives identical numbers to the old one.

For each frame it:
  1. runs the same slice experiment as the notebook (no plotting),
  2. checks the new estimator outputs (fit, point_idx) are consistent,
  3. checks that the same seed reproduces the same results,
  4. saves to .npz, reloads with allow_pickle=False, and compares everything,
  5. writes outputs/check_frame<N>.png: the image with each slice's points and fitted
     profile projected on it, so you can see the slices sit on the curb.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter

import analyze_boundary
import data_loader
import results_io

try:
    import analyze_boundary_old
except ImportError:
    analyze_boundary_old = None


# ---------------------------------------------------------------------------
# Same steps as the notebook (cells 1, 2 and run_experiment), without plotting
# ---------------------------------------------------------------------------
def visible_mask(u, v, z, W, H, k=5, tol=0.7):
    depth = np.full((H, W), np.inf)
    u_i, v_i = u.astype(int), v.astype(int)
    order = np.argsort(-z)
    depth[v_i[order], u_i[order]] = z[order]
    local_min = minimum_filter(depth, size=k)
    return z <= local_min[v_i, u_i] + tol


def load_frame(root, seq, split, cam_id, frame):
    proj_matrix, W, H = data_loader.load_perspective_calibration(root, cam_id)
    poses = data_loader.load_cam0_to_world(root, seq)
    if frame not in poses:
        raise KeyError(f"frame {frame} has no pose in cam0_to_world.txt")
    cam_to_world = poses[frame]
    if cam_id != 0:
        print("  NOTE: cam_id != 0 but the pose is cam0_to_world (same as the notebook)")
    chunks = data_loader.list_static_chunks(root, split, seq)
    cloud = data_loader.load_cloud_for_frame(chunks, frame)
    u, v, z, idx = data_loader.project(cloud["xyz"], cam_to_world, proj_matrix, W, H)
    vis = visible_mask(u, v, z, W, H)
    img_path = data_loader.get_img_path(root, seq, cam_id, frame)
    return dict(cloud=cloud, u=u[vis], v=v[vis], z=z[vis], idx=idx[vis],
                cam_to_world=cam_to_world, proj_matrix=proj_matrix.astype(float),
                W=W, H=H, img_path=img_path, frame=frame, seq=seq, cam_id=cam_id)


def run_slices(fr, road_id, sidewalk_id, n=10, candidate_num=5, seed=0, ab=analyze_boundary):
    """Mirror of the notebook's run_experiment(plot=False)."""
    cloud, z, idx = fr["cloud"], fr["z"], fr["idx"]
    ground_visible = np.isin(cloud["semantic"][idx], [road_id, sidewalk_id])
    if not ground_visible.any():
        return []
    target_depths = np.linspace(*np.percentile(z[ground_visible], [10, 90]), n)

    out = []
    for i, target in enumerate(target_depths, start=1):
        band = ground_visible & (np.abs(z - target) <= 2.0)
        depth_cloud = {k: val[idx[band]] for k, val in cloud.items()}
        band_seed = None if seed is None else seed + i
        anchors = ab.boundary_candidates(depth_cloud, road_id, sidewalk_id,
                                         n=candidate_num, seed=band_seed)
        result, anchor_used = None, None
        for anchor in anchors:
            cand = ab.boundary_estimator(depth_cloud, anchor, road_id, sidewalk_id)
            if cand is not None:
                result, anchor_used = cand, anchor
                break
        cloud_idx = side = None
        if result is not None:
            cloud_idx = idx[np.flatnonzero(band)[result["point_idx"]]]
            side = ab.get_curb_side(result, fr["cam_to_world"], fr["proj_matrix"])
        out.append(dict(
            depth=target,
            height_cm=result["step_cm"] if result else np.nan,
            sidewalk_slope=result["sidewalk_slope"] if result else np.nan,
            fit=result["fit"] if result else None,
            result=result, anchor=anchor_used, side=side, cloud_idx=cloud_idx,
            all_anchors=anchors, depth_cloud_keys=(band,),   # kept for the regression check
        ))
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.fails, self.warns = 0, 0

    def ok(self, cond, msg, warn_only=False):
        tag = "PASS" if cond else ("WARN" if warn_only else "FAIL")
        if not cond:
            if warn_only:
                self.warns += 1
            else:
                self.fails += 1
        print(f"  [{tag}] {msg}")
        return cond


def check_frame(fr, road_id, sidewalk_id, out_dir, n=10, candidate_num=5, seed=0, make_png=True):
    rep = Report()
    cloud = fr["cloud"]
    print(f"\n=== frame {fr['frame']} ({len(cloud['xyz']):,} pts, {len(fr['idx']):,} visible) ===")

    results = run_slices(fr, road_id, sidewalk_id, n, candidate_num, seed)
    good = [r for r in results if r["result"] is not None]
    for i, r in enumerate(results, 1):
        print(f"  #{i:02d} depth={r['depth']:5.1f} m  height={r['height_cm']:6.1f} cm  side={r['side']}")
    if not rep.ok(len(good) > 0, f"{len(good)}/{len(results)} bands produced a fit", warn_only=True):
        print("  nothing to check further for this frame")
        return rep

    # 1. new estimator outputs are internally consistent
    for k, r in enumerate(good):
        res, fit = r["result"], r["result"]["fit"]
        xyz = cloud["xyz"][r["cloud_idx"]]
        tag = f"slice {k} (depth {r['depth']:.1f} m)"
        good_pts = (len(res["point_idx"]) == len(res["pts"]) == len(res["z"]) == len(res["lab"]))
        rep.ok(good_pts, f"{tag}: point_idx length matches pts/z/lab ({len(res['pts'])})")
        rep.ok(np.allclose(xyz[:, 2], res["z"]), f"{tag}: cloud z at cloud_idx == res['z']")
        rep.ok(np.allclose((xyz[:, :2] - res["mean"]) @ res["normal"], res["pts"]),
               f"{tag}: (xy - mean)·normal == res['pts']")
        rep.ok(np.array_equal(cloud["semantic"][r["cloud_idx"]], res["lab"]),
               f"{tag}: labels at cloud_idx == res['lab']")
        rep.ok(np.isclose(fit["beta"][2] * 100, res["step_cm"]) and np.isclose(fit["c"], res["s_step"])
               and np.isclose(fit["width"] * 100, res["ramp_width_cm"]),
               f"{tag}: fit dict agrees with step_cm / s_step / ramp_width_cm")
        cx, cz = res["fit_curve"]
        rep.ok(np.allclose(analyze_boundary.design_matrix(cx, fit["c"], fit["width"]) @ fit["beta"], cz),
               f"{tag}: beta reproduces fit_curve")

    # 2. regression against the original estimator (same anchor -> same numbers)
    if analyze_boundary_old is not None:
        same = True
        for r in good:
            band = r["depth_cloud_keys"][0]
            depth_cloud = {k: val[fr["idx"][band]] for k, val in cloud.items()}
            try:
                old = analyze_boundary_old.boundary_estimator(depth_cloud, r["anchor"], road_id, sidewalk_id)
            except Exception as e:  # old code could crash where the new one returns None
                print(f"    old estimator raised {type(e).__name__} at depth {r['depth']:.1f}")
                same = False
                continue
            for key in ("step_cm", "s_step", "ramp_width_cm", "sidewalk_slope"):
                same &= old is not None and np.isclose(old[key], r["result"][key])
            same &= old is not None and np.allclose(old["pts"], r["result"]["pts"])
        rep.ok(same, "new estimator == original estimator on the same anchors")
    else:
        print("  [skip] no analyze_boundary_old.py found, regression check skipped")

    # 3. same seed -> same results
    again = run_slices(fr, road_id, sidewalk_id, n, candidate_num, seed)
    rep.ok(all(np.array_equal(a["height_cm"], b["height_cm"], equal_nan=True) for a, b in zip(results, again)),
           f"seed={seed} gives identical heights on a second run")

    # 4. curb sides
    sides = [r["side"] for r in good]
    counts = {s: sides.count(s) for s in set(sides)}
    rep.ok(len(counts) == 1 and None not in counts,
           f"all slices on one curb side {counts}" + ("" if len(counts) == 1 else
           " -> slices from different curbs; group by side before interpolating"), warn_only=True)

    # 5. save -> load round trip
    npz = Path(out_dir) / f"{fr['seq']}_cam{fr['cam_id']}_frame{fr['frame']:06d}_slices.npz"
    try:
        results_io.save_frame_results(
            npz, results, cloud, fr["idx"], fr["cam_to_world"], fr["proj_matrix"], fr["img_path"],
            frame=fr["frame"], seq=fr["seq"], cam_id=fr["cam_id"],
            road_id=road_id, sidewalk_id=sidewalk_id)
        meta, slices = results_io.load_frame_results(npz)
        rep.ok(len(slices) == len(good), f"reloaded {len(slices)} slices (allow_pickle=False)")
        rt = True
        for sl, r in zip(slices, good):
            res = r["result"]
            rt &= np.array_equal(sl["cloud_idx"], r["cloud_idx"])
            rt &= np.allclose(sl["s"], res["pts"]) and np.allclose(sl["z"], res["z"])
            rt &= np.array_equal(sl["lab"], res["lab"])
            rt &= np.allclose(sl["fit"]["beta"], res["fit"]["beta"])
            rt &= np.isclose(sl["fit"]["c"], res["fit"]["c"]) and np.isclose(sl["fit"]["width"], res["fit"]["width"])
            rt &= np.isclose(sl["step_cm"], res["step_cm"]) and sl["side"] == (r["side"] or "unknown")
            rt &= np.allclose(sl["mean"], res["mean"]) and np.allclose(sl["normal"], res["normal"])
            # the saved curve must lie on the fitted profile
            s_curve = (sl["curve_xyz"][:, :2] - sl["mean"]) @ sl["normal"]
            z_model = analyze_boundary.design_matrix(s_curve, sl["fit"]["c"], sl["fit"]["width"]) @ sl["fit"]["beta"]
            rt &= np.allclose(sl["curve_xyz"][:, 2], z_model)
        rep.ok(bool(rt), "saved arrays match the in-memory results exactly")
        rep.ok(np.allclose(meta["all_heights_cm"], [r["height_cm"] for r in results], equal_nan=True),
               "failed bands kept as NaN in all_heights_cm")
        uv = meta["anchors_uv"]
        inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < fr["W"]) & (uv[:, 1] >= 0) & (uv[:, 1] < fr["H"])
        rep.ok(inside.all(), f"{inside.sum()}/{len(uv)} curb anchors project inside the image", warn_only=True)
        print(f"  file: {npz} ({npz.stat().st_size / 1e6:.1f} MB)")
    except Exception:
        traceback.print_exc()
        rep.ok(False, "save/load raised an exception")
        return rep

    if make_png:
        save_overlay(fr, meta, slices, road_id, Path(out_dir) / f"check_frame{fr['frame']}.png")
    return rep


def save_overlay(fr, meta, slices, road_id, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    try:
        img = np.asarray(Image.open(fr["img_path"]))
    except Exception:
        img = np.zeros((fr["H"], fr["W"], 3), np.uint8)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.imshow(img)
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(slices)))
    for k, (sl, col) in enumerate(zip(slices, colors)):
        uv = results_io._to_image(sl["xyz"], fr["cam_to_world"], fr["proj_matrix"])
        ax.scatter(uv[:, 0], uv[:, 1], s=0.5, color=col, alpha=0.25)
        cuv = results_io._to_image(sl["curve_xyz"], fr["cam_to_world"], fr["proj_matrix"])
        ax.plot(cuv[:, 0], cuv[:, 1], color="white", lw=2.2)
        ax.plot(cuv[:, 0], cuv[:, 1], color=col, lw=1.2)
        a = meta["anchors_uv"][k]
        ax.scatter(*a, s=40, color=col, edgecolor="black", zorder=5)
        ax.text(a[0] + 5, a[1] - 5, f"{k}: {sl['step_cm']:.1f}cm {sl['side'][0].upper()}",
                color="white", fontsize=8, bbox=dict(facecolor="black", alpha=0.6, pad=1))
    ax.set_xlim(0, fr["W"]); ax.set_ylim(fr["H"], 0); ax.axis("off")
    ax.set_title(f"frame {fr['frame']}: slice points (faint), fitted profiles (lines), "
                 f"curb centre (dot) | label: slice, height, side L/R")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  overlay: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path.cwd().parent / "KITTI-360")
    ap.add_argument("--seq", default="2013_05_28_drive_0000_sync")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", default=[250, 325])
    ap.add_argument("--n", type=int, default=10, help="number of depth bands")
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--road-id", type=int, default=7)
    ap.add_argument("--sidewalk-id", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path.cwd().parent / "outputs" / "checks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for frame in args.frames:
        try:
            fr = load_frame(args.root, args.seq, args.split, args.cam, frame)
        except Exception as e:
            print(f"\n=== frame {frame}: could not load ({type(e).__name__}: {e}) ===")
            summary[frame] = "not loaded"
            continue
        rep = check_frame(fr, args.road_id, args.sidewalk_id, args.out, args.n, args.candidates, args.seed)
        summary[frame] = f"{rep.fails} fail, {rep.warns} warn"

    print("\n=== summary ===")
    for f, s in summary.items():
        print(f"  frame {f}: {s}")
    sys.exit(1 if any("fail" in s and not s.startswith("0 fail") for s in summary.values()) else 0)


if __name__ == "__main__":
    main()