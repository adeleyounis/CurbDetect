"""
Sanity checks for the updated analyze_boundary.py and results_io.py on real KITTI-360 frames.

Run from src/ (next to data_loader.py, analyze_boundary.py, results_io.py):

    python check_pipeline.py --frames 250 325 1000

If you keep a copy of the ORIGINAL analyze_boundary.py as analyze_boundary_old.py next to it,
the script also checks that the new estimator gives identical numbers to the old one.

For each frame it:
  1. runs the slice pipeline (slice_pipeline.py, same logic as the notebook),
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

import analyze_boundary
import results_io
import slice_pipeline as sp
from slice_pipeline import load_frame, run_slices

try:
    import analyze_boundary_old
except ImportError:
    analyze_boundary_old = None


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
            band = r["band"]
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
    try:
        npz = sp.save_frame(fr, results, out_dir, road_id, sidewalk_id)
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
    ap.add_argument("--seq", default=sp.DEFAULT_SEQ)
    ap.add_argument("--split", default=sp.DEFAULT_SPLIT)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", default=[250, 325])
    ap.add_argument("--n", type=int, default=10, help="number of depth bands")
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--road-id", type=int, default=sp.ROAD_ID)
    ap.add_argument("--sidewalk-id", type=int, default=sp.SIDEWALK_ID)
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