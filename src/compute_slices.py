"""
Compute curb slices for one or more KITTI-360 frames and save one .npz per frame.

Run from src/:

    python compute_slices.py --frames 250 325
    python compute_slices.py --range 200 400 25        # frames 200, 225, ..., 375
    python curb_viewer.py ../outputs/slices/<file>.npz  # then look at a result

Frames that can't be loaded (no pose, no static chunk) or give no fit are reported and skipped.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

import slice_pipeline as sp


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--frames", type=int, nargs="+", help="explicit frame numbers")
    which.add_argument("--range", type=int, nargs=3, metavar=("START", "STOP", "STEP"),
                       help="frames START, START+STEP, ... (< STOP)")
    ap.add_argument("--root", type=Path, default=Path.cwd().parent / "KITTI-360")
    ap.add_argument("--seq", default=sp.DEFAULT_SEQ)
    ap.add_argument("--split", default=sp.DEFAULT_SPLIT)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--n", type=int, default=10, help="number of camera-depth bands")
    ap.add_argument("--candidates", type=int, default=5, help="boundary candidates tried per band")
    ap.add_argument("--seed", type=int, default=0, help="-1 = random every run (like the notebook)")
    ap.add_argument("--road-id", type=int, default=sp.ROAD_ID)
    ap.add_argument("--sidewalk-id", type=int, default=sp.SIDEWALK_ID)
    ap.add_argument("--context-max", type=int, default=40_000, help="max LiDAR context points saved")
    ap.add_argument("--out", type=Path, default=Path.cwd().parent / "outputs" / "slices")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    frames = a.frames if a.frames else list(range(*a.range))
    seed = None if a.seed < 0 else a.seed
    print(f"{len(frames)} frame(s) -> {a.out}")

    rows, written = [], []
    for frame in frames:
        try:
            fr = sp.load_frame(a.root, a.seq, a.split, a.cam, frame)
            results = sp.run_slices(fr, a.road_id, a.sidewalk_id, a.n, a.candidates, seed)
            good = [r for r in results if r["result"] is not None]
            if not good:
                rows.append((frame, "no fits", "", ""))
                continue
            path = sp.save_frame(fr, results, a.out, a.road_id, a.sidewalk_id, context_max=a.context_max)
            sides = [r["side"] for r in good]
            h = np.array([r["height_cm"] for r in good])
            rows.append((frame, f"{len(good)}/{len(results)} fits",
                         f"L {sides.count('left')} / R {sides.count('right')}",
                         f"{np.min(h):.1f}–{np.max(h):.1f} cm"))
            written.append(path)
        except Exception as e:
            rows.append((frame, f"FAILED: {type(e).__name__}: {e}", "", ""))
            if not isinstance(e, (KeyError, ValueError, FileNotFoundError)):
                traceback.print_exc()

    print("\n frame  result            sides          heights")
    for frame, res, sides, h in rows:
        print(f" {frame:5d}  {res:<16}  {sides:<13}  {h}")
    if written:
        print(f"\nview with:\n  python curb_viewer.py {written[0]}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())