"""
preprocess_kinematics.py -- per-video kinematic features for SignSegmentationDataset.

    python preprocess_kinematics.py                       # -> processed_data/kinematic_features_v2
    python preprocess_kinematics.py --workers 8           # parallel
    python preprocess_kinematics.py --overwrite           # rebuild files that already exist
    python preprocess_kinematics.py --check 3             # print value ranges of 3 videos, write nothing

Output per video: <out-dir>/<vid>.pt, a dict of float16 tensors, all (T, 65, K):
    base             positions (T, 65, 3) -- IDENTICAL to version 1 (scipy.signal.medfilt, k=5),
                     so runs that only use positions see exactly the same input as before
    velocity         d position / dt          (T, 65, 3)   shoulder widths per SECOND
    acceleration     d^2 position / dt^2      (T, 65, 3)   per second^2
    jerk             d^3 position / dt^3      (T, 65, 3)   per second^3
    velocity-mag     |velocity|               (T, 65, 1)
    angular-vel      turning rate of the 2D motion direction (T, 65, 1), rad/s, clipped
    spatial_angles   direction of each joint from the nose (T, 65, 2)
    temporal_angles  direction of motion (T, 65, 2)
  + "meta": {"version": 2, "fps": ..., "units": "per_second", ...}

What changed from version 1 (processed_data/kinematic_features):
  1. Units. v1 derivatives were per FRAME (x[t] - x[t-1]): ~25x smaller than the positions they
     are concatenated with, and wrong as soon as the frame rate changes (frame-rate augmentation
     resamples the windows). v2 is per SECOND, so a resampled window keeps correct values.
  2. Smoothing. v1 differenced median-filtered positions; a median filter outputs flat steps,
     so the difference was a spike train. v2 takes the derivatives with a Savitzky-Golay filter
     (local polynomial fit; window 7 = 0.14 s at 50 fps), which is centred (no half-frame lag).
  3. Edges. scipy.signal.medfilt zero-pads, which pulled the first/last 2 frames toward 0 and
     created fake movement there. Derivatives now use a median filter with mode="nearest".
     ("base" keeps the v1 filter on purpose -- see above.)
  4. angular-vel used a 3D speed with a 2D cross product and could explode for still joints;
     now 2D throughout, with a speed floor and clipping.
  5. --overwrite / --out-dir / --workers / --check.
Old v1 files are left untouched, so runs trained on them still evaluate identically. New runs
read v2 by setting "kinematic_dir": "processed_data/kinematic_features_v2" in their config.

Known limitation (unchanged): missing detections were linearly interpolated during pose
extraction, so a hand that disappears for a while "moves" at constant speed across the gap.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import medfilt, savgol_filter
from tqdm import tqdm

VERSION = 2
DEFAULTS = dict(keypoints_dir="processed_data/keypoints", split_file="dataset_splits.json",
                out_dir="processed_data/kinematic_features_v2", fps=50.0, window=7)
MIN_SPEED = 0.05      # shoulder widths / s; below this the motion direction is noise
MAX_OMEGA = 50.0      # rad / s


def _angles(vec):
    """vec (..., 3) -> polar (theta / pi) and azimuth (phi / pi) of its direction, (..., 2)."""
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-6
    theta = np.arccos(np.clip(z / r, -1.0, 1.0)) / np.pi
    phi = np.arctan2(y, x) / np.pi
    return np.stack([theta, phi], axis=-1)


def _derivative(pos, order, fps, window):
    """order-th time derivative (per second) of pos (T, V, C), centred, smoothed."""
    T = pos.shape[0]
    poly = max(2, order)
    w = window if window % 2 else window + 1
    if order == 3:
        w = max(w, 9)
    if T > w:
        return savgol_filter(pos, w, poly, deriv=order, delta=1.0 / fps, axis=0, mode="interp")
    d = pos                                    # very short video: plain centred differences
    for _ in range(order):
        d = np.gradient(d, 1.0 / fps, axis=0) if d.shape[0] > 1 else np.zeros_like(d)
    return d


def compute_kinematics(raw, fps=DEFAULTS["fps"], window=DEFAULTS["window"]):
    """raw: (T, 65, 3) float keypoints (as saved by extract_poses.py). Returns a dict of float32 arrays."""
    raw = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if raw.ndim == 2:
        raw = raw.reshape(raw.shape[0], -1, 3)
    base = medfilt(raw, kernel_size=(5, 1, 1)).astype(np.float32)            # = version 1
    pos = median_filter(raw, size=(5, 1, 1), mode="nearest").astype(np.float32)
    v = _derivative(pos, 1, fps, window).astype(np.float32)
    a = _derivative(pos, 2, fps, window).astype(np.float32)
    j = _derivative(pos, 3, fps, window).astype(np.float32)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    sp2 = np.maximum(v[..., 0] ** 2 + v[..., 1] ** 2, MIN_SPEED ** 2)
    omega = np.clip((v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]) / sp2, -MAX_OMEGA, MAX_OMEGA)
    return {
        "base": base,
        "velocity": v,
        "acceleration": a,
        "jerk": j,
        "velocity-mag": speed,
        "angular-vel": omega[..., None].astype(np.float32),
        "spatial_angles": _angles(base - base[:, 0:1]).astype(np.float32),
        "temporal_angles": _angles(v).astype(np.float32),
    }


def _process(job):
    vid, src, dst, fps, window = job
    try:
        out = compute_kinematics(np.load(src), fps, window)
    except Exception as e:                      # report, keep going
        return vid, f"failed: {e}"
    data = {k: torch.from_numpy(np.ascontiguousarray(x)).to(torch.float16) for k, x in out.items()}
    data["meta"] = {"version": VERSION, "fps": float(fps), "units": "per_second",
                    "savgol_window": int(window), "base_filter": "scipy.signal.medfilt k=5 (v1)"}
    tmp = dst + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, dst)                        # never leaves a half-written file behind
    return vid, None


def _video_ids(split_file):
    with open(split_file) as f:
        splits = json.load(f)
    vids = set()
    for name in ("train", "val", "test"):
        vids.update(v.replace(".npy", "").replace(".pt", "") for v in splits.get(name, []))
    return sorted(vids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints-dir", default=DEFAULTS["keypoints_dir"])
    ap.add_argument("--split-file", default=DEFAULTS["split_file"])
    ap.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    ap.add_argument("--fps", type=float, default=DEFAULTS["fps"], help="frame rate of the keypoint files")
    ap.add_argument("--window", type=int, default=DEFAULTS["window"],
                    help="Savitzky-Golay window in frames (odd; 7 = 0.14 s at 50 fps)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", help="rebuild files that already exist")
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="print value ranges for N videos and exit (writes nothing)")
    args = ap.parse_args()
    if os.path.abspath(args.out_dir) == os.path.abspath("processed_data/kinematic_features"):
        raise SystemExit("Refusing to overwrite the version-1 directory: runs trained on it would "
                         "silently change. Use a new --out-dir.")

    vids = _video_ids(args.split_file)
    have = [v for v in vids if os.path.exists(os.path.join(args.keypoints_dir, f"{v}.npy"))]
    print(f"{len(vids)} videos in {args.split_file}, {len(have)} with keypoints in {args.keypoints_dir}")

    if args.check:
        for v in have[:args.check]:
            k = compute_kinematics(np.load(os.path.join(args.keypoints_dir, f"{v}.npy")), args.fps, args.window)
            hands = slice(23, 65)
            print(f"{v}: T={k['base'].shape[0]} | |pos| p99 {np.percentile(np.abs(k['base']), 99):.2f} | "
                  f"hand speed (per s) median {np.median(k['velocity-mag'][:, hands]):.3f}, "
                  f"p99 {np.percentile(k['velocity-mag'][:, hands], 99):.2f}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(v, os.path.join(args.keypoints_dir, f"{v}.npy"), os.path.join(args.out_dir, f"{v}.pt"),
             args.fps, args.window)
            for v in have if args.overwrite or not os.path.exists(os.path.join(args.out_dir, f"{v}.pt"))]
    print(f"Writing {len(jobs)} files to {args.out_dir} ({len(have) - len(jobs)} already there)")
    failed = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for vid, err in tqdm(pool.imap_unordered(_process, jobs, chunksize=4), total=len(jobs)):
                if err:
                    failed.append((vid, err))
    else:
        for job in tqdm(jobs):
            vid, err = _process(job)
            if err:
                failed.append((vid, err))
    if failed:
        print(f"{len(failed)} failed, e.g. {failed[:3]}")
    print("Done.")


if __name__ == "__main__":
    main()