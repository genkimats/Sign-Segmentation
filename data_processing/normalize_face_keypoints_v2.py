"""
normalize_face_keypoints_v2.py -- FACE-LOCAL, 2D normalization of the raw face keypoints.

Input : processed_data/face_keypoints/<video>.npy           (T, 83, 3) raw MediaPipe Face
        Landmarker output from extract_face_keypoints.py (image-normalized x, y + face z)
Output: processed_data/face_keypoints_normalized_v2/<video>.npy  (T, 83, 2) float32
        processed_data/face_keypoints_normalized_v2/_detection/<video>.npy  (T,) bool
        processed_data/face_keypoints_normalized_v2/_report.json

Why v2 (vs normalize_face_keypoints.py):
  v1 put the face in the BODY's frame (minus shoulder midpoint, divided by shoulder
  width). The face points then mostly encode where the head is -- which the body stream
  already has (nose, eyes, ears, mouth corners are body vertices) -- while blinks,
  eyebrow raises and mouth movement become a tiny ripple. v1 also subtracted the POSE
  model's z (hip-relative) from the FACE model's z (face-relative), mixing two unrelated
  depth systems.

What v2 does, per frame:
  1. z is dropped (2D only).
  2. x, y are converted to pixels (MediaPipe normalizes x by width and y by height, so
     raw units are not square on non-square video).
  3. origin = midpoint of the inner eye corners (MediaPipe 133 / 362)
     -> removes head translation.
  4. roll alignment (default on): rotate so the line between the outer eye corners
     (33 -> 263) is horizontal -> removes head tilt (head pose is in the body stream).
  5. scale = distance between the outer eye corners, as the per-video MEDIAN (default)
     or per frame (--scale per_frame). The median keeps one consistent unit per video
     and doesn't amplify frame-to-frame jitter; head turns then show as compression.
  Resulting units: inter-ocular distance (eyes at about x = +-0.5, mouth at about y ~ 1).

Missing detections: extract_face_keypoints.py stored zeros before the first detection
and copied the previous frame (forward-fill) during dropouts. Those frames are detected
(all-zero, or bit-identical to the previous frame), normalized frames are linearly
interpolated across them (nearest value at the ends), and the per-frame detection mask is
saved in _detection/ for analysis.

Run from the repo root:
    python data_processing/normalize_face_keypoints_v2.py
    python data_processing/normalize_face_keypoints_v2.py --no-roll-align --scale per_frame --overwrite
"""
import argparse
import json
import multiprocessing as mproc
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.face_subsets import SAVED_FACE_INDICES  # noqa: E402  (vertex order of the saved files)

RAW_FACE_DIR = "processed_data/face_keypoints"
OUTPUT_DIR = "processed_data/face_keypoints_normalized_v2"
RAW_VIDEO_DIR = "raw_data/videos"

_POS = {raw: i for i, raw in enumerate(SAVED_FACE_INDICES)}
LEFT_INNER, RIGHT_INNER = _POS[133], _POS[362]   # inner eye corners  -> origin
LEFT_OUTER, RIGHT_OUTER = _POS[33], _POS[263]    # outer eye corners  -> scale / roll


def find_video_size(vid):
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        path = os.path.join(RAW_VIDEO_DIR, vid + ext)
        if os.path.exists(path):
            cap = cv2.VideoCapture(path)
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if w > 0 and h > 0:
                return float(w), float(h)
    return None


def detection_mask(raw):
    """True where the face was actually detected (not zero-initialised, not forward-filled)."""
    T = raw.shape[0]
    flat = raw.reshape(T, -1)
    zero = ~np.any(flat != 0, axis=1)
    repeated = np.zeros(T, dtype=bool)
    repeated[1:] = np.all(flat[1:] == flat[:-1], axis=1)
    return ~(zero | repeated)


def interpolate_missing(x, detected):
    """x: (T, V, 2). Linear interpolation over undetected frames, nearest value at the ends."""
    T = x.shape[0]
    idx = np.flatnonzero(detected)
    if len(idx) == T:
        return x
    t = np.arange(T)
    flat = x.reshape(T, -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t, idx, flat[idx, c])
    return out.reshape(x.shape)


def normalize(raw, width, height, roll_align=True, scale_mode="video"):
    detected = detection_mask(raw)
    if not detected.any():
        return None, detected, None

    xy = raw[..., :2].astype(np.float64) * np.array([width, height])   # pixels, z dropped
    origin = (xy[:, LEFT_INNER] + xy[:, RIGHT_INNER]) / 2.0              # (T, 2)
    eye_vec = xy[:, RIGHT_OUTER] - xy[:, LEFT_OUTER]                     # (T, 2)
    eye_dist = np.linalg.norm(eye_vec, axis=1)                           # (T,)

    centered = xy - origin[:, None, :]
    if roll_align:
        ang = np.arctan2(eye_vec[:, 1], eye_vec[:, 0])
        c, s = np.cos(-ang), np.sin(-ang)
        rx = centered[..., 0] * c[:, None] - centered[..., 1] * s[:, None]
        ry = centered[..., 0] * s[:, None] + centered[..., 1] * c[:, None]
        centered = np.stack([rx, ry], axis=-1)

    good = detected & (eye_dist > 1e-6)
    if not good.any():
        return None, detected, None
    if scale_mode == "per_frame":
        scale = np.where(good, eye_dist, np.median(eye_dist[good]))[:, None, None]
    else:
        scale = np.median(eye_dist[good])
    out = centered / scale
    out = interpolate_missing(out, good)
    info = {"frames": int(len(detected)), "missing": int((~good).sum()),
            "median_eye_dist_px": float(np.median(eye_dist[good]))}
    return out.astype(np.float32), good, info


def process_one(args):
    fname, roll_align, scale_mode, overwrite = args
    vid = fname[:-4]
    out_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(out_path) and not overwrite:
        return vid, "exists", None
    raw = np.load(os.path.join(RAW_FACE_DIR, fname))
    if raw.ndim != 3 or raw.shape[1] != len(SAVED_FACE_INDICES) or raw.shape[2] < 2:
        return vid, f"bad_shape {raw.shape}", None
    size = find_video_size(vid)
    if size is None:
        return vid, "no_video_for_size", None
    out, detected, info = normalize(raw, size[0], size[1], roll_align, scale_mode)
    if out is None:
        return vid, "no_face_detected", None
    np.save(out_path, out)
    np.save(os.path.join(OUTPUT_DIR, "_detection", fname), detected)
    return vid, None, info


def main():
    parser = argparse.ArgumentParser(description="Face-local 2D normalization of raw face keypoints.")
    parser.add_argument("--no-roll-align", action="store_true", help="Keep head tilt (don't rotate the eye line level).")
    parser.add_argument("--scale", choices=["video", "per_frame"], default="video",
                        help="Inter-ocular scale: per-video median (default) or per frame.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "_detection"), exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_FACE_DIR) if f.endswith(".npy"))
    print(f"{len(files)} raw face files -> {OUTPUT_DIR} "
          f"(2D, origin=inner eye corners, roll_align={not args.no_roll_align}, scale={args.scale})")

    jobs = [(f, not args.no_roll_align, args.scale, args.overwrite) for f in files]
    skips, infos = {}, {}
    with mproc.Pool(max(1, args.workers)) as pool:
        for vid, reason, info in tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs)):
            if reason == "exists":
                continue
            if reason:
                skips.setdefault(reason.split(" ")[0], []).append(vid)
            else:
                infos[vid] = info

    total = sum(i["frames"] for i in infos.values())
    missing = sum(i["missing"] for i in infos.values())
    worst = sorted(infos.items(), key=lambda kv: -kv[1]["missing"] / max(1, kv[1]["frames"]))[:10]
    report = {
        "roll_align": not args.no_roll_align, "scale": args.scale,
        "written": len(infos), "skipped": {k: v for k, v in skips.items()},
        "frames": total, "missing_frames": missing,
        "missing_share": missing / total if total else None,
        "worst_videos": {v: round(i["missing"] / i["frames"], 4) for v, i in worst},
    }
    with open(os.path.join(OUTPUT_DIR, "_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWritten: {len(infos)} | skipped: { {k: len(v) for k, v in skips.items()} }")
    if total:
        print(f"Frames without a real detection (interpolated): {missing} of {total} ({missing / total:.2%})")
        print("Most affected videos: " + ", ".join(f"{v} {r:.1%}" for v, r in report["worst_videos"].items()))
    print(f"Report: {os.path.join(OUTPUT_DIR, '_report.json')}")


if __name__ == "__main__":
    main()