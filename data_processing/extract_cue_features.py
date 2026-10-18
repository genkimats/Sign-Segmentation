"""
extract_cue_features.py -- precompute the interpretable boundary cues as a training input.

For every video it computes the per-frame cues of boundary_cue_analysis.py (eye
openness, blinks, brow height, mouth opening, head nod / height / speed, wrist height and
speed, holds, hand openness, HaMeR finger flexion and articulation speed, ...) from the
SAME files the models use, and saves them as one (T, C) matrix:

  processed_data/cue_features/<video>.npy     (T, 48) float16, channels in CUE_FILE_ORDER
  processed_data/cue_features/_names.json     channel names (src/cue_sets.py)
  processed_data/cue_features/_report.json    settings + per-source availability

Normalization (per video, no labels involved, identical for train / val / test):
  continuous cues -> robust z ((x - median) / max(1.4826 MAD, 0.1 std)), clipped to +-8
  rate cues (blink, hold, detections) -> kept as 0 / 1
  missing values (face not detected, hand not found) -> 0 (= the video median), and the
  "face detected" / "... detected" channels tell the model which frames were missing.

T always equals the phrase-label length of the video (the dataset requires it).
The training-time cue SET (all / selected / face / nonmanual / manual) is chosen when
loading (cue_set in the config), so this only has to run once.

Run from the repo root:
    python data_processing/extract_cue_features.py
    python data_processing/extract_cue_features.py --workers 8 --overwrite
"""
import argparse
import json
import multiprocessing as mproc
import os
import sys

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import boundary_cue_analysis as bca  # noqa: E402  (cue computation is shared, not copied)
from src.cue_sets import CUE_FILE_ORDER, CUE_INFO, FACE_DETECTED  # noqa: E402
from src.face_subsets import SAVED_FACE_INDICES  # noqa: E402

Z_CLIP = 8.0
_ARGS = None


def compute_one(vid):
    a = _ARGS
    lp = os.path.join(a.labels_dir, f"{vid}.npy")
    out_path = os.path.join(a.out_dir, f"{vid}.npy")
    if os.path.exists(out_path) and not a.overwrite:
        return vid, "exists", None
    if not os.path.exists(lp):
        return vid, "no_labels", None
    T = len(np.load(lp))
    cues, avail = {}, {"body": False, "face": False, "hamer": False}

    kp = os.path.join(a.kinematic_dir, f"{vid}.pt")
    if not os.path.exists(kp):
        return vid, "no_body", None
    try:
        kin = bca.load_pt(kp)
        if isinstance(kin, dict) and "mediapipe" in kin:
            kin = kin["mediapipe"]
        base = bca.to_np(kin["base"]).astype(np.float64)
    except Exception as e:  # noqa: BLE001
        return vid, f"body_load_error {type(e).__name__}", None
    if base.shape[0] != T:
        return vid, f"body_length_mismatch {base.shape[0]}!={T}", None
    c, _ = bca.body_hand_cues(base, a.fps, a.smooth)
    if not c:
        return vid, "no_usable_pose", None
    cues.update(c)
    avail["body"] = True

    face_det = np.zeros(T)
    fp = os.path.join(a.face_dir, f"{vid}.npy")
    if os.path.exists(fp):
        face = np.load(fp)
        if face.ndim == 3 and face.shape[1] == len(SAVED_FACE_INDICES) and face.shape[0] == T:
            dp = os.path.join(a.face_dir, "_detection", f"{vid}.npy")
            det = np.load(dp).astype(bool)[:T] if os.path.exists(dp) else np.ones(T, dtype=bool)
            cues.update(bca.face_cues(face[..., :2], det, a.fps, a.smooth))
            face_det = det.astype(np.float64)
            avail["face"] = True

    hp = os.path.join(a.hamer_dir, f"{vid}_hamer.pt")
    if os.path.exists(hp):
        try:
            hd = bca.load_pt(hp)
            hpose, gor = bca.to_np(hd["hand_pose"]), bca.to_np(hd["global_orient"])
            if len(hpose):
                cues.update(bca.hamer_cues(hpose, gor, int(hd.get("temporal_downsample_factor", 1)),
                                           T, a.fps, a.smooth))
                avail["hamer"] = True
        except Exception:  # noqa: BLE001
            pass

    out = np.zeros((T, len(CUE_FILE_ORDER)), dtype=np.float32)
    empty = []
    for j, name in enumerate(CUE_FILE_ORDER):
        if name == FACE_DETECTED:
            out[:, j] = face_det
            continue
        x = cues.get(name)
        if x is None:
            empty.append(name)
            continue
        x = np.asarray(x, dtype=np.float64)[:T]
        if CUE_INFO[name]["kind"] == "rate":
            v = x
        else:
            v, ms = bca.robust_z(x)
            if ms is None:
                empty.append(name)
                continue
            v = np.clip(v, -Z_CLIP, Z_CLIP)
        out[:, j] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(out_path, out.astype(np.float16))
    return vid, None, {"frames": T, "avail": avail, "empty": empty}


def _init(args):
    global _ARGS
    _ARGS = args
    bca.BLINK_RATIO = args.blink_ratio


def main():
    p = argparse.ArgumentParser(description="Precompute interpretable boundary cues as a training stream.")
    p.add_argument("--split-file", default="dataset_splits.json")
    p.add_argument("--labels-dir", default="processed_data/BIO_tags_phrase_v2",
                   help="defines the frame count of every video (must match training labels)")
    p.add_argument("--kinematic-dir", default="processed_data/kinematic_features")
    p.add_argument("--face-dir", default="processed_data/face_keypoints_normalized_v2")
    p.add_argument("--hamer-dir", default="processed_data/hamer_features")
    p.add_argument("--out-dir", default="processed_data/cue_features")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--blink-ratio", type=float, default=0.75)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    with open(args.split_file) as f:
        splits = json.load(f)
    vids = sorted({v.replace(".npy", "").replace(".pt", "") for s in splits.values() for v in s})
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "_names.json"), "w") as f:
        json.dump(CUE_FILE_ORDER, f, indent=1)
    print(f"{len(vids)} videos (all splits) -> {args.out_dir}  ({len(CUE_FILE_ORDER)} channels)")

    skips, infos = {}, {}
    with mproc.Pool(max(1, args.workers), initializer=_init, initargs=(args,)) as pool:
        for vid, reason, info in tqdm(pool.imap_unordered(compute_one, vids), total=len(vids)):
            if reason == "exists":
                continue
            if reason:
                skips.setdefault(reason.split(" ")[0], []).append(vid)
            else:
                infos[vid] = info

    n = len(infos)
    report = {"written": n, "skipped": {k: len(v) for k, v in skips.items()},
              "skipped_videos": skips,
              "settings": {"fps": args.fps, "smooth": args.smooth, "blink_ratio": args.blink_ratio,
                           "z_clip": Z_CLIP, "labels_dir": args.labels_dir, "face_dir": args.face_dir,
                           "hamer_dir": args.hamer_dir, "kinematic_dir": args.kinematic_dir},
              "videos_with": {s: sum(i["avail"][s] for i in infos.values()) for s in ("body", "face", "hamer")}}
    empty_counts = {}
    for i in infos.values():
        for name in i["empty"]:
            empty_counts[name] = empty_counts.get(name, 0) + 1
    report["videos_where_cue_is_empty"] = empty_counts
    with open(os.path.join(args.out_dir, "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Written: {n} | skipped: {report['skipped']} | with face: {report['videos_with']['face']}, "
          f"with HaMeR: {report['videos_with']['hamer']}")
    if empty_counts:
        worst = sorted(empty_counts.items(), key=lambda kv: -kv[1])[:5]
        print("Cues empty (no data / no variation) in some videos: " + ", ".join(f"{k} ({v})" for k, v in worst))
    print(f"Report: {os.path.join(args.out_dir, '_report.json')}")


if __name__ == "__main__":
    main()