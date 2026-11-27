"""
check_excluded_videos.py -- every video, checked from the raw video: which feature files exist, would the
training loader use it, and how much of it really has a detected hand / face (no interpolated data used).

Videos checked: every camera in raw_data/videos (<doc>_1a1 / _1b1 = <doc>_A / _B), plus every video that has
a keypoints file or is in dataset_splits.json.

Per video:
  files     raw video, keypoints, kinematic, phrase labels, gloss labels, face (the directory training
            loads), raw face, full face, HaMeR (.pt), hand boxes (.pkl); frame counts of each
  loader    would src/dataset.py load it for the proposed model (body + HaMeR + face)? Same rules, same
            order: label + kinematic file, face file, face frames == label frames, HaMeR file loadable and
            not empty. The first failing rule is reported (what training printed as "skipped").
  detection, FRESH, run on the raw video (default; --detect none to skip):
            hands  MediaPipe HandLandmarker, the exact detector and settings of extract_hand_boxes.py (the
                   boxes HaMeR is run on: HaMeR only regresses the hand pose inside these boxes, so a frame
                   without a box is a frame HaMeR cannot see; running the HaMeR network itself would not
                   change which frames are detected)
            face   MediaPipe FaceLandmarker, the settings of extract_face_keypoints.py
            -> % frames with no hand, with fewer than 2 hands, with no face; longest undetected stretch (s)
  detection, SAVED files that are NOT interpolated (for comparison with the fresh run):
            HaMeR .pt  hands with all-zero rotations = not detected (extract_hamer_features.py writes zeros)
            boxes .pkl frames without a box
            face       the raw file is forward-filled: a frame equal to the previous one (or all zero) was
                       not detected; or face_keypoints_normalized_v2/_detection if saved
            body kp    MediaPipe Holistic hands (extract_poses.py interpolates; filled frames are found by
                       analyze_missing_hands.filled_frames) -- this is what the keypoint stream sees

Fresh detections are cached in <out-dir>/detections/<video>.npz, so the run can be stopped and resumed;
--stride 2 (default) checks every 2nd frame = 25 fps, the rate training now uses; --stride 1 for all frames.
Needs hand_landmarker.task and face_landmarker.task in the repo root (the extraction scripts use them).
MediaPipe runs in separate worker processes that never import torch (extract_hand_boxes.py: the two in
one process can segfault).

    python check_excluded_videos.py                       # everything (slow the first time; cached after)
    python check_excluded_videos.py --workers 8 --stride 2
    python check_excluded_videos.py --videos 1582841_A 1180022_B
    python check_excluded_videos.py --detect none         # files + loader + saved detections only (fast)
Output: evaluation_phrase/video_check/{summary.txt, per_video.csv, detections/}
"""
import argparse
import csv
import glob
import json
import multiprocessing as mproc
import os
import pickle
import subprocess
import sys
import time

import numpy as np

VIDEO_DIR = "raw_data/videos"
KP_DIR = "processed_data/keypoints"
KIN_DIR = "processed_data/kinematic_features"
PHRASE_DIR = "processed_data/BIO_tags_phrase_v2"
GLOSS_DIR = "processed_data/BIO_tags"
FACE_DIR = "processed_data/face_keypoints_normalized_v2"
RAW_FACE_DIR = "processed_data/face_keypoints"
FULL_FACE_DIR = "processed_data/face_keypoints_full_normalized_v2"
HAMER_DIR = "processed_data/hamer_features"
BOX_DIR = "processed_data/hand_boxes"
ANN_DIR = "raw_data/annotations"
CAM = {"A": "1a1", "B": "1b1"}
VIDEO_EXTS = (".mp4", ".avi", ".mov")


def doc_part(vid):
    return vid.rsplit("_", 1)


def video_path(vid):
    d, p = doc_part(vid)
    for ext in VIDEO_EXTS:
        q = os.path.join(VIDEO_DIR, f"{d}_{CAM[p]}{ext}")
        if os.path.exists(q):
            return q
    return None


def all_videos(split_of):
    vids = set(split_of)
    for p in glob.glob(os.path.join(VIDEO_DIR, "*")):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in VIDEO_EXTS:
            for cam, part in (("_1a1", "A"), ("_1b1", "B")):
                if stem.endswith(cam):
                    vids.add(f"{stem[:-len(cam)]}_{part}")
    vids |= {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(KP_DIR, "*.npy"))}
    return sorted(vids)


def removal_history(split_file):
    """vid -> 'commit date message' of the commit in which it last left the split."""
    try:
        log = subprocess.run(["git", "log", "--reverse", "--format=%h|%ad|%s", "--date=short", "--", split_file],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:                                            # noqa: BLE001
        return {}
    out, prev = {}, None
    for line in log:
        h, date, msg = line.split("|", 2)
        try:
            d = json.loads(subprocess.run(["git", "show", f"{h}:{split_file}"], capture_output=True,
                                          text=True, check=True).stdout)
        except Exception:                                        # noqa: BLE001
            continue
        cur = {v.replace(".npy", "") for vs in d.values() for v in vs}
        if prev is not None:
            for v in prev - cur:
                out[v] = f"{h} {date} {msg}"
            for v in cur - prev:
                out.pop(v, None)
        prev = cur
    return out


def longest_run(miss):
    if not len(miss):
        return 0
    d = np.diff(np.r_[0, miss.astype(np.int8), 0])
    r = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return int(r.max()) if len(r) else 0


# ============================================================================ fresh detection (workers)
def detect_video(job):
    """Runs in a spawned worker (no torch). Returns (vid, cache_path or None, message)."""
    vid, vpath, cache, stride, root = job
    if os.path.exists(cache):
        return vid, cache, None
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:                                       # noqa: BLE001
        return vid, None, f"import error: {e}"
    hand_opt = mp_vision.HandLandmarkerOptions(              # = extract_hand_boxes.py
        base_options=mp_python.BaseOptions(model_asset_path=os.path.join(root, "hand_landmarker.task")),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.5)
    face_opt = mp_vision.FaceLandmarkerOptions(              # = extract_face_keypoints.py
        base_options=mp_python.BaseOptions(model_asset_path=os.path.join(root, "face_landmarker.task")),
        running_mode=mp_vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5)
    idx, n_hands, left, right, face = [], [], [], [], []
    try:
        with mp_vision.HandLandmarker.create_from_options(hand_opt) as hl, \
                mp_vision.FaceLandmarker.create_from_options(face_opt) as fl:
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
            i = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if i % stride == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ts = int(i * 1000 / fps)
                    hr = hl.detect_for_video(img, ts)
                    fr = fl.detect_for_video(img, ts)
                    labels = [h[0].category_name for h in hr.handedness]
                    idx.append(i)
                    n_hands.append(len(hr.hand_landmarks))
                    left.append("Left" in labels)
                    right.append("Right" in labels)
                    face.append(bool(fr.face_landmarks))
                i += 1
            cap.release()
    except Exception as e:                                       # noqa: BLE001
        return vid, None, f"detection error: {str(e)[:100]}"
    tmp = cache + ".tmp.npz"
    np.savez(tmp, frame_idx=np.array(idx), n_hands=np.array(n_hands, dtype=np.int8), left=np.array(left),
             right=np.array(right), face=np.array(face), stride=stride, total_frames=i, fps=fps)
    os.replace(tmp, cache)
    return vid, cache, None


def fresh_stats(cache):
    z = np.load(cache)
    n, face = z["n_hands"], z["face"]
    if not len(n):
        return {"det_frames": 0}
    step = int(z["stride"]) / float(z["fps"])
    return {"det_frames": int(len(n)), "video_frames": int(z["total_frames"]),
            "fresh_no_hand": float((n == 0).mean()), "fresh_lt2_hands": float((n < 2).mean()),
            "fresh_no_face": float((~face).mean()),
            "fresh_longest_no_hand_s": round(longest_run(n == 0) * step, 1),
            "fresh_longest_no_face_s": round(longest_run(~face) * step, 1)}


# ============================================================================ files, loader, saved detection
def n_frames(path, kind):
    try:
        if kind == "npy":
            return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:                                            # noqa: BLE001
        return -1
    return -1


def file_stats(vid, phrase_dir, face_dir):
    d, p = doc_part(vid)
    raw_face = next((q for q in (os.path.join(RAW_FACE_DIR, f"{vid}.npy"),
                                 os.path.join(RAW_FACE_DIR, f"{d}_{CAM[p]}.npy")) if os.path.exists(q)), None)
    box = next((q for q in (os.path.join(BOX_DIR, f"{vid}_boxes.pkl"),
                            os.path.join(BOX_DIR, f"{d}_{CAM[p]}_boxes.pkl")) if os.path.exists(q)), None)
    paths = {"video": video_path(vid), "keypoints": os.path.join(KP_DIR, f"{vid}.npy"),
             "kinematic": os.path.join(KIN_DIR, f"{vid}.pt"), "phrase": os.path.join(phrase_dir, f"{vid}.npy"),
             "gloss": os.path.join(GLOSS_DIR, f"{vid}.npy"), "face": os.path.join(face_dir, f"{vid}.npy"),
             "raw_face": raw_face, "full_face": os.path.join(FULL_FACE_DIR, f"{vid}.npy"),
             "hamer": os.path.join(HAMER_DIR, f"{vid}_hamer.pt"), "boxes": box}
    r = {f"has_{k}": int(bool(v) and os.path.exists(v)) for k, v in paths.items()}
    for k in ("keypoints", "phrase", "face", "raw_face"):
        r[f"T_{k}"] = n_frames(paths[k], "npy") if r[f"has_{k}"] else np.nan
    return r, paths


def saved_detection(vid, paths, r):
    out = {}
    # HaMeR: zero rotation matrices = hand not detected (no interpolation)
    if r["has_hamer"]:
        try:
            import torch
            h = torch.load(paths["hamer"], weights_only=False)
            go = np.asarray(h["global_orient"]).reshape(len(h["global_orient"]), 2, -1)
            det = np.abs(go).sum(axis=2) > 0                     # (T', 2) left, right
            ds = int(h.get("temporal_downsample_factor", 1))
            out.update(T_hamer=int(len(det)) * ds, hamer_loadable=1, hamer_empty=int(len(det) == 0))
            if len(det):
                out.update(saved_hamer_no_hand=float((~det.any(axis=1)).mean()),
                           saved_hamer_lt2_hands=float((~det.all(axis=1)).mean()))
        except Exception as e:                                   # noqa: BLE001
            out.update(hamer_loadable=0, hamer_error=str(e)[:80])
    if r["has_boxes"]:
        with open(paths["boxes"], "rb") as f:
            n = np.array([len(x["boxes"]) for x in pickle.load(f)["frames"]])
        if len(n):
            out.update(saved_box_no_hand=float((n == 0).mean()), saved_box_lt2=float((n < 2).mean()))
    # face: _detection mask if saved, else the forward-filled raw file
    det_p = os.path.join(FACE_DIR, "_detection", f"{vid}.npy")
    if os.path.exists(det_p):
        out["saved_face_no_face"] = float(1 - np.load(det_p).mean())
        out["saved_face_source"] = "_detection"
    elif r["has_raw_face"]:
        x = np.load(paths["raw_face"]).reshape(r["T_raw_face"], -1)
        miss = ~np.any(x != 0, axis=1)
        miss[1:] |= np.all(x[1:] == x[:-1], axis=1)
        out["saved_face_no_face"] = float(miss.mean())
        out["saved_face_source"] = "raw (forward-fill test)"
    # body keypoints (Holistic hands, interpolated in the file)
    if r["has_keypoints"]:
        try:
            from analyze_missing_hands import filled_frames
            kp = np.load(paths["keypoints"])
            kp = kp.reshape(len(kp), -1, 3) if kp.ndim == 2 else kp
            L, R = filled_frames(kp, "left"), filled_frames(kp, "right")
            out.update(kp_no_hand=float((L & R).mean()), kp_left_missing=float(L.mean()),
                       kp_right_missing=float(R.mean()))
        except Exception as e:                                   # noqa: BLE001
            out["kp_error"] = str(e)[:80]
    return out


def loader_verdict(r):
    """First rule of src/dataset.py (proposed model: body + HaMeR + face) that would skip the video."""
    if not r["has_phrase"]:
        return "missing_label (no phrase label file)"
    if not r["has_kinematic"]:
        return "missing_kinematic"
    if not r["has_face"]:
        return "missing_face"
    if np.isfinite(r.get("T_face", np.nan)) and r["T_face"] != r["T_phrase"]:
        return f"face_frame_mismatch ({r['T_face']} vs {r['T_phrase']})"
    if not r["has_hamer"]:
        return "missing_hamer"
    if not r.get("hamer_loadable", 1):
        return "hamer_load_error"
    if r.get("hamer_empty"):
        return "hamer_empty"
    return "loaded"


def phrase_reason(vid, r):
    if r["has_phrase"]:
        return ""
    d, p = doc_part(vid)
    eaf = os.path.join(ANN_DIR, f"{d}.eaf")
    if not r["has_gloss"]:
        return "no gloss labels"
    if not os.path.exists(eaf):
        return "no .eaf"
    try:
        sys.path.insert(0, "data_processing")
        from create_phrase_bio_tags_v2 import get_elan_sentences
        n = sum(1 for s in get_elan_sentences(eaf) if s.get("participant") == p)
    except Exception as e:                                       # noqa: BLE001
        return f"eaf error: {str(e)[:50]}"
    return "no translation sentence for this participant" if n == 0 else "sentences but no sign inside"


# ============================================================================ main
def pct(x):
    return f"{100 * x:5.1f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else "    -"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--detect", default="hands,face", choices=["hands,face", "none"])
    ap.add_argument("--stride", type=int, default=2, help="check every Nth frame (2 = 25 fps)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--phrase-dir", default=PHRASE_DIR)
    ap.add_argument("--face-dir", default=FACE_DIR)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "video_check"))
    args = ap.parse_args()

    with open(args.split_file) as f:
        split_of = {v.replace(".npy", ""): k for k, vs in json.load(f).items() for v in vs}
    vids = [v.replace(".npy", "") for v in args.videos] if args.videos else all_videos(split_of)
    hist = removal_history(args.split_file)
    os.makedirs(os.path.join(args.out_dir, "detections"), exist_ok=True)

    # 1. fresh detection (spawned workers, no torch)
    caches, det_err = {}, {}
    if args.detect != "none":
        root = os.getcwd()
        jobs = []
        for v in vids:
            vp = video_path(v)
            c = os.path.join(args.out_dir, "detections", f"{v}_s{args.stride}.npz")
            if vp is None:
                det_err[v] = "no raw video"
            elif os.path.exists(c):
                caches[v] = c
            else:
                jobs.append((v, vp, c, args.stride, root))
        print(f"fresh detection: {len(caches)} cached, {len(jobs)} to run with {args.workers} workers "
              f"(stride {args.stride}; roughly a few minutes per video per worker)")
        if jobs:
            t0 = time.time()
            with mproc.get_context("spawn").Pool(args.workers) as pool:
                for i, (v, c, msg) in enumerate(pool.imap_unordered(detect_video, jobs), 1):
                    if c:
                        caches[v] = c
                    else:
                        det_err[v] = msg
                    el = time.time() - t0
                    print(f"  [{i}/{len(jobs)}] {v} {'ok' if c else msg}  "
                          f"(elapsed {el / 60:.0f} min, left ~{el / i * (len(jobs) - i) / 60:.0f} min)", flush=True)

    # 2. files, loader, saved detections
    rows = []
    for v in vids:
        r, paths = file_stats(v, args.phrase_dir, args.face_dir)
        r = {"video": v, "split": split_of.get(v, "-"), **r}
        r.update(saved_detection(v, paths, r))
        if v in caches:
            r.update(fresh_stats(caches[v]))
        elif v in det_err:
            r["fresh_error"] = det_err[v]
        r["loader"] = loader_verdict(r)
        r["phrase_reason"] = phrase_reason(v, r)
        r["removed_in"] = "" if v in split_of else hist.get(v, "never in split")
        rows.append(r)

    # 3. report
    inr = [r for r in rows if r["split"] != "-"]
    L = [f"Video check: {len(rows)} videos ({len(inr)} in {args.split_file}); fresh detection stride "
         f"{args.stride} ({'off' if args.detect == 'none' else 'MediaPipe hands + face on the raw video'})", ""]
    for key, name in (("fresh_no_hand", "no hand (fresh)"), ("fresh_lt2_hands", "< 2 hands (fresh)"),
                      ("fresh_no_face", "no face (fresh)"), ("saved_hamer_no_hand", "no hand (saved HaMeR)"),
                      ("kp_no_hand", "no hand (Holistic keypoints)")):
        a = np.array([r[key] for r in inr if np.isfinite(r.get(key, np.nan))])
        if len(a):
            L.append(f"  videos in split, % frames {name:<30s} median {100 * np.median(a):5.1f}  "
                     f"95th pct {100 * np.percentile(a, 95):5.1f}  max {100 * a.max():5.1f}")
    L += ["", "Loader verdict (proposed model: body + HaMeR + face):"]
    for grp, sel in (("in split", lambda r: r["split"] != "-"), ("not in split", lambda r: r["split"] == "-")):
        c = {}
        for r in rows:
            if sel(r):
                k = r["loader"].split(" (")[0]
                c[k] = c.get(k, 0) + 1
        L.append(f"  {grp:<13s} " + ", ".join(f"{k}: {n}" for k, n in sorted(c.items())))
    L += ["", f"{'video':<30s} {'split':<5s} {'noHand':>6s} {'<2hand':>6s} {'noFace':>6s} {'HaMeR':>6s} "
              f"{'kpHand':>6s} {'svFace':>6s}  files missing | loader | phrase | removed in"]
    for r in rows:
        miss = [k[4:] for k in r if k.startswith("has_") and not r[k]]
        L.append(f"{r['video']:<30s} {r['split']:<5s} {pct(r.get('fresh_no_hand', np.nan)):>6s} "
                 f"{pct(r.get('fresh_lt2_hands', np.nan)):>6s} {pct(r.get('fresh_no_face', np.nan)):>6s} "
                 f"{pct(r.get('saved_hamer_no_hand', np.nan)):>6s} {pct(r.get('kp_no_hand', np.nan)):>6s} "
                 f"{pct(r.get('saved_face_no_face', np.nan)):>6s}  {','.join(miss) or '-'} | {r['loader']} | "
                 f"{r['phrase_reason'] or 'ok'} | {r['removed_in'] or '-'}"
                 + (f" | fresh: {r['fresh_error']}" if r.get("fresh_error") else ""))
    L += ["", "Columns (% of frames): noHand / <2hand / noFace = fresh MediaPipe detection on the video; "
              "HaMeR = no hand in the saved HaMeR file; kpHand = both Holistic hands interpolated in the keypoints; "
              "svFace = no face in the saved raw face file. 'files missing' lists feature files that do not exist."]

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "video", k != "split", k))
    with open(os.path.join(args.out_dir, "per_video.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out_dir}/summary.txt, per_video.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())