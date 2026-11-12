"""
extract_optical_flow.py -- dense optical flow from the videos as a per-frame feature stream
(FLOW_STUDY in queue_train_phrase.py; loaded through the generic cue stream, cue_set "file").

Per frame t (flow from frame t-1 to t; frame 0 = zeros), on the grey frame resized to --height pixels:
  grid cells (--grid x --grid over the whole frame): mean dx, mean dy, mean magnitude   3 * grid^2 channels
  whole frame: mean magnitude, 90th-percentile magnitude                                2 channels
Displacements are in percent of the frame height per frame (independent of the video resolution).
Default grid 6 -> 110 channels. Flow method: DIS (OpenCV, preset FAST; --method farneback for Farneback).

Why this is not Moryossef et al. (2023)'s "optical flow": theirs is computed from the pose keypoints
(keypoint displacement between frames); this one comes from the PIXELS, so it also sees what the
keypoints miss (finger and lip movement between landmarks, clothing / torso motion, keypoint jitter
does not enter it).

Output: processed_data/flow_features/<video id>.npy  (T, C) float16, T = the length of the phrase label
file of that video (BIO_tags_phrase_v2/<id>.npy): a video with a few frames more or less than its label
file is cut / zero-padded at the end (reported); one that differs by more than --max-mismatch frames is
skipped (reported) -- the dataset loader skips a video whose cue length differs from its labels.
         processed_data/flow_features/_names.json  channel names (src/cue_sets.py reads this)

Videos: raw_data/videos/<video id>.<mp4|avi|mov> for every id in dataset_splits.json (all splits).
Existing outputs are skipped (resume). Run from the repo root:
    python data_processing/extract_optical_flow.py --workers 6
    python data_processing/extract_optical_flow.py --only 1176340_A    # one video, for a quick check
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_EXTS = (".mp4", ".avi", ".mov")


def channel_names(grid):
    names = []
    for r in range(grid):
        for c in range(grid):
            names += [f"flow r{r}c{c} dx", f"flow r{r}c{c} dy", f"flow r{r}c{c} mag"]
    return names + ["flow mean magnitude", "flow p90 magnitude"]


def frame_features(flow, grid, height):
    """flow (H, W, 2) in pixels -> (3 * grid^2 + 2,) in percent of the frame height."""
    f = flow * (100.0 / height)
    mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
    H, W = mag.shape
    rows = np.linspace(0, H, grid + 1).astype(int)
    cols = np.linspace(0, W, grid + 1).astype(int)
    out = []
    for r in range(grid):
        for c in range(grid):
            sl = (slice(rows[r], rows[r + 1]), slice(cols[c], cols[c + 1]))
            out += [f[sl][..., 0].mean(), f[sl][..., 1].mean(), mag[sl].mean()]
    out += [mag.mean(), np.percentile(mag, 90)]
    return np.asarray(out, dtype=np.float32)


def video_flow(path, height, grid, method):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    if method == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    feats, prev = [], None
    n_ch = 3 * grid * grid + 2
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        g = cv2.resize(g, (max(1, int(round(w * height / h))), height), interpolation=cv2.INTER_AREA)
        if prev is None:
            feats.append(np.zeros(n_ch, dtype=np.float32))
        else:
            if method == "dis":
                flow = dis.calc(prev, g, None)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            feats.append(frame_features(flow, grid, height))
        prev = g
    cap.release()
    return np.stack(feats) if feats else np.zeros((0, n_ch), dtype=np.float32)


def align(feats, T):
    """Cut / zero-pad at the end to T frames."""
    if len(feats) >= T:
        return feats[:T]
    return np.concatenate([feats, np.zeros((T - len(feats), feats.shape[1]), dtype=feats.dtype)])


def process(job):
    vid, video_path, label_path, out_path, height, grid, method, max_mismatch = job
    try:
        T = len(np.load(label_path))
        feats = video_flow(video_path, height, grid, method)
        diff = len(feats) - T
        if abs(diff) > max_mismatch:
            return vid, f"SKIPPED: {len(feats)} video frames vs {T} label frames"
        np.save(out_path, align(feats, T).astype(np.float16))
        return vid, (f"aligned: {len(feats)} video frames -> {T}" if diff else None)
    except Exception as e:                                   # keep the other workers going
        return vid, f"ERROR: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", default="raw_data/videos")
    ap.add_argument("--labels-dir", default="processed_data/BIO_tags_phrase_v2")
    ap.add_argument("--splits", default=os.path.join(REPO_ROOT, "dataset_splits.json"))
    ap.add_argument("--out-dir", default="processed_data/flow_features")
    ap.add_argument("--height", type=int, default=160, help="frames are resized to this height before the flow")
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--method", choices=("dis", "farneback"), default="dis")
    ap.add_argument("--max-mismatch", type=int, default=10, help="frames; larger video/label differences are skipped")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="+", default=None, help="video ids (default: every id in the splits)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = channel_names(args.grid)
    names_path = os.path.join(args.out_dir, "_names.json")
    if os.path.exists(names_path):
        with open(names_path) as f:
            if json.load(f) != names:
                raise SystemExit(f"{names_path} has other channels (another --grid?) -- use a new --out-dir.")
    with open(names_path, "w") as f:
        json.dump(names, f, indent=1)

    with open(args.splits) as f:
        splits = json.load(f)
    ids = args.only or sorted({v.replace(".npy", "").replace(".pt", "") for s in splits.values() for v in s})
    videos = {os.path.splitext(f)[0]: os.path.join(args.video_dir, f)
              for f in os.listdir(args.video_dir) if f.lower().endswith(VIDEO_EXTS)}
    jobs, no_video, no_label, done = [], [], [], 0
    for vid in ids:
        out = os.path.join(args.out_dir, f"{vid}.npy")
        if os.path.exists(out):
            done += 1
            continue
        lab = os.path.join(args.labels_dir, f"{vid}.npy")
        if vid not in videos:
            no_video.append(vid)
        elif not os.path.exists(lab):
            no_label.append(vid)
        else:
            jobs.append((vid, videos[vid], lab, out, args.height, args.grid, args.method, args.max_mismatch))
    print(f"{len(ids)} videos: {done} already done, {len(jobs)} to process, "
          f"{len(no_video)} without a video file, {len(no_label)} without a label file; {len(names)} channels")
    if no_video:
        print("  no video:", ", ".join(no_video[:10]) + (" ..." if len(no_video) > 10 else ""))

    notes = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x                              # noqa: E731
    with mp.Pool(max(1, args.workers)) as pool:
        for vid, msg in tqdm(pool.imap_unordered(process, jobs), total=len(jobs), desc="optical flow"):
            if msg:
                notes.append(f"{vid}: {msg}")
    for n in notes:
        print(" ", n)
    bad = sum("SKIPPED" in n or "ERROR" in n for n in notes)
    print(f"done; {bad} videos skipped or failed" if bad else "done")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
