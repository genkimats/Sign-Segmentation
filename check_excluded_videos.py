"""
check_excluded_videos.py -- for every video: which split is it in, why is it out if it is out, and are
its hands / face / HaMeR really usable?

For every video that has a keypoints file (default; --excluded-only for the ones not in the split, or the
videos given with --videos), the script reports:
  history   the commit that removed it from dataset_splits.json (git log of that file) and whether it is
            listed in no_phrase.txt (the list clean_no_phrase.py removed on 2026-10-05)
  phrases   phrase label file (processed_data/BIO_tags_phrase_v2) present? number of phrases; if absent,
            the reason create_phrase_bio_tags_v2.py would give (no gloss labels / no translation sentence
            for this participant), number of translation sentences and signs of this participant
  hands     MediaPipe hands: share of frames where the hand was NOT detected (filled by interpolation),
            per hand and both hands at once (analyze_missing_hands.filled_frames)
  face      face file present (face_keypoints_normalized_v2, the one training loads) and its frame count;
            share of frames with a detected face, from _detection/ if saved, else the raw 83-point file,
            else from the normalized file itself (interpolated frames lie exactly on a straight line)
  hamer     HaMeR file present and its frame count; share of frames with at least one hand box
  frames    the training loader skips a video whose face file has a different frame count -- flagged
The same hand / face / box numbers are computed for the videos IN the split, and each excluded video is
placed against them (percentile), so "undetectable" is judged against what the kept videos look like.

    python check_excluded_videos.py                          # all videos with keypoints
    python check_excluded_videos.py --excluded-only          # only those not in the split
    python check_excluded_videos.py --videos 1582841_A 1413236_A 1429910-16075041-16115817_B \
                                             1180022_B 1187154_A 1419122_A
    python check_excluded_videos.py --no-reference           # skip the kept-video statistics (faster)
Output: evaluation_phrase/video_check/{summary.txt, per_video.csv}
"""
import argparse
import csv
import glob
import json
import os
import pickle
import subprocess
import sys

import numpy as np

KP_DIR = "processed_data/keypoints"
PHRASE_DIR = "processed_data/BIO_tags_phrase_v2"
GLOSS_DIR = "processed_data/BIO_tags"
FACE_DIR = "processed_data/face_keypoints_normalized_v2"
RAW_FACE_DIR = "processed_data/face_keypoints"
HAMER_DIR = "processed_data/hamer_features"
BOX_DIR = "processed_data/hand_boxes"
ANN_DIR = "raw_data/annotations"


def doc_part(vid):
    d, p = vid.rsplit("_", 1)
    return d, p


# ----------------------------------------------------------------------------- history
def removal_history(split_file):
    """vid -> (commit, date, message) of the commit in which it left the split (last time)."""
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
                out[v] = (h, date, msg)
            for v in cur - prev:
                out.pop(v, None)
        prev = cur
    return out


# ----------------------------------------------------------------------------- per-video checks
def hand_stats(vid):
    from analyze_missing_hands import filled_frames
    p = os.path.join(KP_DIR, f"{vid}.npy")
    if not os.path.exists(p):
        return {"kp": 0}
    kp = np.load(p)
    if kp.ndim == 2:
        kp = kp.reshape(len(kp), -1, 3)
    L, R = filled_frames(kp, "left"), filled_frames(kp, "right")
    return {"kp": 1, "frames": len(kp), "left_missing": L.mean(), "right_missing": R.mean(),
            "both_missing": (L & R).mean(), "left_never": int(L.all()), "right_never": int(R.all())}


def linear_fill_mask(x, rel_tol=1e-6):
    """(T, ...) -> bool (T,): frames whose values are exactly linear in time with both neighbours
    (np.interp / pandas linear fills, constant edge fills). Real detections are never exactly linear."""
    f = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    T = len(f)
    out = np.zeros(T, dtype=bool)
    if T < 3:
        return out
    d2 = np.abs(f[2:] - 2 * f[1:-1] + f[:-2]).max(axis=1)
    scale = np.abs(f).max() + 1e-12
    lin = d2 <= rel_tol * scale
    out[1:-1] = lin
    out[0], out[-1] = lin[0], lin[-1]
    return out


def face_stats(vid, kp_frames=None):
    norm_p = os.path.join(FACE_DIR, f"{vid}.npy")
    out = {"face_file": int(os.path.exists(norm_p)), "face_frames": np.nan, "face_detected": np.nan,
           "face_detected_source": "", "face_full478_file": int(os.path.exists(
               os.path.join("processed_data/face_keypoints_full_normalized_v2", f"{vid}.npy")))}
    norm = np.load(norm_p, mmap_mode="r") if out["face_file"] else None
    if norm is not None:
        out["face_frames"] = int(norm.shape[0])
    det_p = os.path.join(FACE_DIR, "_detection", f"{vid}.npy")
    raw_p = os.path.join(RAW_FACE_DIR, f"{vid}.npy")
    if os.path.exists(det_p):
        out.update(face_detected=float(np.load(det_p).mean()), face_detected_source="_detection")
    elif os.path.exists(raw_p):
        x = np.load(raw_p)
        flat = x.reshape(len(x), -1)
        nonzero = np.any(flat != 0, axis=1)
        rep = np.r_[False, np.all(flat[1:] == flat[:-1], axis=1)]
        out.update(face_detected=float((nonzero & ~rep).mean()), face_detected_source="raw")
    elif norm is not None:
        out.update(face_detected=float(1 - linear_fill_mask(np.asarray(norm)).mean()),
                   face_detected_source="normalized (linear-fill test)")
    if kp_frames is not None and np.isfinite(out["face_frames"]):
        out["face_frame_mismatch"] = int(out["face_frames"] != kp_frames)
    return out


def box_stats(vid):
    d, p = doc_part(vid)
    cam = {"A": "1a1", "B": "1b1"}[p]
    cands = [os.path.join(BOX_DIR, f"{vid}_boxes.pkl"), os.path.join(BOX_DIR, f"{d}_{cam}_boxes.pkl")]
    hp = os.path.join(HAMER_DIR, f"{vid}_hamer.pt")
    out = {"hamer_file": int(os.path.exists(hp)), "box_any": np.nan, "box_two": np.nan}
    for c in cands:
        if os.path.exists(c):
            with open(c, "rb") as f:
                fr = pickle.load(f)["frames"]
            n = np.array([len(r["boxes"]) for r in fr])
            if len(n):
                out.update(box_any=float((n >= 1).mean()), box_two=float((n >= 2).mean()))
            break
    return out


def phrase_stats(vid):
    d, p = doc_part(vid)
    lab_p = os.path.join(PHRASE_DIR, f"{vid}.npy")
    out = {"phrase_file": int(os.path.exists(lab_p)), "phrases": np.nan, "sentences": np.nan, "signs": np.nan,
           "phrase_reason": ""}
    if out["phrase_file"]:
        lab = np.load(lab_p)
        prev = np.r_[0, lab[:-1]]
        out["phrases"] = int(((lab == 2) | ((lab == 1) & (prev == 0))).sum())
    g = os.path.join(GLOSS_DIR, f"{vid}.npy")
    if os.path.exists(g):
        gl = np.load(g)
        prev = np.r_[0, gl[:-1]]
        out["signs"] = int(((gl == 2) | ((gl == 1) & (prev == 0))).sum())
    eaf = os.path.join(ANN_DIR, f"{d}.eaf")
    if os.path.exists(eaf):
        try:
            sys.path.insert(0, "data_processing")
            from create_phrase_bio_tags_v2 import get_elan_sentences
            out["sentences"] = sum(1 for s in get_elan_sentences(eaf) if s.get("participant") == p)
        except Exception as e:                                   # noqa: BLE001
            out["phrase_reason"] = f"eaf read error: {str(e)[:60]}"
    if not out["phrase_file"] and not out["phrase_reason"]:
        if not os.path.exists(g):
            out["phrase_reason"] = "no gloss labels for this participant"
        elif out["sentences"] == 0:
            out["phrase_reason"] = "no translation sentence for this participant"
        elif not os.path.exists(eaf):
            out["phrase_reason"] = "no .eaf"
        else:
            out["phrase_reason"] = "sentences exist but no sign inside them"
    return out


def check(vid):
    r = {"video": vid}
    for f in (hand_stats, face_stats, box_stats, phrase_stats):
        try:
            r.update(f(vid, r.get("frames")) if f is face_stats else f(vid))
        except Exception as e:                                   # noqa: BLE001
            r[f"{f.__name__}_error"] = str(e)[:80]
    return r


def flags_of(r):
    """[(short key, full text)] of everything that would stop or weaken this video; [] = usable."""
    out = []
    if not r.get("phrase_file"):
        out.append(("no phrase labels", "no phrase labels (" + (r.get("phrase_reason") or "?") + ")"))
    if not r.get("kp"):
        out.append(("no keypoints", "no keypoints"))
    if r.get("left_never") and r.get("right_never"):
        out.append(("hands never detected", "hands never detected"))
    elif r.get("both_missing", 0) > 0.5:
        out.append(("both hands missing > 50% of frames",
                    f"both hands missing in {100 * r['both_missing']:.0f}% of frames"))
    fd = r.get("face_detected", np.nan)
    if not r.get("face_file"):
        out.append(("no face file", "no face file"))
    elif np.isfinite(fd) and fd < 0.5:
        out.append(("face detected < 50% of frames", f"face detected in only {100 * fd:.0f}% of frames"))
    if r.get("face_frame_mismatch"):
        out.append(("face frame count != keypoints (loader skips)",
                    f"face frames {r.get('face_frames')} != keypoint frames {r.get('frames')} (loader skips it)"))
    if not r.get("hamer_file"):
        out.append(("no HaMeR file", "no HaMeR file"))
    return out


def verdict(r):
    f = flags_of(r)
    return "; ".join(t for _, t in f) if f else "looks usable"


def pct_of(x, ref):
    ref = np.asarray([v for v in ref if np.isfinite(v)])
    if not np.isfinite(x) or not len(ref):
        return ""
    return f"{100 * (ref < x).mean():.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--excluded-only", action="store_true", help="only videos with keypoints not in the split")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "video_check"))
    args = ap.parse_args()

    with open(args.split_file) as f:
        splits = json.load(f)
    split_of = {v.replace(".npy", ""): k for k, vs in splits.items() for v in vs}
    kept = sorted(split_of)
    have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(KP_DIR, "*.npy"))}
    if args.videos:
        targets = [v.replace(".npy", "") for v in args.videos]
    elif args.excluded_only:
        targets = sorted(have - set(kept))
    else:
        targets = sorted(have | set(kept))
    hist = removal_history(args.split_file)
    no_phrase = set()
    if os.path.exists("no_phrase.txt"):
        with open("no_phrase.txt") as f:
            no_phrase = {t.strip("'\"[] \n") for t in f.read().replace(",", " ").split() if t.strip("'\"[] \n")}

    rows = []
    for v in targets:
        r = check(v)
        h = hist.get(v)
        r["split"] = split_of.get(v, "-")
        r["removed_in"] = "" if v in split_of else (f"{h[0]} {h[1]} {h[2]}" if h else "never in split")
        r["in_no_phrase_txt"] = int(v in no_phrase)
        r["verdict"] = verdict(r)
        rows.append(r)

    ref = {}
    if not args.no_reference:
        done = {r["video"]: r for r in rows}
        refrows = [done.get(v) or check(v) for v in kept]
        for k in ("left_missing", "right_missing", "both_missing", "face_detected", "box_any"):
            ref[k] = [r.get(k, np.nan) for r in refrows]

    n_in = sum(r["split"] != "-" for r in rows)
    L = [f"Video check: {len(rows)} videos ({n_in} in {args.split_file}, {len(rows) - n_in} not)", ""]
    if ref:
        L.append("Kept videos for reference (median / 95th percentile):")
        for k in ("left_missing", "right_missing", "both_missing"):
            a = np.array([x for x in ref[k] if np.isfinite(x)])
            if len(a):
                L.append(f"  {k:<14s} {100 * np.median(a):5.1f}% / {100 * np.percentile(a, 95):5.1f}% of frames")
        for k in ("face_detected", "box_any"):
            a = np.array([x for x in ref[k] if np.isfinite(x)])
            if len(a):
                L.append(f"  {k:<14s} {100 * np.median(a):5.1f}% / 5th pct {100 * np.percentile(a, 5):5.1f}% of frames")
        L.append("")
    L.append(f"{'video':<30s} {'split':<5s} {'phr':>4s} {'sent':>4s} {'signs':>5s} {'L miss':>6s} {'R miss':>6s} {'both':>5s} "
             f"{'face':>5s} {'box':>5s}  verdict | removed in")
    f = lambda x: f"{100 * x:5.1f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else "    -"
    g = lambda x: f"{int(x):d}" if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x) else "-"
    for r in rows:
        L.append(f"{r['video']:<30s} {r['split']:<5s} {g(r.get('phrases')):>4s} {g(r.get('sentences')):>4s} {g(r.get('signs')):>5s} "
                 f"{f(r.get('left_missing', np.nan)):>6s} {f(r.get('right_missing', np.nan)):>6s} "
                 f"{f(r.get('both_missing', np.nan)):>5s} {f(r.get('face_detected', np.nan)):>5s} "
                 f"{f(r.get('box_any', np.nan)):>5s}  {r['verdict']}" + (("" if r['removed_in'] == "never in split" else " | removed in ") + r['removed_in'] if r['removed_in'] else "").replace("never in split", " | never in split")
                 + (" [no_phrase.txt]" if r["in_no_phrase_txt"] else ""))
        if ref:
            r["pct_both_missing_vs_kept"] = pct_of(r.get("both_missing", np.nan), ref["both_missing"])
            r["pct_face_detected_vs_kept"] = pct_of(r.get("face_detected", np.nan), ref["face_detected"])
    L += ["", "Summary (videos):"]
    for grp, sel in (("in split", lambda r: r["split"] != "-"), ("not in split", lambda r: r["split"] == "-")):
        counts = {}
        for r in rows:
            if not sel(r):
                continue
            key = ("no phrase labels" if not r.get("phrase_file") else
                   "looks usable" if r["verdict"] == "looks usable" else "detection / feature problem")
            counts[key] = counts.get(key, 0) + 1
        L.append(f"  {grp}: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    flags = {}
    for r in rows:
        for key, _ in flags_of(r):
            flags.setdefault(key, [0, 0])[0 if r["split"] != "-" else 1] += 1
    L.append("  flags (in split / not in split):")
    for k, (a, b) in sorted(flags.items()):
        L.append(f"    {k:<45s} {a:4d} / {b:4d}")
    L += [
          "Columns: phr = phrases in the label file, sent = translation sentences of this participant in the "
          ".eaf, signs = gloss signs; L/R miss, both = % frames with the hand not detected; face, box = % frames "
          "with a detected face / at least one hand box."]

    os.makedirs(args.out_dir, exist_ok=True)
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