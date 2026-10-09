"""
check_phrase_labels.py -- audits processed_data/BIO_tags_phrase/ per split.

Checks the things that would make a model predict B for whole phrases:
  - label values and class distribution (expected 0=O, 1=I, 2=B, with B rare)
  - length of B runs (B should be ONE frame per phrase start; long B runs mean the
    label files mark whole phrases as B)
  - videos whose labels look different from the rest
  - label length vs keypoint length

Usage (repo root):  python check_phrase_labels.py
"""
import json
import argparse
import os

import numpy as np

LABELS_DIR = "processed_data/BIO_tags_phrase"
KEYPOINTS_DIR = "processed_data/keypoints"
SPLIT_FILE = "dataset_splits.json"


def runs(mask):
    """Lengths of consecutive True runs."""
    if not mask.any():
        return np.array([], dtype=int)
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def audit_video(vid):
    path = os.path.join(LABELS_DIR, f"{vid}.npy")
    if not os.path.exists(path):
        return None
    lab = np.load(path)
    info = {"vid": vid, "shape": lab.shape, "dtype": str(lab.dtype)}
    lab = np.asarray(lab).reshape(-1) if lab.ndim > 1 and 1 in lab.shape else lab
    if lab.ndim != 1:
        info["error"] = f"labels are {lab.ndim}-D {lab.shape}, expected 1-D"
        return info
    vals, cnts = np.unique(lab, return_counts=True)
    info["values"] = dict(zip(vals.tolist(), cnts.tolist()))
    info["T"] = len(lab)
    b_runs = runs(lab == 2)
    i_runs = runs(lab == 1)
    info["n_b_runs"] = len(b_runs)
    info["b_run_max"] = int(b_runs.max()) if len(b_runs) else 0
    info["b_run_median"] = float(np.median(b_runs)) if len(b_runs) else 0.0
    info["n_i_frames"] = int((lab == 1).sum())
    info["i_run_median"] = float(np.median(i_runs)) if len(i_runs) else 0.0
    kp = os.path.join(KEYPOINTS_DIR, f"{vid}.npy")
    if os.path.exists(kp):
        info["kp_T"] = int(np.load(kp, mmap_mode="r").shape[0])
    return info


def main():
    global LABELS_DIR
    parser = argparse.ArgumentParser(description="Audit phrase BIO label files per split.")
    parser.add_argument("--labels-dir", default=LABELS_DIR,
                        help=f"Label directory to audit (default: {LABELS_DIR}).")
    LABELS_DIR = parser.parse_args().labels_dir
    print(f"Auditing {LABELS_DIR}")
    with open(SPLIT_FILE) as f:
        splits = json.load(f)

    for split in ("train", "val", "test"):
        # The split file lists file names ("1247641_A.npy"); strip the extension to get the video id.
        vids = [os.path.splitext(os.path.basename(v))[0] for v in splits.get(split, [])]
        infos, missing = [], []
        for vid in vids:
            info = audit_video(vid)
            (missing if info is None else infos).append(vid if info is None else info)

        print(f"\n{'=' * 80}\n{split.upper()}: {len(vids)} videos in split, {len(infos)} label files, "
              f"{len(missing)} missing")
        if missing:
            print(f"  missing label files: {missing}")
        if not infos:
            continue

        errors = [i for i in infos if "error" in i]
        for i in errors:
            print(f"  ⚠️ {i['vid']}: {i['error']} (dtype {i['dtype']})")
        good = [i for i in infos if "error" not in i]
        if not good:
            continue

        total = {}
        for i in good:
            for k, v in i["values"].items():
                total[k] = total.get(k, 0) + v
        n = sum(total.values())
        print("  class distribution: " + ", ".join(
            f"{k}={v} ({v / n:.2%})" for k, v in sorted(total.items())))
        unexpected = sorted(set(total) - {0, 1, 2})
        if unexpected:
            print(f"  ⚠️ unexpected label values: {unexpected}")

        long_b = [i for i in good if i["b_run_max"] > 1]
        no_i = [i for i in good if i["n_i_frames"] == 0]
        print(f"  B runs: median length {np.median([i['b_run_median'] for i in good]):.1f}, "
              f"videos with B runs longer than 1 frame: {len(long_b)}/{len(good)}")
        print(f"  videos with NO Inside frames at all: {len(no_i)}/{len(good)}")
        for i in sorted(long_b, key=lambda x: -x["b_run_max"])[:15]:
            print(f"    long B: {i['vid']}  max B run {i['b_run_max']} fr, median {i['b_run_median']:.0f}, "
                  f"I frames {i['n_i_frames']}, values {i['values']}")
        mism = [i for i in good if "kp_T" in i and abs(i["kp_T"] - i["T"]) > 1]
        if mism:
            print(f"  ⚠️ label length != keypoint length for {len(mism)} videos, e.g. "
                  + ", ".join(f"{i['vid']} ({i['T']} vs {i['kp_T']})" for i in mism[:5]))


if __name__ == "__main__":
    main()