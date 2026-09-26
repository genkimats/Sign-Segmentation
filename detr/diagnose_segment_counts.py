"""
Diagnostic: shows the actual distribution of segment counts across your
training videos, and flags any outliers (a real video shouldn't have more
than a few hundred signs/phrases at most -- anything in the thousands means
something upstream produced corrupted or flickering BIO labels for that
video, not a genuine finding).

Run this from inside detr/ before setting num_queries to anything unusual.
"""
import os
import json
import numpy as np

from dataset_segments import bio_to_segments

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
LABELS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "BIO_tags")
SPLIT_FILE = os.path.join(_PROJECT_ROOT, "dataset_splits.json")

with open(SPLIT_FILE) as f:
    splits = json.load(f)

for split_name in ["train", "val", "test"]:
    if split_name not in splits:
        continue
    video_ids = [v.replace(".npy", "").replace(".pt", "") for v in splits[split_name]]
    counts = []
    for vid in video_ids:
        path = os.path.join(LABELS_DIR, f"{vid}.npy")
        if not os.path.exists(path):
            continue
        labels = np.load(path)
        n_segments = len(bio_to_segments(labels))
        counts.append((vid, n_segments, len(labels)))

    counts_only = [c[1] for c in counts]
    if not counts_only:
        continue

    print(f"\n=== {split_name.upper()} ({len(counts)} videos) ===")
    print(f"min={min(counts_only)}, median={int(np.median(counts_only))}, "
          f"95th pct={int(np.percentile(counts_only, 95))}, max={max(counts_only)}")

    # Flag the top 5 by segment count -- the real outliers should be obvious
    top5 = sorted(counts, key=lambda c: -c[1])[:5]
    print("Top 5 by segment count:")
    for vid, n_seg, n_frames in top5:
        print(f"  {vid}: {n_seg} segments over {n_frames} frames "
              f"({n_seg/n_frames*100:.1f}% of frames are 'Begin' -- should be a small fraction)")