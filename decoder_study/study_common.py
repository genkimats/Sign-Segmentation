"""
Shared, torch-free helpers for the decoder study: paths, window planning and
stitching, export I/O, and the training-target reconstruction.

This directory lives in Sign-Segmentation/decoder_study/. All paths are computed
from THIS FILE's location, so scripts work from any terminal directory.

Label convention (confirmed against the real data earlier in this project):
    0 = Outside, 1 = Inside, 2 = Begin
"""
import os
import sys
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
LABELS_DIR = os.path.join(PROJECT_ROOT, "processed_data", "BIO_tags")   # RAW gold labels
SPLIT_FILE = os.path.join(PROJECT_ROOT, "dataset_splits.json")
EXPORTS_DIR = os.path.join(_SCRIPT_DIR, "exports")
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")

O, I, B = 0, 1, 2


# ------------------------------------------------------------------ windows --
def plan_windows(T, window, stride):
    """(start, end) windows covering [0, T). If T <= window: one window (0, T)
    (the caller pads it). Otherwise windows of exactly `window` frames at
    multiples of `stride`, plus one final window flush with the end if the
    grid leaves a tail -- the same tiling src/dataset.py uses."""
    if T <= window:
        return [(0, T)]
    starts = list(range(0, T - window + 1, stride))
    if starts[-1] + window < T:
        starts.append(T - window)
    return [(s, s + window) for s in starts]


def stitch_logits(T, windows, window_logits):
    """Average per-window logits back into one (T, C) array. `window_logits[i]`
    is (C, e-s) -- the VALID part only (padding already cut). Overlapping frames
    (e.g. the flush final window) are averaged instead of duplicated. Note:
    the old evaluate_decoder.py concatenated windows, which double-counts that
    tail region for any video whose length isn't a multiple of the window."""
    C = window_logits[0].shape[0]
    acc = np.zeros((T, C), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    for (s, e), lg in zip(windows, window_logits):
        assert lg.shape == (C, e - s), f"window ({s},{e}) logits {lg.shape}"
        acc[s:e] += lg.T
        cnt[s:e] += 1
    if not (cnt > 0).all():
        raise ValueError("stitch_logits: some frames are not covered by any window")
    return (acc / cnt[:, None]).astype(np.float32)


# ------------------------------------------------------------------- exports --
def export_path(run_name, split):
    return os.path.join(EXPORTS_DIR, run_name, f"{split}.npz")


def save_export(run_name, split, records, meta):
    """records: {vid: {"logits": (T,3) f32, "labels": (T,) int8 RAW gold}}"""
    os.makedirs(os.path.join(EXPORTS_DIR, run_name), exist_ok=True)
    arrays = {}
    for vid, rec in records.items():
        arrays[f"logits__{vid}"] = rec["logits"].astype(np.float32)
        arrays[f"labels__{vid}"] = rec["labels"].astype(np.int8)
    np.savez_compressed(export_path(run_name, split), **arrays)
    with open(os.path.join(EXPORTS_DIR, run_name, f"{split}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_export(run_name, split):
    """-> (dict vid -> {"logits","labels"}, meta dict)"""
    path = export_path(run_name, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run export_logits.py for split '{split}' first.")
    data = np.load(path)
    vids = sorted({k.split("__", 1)[1] for k in data.files if k.startswith("logits__")})
    records = {v: {"logits": data[f"logits__{v}"], "labels": data[f"labels__{v}"]} for v in vids}
    meta_path = os.path.join(EXPORTS_DIR, run_name, f"{split}_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return records, meta


def load_gold_for_split(split):
    """RAW gold BIO arrays for every video id in a split (labels only -- no
    encoder involved, so this is safe to use for fitting label statistics)."""
    with open(SPLIT_FILE) as f:
        ids = [v.replace(".npy", "").replace(".pt", "") for v in json.load(f)[split]]
    out = {}
    for vid in ids:
        p = os.path.join(LABELS_DIR, f"{vid}.npy")
        if os.path.exists(p):
            out[vid] = np.load(p).astype(np.int8)
    return out


# ------------------------------------------------------ training-target view --
def training_targets(raw_labels, tolerance_window=5):
    """What the encoder was actually TRAINED to predict: argmax of the Gaussian-
    smoothed soft labels (src/dataset.py apply_label_smoothing). This dilates
    every 1-frame Begin into ~3 frames. Imported lazily from the project so
    there is a single source of truth for the smoothing."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.dataset import apply_label_smoothing
    soft = apply_label_smoothing(np.asarray(raw_labels).astype(np.int64), tolerance_window)
    return soft, soft.argmax(axis=1).astype(np.int8)