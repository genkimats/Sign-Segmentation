"""
hierarchy_phrase/common.py -- paths, loaders and cache I/O shared by every script in this directory.

This directory lives in Sign-Segmentation/hierarchy_phrase/; all paths are computed from THIS FILE's location, so the
scripts work from any terminal directory. Frame rate: the corpus is 50 fps (src/dataset loader asserts it). The 2023
paper works at 25 fps, so everything here runs at a WORKING rate = native / STRIDE (default STRIDE = 2 -> 25 fps).
Keypoints are sub-sampled with x[::STRIDE]; labels are mapped through segments (segments.resample_segments) so that
phrase edges that coincided with sign edges still do.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from segments import bio_to_segments, resample_segments, working_length  # noqa: E402

SIGN_LABELS_DIR = os.path.join(PROJECT_ROOT, "processed_data", "BIO_tags")
PHRASE_LABELS_DIR = os.path.join(PROJECT_ROOT, "processed_data", "BIO_tags_phrase")
KIN_DIR = os.path.join(PROJECT_ROOT, "processed_data", "kinematic_features")
SPLIT_FILE = os.path.join(PROJECT_ROOT, "dataset_splits.json")
RUNS_DIR = os.path.join(HERE, "runs")
NATIVE_FPS = 50.0
DEFAULT_STRIDE = 2


def fps_of(stride):
    return NATIVE_FPS / stride


def split_ids(split, split_file=SPLIT_FILE):
    with open(split_file) as f:
        d = json.load(f)
    return [v.replace(".npy", "").replace(".pt", "") for v in d[split]]


def have_all_files(vid):
    return all(os.path.exists(p) for p in (os.path.join(SIGN_LABELS_DIR, f"{vid}.npy"),
                                           os.path.join(PHRASE_LABELS_DIR, f"{vid}.npy"),
                                           os.path.join(KIN_DIR, f"{vid}.pt")))


def load_gold_segments(vid, stride=DEFAULT_STRIDE):
    """Gold sign and phrase segments at the working rate, plus the working length. Returns None if a label file
    is missing. Label arrays are read from disk exactly as stored (raw, never smoothed)."""
    sp = os.path.join(SIGN_LABELS_DIR, f"{vid}.npy")
    pp = os.path.join(PHRASE_LABELS_DIR, f"{vid}.npy")
    if not (os.path.exists(sp) and os.path.exists(pp)):
        return None
    sign, phrase = np.load(sp), np.load(pp)
    if len(sign) != len(phrase):
        return None
    return {"sign": resample_segments(bio_to_segments(sign), stride),
            "phrase": resample_segments(bio_to_segments(phrase), stride),
            "T": working_length(len(sign), stride), "T_native": len(sign)}


def load_keypoints(vid, stride=DEFAULT_STRIDE):
    """(T', 65, 3) float32, shoulder-normalised MediaPipe landmarks from the kinematic cache, sub-sampled."""
    import torch
    d = torch.load(os.path.join(KIN_DIR, f"{vid}.pt"), weights_only=False)
    if "mediapipe" in d:
        d = d["mediapipe"]
    x = d["base"]
    x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    x = np.nan_to_num(x[:, :, :3].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return x[::stride]


# ------------------------------------------------------------------ run directories and cache I/O
def run_dir(name):
    d = os.path.join(RUNS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def segs_to_arr(segs):
    return np.array(segs, dtype=np.int32).reshape(-1, 2)


def arr_to_segs(a):
    return [(int(s), int(e)) for s, e in np.asarray(a).reshape(-1, 2)]


def save_cache_video(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


def load_cache_video(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}