"""
visualize_face_subsets.py -- figures of the face subsets (src/face_subsets.py) on a real face.

The face is the MEAN of the face-local v2 files (processed_data/face_keypoints_normalized_v2,
inter-ocular units, roll-aligned, so faces from different videos line up), averaged over
detected frames of a sample of videos. For every subset it draws:
  - all 83 saved points in light grey (context)
  - the subset's points, coloured by region (eyes / brows / mouth)
  - the face-graph edges the ST-GCN uses for that subset (src/graph.py)
  - the two anchor points (inner eye corners 133 / 362) as rings

Outputs (evaluation_phrase/face_subsets/):
  face_subset_<name>.png   one figure per subset
  face_subsets_all.png     all subsets side by side (for the thesis)

Usage (repo root):
  python visualize_face_subsets.py
  python visualize_face_subsets.py --videos 50 --labels   # + MediaPipe index labels
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.face_subsets import (FACE_SUBSETS, SAVED_FACE_INDICES, LEFT_EYE_INDICES, RIGHT_EYE_INDICES,  # noqa: E402
                              LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES, LIPS_INDICES,
                              ANCHOR_LEFT_EYE_INNER, ANCHOR_RIGHT_EYE_INNER, face_subset_raw_indices)
from src.graph import SkeletonGraph  # noqa: E402

REGIONS = [("eyes", set(LEFT_EYE_INDICES) | set(RIGHT_EYE_INDICES), "#1f77b4"),
           ("brows", set(LEFT_EYEBROW_INDICES) | set(RIGHT_EYEBROW_INDICES), "#17a589"),
           ("mouth", set(LIPS_INDICES), "#d6336c")]
ORDER = ["full", "compact", "eyes_brows", "minimal"]


def mean_face(face_dir, n_videos, seed):
    files = sorted(f for f in glob.glob(os.path.join(face_dir, "*.npy")))
    if not files:
        raise SystemExit(f"No face files in {face_dir}.")
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.choice(len(files), min(n_videos, len(files)), replace=False)]
    acc, n = np.zeros((len(SAVED_FACE_INDICES), 2)), 0
    for f in files:
        x = np.load(f)[..., :2].astype(np.float64)                  # (T, 83, 2)
        det_path = os.path.join(face_dir, "_detection", os.path.basename(f))
        if os.path.exists(det_path):
            det = np.load(det_path).astype(bool)[:len(x)]
            x = x[det]
        x = x[np.all(np.isfinite(x), axis=(1, 2))]
        if len(x):
            acc += np.median(x, axis=0)                              # per-video median face
            n += 1
    if n == 0:
        raise SystemExit("No usable face frames.")
    return acc / n, n


def draw(ax, face, subset, labels):
    pos = {r: i for i, r in enumerate(SAVED_FACE_INDICES)}
    raw = face_subset_raw_indices(subset)
    xy = {r: face[pos[r]] for r in SAVED_FACE_INDICES}
    ax.scatter(face[:, 0], face[:, 1], s=10, c="#d0d0d0", zorder=1)

    g = SkeletonGraph(num_vertices=len(raw), face_only=True)
    for a, b in g._get_face_edges(offset=0):
        pa, pb = xy[raw[a]], xy[raw[b]]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], c="#888888", lw=0.8, zorder=2)

    for name, members, color in REGIONS:
        pts = [r for r in raw if r in members]
        if pts:
            p = np.array([xy[r] for r in pts])
            ax.scatter(p[:, 0], p[:, 1], s=34, c=color, edgecolors="white", linewidths=0.6,
                       zorder=3, label=f"{name} ({len(pts)})")
    for r in (ANCHOR_LEFT_EYE_INNER, ANCHOR_RIGHT_EYE_INNER):
        ax.scatter(*xy[r], s=140, facecolors="none", edgecolors="black", linewidths=1.2, zorder=4)
    if labels:
        for r in raw:
            ax.annotate(str(r), xy[r], fontsize=5.5, xytext=(2, 2), textcoords="offset points", zorder=5)

    ax.set_title(f"{subset}  ({len(raw)} points)", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()            # image coordinates: y points down
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)


def main():
    p = argparse.ArgumentParser(description="Draw every face subset on the mean face.")
    p.add_argument("--face-dir", default="processed_data/face_keypoints_normalized_v2")
    p.add_argument("--videos", type=int, default=30, help="videos to average (default 30)")
    p.add_argument("--labels", action="store_true", help="annotate MediaPipe indices")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "face_subsets"))
    args = p.parse_args()

    face, n = mean_face(args.face_dir, args.videos, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    subsets = [s for s in ORDER if s in FACE_SUBSETS] + [s for s in FACE_SUBSETS if s not in ORDER]

    for s in subsets:
        fig, ax = plt.subplots(figsize=(5, 5.4))
        draw(ax, face, s, args.labels)
        fig.text(0.5, 0.01, "ring = anchor (inner eye corners 133 / 362)   grey = other saved points",
                 ha="center", fontsize=7, color="#555555")
        fig.savefig(os.path.join(args.out_dir, f"face_subset_{s}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(subsets), figsize=(4.6 * len(subsets), 5.4))
    for ax, s in zip(np.atleast_1d(axes), subsets):
        draw(ax, face, s, args.labels)
    fig.suptitle(f"Face subsets on the mean face-local face ({n} videos); "
                 "edges = face graph used by the ST-GCN", fontsize=12)
    fig.savefig(os.path.join(args.out_dir, "face_subsets_all.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Mean face from {n} videos -> {args.out_dir}/ (face_subsets_all.png + one per subset)")


if __name__ == "__main__":
    main()