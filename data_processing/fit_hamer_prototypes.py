"""
fit_hamer_prototypes.py -- k-means prototype handshapes for the "proto" HaMeR encoder
(src/frontend.py HandShapePrototypeEncoder, TAILORED_ENCODER_STUDY in queue_train_phrase.py).

Input : processed_data/hamer_features/<video>_hamer.pt   (extract_hamer_features.py)
          hand_pose     (T, 2, 15, 3, 3)   finger-joint rotations, slot 0 = left, 1 = right
          global_orient (T, 2, 3, 3)       wrist orientation (all zeros = hand not detected)
Output: processed_data/hamer_prototypes_k<K>.npz
          centroids (K, 90) float32   prototype hand shapes in the model's 6D layout
          tau       float             mean squared distance of a frame to its own centroid -> the
                                      initial softmax temperature (learned afterwards)
          counts    (K,)              frames per prototype
          + k, n_frames, n_videos, stride, seed (for the record)

Only the TRAINING videos of dataset_splits.json are read (no validation / test leakage). Both hands
are pooled: HaMeR returns left hands in mirrored right-hand form, so one set of prototypes serves both
(the model shares the prototypes across hands too). Only frames where the hand was detected count.
The 6D layout is R[..., :, :2] flattened row-major -- identical to hamer_hand_parts() in
src/frontend.py; change both or neither.

k-means++ seeding + Lloyd iterations in numpy (no scikit-learn needed); an empty cluster is re-seeded
with the frame farthest from its centroid.

Run from the repo root, once, before queueing the "proto" arm:
    python data_processing/fit_hamer_prototypes.py               # K = 32
    python data_processing/fit_hamer_prototypes.py --k 64 --stride 5
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shape_6d(hand_pose):
    """(T, 2, 15, 3, 3) -> (T, 2, 90): first two columns of every finger rotation, row-major."""
    hp = np.asarray(hand_pose, dtype=np.float32)
    return hp[..., :, :2].reshape(hp.shape[0], 2, 90)


def detected(global_orient):
    """(T, 2, 3, 3) -> (T, 2) bool, same rule as the model (non-zero orientation matrix)."""
    g = np.asarray(global_orient, dtype=np.float32)
    return np.abs(g).reshape(g.shape[0], 2, 9).sum(-1) > 1e-6


def sq_dist(x, c, chunk=65536):
    """Squared Euclidean distances (N, K), in chunks."""
    c2 = (c ** 2).sum(1)
    out = np.empty((x.shape[0], c.shape[0]), dtype=np.float32)
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        out[s:s + chunk] = np.maximum((xs ** 2).sum(1, keepdims=True) - 2.0 * xs @ c.T + c2, 0.0)
    return out


def kmeans_pp(x, k, rng):
    n = x.shape[0]
    centers = [x[rng.integers(n)]]
    d = sq_dist(x, centers[0][None])[:, 0]
    for _ in range(1, k):
        p = d / d.sum() if d.sum() > 0 else None
        centers.append(x[rng.choice(n, p=p)])
        d = np.minimum(d, sq_dist(x, centers[-1][None])[:, 0])
    return np.stack(centers).astype(np.float32)


def kmeans(x, k, seed=0, iters=100, tol=1e-5):
    rng = np.random.default_rng(seed)
    c = kmeans_pp(x, k, rng)
    prev = None
    for it in range(iters):
        d = sq_dist(x, c)
        lab = d.argmin(1)
        inertia = float(d[np.arange(len(x)), lab].mean())
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
            else:                                   # empty cluster: re-seed at the worst-fitted frame
                far = int(d[np.arange(len(x)), lab].argmax())
                c[j] = x[far]
                lab[far] = j
        if prev is not None and abs(prev - inertia) <= tol * max(prev, 1e-12):
            break
        prev = inertia
    d = sq_dist(x, c)
    lab = d.argmin(1)
    return c, lab, float(d[np.arange(len(x)), lab].mean()), it + 1


def collect(video_ids, hamer_dir, stride, max_frames, seed):
    import torch                                    # only for reading the .pt files
    feats, used, missing = [], 0, []
    for vid in video_ids:
        path = os.path.join(hamer_dir, f"{vid}_hamer.pt")
        if not os.path.exists(path):
            missing.append(vid)
            continue
        d = torch.load(path, weights_only=False)
        hp, go = np.asarray(d["hand_pose"]), np.asarray(d["global_orient"])
        if hp.shape[0] == 0:
            continue
        s = shape_6d(hp)[::stride]                  # (T', 2, 90)
        m = detected(go)[::stride]                  # (T', 2)
        feats.append(s[m])                          # both hands pooled
        used += 1
    if not feats:
        raise SystemExit(f"No HaMeR frames found in {hamer_dir} for the training videos.")
    x = np.concatenate(feats).astype(np.float32)
    if len(x) > max_frames:
        x = x[np.random.default_rng(seed).choice(len(x), max_frames, replace=False)]
    return x, used, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--splits", default=os.path.join(REPO_ROOT, "dataset_splits.json"))
    ap.add_argument("--hamer-dir", default="processed_data/hamer_features")
    ap.add_argument("--out", default=None, help="default: processed_data/hamer_prototypes_k<K>.npz")
    ap.add_argument("--stride", type=int, default=10, help="every n-th frame (neighbouring frames are near-copies)")
    ap.add_argument("--max-frames", type=int, default=300000, help="random subsample above this many hand frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    with open(args.splits) as f:
        train = [v.replace(".npy", "").replace(".pt", "") for v in json.load(f)["train"]]
    x, used, missing = collect(train, args.hamer_dir, max(1, args.stride), args.max_frames, args.seed)
    print(f"{len(x)} detected hand frames from {used}/{len(train)} training videos"
          + (f" ({len(missing)} without a HaMeR file)" if missing else ""))
    if len(x) < 10 * args.k:
        raise SystemExit(f"Too few frames ({len(x)}) for k = {args.k}.")

    c, lab, tau, n_it = kmeans(x, args.k, args.seed, args.iters)
    counts = np.bincount(lab, minlength=args.k)
    out = args.out or f"processed_data/hamer_prototypes_k{args.k}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, centroids=c.astype(np.float32), tau=np.float32(tau), counts=counts, k=args.k,
             n_frames=len(x), n_videos=used, stride=args.stride, seed=args.seed)
    share = np.sort(counts)[::-1] / counts.sum()
    print(f"k-means: {n_it} iterations, tau (mean squared distance to own centroid) = {tau:.4f}")
    print(f"cluster shares: largest {share[0]:.1%}, smallest {share[-1]:.2%}, "
          f"{int((counts < 0.002 * counts.sum()).sum())} clusters under 0.2%")
    print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
