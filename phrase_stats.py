"""
Shared helpers for the phrase-segmentation analyses (count_oracle_decoding.py, the tempo and
auxiliary-supervision studies): loading saved predictions, grouping runs into configurations,
boundary counts, and a PAIRED bootstrap over test videos.

Predictions come from evaluate_phrase.py --save-predictions:
  evaluation_phrase/predictions/<run>/<split>.npz        {video: (3, T) float16 [O, I, B]}
  evaluation_phrase/predictions/<run>/<split>_meta.json  labels_dir, description, seed, ...

Why a paired bootstrap: with 17 test videos the marginal confidence intervals of two arms
overlap even when one arm is ahead on almost every video. Resampling the SAME videos for both
arms in each draw measures the difference directly. Counts (TP, n_pred, n_gold) are pooled over
the resampled videos rather than averaging per-video F1, which is unstable on short videos.

Command line (quick paired comparison on boundary F1):
  python phrase_stats.py --a multistream_bilstm-301 multistream_bilstm-302 multistream_bilstm-303 \
                         --b multistream_bilstm-208 multistream_bilstm-209 multistream_bilstm-210 \
                         --split test --k 5
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from src.metrics import B_TAG, extract_segments, match_boundaries

PRED_ROOT = os.path.join("evaluation_phrase", "predictions")


# ----------------------------------------------------------------------------- loading
def config_name(desc):
    """Description without the '[seed=...]' suffix (same rule as metric_analysis.py)."""
    return re.sub(r"\s*\[seed=\d+\]\s*$", "", desc or "").strip()


def load_meta(pred_root, run, split):
    path = os.path.join(pred_root, run, f"{split}_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} -- run evaluate_phrase.py --prefixes ... --split {split} --save-predictions")
    with open(path) as f:
        return json.load(f)


def load_predictions(pred_root, run, split, labels_dir=None):
    """
    Returns (probs, meta, gold):
      probs {vid: (3, T) float32}, gold {vid: (T,) int64}, both cropped to the shorter length
      (as evaluate_videos does). Gold comes from meta["labels_dir"] unless labels_dir is given.
    """
    meta = load_meta(pred_root, run, split)
    z = np.load(os.path.join(pred_root, run, f"{split}.npz"))
    ldir = labels_dir or meta.get("labels_dir")
    probs, gold = {}, {}
    for vid in z.files:
        p = z[vid].astype(np.float32)
        if p.shape[0] != 3 and p.shape[-1] == 3:
            p = p.T
        gp = os.path.join(ldir, f"{vid}.npy")
        if not os.path.exists(gp):
            continue
        g = np.load(gp).astype(np.int64)
        T = min(len(g), p.shape[1])
        probs[vid], gold[vid] = p[:, :T], g[:T]
    if not probs:
        raise RuntimeError(f"{run}/{split}: no video with gold labels in {ldir}")
    return probs, meta, gold


def available_runs(pred_root, split):
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(pred_root, "*", f"{split}.npz")))


def group_runs_by_config(metas):
    """metas: {run: meta} -> {config name: [runs]} (description without the seed suffix)."""
    groups = defaultdict(list)
    for run, m in metas.items():
        groups[config_name(m.get("description", run))].append(run)
    return dict(groups)


def expand_prefixes(model, prefixes):
    """('multistream_bilstm', ['301-303', '208']) -> ['multistream_bilstm-301', ..., 'multistream_bilstm-208']"""
    out = []
    for p in prefixes:
        if "-" in str(p):
            a, b = str(p).split("-")
            out += [f"{model}-{i}" for i in range(int(a), int(b) + 1)]
        else:
            out.append(f"{model}-{p}")
    return out


# ----------------------------------------------------------------------------- counts
def gold_starts_of(gold):
    return np.array([s for s, _ in extract_segments(gold)], dtype=np.int64)


def boundary_counts(gold_starts, pred_starts, k):
    """One-to-one matching within +-k frames (src.metrics.match_boundaries) -> (tp, n_pred, n_gold)."""
    g_ok, _, _ = match_boundaries(gold_starts, pred_starts, k)
    return int(g_ok.sum()), int(len(pred_starts)), int(len(gold_starts))


def f1_from_counts(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def argmax_starts(probs, merge_starts=0):
    from src.metrics import merge_close_starts
    segs = extract_segments(probs.argmax(axis=0))
    if merge_starts:
        segs = merge_close_starts(segs, probs[B_TAG], merge_starts)
    return np.array([s for s, _ in segs], dtype=np.int64)


def per_video_counts(run_preds, k, starts_fn=argmax_starts):
    """run_preds: (probs, meta, gold) -> {vid: (tp, n_pred, n_gold)}."""
    probs, _, gold = run_preds
    return {v: boundary_counts(gold_starts_of(gold[v]), starts_fn(probs[v]), k) for v in probs}


# ----------------------------------------------------------------------------- bootstrap
def paired_bootstrap(counts_a, counts_b, n_boot=2000, seed=0, metric=f1_from_counts):
    """
    counts_a / counts_b: lists (one entry per seed/run of the arm) of {vid: (tp, n_pred, n_gold)}.
    In every draw the video IDs are resampled with replacement; for each run the counts of the
    drawn videos are pooled and turned into the metric; runs are averaged within each arm; the
    statistic is mean(A) - mean(B). Only videos present in every run are used.
    Returns {"diff": observed difference, "ci95": (lo, hi), "p_le_0": P(diff <= 0), "n_videos"}.
    """
    vids = sorted(set.intersection(*[set(c) for c in counts_a + counts_b]))
    if not vids:
        raise ValueError("the two arms share no videos")

    def arr(c):
        return np.array([c[v] for v in vids], dtype=np.float64)          # (V, 3)

    A = np.stack([arr(c) for c in counts_a])                            # (Ra, V, 3)
    B = np.stack([arr(c) for c in counts_b])

    def stat(idx):
        sa = np.mean([metric(*A[r, idx].sum(0)) for r in range(A.shape[0])])
        sb = np.mean([metric(*B[r, idx].sum(0)) for r in range(B.shape[0])])
        return sa - sb

    rng = np.random.default_rng(seed)
    V = len(vids)
    obs = stat(np.arange(V))
    draws = np.array([stat(rng.integers(0, V, V)) for _ in range(n_boot)])
    return {"diff": float(obs), "ci95": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))),
            "p_le_0": float(np.mean(draws <= 0)), "n_videos": V, "n_boot": n_boot}


def paired_bootstrap_runs(runs_a, runs_b, split, k=5, pred_root=PRED_ROOT, merge_starts=0, **kw):
    """Convenience wrapper: boundary F1@k of argmax(+merge) starts, arm A minus arm B."""
    fn = (lambda p: argmax_starts(p, merge_starts))
    ca = [per_video_counts(load_predictions(pred_root, r, split), k, fn) for r in runs_a]
    cb = [per_video_counts(load_predictions(pred_root, r, split), k, fn) for r in runs_b]
    return paired_bootstrap(ca, cb, **kw)


def format_bootstrap(res, name="diff"):
    lo, hi = res["ci95"]
    return (f"{name} {res['diff']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(diff<=0) {res['p_le_0']:.3f}  "
            f"({res['n_videos']} videos, {res['n_boot']} draws)")


def main():
    ap = argparse.ArgumentParser(description="Paired bootstrap of boundary F1 between two arms.")
    ap.add_argument("--a", nargs="+", required=True, help="runs of arm A (e.g. multistream_bilstm-301 ...)")
    ap.add_argument("--b", nargs="+", required=True, help="runs of arm B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--merge-starts", type=int, default=0, help="duplicate-start suppression for both arms")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pred-root", default=PRED_ROOT)
    args = ap.parse_args()
    res = paired_bootstrap_runs(args.a, args.b, args.split, args.k, args.pred_root, args.merge_starts,
                                n_boot=args.n_boot)
    print(format_bootstrap(res, f"BndF1@{args.k} (A - B)"))


if __name__ == "__main__":
    main()
