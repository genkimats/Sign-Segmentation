"""
diagnose_score_distributions.py -- why does adding the face shift the start score of the plain
early-fusion models by about -18 percentile points at EVERY start (analyze_nonmanual_pooled.py)?

For every run of every face / no-face pair (default: analyze_nonmanual_pooled.DEFAULT_PAIRS), from the
saved TEST predictions (evaluation_phrase/predictions/<run>/test.npz, float16):
  ties        share of frames whose P(start) equals another frame's value in the same video, and share
              of frames with P(start) == 0 exactly (float16 underflow). Many ties make the per-video
              percentile ("rank") coarse and can pull starts down.
  quantiles   P(start) at the 50 / 90 / 99 / 99.9 % quantile of all frames
  peaks       local maxima (>= 10 frames apart, smoothed sigma 2) above 0.1 / 0.3 / 0.5, per gold start
              -> a model with many high peaks inside phrases pushes the starts down the percentile scale
  start score mean raw max P(start) within +-5 frames of the gold starts, and its mean percentile
  AUC         start vs in-phrase moment (>= 0.5 s from any start), on raw scores and on percentiles
Then per pair: face minus no face for each column. If the raw AUC barely changes while the percentile
drops, the shift is a property of the score distribution (more high frames everywhere), not of the
starts; if the tie share jumps, it is a float16 artefact.

    python diagnose_score_distributions.py
    python diagnose_score_distributions.py --runs bilstm_2023-268 bilstm_2023-259
Output: evaluation_phrase/score_distributions.csv (+ printed table)
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import rankdata

import analyze_nonmanual_v2 as v2
from analyze_nonmanual_pooled import DEFAULT_PAIRS


def run_stats(run, pred_root, tol=5, seed=0):
    from phrase_stats import load_predictions
    probs, _, gold = load_predictions(pred_root, run, "test")
    rng = np.random.default_rng(seed)
    tie, zero, allp, peaks, n_starts = [], [], [], {0.1: 0, 0.3: 0, 0.5: 0}, 0
    raw_s, raw_n, rk_s, rk_n = [], [], [], []
    for v in sorted(probs):
        p = np.asarray(probs[v][2], dtype=np.float64)
        g = gold[v]
        _, counts = np.unique(p, return_counts=True)
        tie.append((counts[counts > 1].sum(), len(p)))
        zero.append(((p == 0).sum(), len(p)))
        allp.append(p)
        s = gaussian_filter1d(p, 2.0)
        idx, _ = find_peaks(s, distance=10)
        for t in peaks:
            peaks[t] += int((s[idx] >= t).sum())
        st = v2.starts_of(g)
        ng = v2.negatives_of(g, st, rng)
        n_starts += len(st)
        rank = (rankdata(s) - 0.5) / len(s)
        raw_s.append(v2.window_max(s, st, tol)); raw_n.append(v2.window_max(s, ng, tol))
        rk_s.append(v2.window_max(rank, st, tol)); rk_n.append(v2.window_max(rank, ng, tol))
    allp = np.concatenate(allp)
    raw_s, raw_n, rk_s, rk_n = map(np.concatenate, (raw_s, raw_n, rk_s, rk_n))
    q = np.quantile(allp, [0.5, 0.9, 0.99, 0.999])
    return {"run": run, "tie_share": sum(a for a, _ in tie) / sum(b for _, b in tie),
            "zero_share": sum(a for a, _ in zero) / sum(b for _, b in zero),
            "q50": q[0], "q90": q[1], "q99": q[2], "q999": q[3],
            **{f"peaks>{t}_per_start": peaks[t] / n_starts for t in peaks},
            "start_raw_mean": raw_s.mean(), "start_rank_mean": rk_s.mean(),
            "auc_raw": v2.auc(raw_s, raw_n), "auc_rank": v2.auc(rk_s, rk_n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, help="explicit run names (default: every run of the pairs)")
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--out", default=os.path.join("evaluation_phrase", "score_distributions.csv"))
    args = ap.parse_args()
    from phrase_stats import PRED_ROOT, expand_prefixes
    pred_root = args.pred_root or PRED_ROOT
    if args.runs:
        groups = [("runs", {"face": args.runs, "noface": []})]
    else:
        groups = [(name, {"face": expand_prefixes(m, wf), "noface": expand_prefixes(m, nf)})
                  for name, m, wf, nf in DEFAULT_PAIRS]
    rows, cols = [], None
    for name, g in groups:
        stats = {}
        for side, runs in g.items():
            for r in runs:
                if not os.path.exists(os.path.join(pred_root, r, "test.npz")):
                    print(f"  (no test predictions: {r})")
                    continue
                s = run_stats(r, pred_root)
                s.update(pair=name, face=int(side == "face"))
                rows.append(s)
                stats.setdefault(side, []).append(s)
        cols = cols or [k for k in rows[-1] if k not in ("run", "pair", "face")]
        print(f"\n{name}")
        print("  " + " ".join(f"{c[:14]:>14s}" for c in ["side"] + cols))
        means = {}
        for side in ("face", "noface"):
            if side in stats:
                means[side] = {c: np.mean([s[c] for s in stats[side]]) for c in cols}
                print("  " + f"{side:>14s} " + " ".join(f"{means[side][c]:>14.4f}" for c in cols))
        if len(means) == 2:
            print("  " + f"{'face - noface':>14s} " + " ".join(f"{means['face'][c] - means['noface'][c]:>+14.4f}" for c in cols))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "run", "face"] + cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
