"""
frame_f1_stats.py -- Frame-level F1 and boundary F1 with 95% CIs for the slides.

  slide 9  (model selection, VALIDATION set, 14 videos): every model, and the CI of each step:
           + hand shape, + face, modality-specific vs early fusion (same inputs)
  slide 10 (main comparison, TEST set, 17 videos): proposed model minus Moryossef et al. (2023)

Metrics
  Frame-level F1   as src.metrics: argmax label per frame, macro F1 over O / I / B on all frames of the
                   split pooled together. Needs no decoder.
  Boundary F1      phrase starts within +-5 frames (+-0.1 s), one-to-one matching (src.metrics.match_boundaries),
                   counts pooled over the videos of the split. Two decoders:
                     threshold 0.5   Moryossef et al. (2023) decoder at its default thresholds (B = O = 0.5),
                                     NOT tuned -> fair on the validation set
                     peak + thr      peak picking + threshold (count_oracle_decoding.tune_peaks_on_val), the
                                     decoder of the results slides, tuned on VAL. On the validation set it is
                                     tuned and scored on the same videos, so validation numbers are optimistic;
                                     on the test set they reproduce the slides.

Two bootstrap intervals for every difference (A - B), both paired over videos (the same videos are drawn for
A and B in every draw; per-video counts are summed and the metric recomputed):
  videos          seeds fixed: each model = the mean of its seeds
  videos + seeds  the seeds of each model are also drawn with replacement in every draw
Also: the exact permutation p over seeds (3 vs 3: 20 splits, smallest possible two-sided p = .10).

Saved predictions are read from evaluation_phrase/predictions/<run>/<split>.npz (phrase_stats.load_predictions).
If a run has no saved predictions, the script prints the evaluate_phrase.py command that makes them.

  python frame_f1_stats.py --body-only multistream_bilstm-130 multistream_bilstm-131 multistream_bilstm-132 \
      --early-fusion bilstm_2023-302 bilstm_2023-303 bilstm_2023-304
  python frame_f1_stats.py --selftest

Model defaults (override any of them):
  --body-only      modality-specific, body + hand keypoints           (no default: pass the run names)
  --hand-shape     modality-specific, + hand shape (HaMeR)            multistream_bilstm-112..114
  --proposed       modality-specific, + face (proposed)               multistream_bilstm-115..117
  --early-fusion   early fusion, same inputs                          (no default: pass the run names)
  --moryossef      Moryossef et al. (2023), re-implemented            bilstm_2023-244..246

Output: evaluation_phrase/frame_f1_stats/frame_f1_summary.txt and frame_f1_runs.csv
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np

from count_oracle_decoding import peaks_with_heights, tune_peaks_on_val
from phrase_stats import PRED_ROOT, boundary_counts, gold_starts_of, load_predictions
from src.metrics import B_TAG, decode_threshold_2023

TOL = 5
MODELS = [  # key, label, default runs
    ("body_only", "Modality-specific: body + hand keypoints", None),
    ("hand_shape", "Modality-specific: + hand shape (HaMeR)",
     ["multistream_bilstm-112", "multistream_bilstm-113", "multistream_bilstm-114"]),
    ("proposed", "Modality-specific: + face (proposed)",
     ["multistream_bilstm-115", "multistream_bilstm-116", "multistream_bilstm-117"]),
    ("early_fusion", "Early fusion (same inputs)", None),
    ("moryossef", "Moryossef et al. (2023), re-implemented",
     ["bilstm_2023-244", "bilstm_2023-245", "bilstm_2023-246"]),
]
COMPARISONS = [  # slide, split, A, B, label
    ("slide 9", "val", "hand_shape", "body_only", "+ hand shape (HaMeR)"),
    ("slide 9", "val", "proposed", "hand_shape", "+ face"),
    ("slide 9", "val", "proposed", "early_fusion", "modality-specific vs early fusion (same inputs)"),
    ("slide 10", "test", "proposed", "moryossef", "proposed - Moryossef et al. (2023)"),
]
SECONDARY = [  # the same steps on the other split, for reference only
    ("test", "hand_shape", "body_only", "+ hand shape (HaMeR)"),
    ("test", "proposed", "hand_shape", "+ face"),
    ("test", "proposed", "early_fusion", "modality-specific vs early fusion (same inputs)"),
    ("val", "proposed", "moryossef", "proposed - Moryossef et al. (2023)"),
]
METRICS = [  # key, title
    ("frame", "Frame-level F1"),
    ("bnd_thr", "Boundary F1 (+-0.1 s), threshold decoder at 0.5 / 0.5 (untuned)"),
    ("bnd_peak", "Boundary F1 (+-0.1 s), peak picking + threshold tuned on val"),
]


# ----------------------------------------------------------------------------- metrics from counts
def confusion(gold, pred):
    """3 x 3 counts, rows = gold, cols = predicted."""
    return np.bincount(gold * 3 + pred, minlength=9).reshape(3, 3).astype(np.float64)


def macro_f1(conf):
    """conf: (..., 3, 3). Macro F1 over O / I / B; a class with no gold and no prediction scores 0, as
    sklearn's f1_score(zero_division=0) does."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    denom = conf.sum(-1) + conf.sum(-2)
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return f1.mean(-1)


def bnd_f1(c):
    """c: (..., 3) = (tp, n_pred, n_gold) -> F1 (0 when undefined)."""
    tp, npred, ngold = c[..., 0], c[..., 1], c[..., 2]
    denom = npred + ngold
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


SCORE = {"frame": macro_f1, "bnd_thr": bnd_f1, "bnd_peak": bnd_f1}


# ----------------------------------------------------------------------------- per-run counts
def run_counts(pred_root, run):
    """{(metric, split): {vid: counts}} for one run. The peak decoder is tuned on this run's val split."""
    val = load_predictions(pred_root, run, "val")
    tuned = tune_peaks_on_val(val)
    out = {}
    for split in ("val", "test"):
        probs, _, gold = val if split == "val" else load_predictions(pred_root, run, split)
        fr, bt, bp = {}, {}, {}
        for v, p in probs.items():
            g = gold[v]
            gs = gold_starts_of(g)
            fr[v] = confusion(g, p.argmax(axis=0))
            st_thr = np.array([s for s, _ in decode_threshold_2023(p, 0.5, 0.5)], dtype=np.int64)
            idx, h = peaks_with_heights(p[B_TAG], tuned["min_dist"], tuned["sigma"])
            st_peak = np.sort(idx[h >= tuned["thr"]])
            bt[v] = np.array(boundary_counts(gs, st_thr, TOL), dtype=np.float64)
            bp[v] = np.array(boundary_counts(gs, st_peak, TOL), dtype=np.float64)
        out[("frame", split)], out[("bnd_thr", split)], out[("bnd_peak", split)] = fr, bt, bp
    return out, tuned


# ----------------------------------------------------------------------------- statistics
def stack(counts_by_run, vids):
    """(n_runs, n_videos, ...)"""
    return np.stack([np.stack([c[v] for v in vids]) for c in counts_by_run])


def bootstrap_diff(CA, CB, score, n_boot=10000, seed=0, resample_seeds=False):
    """CA, CB: (n_runs, n_videos, ...) on the same videos. Returns the point difference and the 95% CI."""
    rng = np.random.default_rng(seed)
    nv = CA.shape[1]

    def model_score(C, w, s_idx):
        return score(np.tensordot(w, C[s_idx], axes=([0], [1]))).mean()

    w0 = np.ones(nv)
    point = model_score(CA, w0, np.arange(len(CA))) - model_score(CB, w0, np.arange(len(CB)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, nv, nv), minlength=nv).astype(np.float64)
        ia = rng.integers(0, len(CA), len(CA)) if resample_seeds else np.arange(len(CA))
        ib = rng.integers(0, len(CB), len(CB)) if resample_seeds else np.arange(len(CB))
        draws[b] = model_score(CA, w, ia) - model_score(CB, w, ib)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def permutation_p(a, b):
    """Exact two-sided permutation p of mean(a) - mean(b) over seeds."""
    pooled = np.r_[a, b]
    obs = abs(np.mean(a) - np.mean(b))
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        hits += abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12
        total += 1
    return hits / total


def fmt(d, lo, hi):
    return f"{d:+.3f} (95% CI: {lo:+.3f} to {hi:+.3f})"


# ----------------------------------------------------------------------------- main
def run(models, pred_root, out_dir, n_boot=10000):
    os.makedirs(out_dir, exist_ok=True)
    missing = [(r, s) for _, _, runs in models if runs for r in runs for s in ("val", "test")
               if not os.path.exists(os.path.join(pred_root, r, f"{s}.npz"))]
    if missing:
        print("Missing saved predictions -- make them first, e.g.:")
        for r, s in missing:
            model, prefix = r.rsplit("-", 1)
            print(f"  python evaluate_phrase.py --model {model} --prefixes {prefix} --split {s} --save-predictions")
        sys.exit(1)

    counts, tuned = {}, {}  # counts[(model, metric, split)] = [ {vid: counts} per run ]
    for key, _, runs in models:
        if not runs:
            continue
        for r in runs:
            print(f"  {r} ...")
            c, tuned[r] = run_counts(pred_root, r)
            for (m, s), d in c.items():
                counts.setdefault((key, m, s), []).append(d)

    labels = {k: l for k, l, _ in models}
    runs_of = {k: r for k, _, r in models}
    per_run = {}  # (model, metric, split) -> array over seeds
    rows = []
    for key, _, runs in models:
        if not runs:
            continue
        for m, _ in METRICS:
            for s in ("val", "test"):
                vals = np.array([float(SCORE[m](sum(d.values()))) for d in counts[(key, m, s)]])
                per_run[(key, m, s)] = vals
                for r, x in zip(runs, vals):
                    rows.append({"model": labels[key], "run": r, "metric": m, "split": s, "value": round(x, 4),
                                 "peak_settings_tuned_on_val": f"d={tuned[r]['min_dist']} sigma={tuned[r]['sigma']} "
                                                               f"t={tuned[r]['thr']}"})

    def compare(m, split, a, b, name):
        if not runs_of.get(a) or not runs_of.get(b):
            miss = labels[a] if not runs_of.get(a) else labels[b]
            return f"  {name:50s} skipped: runs of '{miss}' not given"
        ca, cb = counts[(a, m, split)], counts[(b, m, split)]
        vids = sorted(set.intersection(*[set(c) for c in ca + cb]))
        CA, CB = stack(ca, vids), stack(cb, vids)
        d, lo, hi = bootstrap_diff(CA, CB, SCORE[m], n_boot)
        _, lo2, hi2 = bootstrap_diff(CA, CB, SCORE[m], n_boot, resample_seeds=True)
        p = permutation_p(per_run[(a, m, split)], per_run[(b, m, split)])
        return (f"  {name:50s} {fmt(d, lo, hi)}   videos + seeds: {lo2:+.3f} to {hi2:+.3f}   "
                f"seed permutation p = {p:.2f}   ({len(vids)} videos)")

    L = ["Mean +- SD over seeds. Frame-level F1: argmax per frame, macro F1 over O / I / B, frames pooled. "
         "Boundary F1: phrase starts within +-5 frames (+-0.1 s), counts pooled.",
         f"Differences A - B: 95% CI from a paired bootstrap over videos ({n_boot} draws); "
         "'videos + seeds' also redraws the seeds of each model.", ""]
    for m, title in METRICS:
        L += ["=" * 100, title, "=" * 100,
              f"{'model':45s}{'val (14 videos)':>20s}{'test (17 videos)':>20s}   per seed (val | test)"]
        for key, label, runs in models:
            if not runs:
                L.append(f"{label:45s}{'(runs not given)':>20s}")
                continue
            v, t = per_run[(key, m, "val")], per_run[(key, m, "test")]
            sd = (lambda a: a.std(ddof=1) if len(a) > 1 else 0.0)
            L.append(f"{label:45s}{f'{v.mean():.3f} +- {sd(v):.3f}':>20s}{f'{t.mean():.3f} +- {sd(t):.3f}':>20s}   "
                     + " ".join(f"{x:.3f}" for x in v) + " | " + " ".join(f"{x:.3f}" for x in t))
        L.append("")
        for slide in ("slide 9", "slide 10"):
            sp = "val" if slide == "slide 9" else "test"
            L.append(f"{slide} ({'validation' if sp == 'val' else 'test'} set):")
            L += [compare(m, split, a, b, name) for s, split, a, b, name in COMPARISONS if s == slide]
        L.append("For reference only (the other split):")
        L += [compare(m, split, a, b, f"[{split}] {name}") for split, a, b, name in SECONDARY]
        L.append("")
    L += ["Notes:",
          "- The best epoch of every run was chosen by validation Frame-level F1, so validation scores are the "
          "maximum over epochs and slightly optimistic for every model alike.",
          "- 'peak + thr' is tuned on the validation set: its validation numbers are tuned and scored on the same "
          "videos (optimistic). Use the untuned threshold-0.5 decoder for validation claims.",
          "- With 3 seeds per model the seed permutation p can never be below .10; it is shown only as a check on "
          "seed variation."]
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "frame_f1_summary.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "frame_f1_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(text)
    return per_run


# ----------------------------------------------------------------------------- selftest
def selftest():
    import json
    import tempfile
    from sklearn.metrics import f1_score
    ok = []
    rng = np.random.default_rng(0)
    g = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.7, g, rng.integers(0, 3, 500))
    ok.append(("macro F1 matches sklearn",
               abs(macro_f1(confusion(g, p)) - f1_score(g, p, labels=[0, 1, 2], average="macro", zero_division=0)) < 1e-12))
    ok.append(("boundary F1 from counts", abs(float(bnd_f1(np.array([3., 4, 6]))) - 0.6) < 1e-12))
    ok.append(("permutation p: 3 vs 3 fully separated = .10",
               abs(permutation_p(np.array([3., 4, 5]), np.array([0., 1, 2])) - 0.1) < 1e-12))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/labels")
        golds = {}
        for split, n in (("val", 6), ("test", 8)):
            for i in range(n):
                gg = np.zeros(1500, dtype=np.int64)
                t = 30
                while t < 1400:
                    L = int(rng.integers(40, 120))
                    gg[t], gg[t + 1:t + L] = 2, 1
                    t += L + int(rng.integers(5, 30))
                golds[(split, f"{split}{i}")] = gg
                np.save(f"{d}/labels/{split}{i}.npy", gg)
        models = []
        for key, acc in (("body_only", 0.70), ("hand_shape", 0.80), ("proposed", 0.82),
                         ("early_fusion", 0.75), ("moryossef", 0.60)):
            runs = []
            for sd in range(3):
                name = f"{key}-{sd}"
                os.makedirs(f"{d}/pred/{name}")
                for split in ("val", "test"):
                    arrs = {}
                    for (sp, v), gg in golds.items():
                        if sp != split:
                            continue
                        q = rng.random((3, len(gg))).astype(np.float32) * 0.3
                        right = rng.random(len(gg)) < acc + 0.01 * sd
                        q[gg[right], np.where(right)[0]] += 1.0
                        arrs[v] = q / q.sum(0, keepdims=True)
                    np.savez_compressed(f"{d}/pred/{name}/{split}.npz", **arrs)
                    with open(f"{d}/pred/{name}/{split}_meta.json", "w") as f:
                        json.dump({"run": name, "labels_dir": f"{d}/labels"}, f)
                runs.append(name)
            models.append((key, dict((k, l) for k, l, _ in MODELS)[key], runs))
        res = run(models, f"{d}/pred", f"{d}/out", n_boot=200)
        ok.append(("proposed beats Moryossef on test boundary F1",
                   res[("proposed", "bnd_peak", "test")].mean() > res[("moryossef", "bnd_peak", "test")].mean()))
        txt = open(f"{d}/out/frame_f1_summary.txt").read()
        ok.append(("summary has all three metrics", all(t in txt for _, t in METRICS)))
    for name, c in ok:
        print(f"  [{'ok' if c else 'FAIL'}] {name}")
    return all(c for _, c in ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, label, runs in MODELS:
        ap.add_argument("--" + key.replace("_", "-"), nargs="+", default=runs, metavar="RUN", help=label)
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "frame_f1_stats"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run([(k, l, getattr(a, k)) for k, l, _ in MODELS], a.pred_root, a.out_dir, a.n_boot)


if __name__ == "__main__":
    main()