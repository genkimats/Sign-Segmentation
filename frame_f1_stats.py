"""
frame_f1_stats.py -- Frame-level F1 with 95% CIs for the slides.

  slide 9  (model selection, VALIDATION set, 14 videos): Frame-level F1 of every model, and the CI of
           each step: + hand shape, + face, modality-specific vs early fusion (same inputs)
  slide 10 (main comparison, TEST set, 17 videos): proposed model minus Moryossef et al. (2023)

Frame-level F1 is computed exactly as src.metrics does it: argmax label per frame, macro F1 over O / I / B
on all frames of the split pooled together (not a mean over videos). It needs no decoder.

Two bootstrap intervals for every difference (A - B), both paired over videos (the same videos are drawn
for A and B in every draw; per-video 3 x 3 confusion counts are summed and the macro F1 recomputed):
  videos          seeds fixed: each model = the mean Frame F1 of its 3 seeds
  videos + seeds  the 3 seeds of each model are also drawn with replacement in every draw, so seed
                  variation is part of the interval (use this one for small differences, e.g. + face)
Also: the exact permutation p over seeds (3 vs 3: 20 splits, smallest possible two-sided p = .10).

Saved predictions are read from evaluation_phrase/predictions/<run>/<split>.npz (phrase_stats.load_predictions).
If a run has no val predictions yet, the script prints the evaluate_phrase.py command that makes them.

  python frame_f1_stats.py --body-only <run> <run> <run> --early-fusion <run> <run> <run>
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

from phrase_stats import PRED_ROOT, load_predictions

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


# ----------------------------------------------------------------------------- frame F1
def confusion(gold, pred):
    """3 x 3 counts, rows = gold, cols = predicted."""
    return np.bincount(gold * 3 + pred, minlength=9).reshape(3, 3).astype(np.float64)


def macro_f1(conf):
    """conf: (..., 3, 3). Macro F1 over O / I / B; a class with no gold and no prediction scores 0, as
    sklearn's f1_score(zero_division=0) does."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    n_gold = conf.sum(-1)
    n_pred = conf.sum(-2)
    denom = n_gold + n_pred
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return f1.mean(-1)


def run_confusions(pred_root, run, split):
    probs, _, gold = load_predictions(pred_root, run, split)
    return {v: confusion(gold[v], probs[v].argmax(axis=0)) for v in probs}


# ----------------------------------------------------------------------------- statistics
def stack(confs_by_run, vids):
    """(n_runs, n_videos, 3, 3)"""
    return np.stack([np.stack([c[v] for v in vids]) for c in confs_by_run])


def bootstrap_diff(CA, CB, n_boot=10000, seed=0, resample_seeds=False):
    """CA, CB: (n_runs, n_videos, 3, 3) on the same videos. Returns the point difference and 95% CI."""
    rng = np.random.default_rng(seed)
    nv = CA.shape[1]

    def model_f1(C, w, s_idx):
        conf = np.einsum("v,rvij->rij", w, C[s_idx])
        return macro_f1(conf).mean()

    w0 = np.ones(nv)
    point = model_f1(CA, w0, np.arange(len(CA))) - model_f1(CB, w0, np.arange(len(CB)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, nv, nv), minlength=nv).astype(np.float64)
        ia = rng.integers(0, len(CA), len(CA)) if resample_seeds else np.arange(len(CA))
        ib = rng.integers(0, len(CB), len(CB)) if resample_seeds else np.arange(len(CB))
        draws[b] = model_f1(CA, w, ia) - model_f1(CB, w, ib)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def permutation_p(a, b):
    """Exact two-sided permutation p of mean(a) - mean(b) over seeds."""
    pooled = np.r_[a, b]
    obs = abs(np.mean(a) - np.mean(b))
    n, hits, total = len(a), 0, 0
    for idx in itertools.combinations(range(len(pooled)), n):
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
    confs, missing = {}, []
    for key, label, runs in models:
        if not runs:
            continue
        for split in ("val", "test"):
            for r in runs:
                if not os.path.exists(os.path.join(pred_root, r, f"{split}.npz")):
                    missing.append((r, split))
    if missing:
        print("Missing saved predictions -- make them first, e.g.:")
        for r, split in missing:
            model, prefix = r.rsplit("-", 1)
            print(f"  python evaluate_phrase.py --model {model} --prefixes {prefix} --split {split} --save-predictions")
        sys.exit(1)
    for key, label, runs in models:
        if runs:
            for split in ("val", "test"):
                confs[(key, split)] = [run_confusions(pred_root, r, split) for r in runs]

    labels = {k: l for k, l, _ in models}
    runs_of = {k: r for k, _, r in models}
    rows = []
    L = ["Frame-level F1 (argmax per frame, macro F1 over O / I / B, frames of the split pooled; mean +- SD "
         "over seeds)", "",
         f"{'model':45s}{'val (14 videos)':>20s}{'test (17 videos)':>20s}   per seed (val | test)"]
    f1s = {}
    for key, label, runs in models:
        if not runs:
            L.append(f"{label:45s}{'(runs not given)':>20s}")
            continue
        cells, per_seed = [], []
        for split in ("val", "test"):
            vals = np.array([macro_f1(sum(c.values())) for c in confs[(key, split)]])
            f1s[(key, split)] = vals
            cells.append(f"{vals.mean():.3f} +- {vals.std(ddof=1) if len(vals) > 1 else 0:.3f}")
            per_seed.append(" ".join(f"{x:.3f}" for x in vals))
            for r, x in zip(runs, vals):
                rows.append({"model": label, "run": r, "split": split, "frame_f1": round(float(x), 4)})
        L.append(f"{label:45s}{cells[0]:>20s}{cells[1]:>20s}   {per_seed[0]} | {per_seed[1]}")

    def compare(split, a, b, name):
        if not runs_of.get(a) or not runs_of.get(b):
            return f"  {name:50s} skipped: runs of '{labels[a] if not runs_of.get(a) else labels[b]}' not given"
        vids = sorted(set.intersection(*[set(c) for c in confs[(a, split)] + confs[(b, split)]]))
        CA, CB = stack(confs[(a, split)], vids), stack(confs[(b, split)], vids)
        d, lo, hi = bootstrap_diff(CA, CB, n_boot)
        _, lo2, hi2 = bootstrap_diff(CA, CB, n_boot, resample_seeds=True)
        p = permutation_p(f1s[(a, split)], f1s[(b, split)])
        return (f"  {name:50s} {fmt(d, lo, hi)}   videos + seeds: {lo2:+.3f} to {hi2:+.3f}   "
                f"seed permutation p = {p:.2f}   ({len(vids)} videos)")

    L += ["", "Differences in Frame-level F1 (A - B). 95% CI: paired bootstrap over videos "
          f"({n_boot} draws); 'videos + seeds' also redraws the 3 seeds of each model.", ""]
    for slide in ("slide 9", "slide 10"):
        sp = "val" if slide == "slide 9" else "test"
        L.append(f"{slide} ({'validation' if sp == 'val' else 'test'} set):")
        for s, split, a, b, name in COMPARISONS:
            if s == slide:
                L.append(compare(split, a, b, name))
        L.append("")
    L.append("For reference only (the other split; not for the slides):")
    for split, a, b, name in SECONDARY:
        L.append(compare(split, a, b, f"[{split}] {name}"))
    L += ["", "Notes:",
          "- The best epoch of every run was chosen by validation Frame-level F1, so validation scores are "
          "the maximum over epochs and slightly optimistic for every model alike.",
          "- With 3 seeds per model the seed permutation p can never be below .10; it is shown only as a "
          "check on seed variation."]
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "frame_f1_summary.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "frame_f1_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "run", "split", "frame_f1"])
        w.writeheader()
        w.writerows(rows)
    print(text)
    return f1s


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
    g2 = np.zeros(50, dtype=np.int64)
    ok.append(("empty classes score 0 (sklearn zero_division=0)",
               abs(macro_f1(confusion(g2, g2)) - f1_score(g2, g2, labels=[0, 1, 2], average="macro",
                                                          zero_division=0)) < 1e-12))
    ok.append(("permutation p: 3 vs 3 fully separated = .10",
               abs(permutation_p(np.array([3., 4, 5]), np.array([0., 1, 2])) - 0.1) < 1e-12))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/labels")
        golds = {}
        for split, n in (("val", 6), ("test", 8)):
            for i in range(n):
                gg = np.repeat(rng.integers(0, 2, 40), 25)
                gg[np.r_[np.diff(gg) > 0, False]] = 2
                golds[(split, f"{split}{i}")] = gg
                np.save(f"{d}/labels/{split}{i}.npy", gg)
        models = []
        for key, acc in (("body_only", 0.70), ("hand_shape", 0.80), ("proposed", 0.81),
                         ("early_fusion", 0.75), ("moryossef", 0.65)):
            runs = []
            for sd in range(3):
                run_name = f"{key}-{sd}"
                os.makedirs(f"{d}/pred/{run_name}")
                for split in ("val", "test"):
                    arrs = {}
                    for (sp, v), gg in golds.items():
                        if sp != split:
                            continue
                        q = rng.random((3, len(gg))).astype(np.float32) * 0.5
                        right = rng.random(len(gg)) < acc + 0.01 * sd
                        q[gg[right], np.where(right)[0]] += 1.0
                        arrs[v] = q
                    np.savez_compressed(f"{d}/pred/{run_name}/{split}.npz", **arrs)
                    with open(f"{d}/pred/{run_name}/{split}_meta.json", "w") as f:
                        json.dump({"run": run_name, "labels_dir": f"{d}/labels"}, f)
                runs.append(run_name)
            models.append((key, dict((k, l) for k, l, _ in MODELS)[key], runs))
        f1s = run(models, f"{d}/pred", f"{d}/out", n_boot=300)
        ok.append(("proposed beats Moryossef on test", f1s[("proposed", "test")].mean() > f1s[("moryossef", "test")].mean()))
        txt = open(f"{d}/out/frame_f1_summary.txt").read()
        ok.append(("summary has both slides", "slide 9" in txt and "slide 10" in txt))
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
    models = [(k, l, getattr(a, k)) for k, l, _ in MODELS]
    run(models, a.pred_root, a.out_dir, a.n_boot)


if __name__ == "__main__":
    main()