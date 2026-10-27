"""
metric_analysis.py -- which segmentation metric should be used to compare phrase models?

Reads the *_metrics.csv files written by evaluate_phrase.py (any number of models, val and
test) and asks, over every configuration you trained (seeds averaged):

  1. Agreement      Spearman / Kendall rank correlation between every pair of metrics.
                    Low agreement = papers using different metrics can rank models oppositely.
  2. Reliability    How well a metric separates configurations relative to seed noise
                    (ICC(1): share of the metric's variance that is between-configuration).
                    A metric with low ICC cannot tell your models apart.
  3. Val -> test    Spearman between val and test configuration means per metric = how much
                    model selection on val by this metric carries over to test.
  4. Range          Spread of each metric over the competitive models vs its seed SD.
  5. Blind spots    Configurations a metric scores well although another metric says they
                    fail (e.g. high IoU while predicting many times too many segments).

Configurations = runs with the same description minus the "[seed=...]" suffix.
"Competitive" = Frame F1 within --competitive of the best configuration (default 0.03), so
degenerate models (face only, ...) don't inflate the correlations.

Usage:
  python metric_analysis.py --csv evaluation_phrase/*_metrics.csv
  python metric_analysis.py --csv evaluation_phrase/*_metrics.csv --decoder argmax --out-dir evaluation_phrase/metric_analysis
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

METRICS = [("frame_f1", "Frame F1 (macro)"), ("f1_B", "B-class F1"), ("iou", "IoU"),
           ("pct_dev", "|log %| (segment-count error, lower=better)"), ("segment_f1_05", "Segment F1@0.5"),
           ("bnd_f1_1", "Boundary F1 +-1"), ("bnd_f1_2", "Boundary F1 +-2"), ("bnd_f1_5", "Boundary F1 +-5"),
           ("bnd_f1_10", "Boundary F1 +-10"), ("bnd_f1_25", "Boundary F1 +-25")]
LOWER_IS_BETTER = {"pct_dev"}


def split_of(path, row):
    m = re.search(r"_(val|test)_metrics\.csv$", os.path.basename(path))
    return m.group(1) if m else "unknown"


def config_name(desc):
    return re.sub(r"\s*\[seed=\d+\]\s*$", "", desc or "").strip()


def decoder_kind(label):
    if label == "argmax":
        return "argmax"
    if label.startswith("thr"):
        return "fixed"
    if label.startswith("tuned"):
        return "tuned"
    if label.startswith("swept"):
        return "swept"
    return label


def load(paths, decoder, include_unseeded=False):
    """-> {split: {config: {metric: [values over seeds]}}}, deduplicated by (run, split).
    A configuration = same model + description AND consecutive prefixes with distinct seeds
    (descriptions such as "MS: Body" were reused by different studies; without the prefix
    blocks their runs would be pooled as if they were seeds of one configuration)."""
    raw = defaultdict(list)
    seen = set()
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                if decoder_kind(r["decoder"]) != decoder:
                    continue
                split = split_of(path, r)
                key = (r["run"], split)
                if key in seen:
                    continue
                seen.add(key)
                if not include_unseeded and str(r.get("seed", "")).strip() == "":
                    continue
                model = r["run"].rsplit("-", 1)[0]
                cfg = f"{model} | {config_name(r.get('description', '')) or r['run']}"
                raw[(split, cfg)].append(r)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (split, cfg), rows in raw.items():
        rows.sort(key=lambda r: int(r.get("prefix") or r["run"].rsplit("-", 1)[-1]))
        block, last, seeds = 0, None, set()
        for r in rows:
            pref = int(r.get("prefix") or r["run"].rsplit("-", 1)[-1])
            seed = str(r.get("seed", ""))
            if last is not None and (pref - last > 1 or seed in seeds):
                block, seeds = block + 1, set()
            seeds.add(seed)
            last = pref
            name = f"{cfg} #{block}" if block else cfg
            for m, _ in METRICS:
                if m == "pct_dev":
                    v = float(r["segment_pct"])
                    val = abs(np.log(v)) if v > 0 else np.nan
                elif m in r and r[m] != "":
                    val = float(r[m])
                else:
                    continue
                data[split][name][m].append(val)
    return data


def rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1])


def kendall(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return np.nan
    s = 0
    for i in range(n):
        s += np.sum(np.sign(a[i] - a[i + 1:]) * np.sign(b[i] - b[i + 1:]))
    return float(s / (n * (n - 1) / 2))


def icc1(groups):
    """One-way random-effects ICC(1) over configurations (groups of seed values)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2 and np.all(np.isfinite(g))]
    if len(groups) < 3:
        return np.nan
    k = np.mean([len(g) for g in groups])
    grand = np.mean(np.concatenate(groups))
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / sum(len(g) - 1 for g in groups)
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description="Compare phrase-segmentation metrics over all trained runs.")
    ap.add_argument("--csv", nargs="+", required=True, help="evaluate_phrase.py *_metrics.csv files (globs ok)")
    ap.add_argument("--decoder", default="argmax", choices=["argmax", "fixed", "tuned", "swept"])
    ap.add_argument("--competitive", type=float, default=0.03,
                    help="keep configurations within this Frame F1 of the best (default 0.03)")
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--include-unseeded", action="store_true",
                    help="also use runs without a recorded seed (older experiments; off by default)")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "metric_analysis"))
    args = ap.parse_args()

    paths = sorted({p for pat in args.csv for p in glob.glob(pat)})
    data = load(paths, args.decoder, args.include_unseeded)
    os.makedirs(args.out_dir, exist_ok=True)
    report = {"files": paths, "decoder": args.decoder}
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    for split in ("test", "val"):
        if split not in data:
            continue
        cfgs = {c: d for c, d in data[split].items() if len(d["frame_f1"]) >= args.min_seeds}
        best = max(np.mean(d["frame_f1"]) for d in cfgs.values())
        comp = {c: d for c, d in cfgs.items() if np.mean(d["frame_f1"]) >= best - args.competitive}
        out(f"\n=== {split}: {len(cfgs)} configurations ({len(comp)} competitive, Frame F1 >= {best - args.competitive:.3f}), "
            f"decoder={args.decoder}")
        names = sorted(comp)
        means = {m: np.array([np.mean(comp[c][m]) if comp[c].get(m) else np.nan for c in names]) for m, _ in METRICS}

        out("\nRank agreement between metrics (Spearman, competitive configurations):")
        ms = [m for m, _ in METRICS]
        header = "".join(f"{m[:9]:>10s}" for m in ms)
        out(f"{'':>16s}{header}")
        corr = {}
        for a in ms:
            row = []
            for b in ms:
                sa = -means[a] if a in LOWER_IS_BETTER else means[a]
                sb = -means[b] if b in LOWER_IS_BETTER else means[b]
                v = spearman(sa, sb)
                corr[(a, b)] = v
                row.append(f"{v:10.2f}")
            out(f"{a[:16]:>16s}" + "".join(row))
        report[f"{split}_spearman"] = {f"{a}|{b}": v for (a, b), v in corr.items()}
        report[f"{split}_kendall_frame_f1"] = {
            m: kendall(means["frame_f1"], -means[m] if m in LOWER_IS_BETTER else means[m]) for m in ms}

        out("\nReliability and range per metric (competitive configurations):")
        out(f"{'metric':<44s}{'ICC(1)':>8s}{'range':>9s}{'seed SD':>9s}{'range/SD':>10s}")
        rel = {}
        for m, label in METRICS:
            groups = [comp[c][m] for c in names if comp[c].get(m)]
            sd = np.sqrt(np.mean([np.var(g, ddof=1) for g in groups if len(g) > 1]))
            rng = np.nanmax(means[m]) - np.nanmin(means[m])
            rel[m] = {"icc1": icc1(groups), "range": float(rng), "seed_sd": float(sd),
                      "range_over_sd": float(rng / sd) if sd > 0 else np.nan}
            out(f"{label:<44s}{rel[m]['icc1']:8.2f}{rng:9.4f}{sd:9.4f}{rel[m]['range_over_sd']:10.1f}")
        report[f"{split}_reliability"] = rel

        out("\nTop 5 configurations by each metric:")
        for m, label in [("frame_f1", "Frame F1"), ("iou", "IoU"), ("bnd_f1_1", "Boundary F1 +-1"),
                         ("bnd_f1_5", "Boundary F1 +-5"), ("segment_f1_05", "Segment F1@0.5")]:
            order = np.argsort(-means[m])[:5]
            out(f"  {label}: " + " | ".join(f"{names[i][:48]} ({means[m][i]:.3f})" for i in order))

    if "val" in data and "test" in data:
        out("\nVal -> test agreement of configuration rankings (Spearman over shared competitive configurations):")
        shared = sorted(set(data["val"]) & set(data["test"]))
        shared = [c for c in shared if len(data["test"][c]["frame_f1"]) >= args.min_seeds]
        if shared:
            best = max(np.mean(data["test"][c]["frame_f1"]) for c in shared)
            shared = [c for c in shared if np.mean(data["test"][c]["frame_f1"]) >= best - args.competitive]
        vt = {}
        for m, label in METRICS:
            v = [np.mean(data["val"][c][m]) for c in shared]
            t = [np.mean(data["test"][c][m]) for c in shared]
            vt[m] = spearman(v, t)
            out(f"  {label:<44s} {vt[m]:.2f}   (n={len(shared)})")
        report["val_test_spearman"] = vt

    out("\nBlind spots (all configurations, test): segment-count error vs IoU / Frame F1")
    if "test" in data:
        rows = []
        for c, d in data["test"].items():
            if d.get("pct_dev") and d.get("iou"):
                rows.append((c, np.mean(d["iou"]), np.exp(np.mean(d["pct_dev"])), np.mean(d["frame_f1"]),
                             np.mean(d.get("bnd_f1_5", [np.nan]))))
        rows.sort(key=lambda r: -r[2])
        out(f"  {'configuration':<60s}{'IoU':>7s}{'x gold':>8s}{'F1':>7s}{'BndF1@5':>9s}")
        for c, iou, ratio, f1, b5 in rows[:8]:
            out(f"  {c[:60]:<60s}{iou:7.3f}{ratio:8.2f}{f1:7.3f}{b5:9.3f}")
        out("  ('x gold' = predicted segments / gold segments, as a factor either way)")

    with open(os.path.join(args.out_dir, f"metric_analysis_{args.decoder}.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    with open(os.path.join(args.out_dir, f"metric_analysis_{args.decoder}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to {args.out_dir}/metric_analysis_{args.decoder}.(txt|json)")


if __name__ == "__main__":
    main()