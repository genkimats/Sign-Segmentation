"""
summarize_fact.py -- the go/no-go table for the FACT pilot.

    python fact_phrase/summarize_fact.py                 # test split
    python fact_phrase/summarize_fact.py --split val

1. Mean +- SD over seeds per configuration (fact_phrase/results/<split>_metrics.csv), next to R0
   (multistream_bilstm-310..312) read from evaluation_phrase/predictions.
2. Paired bootstraps over videos on boundary F1@5 (phrase_stats.paired_bootstrap_runs):
     P1 FACT-token  vs P0 frozen BiLSTM    <- THE go/no-go comparison
     P1 FACT-head   vs P0 frozen BiLSTM
     P1 FACT-token  vs P2 frozen BiLSTM, same 1500-frame window
     P0 frozen BiLSTM vs R0                <- sanity: retraining R0's encoder on its own features
     (stage 2, if present) P3 vs P4
Go = FACT-token or FACT-head beats P0 with a 95% CI above 0. Runs are matched to arms by the
"P0:"/"P1:"/... prefix of their description (queue_fact.py).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from phrase_stats import format_bootstrap, paired_bootstrap_runs  # noqa: E402
from src.metrics import evaluate_videos  # noqa: E402

RESULTS = os.path.join("fact_phrase", "results")
R0_RUNS = ["multistream_bilstm-310", "multistream_bilstm-311", "multistream_bilstm-312"]
METRICS = ["frame_f1", "iou", "segment_pct", "bnd_f1_2", "bnd_f1_5", "bnd_f1_10"]


def r0_row(split):
    from phrase_stats import load_predictions, PRED_ROOT
    vals = defaultdict(list)
    for r in R0_RUNS:
        try:
            probs, _, gold = load_predictions(PRED_ROOT, r, split)
        except FileNotFoundError:
            return None
        m = evaluate_videos(probs, gold)
        for k, key in (("frame_f1", "Frame_F1"), ("iou", "IoU"), ("segment_pct", "Pct"), ("bnd_f1_2", "BndF1_2"),
                       ("bnd_f1_5", "BndF1_5"), ("bnd_f1_10", "BndF1_10")):
            vals[k].append(m[key])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    path = os.path.join(RESULTS, f"{args.split}_metrics.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- train / evaluate some runs first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)                                  # (arm, decoder) -> rows
    for r in rows:
        arm = r["description"].split(":", 1)[0].strip()
        groups[(arm, r["decoder"])].append(r)

    print(f"\n{args.split.upper()}  (mean +- SD over seeds)")
    print(f"{'arm / decoder':28s} {'n':>2s} " + " ".join(f"{m:>15s}" for m in METRICS))
    r0 = r0_row(args.split)
    if r0:
        print(f"{'R0 (310-312) argmax':28s} {len(R0_RUNS):2d} " +
              " ".join(f"{np.mean(r0[m]):7.4f}+-{np.std(r0[m]):.4f}" for m in METRICS))
    for (arm, dec), rs in sorted(groups.items()):
        line = f"{arm + ' / ' + dec:28s} {len(rs):2d} "
        for m in METRICS:
            v = np.array([float(r[m]) for r in rs])
            line += f"{v.mean():7.4f}+-{v.std():.4f} "
        print(line)

    def runs(arm, dec):
        return [r["run"] for r in groups.get((arm, dec), [])]

    tests = [(("P1", "token"), ("P0", "head"), "GO/NO-GO  FACT-token - frozen BiLSTM"),
             (("P1", "head"), ("P0", "head"), "FACT-head - frozen BiLSTM"),
             (("P1", "token"), ("P2", "head"), "FACT-token - frozen BiLSTM @1500"),
             (("P2", "head"), ("P0", "head"), "frozen BiLSTM @1500 - @512"),
             (("P3", "token"), ("P4", "head"), "CE features: FACT-token - frozen BiLSTM"),
             (("P3", "head"), ("P4", "head"), "CE features: FACT-head - frozen BiLSTM")]
    print(f"\nPaired bootstrap over {args.split} videos, boundary F1@5 (A - B):")
    for a, b, name in tests:
        ra, rb = runs(*a), runs(*b)
        if ra and rb:
            res = paired_bootstrap_runs(ra, rb, args.split, k=5, n_boot=args.n_boot)
            print(f"  {name:42s} {format_bootstrap(res)}")
    p0 = runs("P0", "head")
    if p0 and r0:
        res = paired_bootstrap_runs(p0, R0_RUNS, args.split, k=5, n_boot=args.n_boot)
        print(f"  {'sanity  frozen BiLSTM - R0 end-to-end':42s} {format_bootstrap(res)}")


if __name__ == "__main__":
    main()
