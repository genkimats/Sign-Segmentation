"""
balance_study/summarize_balance.py -- one table for the modality-imbalance study.

1. Test metrics of every balance arm (IMB-*) and single-stream rerun (SS-*) next to the reference
   (multistream_bilstm-373, seed 42), with the reference's seed spread (373-375: max - min) as the yardstick a
   single-seed difference has to clear. Decoders: argmax and the 2023 threshold decoder (b=0.5, o=0.5).
   Reads evaluation_phrase/multistream_bilstm_<split>_metrics.csv, so evaluate the runs first:
       python evaluate_phrase.py --model multistream_bilstm --prefixes <runs> --split val --save-predictions
       python evaluate_phrase.py --model multistream_bilstm --prefixes <runs> --split test --save-predictions
2. Paired video bootstrap of boundary F1@5 (argmax starts) vs the reference, when predictions are saved.
3. Learning speed: per-stream head losses / conditional speeds per epoch from balance_log.csv (the monitor
   arm shows how fast each stream is learnt INSIDE the joint model), and the training-loss curves of the
   single-stream reruns vs B+H (428) and B+H+Fc (373) from training_metrics.csv. --plot saves both figures.

Usage (repo root):
    python balance_study/summarize_balance.py --split test
    python balance_study/summarize_balance.py --split test --runs multistream_bilstm-431 multistream_bilstm-432 --plot
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXP_DIR = "experiments_phrase"
EVAL_DIR = "evaluation_phrase"
PRED_ROOT = os.path.join(EVAL_DIR, "predictions")
OUT_DIR = os.path.join(EVAL_DIR, "balance")
DECODERS = ("argmax", "thr b=0.5 o=0.5")
METRICS = (("frame_f1", "Frame F1"), ("f1_B", "B F1"), ("iou", "IoU"), ("segment_pct", "%"), ("bnd_f1_5", "BndF1@5"))


def config_of(run):
    p = os.path.join(EXP_DIR, run, "hyperparameters.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def study_runs():
    """multistream_bilstm runs whose config has 'balance' or an IMB / SS description."""
    out = []
    for d in sorted(os.listdir(EXP_DIR)):
        if not d.startswith("multistream_bilstm-"):
            continue
        cfg = config_of(d)
        if cfg and (cfg.get("balance") or str(cfg.get("description", "")).startswith(("IMB", "SS-"))):
            out.append(d)
    return sorted(out, key=lambda r: int(r.split("-")[-1]))


def read_metrics(split):
    path = os.path.join(EVAL_DIR, f"multistream_bilstm_{split}_metrics.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run evaluate_phrase.py on the runs first.")
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r["run"], r["decoder"])] = r
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def table(runs, ref, ref_seeds, split, rows, do_boot):
    for dec in DECODERS:
        refrow = rows.get((ref, dec))
        if refrow is None:
            print(f"\n[{dec}] reference {ref} not evaluated on {split}; skipped.")
            continue
        spread = {}
        for k, _ in METRICS:
            vals = [fnum(rows[(r, dec)][k]) for r in [ref] + ref_seeds if (r, dec) in rows]
            spread[k] = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
        print(f"\n=== {split}, decoder '{dec}'  (reference {ref}; seed spread of {[ref] + ref_seeds}: "
              + ", ".join(f"{n} {spread[k]:.3f}" for k, n in METRICS) + ")")
        head = f"{'run':24s} " + " ".join(f"{n:>8s}" for _, n in METRICS) + f" {'dFrameF1':>9s} {'dBndF1@5':>9s}  description"
        print(head)
        print(f"{ref:24s} " + " ".join(f"{fnum(refrow[k]):8.4f}" for k, _ in METRICS) + f" {'':>9s} {'':>9s}  reference")
        for r in runs:
            row = rows.get((r, dec))
            if row is None:
                print(f"{r:24s} (not evaluated)")
                continue
            d1 = fnum(row["frame_f1"]) - fnum(refrow["frame_f1"])
            d5 = fnum(row["bnd_f1_5"]) - fnum(refrow["bnd_f1_5"])
            flag = lambda d, k: "*" if abs(d) > spread[k] else " "        # noqa: E731
            print(f"{r:24s} " + " ".join(f"{fnum(row[k]):8.4f}" for k, _ in METRICS)
                  + f" {d1:+8.4f}{flag(d1, 'frame_f1')} {d5:+8.4f}{flag(d5, 'bnd_f1_5')}  "
                  + str((config_of(r) or {}).get("description", ""))[:70])
        print("  * = |difference| larger than the reference's seed spread (a hint, not a test)")
    if not do_boot:
        return
    try:
        from phrase_stats import paired_bootstrap_runs, format_bootstrap
    except Exception as e:                                                    # pragma: no cover
        print(f"\n(paired bootstrap skipped: {e})")
        return
    if not os.path.exists(os.path.join(PRED_ROOT, ref, f"{split}.npz")):
        print(f"\n(paired bootstrap skipped: no saved predictions for {ref} on {split})")
        return
    print(f"\n=== paired video bootstrap vs {ref}, BndF1@5 of argmax starts ({split})")
    for r in runs:
        if not os.path.exists(os.path.join(PRED_ROOT, r, f"{split}.npz")):
            print(f"  {r:24s} (no saved predictions)")
            continue
        res = paired_bootstrap_runs([r], [ref], split, k=5, n_boot=2000)
        print(f"  {r:24s} {format_bootstrap(res, 'A - ref')}")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def learning_speed(runs, plot, curve_runs):
    print("\n=== inside the joint model: per-stream head loss (weighted CE) and Wu's conditional speed, per epoch")
    for r in runs:
        p = os.path.join(EXP_DIR, r, "balance_log.csv")
        if not os.path.exists(p):
            continue
        log = read_csv(p)
        if not log:
            continue
        streams = [k[len("mu_enc_"):] for k in log[0] if k.startswith("mu_enc_")]
        method = log[0].get("method", "")
        print(f"\n  {r} [{method}]")
        cols = [f"head_loss_{s}" for s in streams if f"head_loss_{s}" in log[0]]
        cols += [f"speed_{s}" for s in streams]
        extra = [k for k in log[0] if k.split("_")[0] in ("q", "B", "kappa", "distill", "rebal", "reg", "conflict")]
        cols += extra
        print("  " + f"{'ep':>3s} " + " ".join(f"{c[:18]:>18s}" for c in cols))
        for row in log:
            print("  " + f"{int(row['epoch']):3d} " + " ".join(f"{fnum(row.get(c)):18.4f}" for c in cols))

    print("\n=== single-stream learning speed: training loss (weighted CE) per epoch")
    curves = {}
    for r in curve_runs:
        p = os.path.join(EXP_DIR, r, "training_metrics.csv")
        if not os.path.exists(p):
            continue
        curves[r] = [fnum(x["train_loss"]) for x in read_csv(p)]
    if curves:
        n = max(len(v) for v in curves.values())
        eps = [e for e in (1, 2, 3, 5, 10, 15, 20) if e <= n]
        print(f"  {'run':24s} " + " ".join(f"{'ep' + str(e):>7s}" for e in eps) + "  description")
        for r, v in curves.items():
            print(f"  {r:24s} " + " ".join(f"{v[e - 1]:7.3f}" if e <= len(v) else f"{'':>7s}" for e in eps)
                  + "  " + str((config_of(r) or {}).get("description", ""))[:60])
    if plot and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(OUT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for r, v in curves.items():
            desc = str((config_of(r) or {}).get("description", r))
            ax.plot(range(1, len(v) + 1), v, lw=1.8, label=f"{r.split('-')[-1]}: {desc[:34]}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss (weighted CE)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, "learning_speed_train_loss.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  plot {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="*", help="default: every IMB-* / SS-* / balance run in experiments_phrase")
    ap.add_argument("--ref", default="multistream_bilstm-373")
    ap.add_argument("--ref-seeds", nargs="*", default=["multistream_bilstm-374", "multistream_bilstm-375"])
    ap.add_argument("--curve-runs", nargs="*", default=None,
                    help="training-loss curves to compare (default: SS-* reruns + 428 (B+H) + 373 (B+H+Fc))")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    runs = args.runs or study_runs()
    if not runs:
        raise SystemExit("no balance-study runs found")
    rows = read_metrics(args.split)
    table(runs, args.ref, args.ref_seeds, args.split, rows, not args.no_bootstrap)
    curve_runs = args.curve_runs
    if curve_runs is None:
        curve_runs = [r for r in runs if str((config_of(r) or {}).get("description", "")).startswith("SS-")]
        curve_runs += ["multistream_bilstm-428", args.ref]
    learning_speed(runs, args.plot, curve_runs)


if __name__ == "__main__":
    main()
