"""
Learning-curve figure (analysis F) from evaluate_phrase.py's metrics CSV.
  python make_learning_curve.py --csv evaluation_phrase/multistream_bilstm_test_metrics.csv --prefixes 364-375
Runs are grouped by the "LC: ..., X% of training videos" description; mean +- SD over seeds.
Decoders: every decoder present in the CSV for these runs (argmax, thr ..., tuned(val) ...).
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_TRAIN = 500
LABELS = {"argmax": "Argmax", "thr": "Threshold (0.5 / 0.5)", "tuned(val)": "Threshold (tuned on val)"}
COLORS = {"argmax": "#C0504D", "thr": "#4F81BD", "tuned(val)": "#4F81BD"}


def expand(ps):
    out = []
    for p in ps:
        a, _, b = str(p).partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--out", default="fig_F_learning_curve.png")
    ap.add_argument("--metrics", nargs="+", default=["frame_f1", "bnd_f1_5"])
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d = d[d.prefix.astype(int).isin(expand(a.prefixes))].copy()
    d["frac"] = d.description.str.extract(r"([\d.]+)% of training videos", expand=False).astype(float)
    d["videos"] = np.ceil(d.frac / 100 * N_TRAIN).astype(int)
    d["dec"] = d.decoder.str.replace(r"\s*b=.*", "", regex=True)
    g = d.groupby(["dec", "videos"])[a.metrics].agg(["mean", "std", "count"])
    print(g.round(4).to_string())
    names = {"frame_f1": "Frame F1", "bnd_f1_5": "Boundary F1 (±0.1 s)", "iou": "IoU"}
    fig, axes = plt.subplots(1, len(a.metrics), figsize=(4.6 * len(a.metrics), 3.4))
    for ax, m in zip(np.atleast_1d(axes), a.metrics):
        for dec, sub in g[m].groupby(level=0):
            if m == "frame_f1" and dec != "argmax":
                continue                                   # Frame F1 is always the raw argmax labels
            x = sub.index.get_level_values(1)
            ax.errorbar(x, sub["mean"], yerr=sub["std"], marker="o", capsize=3,
                        label=LABELS.get(dec, dec), color=COLORS.get(dec))
        ax.set_xscale("log", base=2)
        xs = sorted(d.videos.unique())
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.set_xlabel("training videos (log scale)")
        ax.set_ylabel(names.get(m, m))
        if d.dec.nunique() > 1 and m != "frame_f1":
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()