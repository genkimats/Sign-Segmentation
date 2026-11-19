"""
plot_hand_trace.py -- slide 13 figure: height of the dominant wrist around phrase starts that the proposed model
found (all 3 seeds) vs missed (all 3 seeds). Shows the "hand reset": at found starts the hand drops and rises
again; at missed starts it stays up.

Reads the per_start.csv written by characterize_missed_starts.py (columns video, frame, group, pause_s) and the
cue features (processed_data/cue_features/<video>.npy, channel "dom wrist height" from _names.json).

  python plot_hand_trace.py
  python plot_hand_trace.py --per-start evaluation_phrase/missed_starts/per_start.csv
Output: evaluation_phrase/figures/hand_trace.png
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 50
GREEN, RED, INK = "#2E8B57", "#C0392B", "#1F2933"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-start", default=os.path.join("evaluation_phrase", "missed_starts", "per_start.csv"))
    ap.add_argument("--cue-dir", default=os.path.join("processed_data", "cue_features"))
    ap.add_argument("--channel", default="dom wrist height")
    ap.add_argument("--window", type=float, default=1.0, help="seconds before and after the start")
    ap.add_argument("--max-pause", type=float, default=0.5, help="right panel: only starts after a pause <= this (s)")
    ap.add_argument("--out", default=os.path.join("evaluation_phrase", "figures", "hand_trace.png"))
    a = ap.parse_args()

    names = json.load(open(os.path.join(a.cue_dir, "_names.json")))
    ch = names.index(a.channel)
    W = int(a.window * FPS)
    rows = [r for r in csv.DictReader(open(a.per_start)) if r["group"] in ("found", "missed")]
    cache, traces = {}, {("all", g): [] for g in ("found", "missed")}
    traces.update({("short", g): [] for g in ("found", "missed")})
    for r in rows:
        v, s = r["video"], int(r["frame"])
        if v not in cache:
            fp = os.path.join(a.cue_dir, f"{v}.npy")
            cache[v] = np.load(fp)[:, ch].astype(float) if os.path.exists(fp) else None
        x = cache[v]
        if x is None or s - W < 0 or s + W >= len(x):
            continue
        seg = x[s - W:s + W + 1]
        traces[("all", r["group"])].append(seg)
        try:
            pause = float(r.get("pause_s", "nan"))
        except ValueError:
            pause = np.nan
        if np.isfinite(pause) and pause <= a.max_pause:
            traces[("short", r["group"])].append(seg)

    t = np.arange(-W, W + 1) / FPS
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, (("all", "All phrase starts"),
                                        ("short", f"Only after a short pause (≤ {a.max_pause:g} s)"))):
        ns = []
        for g, c, lab in (("found", GREEN, "found by all 3 seeds"), ("missed", RED, "missed by all 3 seeds")):
            arr = np.array(traces[(key, g)])
            if len(arr) == 0:
                continue
            med = np.nanmedian(arr, 0)
            q1, q3 = np.nanpercentile(arr, 25, 0), np.nanpercentile(arr, 75, 0)
            ax.plot(t, med, color=c, lw=2.4, label=lab)
            ns.append(f"{g} n = {len(arr)}")
            ax.fill_between(t, q1, q3, color=c, alpha=0.12)
        ax.axvline(0, color=INK, lw=1, ls="--")
        ax.text(0.02, 0.03, ", ".join(ns), transform=ax.transAxes, fontsize=9, color="#5F6B78")
        ax.set_title(title, fontsize=12); ax.set_xlabel("time from the phrase start (s)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        pass
    axes[0].set_ylabel("dominant wrist height\n(higher = hand raised)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.1, "Median and interquartile range; wrist height relative to the shoulders, per-video robust z-score",
             ha="center", fontsize=10, color="#5F6B78")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()