"""
plot_decoding_example.py -- slide 10 figure from REAL predictions: P(B) of one test video over a short stretch,
the annotated phrase starts, and the starts of three decoders (argmax, threshold, peak picking + threshold).

Decoders are tuned on validation exactly as in decoder_ablation.py (threshold: IoU + %-score; peaks: boundary F1).
By default the script picks the 10-s stretch of the test set with the most annotated phrase starts where argmax
also over-segments; pass --video / --start to choose one yourself.

  python plot_decoding_example.py                       # run multistream_bilstm-115, automatic stretch
  python plot_decoding_example.py --video 1209077_B --start 12000 --seconds 10
Output: evaluation_phrase/figures/decoding_example.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from count_oracle_decoding import tune_peaks_on_val
from decoder_ablation import thr_peaks, tune_threshold
from phrase_stats import PRED_ROOT, gold_starts_of, load_predictions
from src.metrics import B_TAG, decode_threshold_2023, extract_segments

FPS = 50
INK, GREEN, RED, ORANGE, BLUE, GREY = "#1F2933", "#2E8B57", "#C0392B", "#E07B39", "#2F6DB5", "#9AA3AD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="multistream_bilstm-115")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--video", default=None)
    ap.add_argument("--start", type=int, default=None, help="first frame of the stretch")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join("evaluation_phrase", "figures", "decoding_example.png"))
    a = ap.parse_args()

    val = load_predictions(a.pred_root, a.run, "val")
    th, pk = tune_threshold(val), tune_peaks_on_val(val)
    probs, _, gold = load_predictions(a.pred_root, a.run, "test")
    L = int(a.seconds * FPS)

    dec = {}
    for v, p in probs.items():
        dec[v] = {"gold": np.asarray(gold_starts_of(gold[v])),
                  "argmax": np.array([s for s, _ in extract_segments(p.argmax(axis=0))]),
                  "thr": np.array([s for s, _ in decode_threshold_2023(p, th["b"], th["o"])]),
                  "peak": np.asarray(thr_peaks(p[B_TAG], pk))}

    if a.video is None or a.start is None:          # best stretch: many gold starts, argmax over-segments
        best = None
        for v, d in dec.items():
            T = probs[v].shape[1]
            for s0 in range(0, max(1, T - L), FPS):
                inwin = lambda x: int(((x >= s0) & (x < s0 + L)).sum())
                g, am = inwin(d["gold"]), inwin(d["argmax"])
                score = g + 0.2 * min(am - g, 10)
                if g >= 3 and (best is None or score > best[0]):
                    best = (score, v, s0)
        _, a.video, a.start = best
    v, s0 = a.video, a.start
    p, d = probs[v], dec[v]
    t = np.arange(s0, min(s0 + L, p.shape[1]))
    sec = (t - s0) / FPS

    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_axes([0.18, 0.5, 0.66, 0.46])
    ax.plot(sec, p[B_TAG, t], color=INK, lw=1.6)
    for g in d["gold"][(d["gold"] >= t[0]) & (d["gold"] <= t[-1])]:
        ax.axvline((g - s0) / FPS, color=GREEN, lw=2, alpha=0.6)
    ax.set_ylabel("P(B)"); ax.set_ylim(0, 1.05); ax.set_xlim(0, sec[-1]); ax.set_xticklabels([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{v}, test set, {a.run} (green = annotated phrase starts)", fontsize=10, color="#5F6B78")
    rows = [("Argmax", d["argmax"], RED), ("Threshold", d["thr"], ORANGE), ("Peak picking\n+ threshold", d["peak"], BLUE)]
    for r, (name, st, c) in enumerate(rows):
        a2 = fig.add_axes([0.18, 0.34 - r * 0.13, 0.66, 0.09]); a2.set_xlim(0, sec[-1]); a2.set_ylim(0, 1); a2.axis("off")
        for g in d["gold"][(d["gold"] >= t[0]) & (d["gold"] <= t[-1])]:
            a2.axvline((g - s0) / FPS, color=GREEN, lw=1.5, alpha=0.35)
        sel = st[(st >= t[0]) & (st <= t[-1])]
        for s in sel:
            a2.plot([(s - s0) / FPS] * 2, [0.05, 0.95], color=c, lw=2.5)
        a2.text(-0.02 * sec[-1], 0.5, name, ha="right", va="center", fontsize=12, weight="bold", color=c)
        a2.text(sec[-1] * 1.01, 0.5, f"{len(sel)} starts", va="center", fontsize=11, color="#5F6B78")
    ng = int(((d["gold"] >= t[0]) & (d["gold"] <= t[-1])).sum())
    fig.text(0.18, 0.0, f"{a.seconds:.0f} s of video; {ng} annotated phrase starts. Decoders tuned on validation.",
             fontsize=10, color="#5F6B78")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}  (video {v}, frames {t[0]}-{t[-1]})")


if __name__ == "__main__":
    main()