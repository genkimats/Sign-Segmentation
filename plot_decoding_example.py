"""
plot_decoding_example.py -- slide 10 figure from REAL predictions: P(B) of one test video over a short stretch,
the annotated phrase starts, and the starts of three decoders (argmax, threshold, peak picking + threshold).

Decoders are tuned on validation exactly as in decoder_ablation.py (threshold: IoU + %-score; peaks: boundary F1).
By default the script picks a random test video and, in it, a random 10-s stretch (new each run; the seed is
printed) that meets these rules:
  - at least 3 annotated phrase starts, and argmax predicts more starts than that (over-segmentation)
  - both threshold and peak picking + threshold have a start within --match-sec (0.5 s) of every annotated start
  - peak picking + threshold is closer to the annotated start than threshold at every annotated start but exactly
    one, where threshold is closer (no ties unless --allow-ties)
  - exactly one prediction of peak picking + threshold (--far-rows) is more than --far-sec (1 s) from any
    annotated start
Each threshold / peak-picking start is labelled with its distance to the nearest annotated start (ms).
This is a selected example: say so on the slide. Pass --video / --start, or --seed, to fix it.

  python plot_decoding_example.py                       # run multistream_bilstm-115, random video and stretch
  python plot_decoding_example.py --video 1209077_B --start 12000 --seconds 10
Output: evaluation_phrase/figures/decoding_example.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from count_oracle_decoding import peaks_with_heights, tune_peaks_on_val
from phrase_stats import PRED_ROOT, gold_starts_of, load_predictions
from src.metrics import (B_TAG, decode_threshold_2023, extract_segments, pct_score, segment_iou,
                         segment_percentage)

THR_GRID = [round(x, 1) for x in np.arange(0.1, 0.91, 0.1)]


def thr_peaks(p_b, tuned):
    """Peak picking + threshold: smoothed P(B) peaks >= tuned threshold, >= d frames apart."""
    idx, h = peaks_with_heights(p_b, tuned["min_dist"], tuned["sigma"])
    return np.sort(idx[h >= tuned["thr"]])


def tune_threshold(val):
    """Moryossef et al. (2023) decoder: B / O thresholds by IoU + %-score on val (the 2023 criterion)."""
    probs, _, gold = val
    best = None
    for b in THR_GRID:
        for o in THR_GRID:
            ious, pcts = [], []
            for v, p in probs.items():
                segs, gsegs = decode_threshold_2023(p, b, o), extract_segments(gold[v])
                ious.append(segment_iou(segs, gsegs, len(gold[v])))
                pc = segment_percentage(segs, gsegs)
                if not np.isnan(pc):
                    pcts.append(pc)
            crit = np.mean(ious) + pct_score(np.mean(pcts) if pcts else float("nan"))
            if best is None or crit > best[0]:
                best = (crit, b, o)
    return {"b": best[1], "o": best[2]}

FPS = 50
INK, GREEN, RED, ORANGE, BLUE, GREY = "#1F2933", "#2E8B57", "#C0392B", "#E07B39", "#2F6DB5", "#9AA3AD"


def nearest_err(starts, g):
    """Distance in frames from annotated start g to the nearest predicted start (inf if none)."""
    return float(np.min(np.abs(starts - g))) if len(starts) else np.inf


def stretch_ok(d, s0, L, match, far, far_rows, allow_ties):
    """The selection rules for the slide example. Returns (ok, details)."""
    win = lambda x: x[(x >= s0) & (x < s0 + L)]
    gold, thr, pk, am = win(d["gold"]), win(d["thr"]), win(d["peak"]), win(d["argmax"])
    if len(gold) < 3 or len(am) <= len(gold):            # >= 3 annotated starts; argmax over-segments
        return False, None
    e_thr = np.array([nearest_err(thr, g) for g in gold])
    e_pk = np.array([nearest_err(pk, g) for g in gold])
    if np.any(e_thr > match) or np.any(e_pk > match):   # both decoders find every annotated start
        return False, None
    thr_better = int(np.sum(e_thr < e_pk))
    pk_better = int(np.sum(e_pk < e_thr))
    ties = len(gold) - thr_better - pk_better
    if thr_better != 1 or (ties and not allow_ties) or pk_better < 1:
        return False, None
    n_far = 0                                           # predictions nowhere near an annotated start
    for row in far_rows:
        x = {"thr": thr, "peak": pk}[row]
        n_far += int(sum(nearest_err(d["gold"], p) > far for p in x))
    if n_far != 1:
        return False, None
    return True, (gold, e_thr, e_pk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="multistream_bilstm-115")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--video", default=None, help="default: a random test video")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: new each run, printed)")
    ap.add_argument("--match-sec", type=float, default=0.5,
                    help="every annotated start must have a start of both decoders within this (s)")
    ap.add_argument("--far-sec", type=float, default=1.0,
                    help="a prediction this far (s) from every annotated start counts as 'not even close'")
    ap.add_argument("--far-rows", default="peak", choices=["peak", "thr", "both"],
                    help="which decoder row(s) must contain exactly one far prediction (default: peak)")
    ap.add_argument("--allow-ties", action="store_true",
                    help="also accept stretches where the two decoders are equally close at some starts")
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

    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 100000)
    rng = np.random.default_rng(seed)
    far_rows = ["thr", "peak"] if a.far_rows == "both" else [a.far_rows]
    match, far = a.match_sec * FPS, a.far_sec * FPS
    if a.start is None:
        # random stretch where: >= 3 annotated starts, argmax over-segments, peak picking + threshold is closer to
        # the annotated start than threshold at every start but exactly one, and exactly one prediction is far
        # (> --far-sec) from any annotated start
        good = []
        for v, d in dec.items():
            if a.video is not None and v != a.video:
                continue
            T = probs[v].shape[1]
            for s0 in range(0, max(1, T - L), FPS // 2):
                if stretch_ok(d, s0, L, match, far, far_rows, a.allow_ties)[0]:
                    good.append((v, s0))
        if not good:
            raise SystemExit("no stretch meets the rules -- try --allow-ties, a larger --match-sec, "
                             "--far-rows both, or --seconds 15")
        vids = sorted({v for v, _ in good})
        print(f"  {len(good)} qualifying stretches in {len(vids)} test videos")
        pick_v = a.video or str(rng.choice(vids))     # first a random video, then a random stretch in it
        cands = [s0 for v, s0 in good if v == pick_v]
        a.video, a.start = pick_v, int(rng.choice(cands))
    elif a.video is None:
        raise SystemExit("--start needs --video")
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
            if name != "Argmax":                       # how far each start is from the annotated one
                gd = d["gold"]
                k = int(np.argmin(np.abs(gd - s))) if len(gd) else None
                err = abs(int(gd[k]) - int(s)) if k is not None else 10 ** 9
                lab = "no phrase start nearby" if err > far else f"{err * 1000 / FPS:.0f} ms"
                a2.text((s - s0) / FPS, -0.12, lab, ha="center", va="top", fontsize=8.5,
                        color=RED if err > far else "#5F6B78")
        a2.text(-0.02 * sec[-1], 0.5, name, ha="right", va="center", fontsize=12, weight="bold", color=c)
        a2.text(sec[-1] * 1.01, 0.5, f"{len(sel)} starts", va="center", fontsize=11, color="#5F6B78")
    ng = int(((d["gold"] >= t[0]) & (d["gold"] <= t[-1])).sum())
    fig.text(0.18, -0.03, f"{a.seconds:.0f} s of video; {ng} annotated phrase starts; numbers = distance to the "
             f"annotated start. Selected example; decoders tuned on validation.", fontsize=10, color="#5F6B78")
    ok, det = stretch_ok(d, s0, L, match, far, far_rows, True)
    if det is not None:
        for g, et, ep in zip(*det):
            print(f"    annotated start {g}: threshold off by {et * 1000 / FPS:.0f} ms, "
                  f"peak picking + threshold off by {ep * 1000 / FPS:.0f} ms")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}  (video {v}, frames {t[0]}-{t[-1]})")
    print(f"  to redraw exactly this: --video {v} --start {s0}   (or --seed {seed})")


if __name__ == "__main__":
    main()