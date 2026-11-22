"""
decoder_ablation.py -- which part of the decoder matters? A systematic decoder comparison on the saved
predictions of one model (default: the proposed model, multistream_bilstm-115..117).

Every decoder reads the same per-frame probabilities P(O), P(I), P(B). Parameters are tuned on VAL, scores
are on TEST.

  decoder                          phrase starts at                              phrase ends at
  argmax                           every frame whose most likely label is B      before the next O or B frame
                                   (or I right after O)                          (src.metrics.extract_segments)
  threshold                        first frame with P(B) > b                     first later frame with P(B) > b
  (Moryossef et al., 2023)         (src.metrics.decode_threshold_2023; b / o     or P(O) > o
                                   tuned by IoU + %-score, the 2023 criterion)
  peak picking                     every local maximum of the Gaussian-smoothed  PAUSE END
                                   P(B), >= d frames apart, NO height threshold
                                   (sigma / d tuned by boundary F1 +-5)
  peak picking + threshold         the same, only peaks with smoothed P(B) >= t  PAUSE END
                                   (sigma / d / t tuned by boundary F1 +-5:
                                   count_oracle_decoding.tune_peaks_on_val, the decoder of the results slides)
  peak picking + threshold,        the same starts as the row above              THRESHOLD END
  threshold end                                                                  (o tuned by IoU + %-score)
  peak picking, threshold end      the same starts as "peak picking"             THRESHOLD END
  (extra: completes the 2 x 2)

  PAUSE END      before the next start, or earlier: before the first run of >= o_min (default 5 frames = 0.1 s)
                 consecutive frames whose most likely label is O (count_oracle_decoding.starts_to_segments,
                 written for the count-oracle analysis; o_min was set by hand, never tuned)
  THRESHOLD END  before the next start, or earlier: before the first later frame with P(O) > o. This is the
                 end rule of the 2023 decoder; its other half ("P(B) > b opens a new phrase") is the next start.

Both end rules give exactly one phrase per start, so with the same starts the end rule changes only IoU and
Segment F1, never boundary F1 or the Percentage of Segments (checked and printed).

Pure peak picking has no threshold, so it gets a wider sigma / d grid (a superset of the thesis grid); the
extra settings can only help it. The thresholded rows use the thesis tuning unchanged, so "peak picking +
threshold" reproduces the numbers of the results slides.

Also reported: the pause-end length o_min (fixed at 5) for the "peak picking + threshold" starts, o_min = 1 /
3 / 5 / 10 / 25 frames: IoU and Segment F1 on val and test.

Statistics: mean +- SD over runs; every decoder minus "peak picking + threshold": paired bootstrap over test
videos (runs averaged within each decoder), boundary F1 from pooled counts, IoU / % / Segment F1@0.5 as
per-video means (2023 style).

Outputs (evaluation_phrase/decoder_ablation/): decoder_ablation_summary.txt, decoder_ablation_runs.csv

  python decoder_ablation.py
  python decoder_ablation.py --runs multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117
  python decoder_ablation.py --selftest
"""
import argparse
import csv
import os
import sys

import numpy as np

from count_oracle_decoding import bnd_f1_counts, peaks_with_heights, starts_to_segments, tune_peaks_on_val
from phrase_stats import PRED_ROOT, f1_from_counts, gold_starts_of, load_predictions, paired_bootstrap
from src.metrics import (B_TAG, O_TAG, decode_threshold_2023, extract_segments, match_boundaries, pct_score,
                         segment_f1_at, segment_iou, segment_percentage)

DEFAULT_RUNS = ["multistream_bilstm-115", "multistream_bilstm-116", "multistream_bilstm-117"]
THR_GRID = [round(x, 1) for x in np.arange(0.1, 0.91, 0.1)]       # b / o of the threshold rules
PURE_SIGMA_GRID = (0, 2, 5, 10)                                     # thesis grid: (0, 2, 5)
PURE_MIN_DIST_GRID = (10, 25, 50, 75, 100, 150)                     # thesis grid: (10, 25, 50)
O_MIN_GRID = (1, 3, 5, 10, 25)
TOLS = (2, 5, 10)

ARGMAX, THRESH = "argmax", "threshold (2023)"
PEAK, PEAK_T = "peak picking", "peak picking + threshold"
PEAK_T_TE, PEAK_TE = "peak picking + threshold, threshold end", "peak picking, threshold end (extra)"
ORDER = [ARGMAX, THRESH, PEAK, PEAK_T, PEAK_T_TE, PEAK_TE]
REF = PEAK_T


# ----------------------------------------------------------------------------- decoders
def threshold_end_segments(starts, probs, o_thr):
    """One phrase per start: ends before the next start, or earlier before the first frame after the start
    with P(O) > o_thr (the end rule of the 2023 decoder)."""
    starts = np.sort(np.asarray(starts, dtype=np.int64))
    T = probs.shape[1]
    if T == 0 or len(starts) == 0:
        return []
    is_o = np.asarray(probs[O_TAG], dtype=np.float64) > o_thr
    idx = np.where(is_o, np.arange(T), T)
    nxt = np.r_[np.minimum.accumulate(idx[::-1])[::-1], T]          # first t' >= t with P(O) > o
    segs = []
    for j, s in enumerate(starts):
        s = int(min(max(s, 0), T - 1))
        lim = int(starts[j + 1]) - 1 if j + 1 < len(starts) else T - 1
        end = min(lim, int(nxt[s + 1]) - 1)
        segs.append((s, max(s, end)))
    return segs


def pure_peaks(p_b, min_dist, sigma):
    idx, _ = peaks_with_heights(p_b, min_dist, sigma)
    return np.sort(idx)


def thr_peaks(p_b, tuned):
    idx, h = peaks_with_heights(p_b, tuned["min_dist"], tuned["sigma"])
    return np.sort(idx[h >= tuned["thr"]])


# ----------------------------------------------------------------------------- scoring
def score(segs_by_vid, gold):
    tp = {k: 0 for k in TOLS}
    npred = ngold = 0
    ious, pcts, sf1s, per_vid = [], [], [], {}
    for v, segs in segs_by_vid.items():
        g = gold[v]
        gsegs = extract_segments(g)
        gs = np.array([s for s, _ in gsegs], dtype=np.int64)
        st = np.array([s for s, _ in segs], dtype=np.int64)
        bnd5 = None
        for k in TOLS:
            ok, _, _ = match_boundaries(gs, st, k)
            tp[k] += int(ok.sum())
            if k == 5:
                bnd5 = (int(ok.sum()), len(st), len(gs))
        npred += len(st)
        ngold += len(gs)
        iou = segment_iou(segs, gsegs, len(g))
        pc = segment_percentage(segs, gsegs)
        sf = segment_f1_at(segs, gsegs, 0.5)
        ious.append(iou)
        if not np.isnan(pc):
            pcts.append(pc)
        sf1s.append(sf)
        per_vid[v] = {"bnd5": bnd5, "seg_f1_05": sf, "iou": iou, "pct": pc}
    out = {f"bnd_f1_{k}": f1_from_counts(tp[k], npred, ngold) for k in TOLS}
    out.update({"bnd_p_5": tp[5] / npred if npred else 0.0, "bnd_r_5": tp[5] / ngold if ngold else 0.0,
                "iou": float(np.mean(ious)), "pct": float(np.mean(pcts)) if pcts else float("nan"),
                "seg_f1_05": float(np.mean(sf1s)), "n_pred": npred, "n_gold": ngold})
    out["iou_pct_score"] = out["iou"] + pct_score(out["pct"])
    return out, per_vid


# ----------------------------------------------------------------------------- tuning (val)
def tune_threshold(val):
    """The 2023 decoder: (b, o) by IoU + %-score on val."""
    probs, _, gold = val
    best = None
    for b in THR_GRID:
        for o in THR_GRID:
            m, _ = score({v: decode_threshold_2023(probs[v], b, o) for v in probs}, gold)
            if best is None or m["iou_pct_score"] > best[0]:
                best = (m["iou_pct_score"], b, o)
    return {"b": best[1], "o": best[2]}


def tune_pure_peaks(val):
    """Pure peak picking: (d, sigma) by boundary F1 +-5 on val."""
    probs, _, gold = val
    gst = {v: gold_starts_of(gold[v]) for v in probs}
    best = None
    for md in PURE_MIN_DIST_GRID:
        for sg in PURE_SIGMA_GRID:
            f1 = bnd_f1_counts({v: pure_peaks(probs[v][B_TAG], md, sg) for v in probs}, gst, 5)
            if best is None or f1 > best[0]:
                best = (f1, md, sg)
    return {"min_dist": best[1], "sigma": best[2]}


def tune_end_threshold(val, starts):
    """Threshold end: o by IoU + %-score on val, starts fixed (so only the ends change)."""
    probs, _, gold = val
    best = None
    for o in THR_GRID:
        m, _ = score({v: threshold_end_segments(starts[v], probs[v], o) for v in probs}, gold)
        if best is None or m["iou_pct_score"] > best[0]:
            best = (m["iou_pct_score"], o)
    return best[1]


# ----------------------------------------------------------------------------- one run
def decode_run(run, split, pred_root, o_min):
    val = load_predictions(pred_root, run, "val")
    probs, _, gold = load_predictions(pred_root, run, split)
    vprobs = val[0]
    th = tune_threshold(val)
    pure = tune_pure_peaks(val)
    tpk = tune_peaks_on_val(val)
    v_pure = {v: pure_peaks(vprobs[v][B_TAG], pure["min_dist"], pure["sigma"]) for v in vprobs}
    v_thr = {v: thr_peaks(vprobs[v][B_TAG], tpk) for v in vprobs}
    o_t = tune_end_threshold(val, v_thr)
    o_p = tune_end_threshold(val, v_pure)
    settings = {ARGMAX: "-", THRESH: f"b={th['b']} o={th['o']}",
                PEAK: f"d={pure['min_dist']} sigma={pure['sigma']} o_min={o_min}",
                PEAK_T: f"d={tpk['min_dist']} sigma={tpk['sigma']} t={tpk['thr']} o_min={o_min}",
                PEAK_T_TE: f"d={tpk['min_dist']} sigma={tpk['sigma']} t={tpk['thr']} o={o_t}",
                PEAK_TE: f"d={pure['min_dist']} sigma={pure['sigma']} o={o_p}"}
    dec = {k: {} for k in ORDER}
    for v, p in probs.items():
        sp = pure_peaks(p[B_TAG], pure["min_dist"], pure["sigma"])
        st = thr_peaks(p[B_TAG], tpk)
        dec[ARGMAX][v] = extract_segments(p.argmax(axis=0))
        dec[THRESH][v] = decode_threshold_2023(p, th["b"], th["o"])
        dec[PEAK][v] = starts_to_segments(sp, p, o_min)
        dec[PEAK_T][v] = starts_to_segments(st, p, o_min)
        dec[PEAK_T_TE][v] = threshold_end_segments(st, p, o_t)
        dec[PEAK_TE][v] = threshold_end_segments(sp, p, o_p)
    res = {k: score(dec[k], gold) for k in ORDER}
    # pause-end length o_min, "peak picking + threshold" starts, val and test
    omin = {}
    for name, (pr, gd, sts) in {"val": (vprobs, val[2], v_thr),
                                split: (probs, gold, {v: thr_peaks(probs[v][B_TAG], tpk) for v in probs})}.items():
        for om in O_MIN_GRID:
            m, _ = score({v: starts_to_segments(sts[v], pr[v], om) for v in pr}, gd)
            omin[(name, om)] = (m["iou"], m["seg_f1_05"])
    return res, settings, omin


# ----------------------------------------------------------------------------- bootstrap
def _bnd(per_vids):
    return [{v: d["bnd5"] for v, d in pv.items()} for pv in per_vids]


def mean_bootstrap(pv_a, pv_b, key, n_boot=2000, seed=0):
    """Per-video metric averaged over videos, runs averaged within each arm; A - B, videos resampled together."""
    vids = sorted(set.intersection(*[set(p) for p in pv_a + pv_b]))
    A = np.array([[p[v][key] for v in vids] for p in pv_a], dtype=float)
    B = np.array([[p[v][key] for v in vids] for p in pv_b], dtype=float)

    def stat(idx):
        return float(np.nanmean(np.nanmean(A[:, idx], 1)) - np.nanmean(np.nanmean(B[:, idx], 1)))
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.integers(0, len(vids), len(vids))) for _ in range(n_boot)])
    return {"diff": stat(np.arange(len(vids))),
            "ci95": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))}


def fmt_ci(b, nd=3):
    return f"{b['diff']:+.{nd}f} ({b['ci95'][0]:+.{nd}f} to {b['ci95'][1]:+.{nd}f})"


# ----------------------------------------------------------------------------- report
def run(runs, split, pred_root, out_dir, o_min=5, n_boot=2000):
    os.makedirs(out_dir, exist_ok=True)
    per_run, rows, omins = {}, [], {}
    for r in runs:
        print(f"  tuning + decoding {r} (the threshold grid takes a few minutes per run) ...")
        res, settings, omin = decode_run(r, split, pred_root, o_min)
        per_run[r], omins[r] = res, omin
        for k in ORDER:
            rows.append({"run": r, "decoder": k, "settings_tuned_on_val": settings[k],
                         **{m: round(float(v), 4) for m, v in res[k][0].items()}})
    with open(os.path.join(out_dir, "decoder_ablation_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def M(k, m):
        return np.array([per_run[r][k][0][m] for r in runs], dtype=float)

    def ms(k, m, nd=3):
        a = M(k, m)
        sd = a.std(ddof=1) if len(a) > 1 else 0.0
        return f"{a.mean():.{nd}f} +- {sd:.{nd}f}"

    L = [f"Decoder ablation ({split}; every decoder tuned on val); runs: {', '.join(runs)}",
         "Boundary F1 / P / R: phrase starts within +-5 frames (+-0.1 s) or +-2 frames (+-0.04 s), counts pooled "
         "over videos. IoU, Percentage of Segments (%) and Segment F1 (IoU >= 0.5): per-video means. "
         "Mean +- SD over runs.", "",
         f"{'decoder':42s}{'bF1 +-0.1 s':>16s}{'bF1 +-0.04 s':>16s}{'P':>7s}{'R':>7s}{'IoU':>16s}{'%':>7s}"
         f"{'Segment F1':>16s}"]
    for k in ORDER:
        L.append(f"{k:42s}{ms(k, 'bnd_f1_5'):>16s}{ms(k, 'bnd_f1_2'):>16s}{M(k, 'bnd_p_5').mean():7.3f}"
                 f"{M(k, 'bnd_r_5').mean():7.3f}{ms(k, 'iou'):>16s}{M(k, 'pct').mean():7.2f}{ms(k, 'seg_f1_05'):>16s}")
    L += ["", f"Difference to '{REF}' (95% CI: paired bootstrap over test videos, runs averaged):",
          f"{'decoder':42s}{'bF1 +-0.1 s':>26s}{'IoU':>26s}{'Segment F1':>26s}{'%':>26s}"]
    ref_pv = [per_run[r][REF][1] for r in runs]
    for k in ORDER:
        if k == REF:
            continue
        pv = [per_run[r][k][1] for r in runs]
        cells = [fmt_ci(paired_bootstrap(_bnd(pv), _bnd(ref_pv), n_boot=n_boot))]
        cells += [fmt_ci(mean_bootstrap(pv, ref_pv, m, n_boot), 3 if m != "pct" else 2)
                  for m in ("iou", "seg_f1_05", "pct")]
        L.append(f"{k:42s}" + "".join(f"{c:>26s}" for c in cells))

    # the two contrasts of the 2 x 2, in words
    L += ["", "Contrasts:"]
    for a, b, what in ((PEAK_T, PEAK, "start threshold (pause end)"),
                       (PEAK_T_TE, PEAK_TE, "start threshold (threshold end)"),
                       (PEAK_T_TE, PEAK_T, "threshold end vs pause end (thresholded starts)"),
                       (PEAK_TE, PEAK, "threshold end vs pause end (pure starts)")):
        pa, pb = [per_run[r][a][1] for r in runs], [per_run[r][b][1] for r in runs]
        L.append(f"  {what:50s} bF1 {fmt_ci(paired_bootstrap(_bnd(pa), _bnd(pb), n_boot=n_boot))}"
                 f" | IoU {fmt_ci(mean_bootstrap(pa, pb, 'iou', n_boot))}"
                 f" | Segment F1 {fmt_ci(mean_bootstrap(pa, pb, 'seg_f1_05', n_boot))}")

    same = all(per_run[r][a][0][m] == per_run[r][b][0][m] for r in runs
               for a, b in ((PEAK_T, PEAK_T_TE), (PEAK, PEAK_TE)) for m in ("bnd_f1_5", "bnd_f1_2", "pct", "n_pred"))
    L += ["", f"Check: with the same starts, the end rule leaves boundary F1 and % unchanged: {'yes' if same else 'NO'}",
          "", f"Pause-end length o_min ('{PEAK_T}' starts; the decoder uses {o_min}); mean over runs:",
          f"  {'o_min (frames)':16s}" + "".join(f"{om:>16d}" for om in O_MIN_GRID)]
    for sp in ("val", split):
        for j, name in ((0, "IoU"), (1, "Segment F1")):
            vals = [np.mean([omins[r][(sp, om)][j] for r in runs]) for om in O_MIN_GRID]
            L.append(f"  {sp + ' ' + name:16s}" + "".join(f"{x:>16.3f}" for x in vals))
    L += ["", "Settings tuned on val (per run): d = minimum peak distance (frames), sigma = Gaussian smoothing "
          "(frames), t = peak threshold, b / o = P(B) / P(O) thresholds"]
    for r in runs:
        L.append(f"  {r}:")
        for row in rows:
            if row["run"] == r:
                L.append(f"    {row['decoder']:42s} {row['settings_tuned_on_val']}")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "decoder_ablation_summary.txt"), "w") as f:
        f.write(text)
    print(text)
    return per_run


# ----------------------------------------------------------------------------- selftest
def selftest():
    import json
    import tempfile
    ok = []
    p = np.full((3, 60), 0.05)
    p[1] = 0.9
    p[0, 30:40] = 0.8                                     # a pause at 30-39
    segs = threshold_end_segments([5, 20, 45], p, 0.5)
    ok.append(("threshold end: before the pause, before the next start, to the end",
               segs == [(5, 19), (20, 29), (45, 59)]))
    segs = threshold_end_segments([5, 20, 45], p, 0.9)
    ok.append(("threshold end: no P(O) above o -> next start", segs == [(5, 19), (20, 44), (45, 59)]))
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/labels")
        golds = {}
        for i in range(5):
            g = np.zeros(2500, dtype=np.int64)
            t = 40
            while t < 2400:
                L = int(rng.integers(50, 150))
                g[t], g[t + 1:t + L] = 2, 1
                t += L + int(rng.integers(10, 40))
            golds[f"v{i}"] = g
            np.save(f"{d}/labels/v{i}.npy", g)
        os.makedirs(f"{d}/pred/m-1")
        for split in ("val", "test"):
            arrs = {}
            for v, g in golds.items():
                q = np.full((3, len(g)), 0.05, dtype=np.float32)
                q[0, g == 0], q[1, g != 0] = 0.9, 0.9
                q[2] += rng.uniform(0, 0.04, len(g))          # low noise: pure peaks find false starts
                for s in np.where(g == 2)[0]:
                    for j, hgt in zip(range(s, s + 5), (0.4, 0.6, 0.8, 0.6, 0.4)):
                        q[2, j] = hgt
                arrs[v] = (q / q.sum(0, keepdims=True)).astype(np.float16)
            np.savez_compressed(f"{d}/pred/m-1/{split}.npz", **arrs)
            with open(f"{d}/pred/m-1/{split}_meta.json", "w") as f:
                json.dump({"run": "m-1", "labels_dir": f"{d}/labels", "description": "m"}, f)
        res = run(["m-1"], "test", f"{d}/pred", f"{d}/out", n_boot=100)["m-1"]
        ok.append(("thresholded peaks find the bumps (bF1 +-5 > .95)", res[PEAK_T][0]["bnd_f1_5"] > 0.95))
        ok.append(("pure peaks: at least as many starts", res[PEAK][0]["n_pred"] >= res[PEAK_T][0]["n_pred"]))
        ok.append(("end rule leaves boundary F1 unchanged",
                   res[PEAK_T][0]["bnd_f1_5"] == res[PEAK_T_TE][0]["bnd_f1_5"]))
        ok.append(("threshold end: Segment F1 > .9", res[PEAK_T_TE][0]["seg_f1_05"] > 0.9))
        ok.append(("summary written", os.path.exists(f"{d}/out/decoder_ablation_summary.txt")))
    for name, c in ok:
        print(f"  [{'ok' if c else 'FAIL'}] {name}")
    return all(c for _, c in ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--o-min", type=int, default=5, help="pause end: minimum run of argmax-O frames (default 5)")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "decoder_ablation"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run(a.runs, a.split, a.pred_root, a.out_dir, a.o_min, a.n_boot)


if __name__ == "__main__":
    main()