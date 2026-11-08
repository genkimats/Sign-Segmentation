"""
Result statistics for the thesis (analyses B-E), from saved predictions:
  evaluation_phrase/predictions/<run>/<split>.npz (+ val.npz for the peak decoder's tuning)
  -> python evaluate_phrase.py --model <model> --prefixes ... --split val  --save-predictions
     python evaluate_phrase.py --model <model> --prefixes ... --split test --save-predictions

Decoders: argmax | threshold (the 2023 paper's decoder, src.metrics.decode_threshold_2023, with the B / O
thresholds tuned on VAL: the run's own tuned_thresholds from evaluate_phrase.py --tune-on val if saved,
otherwise tuned here by boundary F1@5 on a 0.1..0.9 grid) | peak (peak picking; min_dist / sigma /
threshold tuned on VAL by boundary F1@5).

  B  SEGMENT LENGTHS   phrase durations of gold vs every decoder (pooled over the runs of a group):
                       median / quartiles / share < 0.3 s, Wasserstein distance and KS statistic on
                       log duration; also the gold pauses between phrases. fig_B_lengths.png
  C  TOLERANCE         boundary P / R / F1 at +-0..25 frames per decoder (mean +- SD over runs), and
                       the signed offsets (pred - gold) of starts matched within +-25 frames: median,
                       MAD, share within +-1 / +-2 frames. fig_C_tolerance.png
  D  PRECISION-RECALL  peak-height threshold sweep at +-5 frames: AP, recall at precision .7/.8/.9,
                       precision at recall .5/.6, and a reliability table (share of peaks that are
                       correct per peak-height bin). The "annotation assistant" operating points.
                       fig_D_pr.png
  E  PER VIDEO         boundary F1@5 / recall (peak decoder, mean over runs) per video next to
                       descriptors of the video's gold annotation (phrases per minute, median
                       phrase duration, median pause before a phrase, share of phrase starts with a
                       pause < 0.25 s; signs per second / median pause between signs with
                       --sign-labels-dir; share of filled hand frames inside phrases with
                       --missing-hands per_video.csv). Spearman rho + p (n = number of videos:
                       descriptive). Variance split: SD between videos vs SD between runs.
                       fig_E_per_video.png

Groups: --runs is one group ("model"); --compare adds a second group (e.g. the 2023 baseline),
plotted next to it. Every number is pooled over videos the way evaluate_phrase.py pools it.

Example:
  python thesis_result_stats.py --runs multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117 \
      --compare bilstm_2023-244 bilstm_2023-245 bilstm_2023-246 --labels "Multi-stream" "2023 (retrained)" \
      --sign-labels-dir processed_data/BIO_tags --missing-hands missing_hands/per_video.csv
  python thesis_result_stats.py --selftest
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

from count_oracle_decoding import peaks_with_heights, starts_to_segments, tune_peaks_on_val
from phrase_stats import PRED_ROOT, f1_from_counts, gold_starts_of, load_predictions
from src.metrics import B_TAG, O_TAG, decode_threshold_2023, extract_segments, match_boundaries

FPS = 50.0
TOL_CURVE = list(range(0, 26))
DECODERS = ("argmax", "threshold", "peak")
DEC_LABEL = {"argmax": "Argmax", "threshold": "Threshold", "peak": "Peak picking", "gold": "Gold"}
THR_TUNE_GRID = [round(x, 1) for x in np.arange(0.1, 0.91, 0.1)]   # B / O thresholds, if the run has none tuned
SHORT_S = 0.3
LEN_MAX_S = 15.0     # figure B: axis end (s); longer segments go into the last bin. None = the longest segment
LEN_BINS = 60        # figure B: number of bins over 0..LEN_MAX_S (60 -> 0.25 s per bin)


def len_axis_max(dist, gname):
    """Axis end of figure B: LEN_MAX_S if set, else the longest segment of this group (rounded up to 1 s)."""
    if LEN_MAX_S:
        return float(LEN_MAX_S)
    longest = max(float(x.max()) for (g, d), x in dist.items() if g == gname and len(x))
    return float(np.ceil(longest))


# ----------------------------------------------------------------------------- decoding
def tune_threshold_on_val(val):
    """B / O thresholds of the 2023 decoder by pooled boundary F1@5 on val."""
    probs, _, gold = val
    gstarts = {v: gold_starts_of(gold[v]) for v in probs}
    best = None
    for b in THR_TUNE_GRID:
        for o in THR_TUNE_GRID:
            tp = npred = ngold = 0
            for v in probs:
                st = np.array([a for a, _ in decode_threshold_2023(probs[v], b, o)], dtype=np.int64)
                x, y, z = counts(gstarts[v], st, 5)
                tp, npred, ngold = tp + x, npred + y, ngold + z
            f1 = f1_from_counts(tp, npred, ngold)
            if best is None or f1 > best[0]:
                best = (f1, b, o)
    return best[1], best[2], round(best[0], 4)


def decode_run(run, split, pred_root):
    """-> {"gold": {vid: gold}, "probs": ..., "dec": {decoder: {vid: (starts, segments)}},
           "peaks": {vid: (idx, heights)}, "tuned": {...}}"""
    probs, meta, gold = load_predictions(pred_root, run, split)
    val = load_predictions(pred_root, run, "val")
    tuned = tune_peaks_on_val(val)
    md, sg, thr = tuned["min_dist"], tuned["sigma"], tuned["thr"]
    saved = (val[1].get("tuned_thresholds") if val[1].get("tuned_on") == "val" else None) or meta.get("tuned_thresholds")
    if saved:
        b_thr, o_thr, src = float(saved[0]), float(saved[1]), "saved (evaluate_phrase.py --tune-on val)"
    else:
        b_thr, o_thr, f1 = tune_threshold_on_val(val)
        src = f"tuned here on val (boundary F1@5 {f1})"
    dec = {d: {} for d in DECODERS}
    peaks = {}
    for v in probs:
        p = probs[v]
        segs = extract_segments(p.argmax(axis=0))
        dec["argmax"][v] = (np.array([s for s, _ in segs], dtype=np.int64), segs)
        ts = decode_threshold_2023(p, b_thr, o_thr)
        dec["threshold"][v] = (np.array([s for s, _ in ts], dtype=np.int64), ts)
        idx, h = peaks_with_heights(p[B_TAG], md, sg)
        peaks[v] = (idx, h)
        st = np.sort(idx[h >= thr]).astype(np.int64)
        dec["peak"][v] = (st, starts_to_segments(st, p))
    return {"run": run, "gold": gold, "probs": probs, "dec": dec, "peaks": peaks,
            "tuned": {**tuned, "threshold_b": b_thr, "threshold_o": o_thr, "threshold_source": src}}


def counts(gs, ps, k):
    g_ok, _, _ = match_boundaries(gs, ps, k)
    return int(g_ok.sum()), len(ps), len(gs)


def pooled_prf(dec_v, gold, k):
    tp = npred = ngold = 0
    for v, (st, _) in dec_v.items():
        a, b, c = counts(gold_starts_of(gold[v]), st, k)
        tp, npred, ngold = tp + a, npred + b, ngold + c
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return p, r, f1_from_counts(tp, npred, ngold)


def q(x, *qs):
    x = np.asarray(x, dtype=float)
    return [float(np.quantile(x, a)) if len(x) else float("nan") for a in qs]


def mad(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


# ----------------------------------------------------------------------------- B
def gold_pauses(g):
    """Lengths (frames) of the O runs that end right before a gold phrase start (not the video start)."""
    segs = extract_segments(g)
    return [s - e - 1 for (_, e), (s, _) in zip(segs[:-1], segs[1:]) if s - e - 1 > 0]


HIST_ROWS = []


def analysis_B(groups):
    rows, dist = [], {}
    for gname, runs in groups.items():
        gold_d = [(e - s + 1) / FPS for v, g in runs[0]["gold"].items() for s, e in extract_segments(g)]
        dist[(gname, "gold")] = np.array(gold_d)
        for d in DECODERS:
            dist[(gname, d)] = np.array([(e - s + 1) / FPS for r in runs for st, segs in r["dec"][d].values()
                                         for s, e in segs])
        pauses = np.array([x / FPS for v, g in runs[0]["gold"].items() for x in gold_pauses(g)])
        for d in ("gold",) + DECODERS:
            x = dist[(gname, d)]
            n_runs = 1 if d == "gold" else len(runs)
            row = {"group": gname, "segments": DEC_LABEL[d], "n_per_run": round(len(x) / n_runs, 1),
                   "median_s": round(q(x, .5)[0], 3), "q25_s": round(q(x, .25)[0], 3), "q75_s": round(q(x, .75)[0], 3),
                   "q10_s": round(q(x, .1)[0], 3), "q90_s": round(q(x, .9)[0], 3),
                   "share_lt_0.3s": round(float((x < SHORT_S).mean()), 4) if len(x) else float("nan")}
            if d != "gold" and len(x):
                lg, lp = np.log10(dist[(gname, "gold")]), np.log10(x)
                row["wasserstein_log10"] = round(float(wasserstein_distance(lg, lp)), 4)
                row["ks_stat"] = round(float(ks_2samp(lg, lp).statistic), 4)
            rows.append(row)
            x_max = len_axis_max(dist, gname)
            edges = np.linspace(0, x_max, LEN_BINS + 1)
            if len(x):
                h = np.histogram(np.minimum(x, x_max - 1e-9), bins=edges)[0] / len(x)
                HIST_ROWS.append({"group": gname, "segments": DEC_LABEL[d],
                                  **{f"{a:.2f}-{b:.2f}s": round(float(v), 4)
                                     for a, b, v in zip(edges[:-1], edges[1:], h)}})
        rows.append({"group": gname, "segments": "Gold pauses between phrases", "n_per_run": len(pauses),
                     "median_s": round(q(pauses, .5)[0], 3), "q25_s": round(q(pauses, .25)[0], 3),
                     "q75_s": round(q(pauses, .75)[0], 3), "q10_s": round(q(pauses, .1)[0], 3),
                     "q90_s": round(q(pauses, .9)[0], 3), "share_lt_0.3s": round(float((pauses < SHORT_S).mean()), 4)})
    return rows, dist


# ----------------------------------------------------------------------------- C
def analysis_C(groups):
    rows, curves, offsets = [], {}, {}
    for gname, runs in groups.items():
        for d in DECODERS:
            per_k = np.array([[pooled_prf(r["dec"][d], r["gold"], k) for k in TOL_CURVE] for r in runs])  # (R, K, 3)
            curves[(gname, d)] = per_k
            off = []
            for r in runs:
                for v, (st, _) in r["dec"][d].items():
                    _, _, o = match_boundaries(gold_starts_of(r["gold"][v]), st, 25)
                    off += list(o[~np.isnan(o)])
            off = np.array(off)
            offsets[(gname, d)] = off
            for j, k in enumerate(TOL_CURVE):
                m, s = per_k[:, j].mean(0), per_k[:, j].std(0, ddof=1) if len(runs) > 1 else np.zeros(3)
                rows.append({"group": gname, "decoder": DEC_LABEL[d], "tol_frames": k, "tol_s": round(k / FPS, 2),
                             "precision": round(m[0], 4), "recall": round(m[1], 4), "f1": round(m[2], 4),
                             "f1_sd": round(s[2], 4)})
    off_rows = []
    for (gname, d), off in offsets.items():
        off_rows.append({"group": gname, "decoder": DEC_LABEL[d], "n_matched_25": len(off),
                         "median_offset_fr": float(np.median(off)) if len(off) else float("nan"),
                         "mean_offset_fr": round(float(off.mean()), 3) if len(off) else float("nan"),
                         "mad_fr": mad(off), "share_within_1": round(float((np.abs(off) <= 1).mean()), 4) if len(off) else "",
                         "share_within_2": round(float((np.abs(off) <= 2).mean()), 4) if len(off) else "",
                         "share_within_5": round(float((np.abs(off) <= 5).mean()), 4) if len(off) else ""})
    return rows, curves, offsets, off_rows


# ----------------------------------------------------------------------------- D
THR_SWEEP = np.round(np.arange(0.0, 1.0001, 0.01), 2)
HEIGHT_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def pr_curve(run, k=5):
    """-> (P, R) over THR_SWEEP and the per-peak (height, correct) list; peaks matched one-to-one at +-k."""
    tp = np.zeros(len(THR_SWEEP))
    npred = np.zeros(len(THR_SWEEP))
    ngold = 0
    hc = []
    for v, (idx, h) in run["peaks"].items():
        gs = gold_starts_of(run["gold"][v])
        ngold += len(gs)
        for j, t in enumerate(THR_SWEEP):
            st = np.sort(idx[h >= t])
            a, b, _ = counts(gs, st, k)
            tp[j] += a
            npred[j] += b
        _, p_ok, _ = match_boundaries(gs, idx, k)
        order = np.argsort(idx)
        hc += list(zip(h[order], p_ok))                      # match_boundaries sorts the predictions
    P = np.where(npred > 0, tp / np.maximum(npred, 1), 1.0)
    R = tp / max(ngold, 1)
    return P, R, hc


def ap_from_pr(P, R):
    """Step-wise AP: sum over decreasing thresholds of (R_i - R_{i-1}) * P_i."""
    o = np.argsort(R)
    R, P = R[o], P[o]
    Pm = np.maximum.accumulate(P[::-1])[::-1]                # interpolated precision
    return float(np.sum(np.diff(np.concatenate([[0.0], R])) * Pm))


def analysis_D(groups):
    rows, rel_rows, curves = [], [], {}
    for gname, runs in groups.items():
        per = [pr_curve(r) for r in runs]
        Ps, Rs = np.array([p for p, _, _ in per]), np.array([r for _, r, _ in per])
        curves[gname] = (Ps, Rs)
        aps = [ap_from_pr(p, r) for p, r, _ in per]
        row = {"group": gname, "runs": len(runs), "AP_5": round(float(np.mean(aps)), 4),
               "AP_5_sd": round(float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0, 4)}
        for pt in (0.7, 0.8, 0.9):
            vals = [float(r[p >= pt].max()) if (p >= pt).any() else 0.0 for p, r in zip(Ps, Rs)]
            row[f"recall_at_P{pt}"] = round(float(np.mean(vals)), 4)
        for rt in (0.5, 0.6):
            vals = [float(p[r >= rt].max()) if (r >= rt).any() else float("nan") for p, r in zip(Ps, Rs)]
            row[f"precision_at_R{rt}"] = round(float(np.nanmean(vals)), 4) if not all(np.isnan(vals)) else ""
        rows.append(row)
        hc = np.array([x for _, _, h in per for x in h], dtype=float).reshape(-1, 2)
        for lo, hi in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:]):
            m = (hc[:, 0] >= lo) & (hc[:, 0] < hi)
            rel_rows.append({"group": gname, "peak_height": f"{lo:.1f}-{min(hi, 1):.1f}", "peaks": int(m.sum()),
                             "share_correct_5": round(float(hc[m, 1].mean()), 4) if m.any() else ""})
    return rows, rel_rows, curves


# ----------------------------------------------------------------------------- E
def video_descriptors(vid, g, sign_dir=None, missing=None):
    segs = extract_segments(g)
    T = len(g)
    pauses = []
    for (_, e), (s, _) in zip(segs[:-1], segs[1:]):
        pauses.append((s - e - 1) / FPS)
    d = {"video": vid, "minutes": round(T / FPS / 60, 2), "phrases": len(segs),
         "phrases_per_min": round(len(segs) / (T / FPS / 60), 2) if T else float("nan"),
         "median_phrase_s": round(float(np.median([(e - s + 1) / FPS for s, e in segs])), 3) if segs else float("nan"),
         "median_pause_before_phrase_s": round(float(np.median(pauses)), 3) if pauses else float("nan"),
         "share_starts_pause_lt_0.25s": round(float(np.mean(np.array(pauses) < 0.25)), 4) if pauses else float("nan")}
    if sign_dir:
        path = os.path.join(sign_dir, f"{vid}.npy")
        if os.path.exists(path):
            ssegs = extract_segments(np.load(path))
            signing = float((g[:len(g)] != O_TAG).sum()) / FPS
            d["signs_per_s_signing"] = round(len(ssegs) / signing, 3) if signing else float("nan")
            gaps = [(s - e - 1) / FPS for (_, e), (s, _) in zip(ssegs[:-1], ssegs[1:]) if s - e - 1 >= 0]
            d["median_pause_between_signs_s"] = round(float(np.median(gaps)), 3) if gaps else float("nan")
    if missing is not None and vid in missing:
        d["filled_share_in_phrases"] = missing[vid]
    return d


def load_missing(path):
    if not path:
        return None
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            n = float(row["in_n"])
            out[row["video"]] = round(float(row["in_f"]) / n, 4) if n else float("nan")
    return out


def analysis_E(groups, sign_dir=None, missing_path=None):
    missing = load_missing(missing_path)
    vrows, corr_rows, var_rows = [], [], []
    for gname, runs in groups.items():
        vids = sorted(set.intersection(*[set(r["gold"]) for r in runs]))
        f1 = np.zeros((len(runs), len(vids)))
        rec = np.zeros_like(f1)
        for i, r in enumerate(runs):
            for j, v in enumerate(vids):
                a, b, c = counts(gold_starts_of(r["gold"][v]), r["dec"]["peak"][v][0], 5)
                f1[i, j] = f1_from_counts(a, b, c)
                rec[i, j] = a / c if c else np.nan
        for j, v in enumerate(vids):
            d = video_descriptors(v, runs[0]["gold"][v], sign_dir, missing)
            vrows.append({"group": gname, **d, "bnd_f1_5_peak": round(float(f1[:, j].mean()), 4),
                          "recall_5_peak": round(float(np.nanmean(rec[:, j])), 4),
                          "bnd_f1_5_sd_runs": round(float(f1[:, j].std(ddof=1)), 4) if len(runs) > 1 else ""})
        g_rows = [r for r in vrows if r["group"] == gname]
        keys = [k for k in g_rows[0] if k not in ("group", "video", "bnd_f1_5_peak", "recall_5_peak",
                                                 "bnd_f1_5_sd_runs", "minutes", "phrases")]
        for target in ("bnd_f1_5_peak", "recall_5_peak"):
            y = np.array([r[target] for r in g_rows], dtype=float)
            for k in keys:
                x = np.array([r.get(k, np.nan) for r in g_rows], dtype=float)
                ok = ~np.isnan(x) & ~np.isnan(y)
                if ok.sum() >= 5:
                    rho, p = spearmanr(x[ok], y[ok])
                    corr_rows.append({"group": gname, "target": target, "descriptor": k, "n_videos": int(ok.sum()),
                                      "spearman_rho": round(float(rho), 3), "p": round(float(p), 4)})
        var_rows.append({"group": gname, "videos": len(vids), "runs": len(runs),
                         "mean_bnd_f1_5": round(float(f1.mean()), 4),
                         "sd_between_videos": round(float(f1.mean(0).std(ddof=1)), 4),
                         "sd_between_runs_within_video": round(float(f1.std(0, ddof=1).mean()), 4) if len(runs) > 1 else "",
                         "min_video": round(float(f1.mean(0).min()), 4), "max_video": round(float(f1.mean(0).max()), 4)})
    return vrows, corr_rows, var_rows


# ----------------------------------------------------------------------------- output
def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt_table(rows, cols, widths=None):
    widths = widths or {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    out = ["  " + "  ".join(c.rjust(widths[c]) for c in cols)]
    for r in rows:
        out.append("  " + "  ".join(str(r.get(c, "")).rjust(widths[c]) for c in cols))
    return out


def make_figures(out, groups, dist, curves_c, offsets, curves_d, vrows, corr_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: figures skipped")
        return
    colors = {"gold": "#222222", "argmax": "#C0504D", "threshold": "#4F81BD", "peak": "#2E8B57"}
    vivid = {"gold": "#222222", "argmax": "#E8382B", "threshold": "#1F78E0", "peak": "#12A150"}   # figure B
    gnames = list(groups)
    # B: linear axis, LEN_BINS bins from 0 to the axis end (the longest segment unless LEN_MAX_S is set)
    # main group (--runs) only; the --compare group stays in the tables and in figures C / D
    fig, axes = plt.subplots(1, 1, figsize=(6.0, 3.6), squeeze=False)
    for ax, gname in zip(axes[0], gnames[:1]):
        x_max = len_axis_max(dist, gname)
        edges = np.linspace(0, x_max, LEN_BINS + 1)
        for d in ("gold",) + DECODERS:
            x = dist[(gname, d)]
            if len(x):
                share = np.histogram(np.minimum(x, x_max - 1e-9), bins=edges)[0] / len(x) * 100
                ax.stairs(share, edges, fill=True, color=vivid[d], alpha=0.08 if d == "gold" else 0.38,
                          lw=0, zorder=1)                         # shaded area (gold lighter: it is black)
                ax.stairs(share, edges, lw=2.2 if d == "gold" else 1.4, color=vivid[d], label=DEC_LABEL[d],
                          zorder=5 if d == "gold" else 3)          # gold outline drawn on top of everything
        ax.set_xlim(-0.02 * x_max, x_max)
        ax.set_xticks(np.arange(0, x_max + 1e-9, 1))
        ax.set_xlabel(f"phrase duration (s); {LEN_BINS} bins of {x_max / LEN_BINS:.2g} s"
                      + ("; last bin includes longer phrases" if LEN_MAX_S else ""))
        ax.set_ylabel("share of segments (%)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_B_lengths.png"), dpi=200)
    plt.close(fig)
    # C
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ls = ["-", "--", ":"]
    for gi, gname in enumerate(gnames):
        for d in DECODERS:
            c = curves_c[(gname, d)][:, :, 2]
            axes[0].plot(np.array(TOL_CURVE) / FPS, c.mean(0), ls=ls[gi % 3], color=colors[d],
                         label=f"{DEC_LABEL[d]} ({gname})" if len(gnames) > 1 else DEC_LABEL[d])
    axes[0].axvline(5 / FPS, color="#999999", ls=":", lw=1)
    axes[0].set_xlabel("tolerance (s)")
    axes[0].set_ylabel("boundary F1")
    axes[0].legend(frameon=False, fontsize=7)
    off = offsets[(gnames[0], "peak")]
    axes[1].hist(off, bins=np.arange(-25.5, 26.5, 1), color=colors["peak"])
    axes[1].set_xlabel(f"offset predicted - gold (frames), peak picking, {gnames[0]}")
    axes[1].set_ylabel("matched starts")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_C_tolerance.png"), dpi=200)
    plt.close(fig)
    # D
    fig, ax = plt.subplots(figsize=(4.8, 4))
    for gi, gname in enumerate(gnames):
        Ps, Rs = curves_d[gname]
        for i in range(len(Ps)):
            ax.plot(Rs[i], Ps[i], color=["#2E8B57", "#C0504D", "#4F81BD"][gi % 3], alpha=0.8, lw=1.2,
                    label=gname if i == 0 else None)
    ax.set_xlabel("recall (boundary, +-0.1 s)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_D_pr.png"), dpi=200)
    plt.close(fig)
    # E: the descriptor with the largest |rho| for the first group
    cr = [r for r in corr_rows if r["group"] == gnames[0] and r["target"] == "bnd_f1_5_peak"]
    if cr:
        best = max(cr, key=lambda r: abs(r["spearman_rho"]))
        g = [r for r in vrows if r["group"] == gnames[0]]
        x = [r.get(best["descriptor"], np.nan) for r in g]
        y = [r["bnd_f1_5_peak"] for r in g]
        fig, ax = plt.subplots(figsize=(4.8, 4))
        ax.scatter(x, y, color="#2E8B57")
        ax.set_xlabel(best["descriptor"])
        ax.set_ylabel("boundary F1@5 per video (peak)")
        ax.set_title(f"rho = {best['spearman_rho']:+.2f}, p = {best['p']:.3f}, n = {best['n_videos']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_E_per_video.png"), dpi=200)
        plt.close(fig)


def run_all(groups_runs, split, pred_root, out, sign_dir=None, missing=None):
    os.makedirs(out, exist_ok=True)
    groups = {}
    for gname, runs in groups_runs.items():
        groups[gname] = []
        for r in runs:
            print(f"  decoding {r} ...")
            groups[gname].append(decode_run(r, split, pred_root))
    b_rows, dist = analysis_B(groups)
    c_rows, curves_c, offsets, off_rows = analysis_C(groups)
    d_rows, rel_rows, curves_d = analysis_D(groups)
    v_rows, corr_rows, var_rows = analysis_E(groups, sign_dir, missing)
    for name, rows in (("B_lengths", b_rows), ("B_length_hist", HIST_ROWS), ("C_tolerance", c_rows), ("C_offsets", off_rows),
                       ("D_pr", d_rows), ("D_reliability", rel_rows), ("E_per_video", v_rows),
                       ("E_correlations", corr_rows), ("E_variance", var_rows)):
        write_csv(os.path.join(out, f"{name}.csv"), rows)
    tune = [{"run": r["run"], **r["tuned"]} for g in groups.values() for r in g]
    write_csv(os.path.join(out, "decoder_settings.csv"), tune)

    L = [f"Thesis result statistics ({split}); groups: " +
         "; ".join(f"{g} = {', '.join(r)}" for g, r in groups_runs.items()),
         "Peak decoder tuned on val per run (decoder_settings.csv). Boundaries counted at +-5 frames = 0.1 s "
         "unless stated; counts pooled over videos.", ""]
    L += ["B. PHRASE DURATIONS (seconds; Wasserstein / KS on log10 duration vs gold)"]
    L += fmt_table(b_rows, ["group", "segments", "n_per_run", "median_s", "q25_s", "q75_s", "q10_s", "q90_s",
                            "share_lt_0.3s", "wasserstein_log10", "ks_stat"])
    L += ["", "C. BOUNDARY F1 vs TOLERANCE (mean over runs)"]
    sel = [r for r in c_rows if r["tol_frames"] in (0, 1, 2, 5, 10, 15, 25)]
    L += fmt_table(sel, ["group", "decoder", "tol_frames", "tol_s", "precision", "recall", "f1", "f1_sd"])
    L += ["", "   offsets of starts matched within +-25 frames (predicted - gold)"]
    L += fmt_table(off_rows, ["group", "decoder", "n_matched_25", "median_offset_fr", "mean_offset_fr", "mad_fr",
                              "share_within_1", "share_within_2", "share_within_5"])
    L += ["", "D. PRECISION-RECALL of peak heights (+-5 frames; mean over runs)"]
    L += fmt_table(d_rows, list(d_rows[0].keys()))
    L += ["", "   reliability: share of peaks that match a gold start, by peak height"]
    L += fmt_table(rel_rows, ["group", "peak_height", "peaks", "share_correct_5"])
    L += ["", "E. PER VIDEO (peak decoder). Spearman over videos -- n is small: descriptive only"]
    L += fmt_table(var_rows, list(var_rows[0].keys()))
    cr = sorted([r for r in corr_rows if r["target"] == "bnd_f1_5_peak"], key=lambda r: (r["group"], -abs(r["spearman_rho"])))
    L += fmt_table(cr, ["group", "descriptor", "n_videos", "spearman_rho", "p"])
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text)
    print(text)
    make_figures(out, groups, dist, curves_c, offsets, curves_d, v_rows, corr_rows)
    print(f"written to {out}/ (summary.txt, *.csv, fig_*.png)")
    return {"B": b_rows, "C": c_rows, "C_off": off_rows, "D": d_rows, "E_var": var_rows, "E_corr": corr_rows}


# ----------------------------------------------------------------------------- selftest
def selftest():
    """Synthetic predictions with a known answer: B at every gold start shifted by a fixed offset."""
    import json
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/labels")
        pred_root = f"{d}/pred"
        vids = [f"v{i}" for i in range(6)]
        golds = {}
        for v in vids:
            g = np.zeros(3000, dtype=np.int64)
            t = 50
            while t < 2900:
                L = int(rng.integers(40, 200))
                g[t] = 2
                g[t + 1:min(t + L, 3000)] = 1
                t += L + int(rng.integers(5, 60))
            golds[v] = g
            np.save(f"{d}/labels/{v}.npy", g)
        for run, shift in (("model-1", 2), ("model-2", 2), ("base-1", 0)):
            os.makedirs(f"{pred_root}/{run}")
            for split in ("val", "test"):
                arrs = {}
                for v in vids:
                    g = golds[v]
                    p = np.zeros((3, len(g)), dtype=np.float32)
                    p[0] = (g == 0) * 0.9 + 0.05
                    p[1] = (g != 0) * 0.9 + 0.05
                    for s in np.where(g == 2)[0]:
                        s2 = min(s + shift, len(g) - 1)
                        p[2, s2] = 0.95 if run.startswith("model") else (0.95 if rng.random() < 0.7 else 0.2)
                        p[:2, s2] = 0.025
                    if run.startswith("base"):                     # false peaks away from every start
                        for f_ in rng.choice(np.where(g == 1)[0], 15, replace=False):
                            if np.abs(np.where(g == 2)[0] - f_).min() > 30:
                                p[2, f_], p[0, f_], p[1, f_] = 0.95, 0.025, 0.025
                    p = p / p.sum(0, keepdims=True)
                    arrs[v] = p.astype(np.float16)
                np.savez_compressed(f"{pred_root}/{run}/{split}.npz", **arrs)
                with open(f"{pred_root}/{run}/{split}_meta.json", "w") as f:
                    json.dump({"run": run, "labels_dir": f"{d}/labels", "description": run,
                               "merge_starts": {"argmax_merge": 10}}, f)
        res = run_all({"model": ["model-1", "model-2"], "base": ["base-1"]}, "test", pred_root, f"{d}/out")
        ok = []
        c = {(r["group"], r["decoder"], r["tol_frames"]): r for r in res["C"]}
        ok.append(("perfect model: F1@2 of peak == 1", c[("model", "Peak picking", 2)]["f1"] > 0.999))
        ok.append(("perfect model: F1@1 of peak == 0 (all offsets are +2)", c[("model", "Peak picking", 1)]["f1"] < 1e-6))
        off = {(r["group"], r["decoder"]): r for r in res["C_off"]}
        ok.append(("median offset == +2", off[("model", "Peak picking")]["median_offset_fr"] == 2.0))
        b = {(r["group"], r["segments"]): r for r in res["B"]}
        ok.append(("gold-like segments: Wasserstein ~ 0", b[("model", "Peak picking")]["wasserstein_log10"] < 0.05))
        dd = {r["group"]: r for r in res["D"]}
        ok.append(("AP of perfect model == 1", dd["model"]["AP_5"] > 0.999))
        ok.append(("baseline with false peaks: 0.3 < AP < 0.99", 0.3 < dd["base"]["AP_5"] < 0.99))
        ok.append(("figures written", all(os.path.exists(f"{d}/out/fig_{x}.png")
                                          for x in ("B_lengths", "C_tolerance", "D_pr", "E_per_video"))))
        for name, cond in ok:
            print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        return all(c for _, c in ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", help="runs of the main group (e.g. the 3 seeds of the best model)")
    ap.add_argument("--compare", nargs="+", default=None, help="runs of a second group (e.g. the 2023 baseline)")
    ap.add_argument("--labels", nargs="+", default=None, help="group names (default: Model / Comparison)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "thesis_result_stats"))
    ap.add_argument("--sign-labels-dir", default=None, help="gold SIGN BIO labels (processed_data/BIO_tags) for E")
    ap.add_argument("--missing-hands", default=None, help="per_video.csv of analyze_missing_hands.py for E")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.runs:
        ap.error("--runs is required (or --selftest)")
    labels = a.labels or ["Model", "Comparison"]
    groups = {labels[0]: a.runs}
    if a.compare:
        groups[labels[1] if len(labels) > 1 else "Comparison"] = a.compare
    run_all(groups, a.split, a.pred_root, a.out_dir, a.sign_labels_dir, a.missing_hands)


if __name__ == "__main__":
    main()