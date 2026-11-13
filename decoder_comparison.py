"""
Decoder comparison on saved predictions: is peak picking better than the 2023 threshold decoder, and does a
threshold -> peak-picking hybrid beat both? Every decoder is tuned on VAL and scored on the test split.

  threshold (IoU+%)   the 2023 decoder (src.metrics.decode_threshold_2023), B / O thresholds tuned the 2023 way:
                      max IoU + %-score (1 - |% - 1|), the same criterion as evaluate_phrase.py --sweep-metric iou_pct
  threshold (bF1)     the same decoder, thresholds tuned by boundary F1@5 (for the best model: 0.3 / 0.3)
  peak picking        peaks of P(B) (Gaussian smoothing sigma, peaks >= min_dist apart, height >= thr); sigma /
                      min_dist / thr tuned by boundary F1@5 (count_oracle_decoding.tune_peaks_on_val); segments
                      end at the next start or at the first run of >= 5 argmax-O frames
  hybrid (bF1)        threshold first, then peak picking: the 2023 decoder gives the segments (B / O thresholds);
                      each segment's start is moved to the P(B) peak inside its opening B run (P(B) > b, smoothed
                      with sigma); starts closer than min_dist are merged into one segment that starts at the
                      higher peak and ends at the later end. b / o / sigma / min_dist tuned by boundary F1@5
  hybrid (IoU+%)      the same, tuned by IoU + %-score

Reported on test: boundary F1 at +-2 / +-5 / +-10, P / R at +-5, IoU and % (per-video means, 2023 style),
Segment F1@0.5 (each predicted phrase matched one-to-one to a gold phrase with IoU >= 0.5:
the quality of whole phrases), and every decoder minus peak picking on bF1@5 and Segment F1@0.5: paired
bootstrap over test videos (runs averaged). --compare: a second configuration, compared decoder by decoder.

  python decoder_comparison.py --runs multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117
  python decoder_comparison.py --selftest
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

from count_oracle_decoding import peaks_with_heights, starts_to_segments, tune_peaks_on_val
from phrase_stats import PRED_ROOT, f1_from_counts, gold_starts_of, load_predictions, paired_bootstrap
from src.metrics import (B_TAG, decode_threshold_2023, extract_segments, match_boundaries, pct_score,
                         segment_f1_at, segment_iou, segment_percentage)

THR_GRID = [round(x, 1) for x in np.arange(0.1, 0.91, 0.1)]
SIGMA_GRID = (0, 2, 5)
MIN_DIST_GRID = (0, 10, 25, 50)
TOLS = (2, 5, 10)
ORDER = ["threshold (IoU+%)", "threshold (bF1)", "peak picking", "hybrid (bF1)", "hybrid (IoU+%)"]


# ----------------------------------------------------------------------------- decoders
def refine(segs, p_b, b_thr, sigma, min_dist):
    """Hybrid step: start -> P(B) peak inside the opening B run; merge starts closer than min_dist."""
    if not segs:
        return []
    p = gaussian_filter1d(np.asarray(p_b, dtype=np.float64), sigma) if sigma else np.asarray(p_b, dtype=np.float64)
    raw = np.asarray(p_b, dtype=np.float64)
    out = []
    for s, e in segs:
        t = s
        while t + 1 <= e and raw[t + 1] > b_thr:        # opening run where the 2023 decoder still sees B
            t += 1
        pk = s + int(np.argmax(p[s:t + 1]))
        out.append([pk, e, p[pk]])
    merged = [out[0]]
    for pk, e, h in out[1:]:
        if min_dist and pk - merged[-1][0] < min_dist:
            m = merged[-1]
            if h > m[2]:
                m[0], m[2] = pk, h
            m[1] = max(m[1], e)
        else:
            merged.append([pk, e, h])
    res = []
    for pk, e, _ in merged:                              # keep segments non-overlapping and ordered
        if res and pk <= res[-1][1]:
            res[-1] = (res[-1][0], pk - 1)
        res.append((pk, max(pk, e)))
    return res


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
        ious.append(iou)
        pc = segment_percentage(segs, gsegs)
        if not np.isnan(pc):
            pcts.append(pc)
        sf = segment_f1_at(segs, gsegs, 0.5)
        sf1s.append(sf)
        per_vid[v] = {"bnd5": bnd5, "seg_f1_05": sf, "iou": iou, "pct": pc}
    out = {f"bnd_f1_{k}": f1_from_counts(tp[k], npred, ngold) for k in TOLS}
    out.update({"bnd_p_5": tp[5] / npred if npred else 0.0, "bnd_r_5": tp[5] / ngold if ngold else 0.0,
                "iou": float(np.mean(ious)), "pct": float(np.mean(pcts)) if pcts else float("nan"),
                "seg_f1_05": float(np.mean(sf1s))})
    out["iou_pct_score"] = out["iou"] + pct_score(out["pct"])
    return out, per_vid


def tune_threshold_family(val):
    """One pass over (b, o): best plain-threshold (b, o) and best hybrid (b, o, sigma, min_dist), each by both
    criteria ("bnd" = boundary F1@5, "ioupct" = IoU + %-score). Every (b, o) is decoded once."""
    probs, _, gold = val
    keys = {"bnd": lambda m: m["bnd_f1_5"], "ioupct": lambda m: m["iou_pct_score"]}
    best = {(fam, c): None for fam in ("t", "h") for c in keys}
    for b in THR_GRID:
        for o in THR_GRID:
            base = {v: decode_threshold_2023(probs[v], b, o) for v in probs}
            m, _ = score(base, gold)
            for c, f in keys.items():
                if best[("t", c)] is None or f(m) > best[("t", c)][0]:
                    best[("t", c)] = (f(m), b, o)
            for sg in SIGMA_GRID:
                for md in MIN_DIST_GRID:
                    m, _ = score({v: refine(base[v], probs[v][B_TAG], b, sg, md) for v in probs}, gold)
                    for c, f in keys.items():
                        if best[("h", c)] is None or f(m) > best[("h", c)][0]:
                            best[("h", c)] = (f(m), b, o, sg, md)
    return best


def decode_all(run, split, pred_root):
    val = load_predictions(pred_root, run, "val")
    probs, _, gold = load_predictions(pred_root, run, split)
    best = tune_threshold_family(val)
    _, tb_i, to_i = best[("t", "ioupct")]
    _, tb_b, to_b = best[("t", "bnd")]
    _, hb_i, ho_i, hs_i, hm_i = best[("h", "ioupct")]
    _, hb_b, ho_b, hs_b, hm_b = best[("h", "bnd")]
    pk = tune_peaks_on_val(val)
    settings = {"threshold (IoU+%)": f"b={tb_i} o={to_i}", "threshold (bF1)": f"b={tb_b} o={to_b}",
                "peak picking": f"min_dist={pk['min_dist']} sigma={pk['sigma']} thr={pk['thr']}",
                "hybrid (bF1)": f"b={hb_b} o={ho_b} sigma={hs_b} min_dist={hm_b}",
                "hybrid (IoU+%)": f"b={hb_i} o={ho_i} sigma={hs_i} min_dist={hm_i}"}
    dec = {k: {} for k in ORDER}
    for v in probs:
        p = probs[v]
        dec["threshold (IoU+%)"][v] = decode_threshold_2023(p, tb_i, to_i)
        dec["threshold (bF1)"][v] = decode_threshold_2023(p, tb_b, to_b)
        idx, h = peaks_with_heights(p[B_TAG], pk["min_dist"], pk["sigma"])
        dec["peak picking"][v] = starts_to_segments(np.sort(idx[h >= pk["thr"]]), p)
        dec["hybrid (bF1)"][v] = refine(decode_threshold_2023(p, hb_b, ho_b), p[B_TAG], hb_b, hs_b, hm_b)
        dec["hybrid (IoU+%)"][v] = refine(decode_threshold_2023(p, hb_i, ho_i), p[B_TAG], hb_i, hs_i, hm_i)
    res = {k: score(dec[k], gold) for k in ORDER}
    return res, settings


# ----------------------------------------------------------------------------- bootstrap
def _bnd(per_vids):
    return [{v: d["bnd5"] for v, d in pv.items()} for pv in per_vids]


def mean_bootstrap(pv_a, pv_b, key, n_boot=2000, seed=0):
    """Per-video metric (key) averaged over videos, runs averaged within each arm; A - B, videos resampled
    together (paired). NaN values (e.g. % of a video without gold phrases) are skipped."""
    vids = sorted(set.intersection(*[set(p) for p in pv_a + pv_b]))
    A = np.array([[p[v][key] for v in vids] for p in pv_a], dtype=float)
    B = np.array([[p[v][key] for v in vids] for p in pv_b], dtype=float)

    def stat(idx):
        return float(np.nanmean(np.nanmean(A[:, idx], 1)) - np.nanmean(np.nanmean(B[:, idx], 1)))
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.integers(0, len(vids), len(vids))) for _ in range(n_boot)])
    return {"diff": stat(np.arange(len(vids))), "ci95": (float(np.percentile(draws, 2.5)),
                                                           float(np.percentile(draws, 97.5)))}


def fmt_ci(b):
    return f"{b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"


# ----------------------------------------------------------------------------- report
def run(runs, split, pred_root, out_dir, n_boot=2000):
    os.makedirs(out_dir, exist_ok=True)
    per_run, rows = {}, []
    for r in runs:
        print(f"  tuning + decoding {r} (the threshold grids take a few minutes per run) ...")
        res, settings = decode_all(r, split, pred_root)
        per_run[r] = res
        for k in ORDER:
            rows.append({"run": r, "decoder": k, "settings_tuned_on_val": settings[k],
                         **{m: round(v, 4) for m, v in res[k][0].items()}})
    with open(os.path.join(out_dir, "decoder_comparison_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = [f"Decoder comparison ({split}; every decoder tuned on val); runs: {', '.join(runs)}",
         "Boundary F1 pooled over videos; IoU / % / Segment F1@0.5 (one-to-one phrase matches with IoU >= 0.5) =",
         "per-video means (2023 style). Mean +- SD over runs.", "",
         f"{'decoder':20s} {'bF1@2':>13s} {'bF1@5':>13s} {'bF1@10':>13s} {'P@5':>6s} {'R@5':>6s} {'IoU':>6s} "
         f"{'%':>6s} {'SegF1@.5':>13s}   vs peak picking [95% CI]: bF1@5 | SegF1@.5"]
    peak_pv = [per_run[r]["peak picking"][1] for r in runs]
    for k in ORDER:
        M = {m: np.array([per_run[r][k][0][m] for r in runs]) for m in per_run[runs[0]][k][0]}

        def ms(m):
            sd = M[m].std(ddof=1) if len(runs) > 1 else 0.0
            return f"{M[m].mean():.3f}+-{sd:.3f}"
        if k == "peak picking":
            cmp = "(reference)"
        else:
            pv = [per_run[r][k][1] for r in runs]
            cmp = (fmt_ci(paired_bootstrap(_bnd(pv), _bnd(peak_pv), n_boot=n_boot)) + " | "
                   + fmt_ci(mean_bootstrap(pv, peak_pv, "seg_f1_05", n_boot)))
        L.append(f"{k:20s} {ms('bnd_f1_2'):>13s} {ms('bnd_f1_5'):>13s} {ms('bnd_f1_10'):>13s} "
                 f"{M['bnd_p_5'].mean():6.3f} {M['bnd_r_5'].mean():6.3f} {M['iou'].mean():6.3f} "
                 f"{M['pct'].mean():6.2f} {ms('seg_f1_05'):>13s}   {cmp}")
    L += ["", "Settings tuned on val (per run): see decoder_comparison_runs.csv"]
    for r in runs:
        L.append(f"  {r}: " + " | ".join(f"{row['decoder']}: {row['settings_tuned_on_val']}"
                                        for row in rows if row["run"] == r))
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "decoder_comparison_summary.txt"), "w") as f:
        f.write(text)
    print(text)
    return per_run


def compare_groups(res_a, res_b, label_a, label_b, out_dir, n_boot=2000):
    """Every decoder: group A minus group B (each run tuned on its own val), paired over test videos."""
    L = [f"GROUP COMPARISON: {label_a} minus {label_b} (paired bootstrap over test videos, runs averaged)",
         f"{'decoder':20s} {'bF1@5':>24s} {'SegF1@.5':>24s} {'IoU':>24s} {'%':>24s}"]
    for k in ORDER:
        pa = [res_a[r][k][1] for r in res_a]
        pb = [res_b[r][k][1] for r in res_b]
        cells = [fmt_ci(paired_bootstrap(_bnd(pa), _bnd(pb), n_boot=n_boot))]
        cells += [fmt_ci(mean_bootstrap(pa, pb, m, n_boot)) for m in ("seg_f1_05", "iou", "pct")]
        L.append(f"{k:20s} " + " ".join(f"{c:>24s}" for c in cells))
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "group_comparison.txt"), "w") as f:
        f.write(text)
    print(text)


def selftest():
    """refine(): starts move to the P(B) peak, close starts merge; a full run on synthetic predictions."""
    import json
    import tempfile
    ok = []
    p_b = np.zeros(200)
    p_b[[10, 11, 12, 13]] = [0.6, 0.7, 0.9, 0.6]          # opening B run, peak at 12
    p_b[[60, 66]] = [0.7, 0.8]                              # two starts 6 frames apart, the second higher
    segs = [(10, 50), (60, 65), (66, 120)]
    r = refine(segs, p_b, 0.5, 0, 10)
    ok.append(("start moved to the peak inside the B run", r[0][0] == 12))
    ok.append(("close starts merged at the higher peak, end = later end", r[1] == (66, 120) and len(r) == 2))
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
                p = np.full((3, len(g)), 0.05, dtype=np.float32)
                p[0, g == 0], p[1, g != 0] = 0.9, 0.9
                for s in np.where(g == 2)[0]:                # a B bump a few frames wide, peak 2 frames late
                    for j, hgt in zip(range(s, s + 5), (0.4, 0.6, 0.8, 0.6, 0.4)):
                        p[2, j] = hgt
                arrs[v] = (p / p.sum(0, keepdims=True)).astype(np.float16)
            np.savez_compressed(f"{d}/pred/m-1/{split}.npz", **arrs)
            with open(f"{d}/pred/m-1/{split}_meta.json", "w") as f:
                json.dump({"run": "m-1", "labels_dir": f"{d}/labels", "description": "m"}, f)
        full = run(["m-1"], "test", f"{d}/pred", f"{d}/out", n_boot=200)
        res = full["m-1"]
        ok.append(("Segment F1@0.5 of the peak decoder > .9", res["peak picking"][0]["seg_f1_05"] > 0.9))
        compare_groups(full, full, "m", "m", f"{d}/out", n_boot=100)
        ok.append(("group comparison written (a group against itself)", os.path.exists(f"{d}/out/group_comparison.txt")
                   and "+0.000 [+0.000, +0.000]" in open(f"{d}/out/group_comparison.txt").read()))
        ok.append(("peak picking finds the bumps (bF1@5 > .95)", res["peak picking"][0]["bnd_f1_5"] > 0.95))
        ok.append(("hybrid finds the bumps (bF1@5 > .95)", res["hybrid (bF1)"][0]["bnd_f1_5"] > 0.95))
        ok.append(("hybrid starts are on the peak (bF1@1-level: bF1@2 > .95)", res["hybrid (bF1)"][0]["bnd_f1_2"] > 0.95))
    for name, c in ok:
        print(f"  [{'ok' if c else 'FAIL'}] {name}")
    return all(c for _, c in ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="runs of a second configuration: adds group_comparison.txt (--runs minus --compare)")
    ap.add_argument("--labels", nargs=2, default=["A", "B"], help="names of --runs and --compare")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--compare-pred-root", default=None,
                    help="prediction folder of the --compare runs (default: --pred-root). Lets the SAME runs be "
                         "compared under two inference settings, e.g. evaluation_phrase/predictions_ov256 vs "
                         "evaluation_phrase/predictions")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "decoder_comparison"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.runs:
        ap.error("--runs is required (or --selftest)")
    res_a = run(a.runs, a.split, a.pred_root, a.out_dir, a.n_boot)
    if a.compare:
        res_b = run(a.compare, a.split, a.compare_pred_root or a.pred_root, os.path.join(a.out_dir, "compare"), a.n_boot)
        compare_groups(res_a, res_b, a.labels[0], a.labels[1], a.out_dir, a.n_boot)


if __name__ == "__main__":
    main()