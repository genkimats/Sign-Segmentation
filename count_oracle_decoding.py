"""
Idea C: separate "how many phrase boundaries" (count) from "where" (placement).

For every run it decodes the saved P(B) tracks with several decoders, some of which are given
the TRUE number of phrases (oracle K), and scores each with boundary F1, Segment F1@0.5, IoU
and %. The difference between a tuned peak decoder and the same peaks with the true count is
what the run loses to the COUNT; what is still missing with the true count is PLACEMENT.

Decoders (one row each per run):
  argmax            plain argmax segments (what the metric CSVs report as "argmax")
  argmax_merge      + duplicate-start suppression with the run's val-chosen merge distance
                    (from its _meta.json; skipped if the run was evaluated without --merge-starts)
  peak_thr          peaks of (optionally smoothed) P(B) above a threshold; threshold, minimum
                    peak distance and smoothing tuned on VAL by boundary F1@5
  oracleK_video     the K highest peaks, K = number of gold phrases in the video
  oracleK_chunk_L   the video cut into chunks of L frames, each keeping its own gold count
                    (L = 3000 / 1500 / 500 = 60 / 30 / 10 s): count errors removed at finer scales
  estK_duration     K = round(predicted non-O frames / mean gold phrase length on TRAIN)
  estK_mass         K = round(c * sum_t P(B)_t), c fitted on VAL (ratio of gold count to P(B) mass)
Peak decoders turn starts into segments with starts_to_segments(), so the decoded count is
exactly the number of starts kept. peak_thr, oracle and estK decoders share the val-tuned
minimum distance and smoothing.

Also per run: AP@5, a count-free score (all peaks pooled over the split, ranked by height,
greedily matched one-to-one to gold starts within +-5 frames), and recall by boundary type
(continuous / short <= 12 frames / long pause) for argmax, peak_thr and oracleK_video.

Decomposition per configuration (mean over seeds):
  count component      = F1_oracleK_video - F1_peak_thr
  placement component  = 1 - F1_oracleK_video

Inputs: evaluate_phrase.py --save-predictions on BOTH val and test for every run.
Outputs (evaluation_phrase/count_oracle/): rows_<split>.csv, summary_<split>.txt

  python count_oracle_decoding.py                         # the plan's default runs
  python count_oracle_decoding.py --runs multistream_bilstm-301 multistream_bilstm-302
  python count_oracle_decoding.py --selftest
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from phrase_stats import (PRED_ROOT, available_runs, config_name, f1_from_counts, gold_starts_of,
                          load_predictions)
from src.metrics import (B_TAG, O_TAG, classify_gold_boundaries, extract_segments, match_boundaries,
                         merge_close_starts, segment_f1_at, segment_iou, segment_percentage)

TOLS = (2, 5, 10)
THR_GRID = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
MIN_DIST_GRID = (10, 25, 50)
SIGMA_GRID = (0, 2, 5)
CHUNKS = (3000, 1500, 500)
DEFAULT_RUNS = ([f"multistream_bilstm-{i}" for i in range(208, 211)] +            # CE, best epoch by bnd5
                [f"bilstm_2023-{i}" for i in range(244, 250)] +                   # 2023 E1s / E2s re-impl
                [f"multistream_bilstm-{i}" for i in range(301, 310)])            # count loss, soft+count


# ============================================================================ decoding
def pick_peaks(p_b, k=None, thr=None, min_dist=25, sigma=0):
    """
    Peaks of P(B) (Gaussian-smoothed if sigma > 0) at least min_dist frames apart. k: keep the k
    highest; thr: keep peaks >= thr. Returns sorted frame indices. The track is padded with -1 at
    both ends so a start on the first or last frame can be a peak.
    """
    p = np.asarray(p_b, dtype=np.float64)
    if sigma and sigma > 0:
        p = gaussian_filter1d(p, sigma)
    padded = np.r_[-1.0, p, -1.0]
    idx, _ = find_peaks(padded, distance=max(1, int(min_dist)))
    idx = idx - 1
    idx = idx[(idx >= 0) & (idx < len(p))]
    h = p[idx]
    if thr is not None:
        keep = h >= thr
        idx, h = idx[keep], h[keep]
    if k is not None:
        k = int(max(0, k))
        if len(idx) > k:
            top = np.argsort(-h, kind="stable")[:k]
            idx = idx[top]
    return np.sort(idx).astype(np.int64)


def peaks_with_heights(p_b, min_dist, sigma):
    p = np.asarray(p_b, dtype=np.float64)
    if sigma and sigma > 0:
        p = gaussian_filter1d(p, sigma)
    idx = pick_peaks(p, min_dist=min_dist, sigma=0)
    return idx, p[idx]


def starts_to_segments(starts, probs, o_min=5):
    """
    Exactly len(starts) non-overlapping segments [(start, end)], ends inclusive. Each segment ends
    at the earlier of (next start - 1) and the last frame before the first run of >= o_min
    consecutive argmax-O frames after its start (or the video end).
    """
    starts = np.sort(np.asarray(starts, dtype=np.int64))
    T = probs.shape[1]
    is_o = probs.argmax(axis=0) == O_TAG
    run = np.zeros(T + 1, dtype=np.int64)            # length of the O run starting at t
    for t in range(T - 1, -1, -1):
        run[t] = run[t + 1] + 1 if is_o[t] else 0
    nxt = np.full(T + 1, T, dtype=np.int64)          # first t' >= t starting an O run >= o_min
    for t in range(T - 1, -1, -1):
        nxt[t] = t if run[t] >= o_min else nxt[t + 1]
    segs = []
    for j, s in enumerate(starts):
        s = int(min(max(s, 0), T - 1))
        lim = int(starts[j + 1]) - 1 if j + 1 < len(starts) else T - 1
        o_start = int(nxt[s + 1]) if s + 1 <= T else T  # an O run beginning AFTER the start frame
        end = min(lim, o_start - 1)
        segs.append((s, max(s, end)))
    return segs


def chunk_oracle_starts(idx, h, gold_starts, T, L):
    keep = []
    for c0 in range(0, T, L):
        c1 = c0 + L
        kc = int(np.sum((gold_starts >= c0) & (gold_starts < c1)))
        if kc == 0:
            continue
        m = (idx >= c0) & (idx < c1)
        ci, ch = idx[m], h[m]
        if len(ci) > kc:
            ci = ci[np.argsort(-ch, kind="stable")[:kc]]
        keep += list(ci)
    return np.sort(np.array(keep, dtype=np.int64))


def topk(idx, h, k):
    k = int(max(0, k))
    if len(idx) <= k:
        return np.sort(idx)
    return np.sort(idx[np.argsort(-h, kind="stable")[:k]])


# ============================================================================ scoring
def score(per_video, tols=TOLS, types=True):
    """
    per_video: {vid: (pred_starts, pred_segments, gold, probs_T)}. Boundary counts pooled over
    videos; IoU / % / Segment F1@0.5 averaged per video (as evaluate_videos); K ratio pooled.
    """
    tp = {k: 0 for k in tols}
    n_pred = n_gold = 0
    ious, pcts, sf1 = [], [], []
    by_type = defaultdict(lambda: [0, 0])
    for vid, (starts, segs, gold) in per_video.items():
        gsegs = extract_segments(gold)
        gs = np.array([s for s, _ in gsegs], dtype=np.int64)
        n_pred += len(starts)
        n_gold += len(gs)
        for k in tols:
            g_ok, _, _ = match_boundaries(gs, starts, k)
            tp[k] += int(g_ok.sum())
        T = len(gold)
        ious.append(segment_iou(segs, gsegs, T))
        pc = segment_percentage(segs, gsegs)
        if not np.isnan(pc):
            pcts.append(pc)
        sf1.append(segment_f1_at(segs, gsegs, 0.5))
        if types:
            g_ok, _, _ = match_boundaries(gs, starts, 5)
            found = dict(zip(gs.tolist(), g_ok.tolist()))
            for s, typ, _ in classify_gold_boundaries(gold, 12):
                by_type[typ][0] += int(found.get(s, False))
                by_type[typ][1] += 1
    out = {}
    for k in tols:
        p = tp[k] / n_pred if n_pred else 0.0
        r = tp[k] / n_gold if n_gold else 0.0
        out[f"bnd_p_{k}"], out[f"bnd_r_{k}"] = round(p, 4), round(r, 4)
        out[f"bnd_f1_{k}"] = round(f1_from_counts(tp[k], n_pred, n_gold), 4)
    out.update({"segment_f1_05": round(float(np.mean(sf1)), 4), "iou": round(float(np.mean(ious)), 4),
                "pct": round(float(np.mean(pcts)), 4) if pcts else float("nan"),
                "k_pred": n_pred, "k_gold": n_gold, "k_ratio": round(n_pred / n_gold, 4) if n_gold else float("nan")})
    for typ in ("continuous", "short_pause", "long_pause"):
        f, n = by_type.get(typ, (0, 0))
        out[f"recall_{typ}"] = round(f / n, 4) if n else float("nan")
        out[f"n_{typ}"] = n
    return out


def bnd_f1_counts(starts_by_vid, gold_starts_by_vid, k=5):
    tp = n_pred = n_gold = 0
    for v, st in starts_by_vid.items():
        gs = gold_starts_by_vid[v]
        g_ok, _, _ = match_boundaries(gs, st, k)
        tp += int(g_ok.sum())
        n_pred += len(st)
        n_gold += len(gs)
    return f1_from_counts(tp, n_pred, n_gold)


def average_precision(peaks_by_vid, gold_starts_by_vid, k=5):
    """Count-free AP@k: all peaks pooled, ranked by height, greedy one-to-one matching."""
    allp = [(h, v, int(t)) for v, (idx, hs) in peaks_by_vid.items() for t, h in zip(idx, hs)]
    allp.sort(key=lambda x: -x[0])
    n_gold = sum(len(g) for g in gold_starts_by_vid.values())
    if n_gold == 0 or not allp:
        return float("nan")
    used = {v: np.zeros(len(g), dtype=bool) for v, g in gold_starts_by_vid.items()}
    tp, ap, prev_r = 0, 0.0, 0.0
    for i, (_, v, t) in enumerate(allp, start=1):
        gs = gold_starts_by_vid[v]
        cand = np.where((~used[v]) & (np.abs(gs - t) <= k))[0]
        if len(cand):
            j = cand[np.argmin(np.abs(gs[cand] - t))]
            used[v][j] = True
            tp += 1
            r = tp / n_gold
            ap += (r - prev_r) * (tp / i)
            prev_r = r
    return round(ap, 4)


# ============================================================================ per run
def train_mean_phrase_len(labels_dir, split_file):
    import re
    with open(split_file) as f:
        vids = [re.sub(r"\.(npy|pt)$", "", v) for v in json.load(f)["train"]]
    lens = []
    for v in vids:
        p = os.path.join(labels_dir, f"{v}.npy")
        if os.path.exists(p):
            lens += [e - s + 1 for s, e in extract_segments(np.load(p))]
    if not lens:
        raise RuntimeError(f"no train phrases found in {labels_dir}")
    return float(np.mean(lens))


def tune_peaks_on_val(val):
    """Grid over (min_dist, sigma, thr) on val by pooled boundary F1@5."""
    probs, _, gold = val
    gstarts = {v: gold_starts_of(gold[v]) for v in probs}
    best = None
    for md in MIN_DIST_GRID:
        for sg in SIGMA_GRID:
            pk = {v: peaks_with_heights(probs[v][B_TAG], md, sg) for v in probs}
            for thr in THR_GRID:
                st = {v: idx[h >= thr] for v, (idx, h) in pk.items()}
                f1 = bnd_f1_counts(st, gstarts, 5)
                if best is None or f1 > best[0]:
                    best = (f1, md, sg, thr)
    return {"val_bnd_f1_5": round(best[0], 4), "min_dist": best[1], "sigma": best[2], "thr": best[3]}


def fit_mass_ratio(val, min_dist, sigma):
    probs, _, gold = val
    k_gold = sum(len(gold_starts_of(gold[v])) for v in probs)
    mass = sum(float(probs[v][B_TAG].sum()) for v in probs)
    return k_gold / mass if mass > 0 else 1.0


def analyse_run(run, split, pred_root, mean_len):
    test = load_predictions(pred_root, run, split)
    val = load_predictions(pred_root, run, "val")
    probs, meta, gold = test
    tuned = tune_peaks_on_val(val)
    md, sg, thr = tuned["min_dist"], tuned["sigma"], tuned["thr"]
    c_mass = fit_mass_ratio(val, md, sg)
    merge_d = (meta.get("merge_starts") or {}).get("argmax_merge")

    dec = defaultdict(dict)
    peaks, gstarts = {}, {}
    for v in probs:
        p, g = probs[v], gold[v]
        gs = gold_starts_of(g)
        gstarts[v] = gs
        segs = extract_segments(p.argmax(axis=0))
        dec["argmax"][v] = (np.array([s for s, _ in segs], dtype=np.int64), segs, g)
        if merge_d:
            ms = merge_close_starts(segs, p[B_TAG], int(merge_d))
            dec["argmax_merge"][v] = (np.array([s for s, _ in ms], dtype=np.int64), ms, g)
        idx, h = peaks_with_heights(p[B_TAG], md, sg)
        peaks[v] = (idx, h)

        def add(name, st):
            st = np.sort(np.asarray(st, dtype=np.int64))
            dec[name][v] = (st, starts_to_segments(st, p), g)

        add("peak_thr", idx[h >= thr])
        add("oracleK_video", topk(idx, h, len(gs)))
        for L in CHUNKS:
            add(f"oracleK_chunk_{L}", chunk_oracle_starts(idx, h, gs, p.shape[1], L))
        add("estK_duration", topk(idx, h, round(int((p.argmax(axis=0) != O_TAG).sum()) / mean_len)))
        add("estK_mass", topk(idx, h, round(c_mass * float(p[B_TAG].sum()))))

    ap5 = average_precision(peaks, gstarts, 5)
    rows = []
    for name, pv in dec.items():
        r = {"run": run, "config": config_name(meta.get("description", run)), "seed": meta.get("seed", ""),
             "split": split, "decoder": name}
        r.update(score(pv))
        r.update({"ap_5": ap5, "tuned_min_dist": md, "tuned_sigma": sg, "tuned_thr": thr,
                  "val_bnd_f1_5_peak": tuned["val_bnd_f1_5"], "mass_ratio_c": round(c_mass, 4),
                  "merge_d": merge_d if merge_d else ""})
        rows.append(r)
    return rows


# ============================================================================ summary
SUMMARY_DECODERS = ["argmax", "argmax_merge", "peak_thr", "oracleK_video", "oracleK_chunk_1500",
                    "estK_duration", "estK_mass"]


def summarise(rows, split):
    by_cfg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cfg[r["config"]][r["decoder"]].append(r)
    lines = [f"Count vs placement ({split}). Boundary F1@5, mean +- SD over seeds; AP@5 is count-free.",
             "count component = F1(oracleK_video) - F1(peak_thr); placement component = 1 - F1(oracleK_video)", ""]
    cfg_stats = {}
    for cfg, d in by_cfg.items():
        def ms(dec, key="bnd_f1_5"):
            v = [x[key] for x in d.get(dec, []) if isinstance(x.get(key), (int, float)) and not np.isnan(x[key])]
            return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)
        n = ms("argmax")[2]
        lines.append(f"{cfg}  (n={n})")
        parts = []
        for dec in SUMMARY_DECODERS:
            m, s, k = ms(dec)
            if k:
                parts.append(f"{dec} {m:.3f}+-{s:.3f}")
        lines.append("  F1@5: " + " | ".join(parts))
        ap = ms("argmax", "ap_5")
        kr = ms("peak_thr", "k_ratio")
        lines.append(f"  AP@5 {ap[0]:.3f}+-{ap[1]:.3f} | K_pred/K_gold: argmax {ms('argmax', 'k_ratio')[0]:.2f}, "
                     f"peak_thr {kr[0]:.2f}, estK_duration {ms('estK_duration', 'k_ratio')[0]:.2f}, "
                     f"estK_mass {ms('estK_mass', 'k_ratio')[0]:.2f}")
        f_thr, f_ok = ms("peak_thr")[0], ms("oracleK_video")[0]
        count_c, place_c = f_ok - f_thr, 1 - f_ok
        lines.append(f"  decomposition: count {count_c:+.3f} | placement {place_c:.3f}")
        rec = [f"{t} {ms('oracleK_video', f'recall_{t}')[0]:.2f} (argmax {ms('argmax', f'recall_{t}')[0]:.2f})"
               for t in ("continuous", "short_pause", "long_pause")]
        lines.append("  recall@5 under oracle K: " + ", ".join(rec))
        lines.append("")
        cfg_stats[cfg] = {"count": count_c, "placement": place_c,
                          "cont_oracle": ms("oracleK_video", "recall_continuous")[0]}

    # interpretation (thresholds are stated so the reading is reproducible)
    lines.append("Interpretation (count gain > 0.05 = 'substantial', < 0.03 = 'small'; continuous recall < 0.2 = 'rarely found'):")
    for cfg, s in cfg_stats.items():
        tag = "count-loss" if "count loss" in cfg.lower() or "cnt" in cfg.lower() else "CE/other"
        verdict = ("count dominates" if s["count"] > 0.05 else
                   "placement dominates" if s["count"] < 0.03 else "mixed")
        lines.append(f"  [{tag}] {cfg}: oracle-K gain {s['count']:+.3f} -> {verdict}; "
                     f"continuous recall under oracle K {s['cont_oracle']:.2f}"
                     + (" (H-C3: rarely found)" if s["cont_oracle"] < 0.2 else ""))
    return "\n".join(lines) + "\n"


# ============================================================================ self test
def selftest():
    rng = np.random.default_rng(0)
    T = 3000
    gold = np.zeros(T, dtype=np.int64)
    s = 0
    starts = []
    while s < T - 80:
        L = int(rng.integers(40, 150))
        gold[s:s + L] = 1
        gold[s] = 2
        starts.append(s)
        gap = int(rng.choice([0, 3, 20, 60]))
        s += L + gap
    probs = np.zeros((3, T), dtype=np.float32)
    probs[gold, np.arange(T)] = 1.0
    gs = gold_starts_of(gold)
    assert list(gs) == starts, "gold starts"
    # 1. pick_peaks on a one-hot B track returns exactly the gold starts (min_dist 1 so close starts survive)
    pk = pick_peaks(probs[B_TAG], min_dist=1)
    assert list(pk) == starts, f"pick_peaks: {pk[:5]} vs {starts[:5]}"
    # 2. oracle-K decoding of one-hot probabilities: BndF1@0 = 1 and Segment F1 = 1
    idx, h = peaks_with_heights(probs[B_TAG], 1, 0)
    st = topk(idx, h, len(gs))
    segs = starts_to_segments(st, probs)
    res = score({"v": (st, segs, gold)}, tols=(0,), types=False)
    assert res["bnd_f1_0"] == 1.0 and res["segment_f1_05"] == 1.0, res
    # 3. starts_to_segments: len(starts) segments, ordered, non-overlapping, end >= start
    rand_starts = np.sort(rng.choice(T, 40, replace=False))
    segs = starts_to_segments(rand_starts, probs)
    assert len(segs) == len(rand_starts)
    assert all(a <= b for a, b in segs)
    assert all(segs[i][1] < segs[i + 1][0] for i in range(len(segs) - 1))
    # 4. a start on frame 0 is found
    assert pick_peaks(np.r_[1.0, np.zeros(10)], min_dist=1).tolist() == [0]
    # 5. AP of a perfect ranking is 1
    assert average_precision({"v": (idx, h)}, {"v": gs}, 5) == 1.0
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run names (default: the plan's runs)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "count_oracle"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    runs = args.runs or DEFAULT_RUNS
    have_test, have_val = set(available_runs(args.pred_root, args.split)), set(available_runs(args.pred_root, "val"))
    missing = [r for r in runs if r not in have_test or r not in have_val]
    if missing:
        print(f"skipping {len(missing)} runs without saved {args.split} AND val predictions: {missing}")
    runs = [r for r in runs if r not in missing]
    if not runs:
        sys.exit("no runs to analyse (evaluate_phrase.py --prefixes ... --split val/test --save-predictions)")
    os.makedirs(args.out_dir, exist_ok=True)
    mean_len = {}
    rows = []
    for run in runs:
        ldir = json.load(open(os.path.join(args.pred_root, run, f"{args.split}_meta.json")))["labels_dir"]
        if ldir not in mean_len:
            mean_len[ldir] = train_mean_phrase_len(ldir, args.split_file)
            print(f"mean train phrase length ({ldir}): {mean_len[ldir]:.1f} frames")
        print(f"{run} ...")
        rows += analyse_run(run, args.split, args.pred_root, mean_len[ldir])
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(args.out_dir, f"rows_{args.split}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    text = summarise(rows, args.split)
    with open(os.path.join(args.out_dir, f"summary_{args.split}.txt"), "w") as f:
        f.write(text)
    print(text)
    print(f"saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
