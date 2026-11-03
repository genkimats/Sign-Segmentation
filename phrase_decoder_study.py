"""
phrase_decoder_study.py -- decoder ideas for phrase segmentation, compared on the best phrase model.

Ideas 3 (snapping to sign onsets) and 4 (sign -> phrase decoding) from the brainstorm were dropped: no
decoder uses predicted signs. Everything runs on SAVED predictions (evaluate_phrase.py --save-predictions,
val AND test); no model is retrained. Every setting is tuned on VAL (pooled boundary F1 within +-5 frames)
and applied once to TEST.

Reference decoders
  argmax           plain argmax segments
  argmax_merge     + duplicate-start removal, distance D tuned on val
  peak_thr         peaks of smoothed P(B) above a threshold (count_oracle_decoding.py; current best)
  oracleK_video    peak_thr peaks, the true number of phrases kept (upper bound for count-aware decoding)
Ideas
  1 rescore        candidate rescoring: every P(B) peak is a candidate, a logistic regression on
                   peak features (height, prominence, video-relative height, spacing, pause evidence
                   from P(O)) decides; fitted on val with leave-one-video-out (val scores are out-of-fold)
  2 ENSEMBLE run   every decoder also on P averaged over the given runs (seeds; --ensemble-extra adds
                   other configurations)
  5 adapt_quant    signer-adaptive threshold: keep peaks above the video's own q-quantile of peak heights
    adapt_z        keep peaks whose height is z MADs above the video's median peak height
  6 tta variant    all decoders on predictions from predict_tta.py (overlapping windows, centre-weighted,
                   optional mirror averaging): --tta-root evaluation_phrase/predictions_tta
  7 mbr            expected-F1 decoding: peaks of the P(B) mass within +-5 frames, then per video the number
                   of starts that maximises expected boundary F1 (window "hit" probability 1 - exp(-mass);
                   expected gold count = alpha * c * sum P(B), c fitted on val; alpha and a floor tuned on val)
  8 end_aware      start score = P(B) + beta * (phrase-end evidence in the G frames before), peaks + threshold
  9 cal_peak_thr   temperature scaling fitted on val (NLL), then peak_thr / mbr on calibrated P
    cal_mbr

Outputs (evaluation_phrase/decoder_study/): rows.csv (variant x run x decoder, val + test metrics) and
summary.txt (test mean +- SD over seeds, the ensemble, and a paired bootstrap of each decoder vs peak_thr).

  python phrase_decoder_study.py --runs multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117
  python phrase_decoder_study.py --selftest
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks, peak_prominences

from count_oracle_decoding import (THR_GRID, bnd_f1_counts, peaks_with_heights, score, starts_to_segments,
                                   topk, tune_peaks_on_val)
from phrase_stats import (PRED_ROOT, boundary_counts, f1_from_counts, gold_starts_of, load_predictions,
                          paired_bootstrap)
from src.metrics import B_TAG, I_TAG, O_TAG, extract_segments, merge_close_starts

DEFAULT_RUNS = [f"multistream_bilstm-{i}" for i in (115, 116, 117)]   # Body + HaMeR + Face(compact), Frame F1 .689
K = 5                                                                    # tolerance of the tuning objective
MERGE_GRID = (0, 3, 5, 10, 15, 20, 30)
QUANT_GRID = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
Z_GRID = [round(x, 2) for x in np.arange(-1.0, 4.01, 0.25)]
BETA_GRID = (0.1, 0.25, 0.5, 1.0)
GAP_GRID = (10, 25, 50)
TEMP_GRID = [round(x, 2) for x in np.arange(0.5, 3.01, 0.1)]
EPS = 1e-6


# ============================================================================ data
class Split:
    """probs {vid: (3, T)}, gold {vid: (T,)}, gs {vid: gold starts}."""

    def __init__(self, probs, gold, meta=None):
        self.probs, self.gold, self.meta = probs, gold, meta or {}
        self.vids = sorted(probs)
        self.gs = {v: gold_starts_of(gold[v]) for v in self.vids}


def ensemble(splits):
    vids = sorted(set.intersection(*[set(s.vids) for s in splits]))
    probs, gold = {}, {}
    for v in vids:
        T = min(s.probs[v].shape[1] for s in splits)
        probs[v] = np.mean([s.probs[v][:, :T] for s in splits], axis=0)
        gold[v] = splits[0].gold[v][:T]
    return Split(probs, gold, {"description": "ENSEMBLE"})


# ============================================================================ helpers
def segs_from_starts(starts_by_vid, data):
    return {v: starts_to_segments(st, data.probs[v]) for v, st in starts_by_vid.items()}


def segs_to_bio(segs, T):
    lab = np.zeros(T, dtype=np.int64)
    for s, e in segs:
        lab[s:e + 1] = I_TAG
    for s, _ in segs:
        lab[s] = B_TAG
    return lab


def decoded_frame_f1(segs_by_vid, data, labels_by_vid=None):
    """Frame F1 of the DECODED labelling (B at each start, I to the segment end, O elsewhere), pooled over
    videos, macro over O / I / B -- the same definition as evaluate_videos, which scores the argmax labels."""
    from sklearn.metrics import f1_score
    y = np.concatenate([data.gold[v] for v in data.vids])
    yp = np.concatenate([labels_by_vid[v] if labels_by_vid is not None else segs_to_bio(segs_by_vid[v], len(data.gold[v]))
                         for v in data.vids])
    per = f1_score(y, yp, labels=[0, 1, 2], average=None, zero_division=0)
    return {"frame_f1": round(float(per.mean()), 4), "f1_O": round(float(per[0]), 4),
            "f1_I": round(float(per[1]), 4), "f1_B": round(float(per[2]), 4)}


def evaluate(starts_by_vid, data, segs_by_vid=None):
    segs = segs_by_vid or segs_from_starts(starts_by_vid, data)
    m = score({v: (np.asarray(starts_by_vid[v], dtype=np.int64), segs[v], data.gold[v]) for v in data.vids})
    m.update(decoded_frame_f1(segs, data))
    counts = {v: boundary_counts(data.gs[v], np.asarray(starts_by_vid[v], dtype=np.int64), K) for v in data.vids}
    return m, counts


def f1_of(starts_by_vid, data):
    return bnd_f1_counts(starts_by_vid, data.gs, K)


def grid_best(data, cand_fn, grid):
    """cand_fn(params) -> {vid: starts}; returns (best f1, best params)."""
    best = (-1.0, None)
    for p in grid:
        f = f1_of(cand_fn(p), data)
        if f > best[0]:
            best = (f, p)
    return best


def smooth(p, sigma):
    return gaussian_filter1d(p, sigma) if sigma and sigma > 0 else p


def peaks_of(track, md):
    padded = np.r_[-1.0, track, -1.0]
    idx, _ = find_peaks(padded, distance=max(1, int(md)))
    idx = idx - 1
    return idx[(idx >= 0) & (idx < len(track))]


def end_evidence(p):
    """P(a phrase ends at t-1) ~ P(not O at t-1) * P(O at t)."""
    e = np.zeros(p.shape[1])
    e[1:] = (1.0 - p[O_TAG, :-1]) * p[O_TAG, 1:]
    return e


def prev_max(x, G):
    """max of x over [t-G, t] (window clipped at the video start)."""
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.r_[np.full(G, -np.inf), np.asarray(x, dtype=np.float64)]
    out = sliding_window_view(padded, G + 1).max(axis=1)
    return np.maximum(out, 0.0) if np.all(np.asarray(x) >= 0) else out


# ============================================================================ decoders
# Each decoder: fit(val) -> params ; decode(data, params) -> {vid: starts}.

def dec_argmax(data, _):
    return {v: np.array([s for s, _ in extract_segments(data.probs[v].argmax(0))], dtype=np.int64) for v in data.vids}


def segs_argmax(data, _):
    return {v: extract_segments(data.probs[v].argmax(0)) for v in data.vids}


def fit_merge(val):
    def run(d):
        return {v: np.array([s for s, _ in merge_close_starts(extract_segments(val.probs[v].argmax(0)),
                                                              val.probs[v][B_TAG], d)]) for v in val.vids}
    return {"d": grid_best(val, run, MERGE_GRID)[1]}


def segs_merge(data, p):
    return {v: merge_close_starts(extract_segments(data.probs[v].argmax(0)), data.probs[v][B_TAG], p["d"])
            for v in data.vids}


def dec_merge(data, p):
    return {v: np.array([s for s, _ in sg], dtype=np.int64) for v, sg in segs_merge(data, p).items()}


def fit_peak(val):
    t = tune_peaks_on_val((val.probs, None, val.gold))
    return {"md": t["min_dist"], "sg": t["sigma"], "thr": t["thr"]}


def fit_peak_wide(val):
    """peak_thr with a threshold grid that also covers the val peak-height quantiles (for calibrated P,
    whose peaks can sit below the fixed grid's 0.05)."""
    from count_oracle_decoding import MIN_DIST_GRID, SIGMA_GRID
    best = (-1.0, None)
    for md in MIN_DIST_GRID:
        for sg in SIGMA_GRID:
            pk = {v: peaks_with_heights(val.probs[v][B_TAG], md, sg) for v in val.vids}
            hs = np.concatenate([h for _, h in pk.values()]) if pk else np.zeros(1)
            grid = sorted(set(THR_GRID) | set(np.round(np.quantile(hs, np.linspace(0.05, 0.99, 40)), 5).tolist()))
            for thr in grid:
                f = f1_of({v: idx[h >= thr] for v, (idx, h) in pk.items()}, val)
                if f > best[0]:
                    best = (f, {"md": md, "sg": sg, "thr": float(thr)})
    return best[1]


def peak_cache(data, p):
    return {v: peaks_with_heights(data.probs[v][B_TAG], p["md"], p["sg"]) for v in data.vids}


def dec_peak(data, p):
    return {v: idx[h >= p["thr"]] for v, (idx, h) in peak_cache(data, p).items()}


def dec_oracle_k(data, p):
    return {v: topk(idx, h, len(data.gs[v])) for v, (idx, h) in peak_cache(data, p).items()}


# --- idea 5: signer-adaptive thresholds
def fit_quant(val, base):
    pk = peak_cache(val, base)
    def run(q):
        return {v: idx[h >= (np.quantile(h, q) if len(h) else 1)] for v, (idx, h) in pk.items()}
    return {**base, "q": grid_best(val, run, QUANT_GRID)[1]}


def dec_quant(data, p):
    return {v: idx[h >= (np.quantile(h, p["q"]) if len(h) else 1)] for v, (idx, h) in peak_cache(data, p).items()}


def zscores(h):
    if len(h) == 0:
        return h
    med = np.median(h)
    mad = 1.4826 * np.median(np.abs(h - med)) + EPS
    return (h - med) / mad


def fit_z(val, base):
    pk = {v: (idx, zscores(h)) for v, (idx, h) in peak_cache(val, base).items()}
    def run(zt):
        return {v: idx[z >= zt] for v, (idx, z) in pk.items()}
    return {**base, "z": grid_best(val, run, Z_GRID)[1]}


def dec_z(data, p):
    return {v: idx[zscores(h) >= p["z"]] for v, (idx, h) in peak_cache(data, p).items()}


# --- idea 7: expected-F1 (MBR) decoding
def mbr_starts(pb, c, alpha, floor=0.0):
    mass = uniform_filter1d(pb, size=2 * K + 1, mode="constant") * (2 * K + 1)
    idx = peaks_of(mass, 2 * K + 1)
    m = 1.0 - np.exp(-mass[idx])             # P(at least one start in the +-K window), Poisson view
    keep = m >= floor
    idx, m = idx[keep], m[keep]
    if len(idx) == 0:
        return idx
    order = np.argsort(-m, kind="stable")
    exp_gold = alpha * c * pb.sum()
    cum = np.cumsum(m[order])
    k = np.arange(1, len(order) + 1)
    ef1 = 2 * cum / (k + exp_gold)
    kbest = int(np.argmax(ef1)) + 1
    return np.sort(idx[order[:kbest]])


def fit_mbr(val):
    k_gold = sum(len(val.gs[v]) for v in val.vids)
    mass = sum(float(val.probs[v][B_TAG].sum()) for v in val.vids)
    c = k_gold / max(mass, EPS)
    def run(p):
        return {v: mbr_starts(val.probs[v][B_TAG].astype(np.float64), c, *p) for v in val.vids}
    grid = [(a, fl) for a in (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5) for fl in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)]
    a, fl = grid_best(val, run, grid)[1]
    return {"c": c, "alpha": a, "floor": fl}


def dec_mbr(data, p):
    return {v: mbr_starts(data.probs[v][B_TAG].astype(np.float64), p["c"], p["alpha"], p["floor"]) for v in data.vids}


# --- idea 8: phrase-end evidence
def end_track(prob, sg, beta, G):
    return smooth(prob[B_TAG].astype(np.float64), sg) + beta * prev_max(end_evidence(prob), G)


def fit_end(val, base):
    best = (-1.0, None)
    for beta in BETA_GRID:
        for G in GAP_GRID:
            pk = {}
            for v in val.vids:
                tr = end_track(val.probs[v], base["sg"], beta, G)
                idx = peaks_of(tr, base["md"])
                pk[v] = (idx, tr[idx])
            for thr in THR_GRID + [1.1, 1.25, 1.5]:
                f = f1_of({v: idx[h >= thr] for v, (idx, h) in pk.items()}, val)
                if f > best[0]:
                    best = (f, {**base, "beta": beta, "G": G, "thr": thr})
    return best[1]


def dec_end(data, p):
    out = {}
    for v in data.vids:
        tr = end_track(data.probs[v], p["sg"], p["beta"], p["G"])
        idx = peaks_of(tr, p["md"])
        out[v] = idx[tr[idx] >= p["thr"]]
    return out


# --- idea 9: temperature scaling
def fit_temperature(val):
    lp = np.concatenate([np.log(np.clip(val.probs[v], EPS, 1)).T for v in val.vids])
    y = np.concatenate([val.gold[v] for v in val.vids])
    best = (np.inf, 1.0)
    for T in TEMP_GRID:
        z = lp / T
        z = z - z.max(1, keepdims=True)
        nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
        if nll < best[0]:
            best = (nll, T)
    return best[1]


def calibrate(data, T):
    probs = {}
    for v in data.vids:
        z = np.log(np.clip(data.probs[v].astype(np.float64), EPS, 1)) / T
        z = z - z.max(0, keepdims=True)
        e = np.exp(z)
        probs[v] = (e / e.sum(0, keepdims=True)).astype(np.float32)
    return Split(probs, data.gold, data.meta)


# --- idea 1: candidate rescoring
MODEL_FEATS = ["h", "log_h", "prom", "h_z_video", "h_rank_video", "log_gap_prev_cand", "log_gap_next_cand",
               "pO_before", "pO_prev_frame", "end_ev_before"]
def candidate_table(data, v, md, sg):
    prob = data.probs[v].astype(np.float64)
    T = prob.shape[1]
    tr = smooth(prob[B_TAG], sg)
    idx = peaks_of(tr, md)
    idx = idx[tr[idx] >= 0.02]
    if len(idx) == 0:
        return idx, np.zeros((0, len(MODEL_FEATS)))
    h = tr[idx]
    prom = peak_prominences(np.r_[0, tr, 0], idx + 1)[0]
    rank = np.argsort(np.argsort(h)) / max(len(h) - 1, 1)
    gp = np.diff(np.r_[-1000, idx])
    gn = np.diff(np.r_[idx, T + 1000])
    pO = prob[O_TAG]
    endev = prev_max(end_evidence(prob), 25)
    before = np.array([pO[max(0, t - 25):t].mean() if t > 0 else 1.0 for t in idx])
    rows = [h, np.log(h + EPS), prom, zscores(h), rank, np.log1p(gp), np.log1p(gn), before,
            pO[np.maximum(idx - 1, 0)], endev[idx]]
    return idx, np.stack(rows, axis=1)


def fit_rescore(val, base, C=0.3):
    """Leave-one-video-out on val: out-of-fold probabilities -> threshold; final model on all val."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.metrics import match_boundaries
    tabs = {}
    for v in val.vids:
        idx, X = candidate_table(val, v, base["md"], base["sg"])
        _, p_ok, _ = match_boundaries(val.gs[v], idx, K)
        tabs[v] = (idx, X, p_ok.astype(int))
    oof = {}
    for v in val.vids:
        Xtr = np.concatenate([tabs[u][1] for u in val.vids if u != v])
        ytr = np.concatenate([tabs[u][2] for u in val.vids if u != v])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(Xtr, ytr)
        oof[v] = clf.predict_proba(tabs[v][1])[:, 1] if len(tabs[v][1]) else np.zeros(0)
    best = (-1.0, 0.5)
    for thr in THR_GRID:
        f = f1_of({v: tabs[v][0][oof[v] >= thr] for v in val.vids}, val)
        if f > best[0]:
            best = (f, thr)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    clf.fit(np.concatenate([tabs[u][1] for u in val.vids]), np.concatenate([tabs[u][2] for u in val.vids]))
    coef = dict(zip(MODEL_FEATS, np.round(clf[-1].coef_[0], 3).tolist()))
    val_starts = {v: tabs[v][0][oof[v] >= best[1]] for v in val.vids}
    return {**base, "thr": best[1], "clf": clf, "coef": coef, "val_starts": val_starts}


def dec_rescore(data, p):
    out = {}
    for v in data.vids:
        idx, X = candidate_table(data, v, p["md"], p["sg"])
        out[v] = idx[p["clf"].predict_proba(X)[:, 1] >= p["thr"]] if len(idx) else idx
    return out


# ============================================================================ one run
def analyse(val, test):
    """All decoders for one run (or the ensemble). Returns {decoder: (val_metrics, test_metrics, test_counts, params)}."""
    res = {}

    def add(name, params, decode, segs_fn=None, val_starts=None):
        vs = val_starts if val_starts is not None else decode(val, params)
        ts = decode(test, params)
        vm, _ = evaluate(vs, val, segs_fn(val, params) if segs_fn and val_starts is None else None)
        tm, tc = evaluate(ts, test, segs_fn(test, params) if segs_fn else None)
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()
                 if k not in ("clf", "val_starts")} if isinstance(params, dict) else params
        res[name] = (vm, tm, tc, shown)

    add("argmax", {}, dec_argmax, segs_argmax)
    for d, m in ((val, res["argmax"][0]), (test, res["argmax"][1])):   # the usual Frame F1: raw argmax labels
        m.update(decoded_frame_f1(None, d, {v: d.probs[v].argmax(0) for v in d.vids}))
    pm = fit_merge(val)
    add("argmax_merge", pm, dec_merge, segs_merge)
    base = fit_peak(val)
    add("peak_thr", base, dec_peak)
    add("oracleK_video", base, dec_oracle_k)
    add("adapt_quant", fit_quant(val, base), dec_quant)
    add("adapt_z", fit_z(val, base), dec_z)
    add("mbr", fit_mbr(val), dec_mbr)
    add("end_aware", fit_end(val, base), dec_end)
    T = fit_temperature(val)
    cval, ctest = calibrate(val, T), calibrate(test, T)
    cbase = fit_peak_wide(cval)
    vm, _ = evaluate(dec_peak(cval, cbase), cval)
    tm, tc = evaluate(dec_peak(ctest, cbase), ctest)
    res["cal_peak_thr"] = (vm, tm, tc, {**cbase, "T": T})
    cm = fit_mbr(cval)
    vm, _ = evaluate(dec_mbr(cval, cm), cval)
    tm, tc = evaluate(dec_mbr(ctest, cm), ctest)
    res["cal_mbr"] = (vm, tm, tc, {**cm, "T": T})
    pr = fit_rescore(val, base)
    add("rescore", pr, dec_rescore, val_starts=pr["val_starts"])
    return res


# ============================================================================ main
def load_split(pred_root, run, split):
    probs, meta, gold = load_predictions(pred_root, run, split)
    return Split(probs, gold, meta)


ORDER = ["argmax", "argmax_merge", "peak_thr", "adapt_quant", "adapt_z", "mbr", "end_aware", "cal_peak_thr",
         "cal_mbr", "rescore", "oracleK_video"]
IDEA = {"rescore": "1", "adapt_quant": "5", "adapt_z": "5", "mbr": "7", "end_aware": "8", "cal_peak_thr": "9",
        "cal_mbr": "9"}
METRICS = ["frame_f1", "f1_O", "f1_I", "f1_B", "bnd_f1_2", "bnd_f1_5", "bnd_f1_10", "bnd_p_5", "bnd_r_5", "iou", "pct", "k_ratio",
           "recall_short_pause", "recall_long_pause"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="seeds of the best phrase model")
    ap.add_argument("--ensemble-extra", nargs="*", default=[], help="other runs added to a cross-config ensemble")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--tta-root", default=os.path.join("evaluation_phrase", "predictions_tta"),
                    help="predictions from predict_tta.py (variant 'tta'; skipped if missing)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "decoder_study"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    os.makedirs(args.out_dir, exist_ok=True)

    variants = {"base": args.pred_root}
    if all(os.path.exists(os.path.join(args.tta_root, r, f"{s}.npz")) for r in args.runs for s in ("val", "test")):
        variants["tta"] = args.tta_root
    else:
        print(f"(no complete TTA predictions in {args.tta_root}: idea 6 skipped -- run predict_tta.py)")

    rows, results = [], {}
    for vname, root in variants.items():
        splits = {}
        for r in args.runs + args.ensemble_extra:
            src = root if r in args.runs else args.pred_root
            splits[r] = (load_split(src, r, "val"), load_split(src, r, "test"))
        targets = {r: splits[r] for r in args.runs}
        targets["ENSEMBLE(seeds)"] = (ensemble([splits[r][0] for r in args.runs]),
                                      ensemble([splits[r][1] for r in args.runs]))
        if args.ensemble_extra:
            allr = args.runs + args.ensemble_extra
            targets["ENSEMBLE(+extra)"] = (ensemble([splits[r][0] for r in allr]), ensemble([splits[r][1] for r in allr]))
        for name, (val, test) in targets.items():
            print(f"[{vname}] {name}: {len(val.vids)} val / {len(test.vids)} test videos")
            res = analyse(val, test)
            results[(vname, name)] = res
            for dec, (vm, tm, _, params) in res.items():
                rows.append({"variant": vname, "run": name, "decoder": dec, "idea": IDEA.get(dec, "ref"),
                             "val_bnd_f1_5": vm["bnd_f1_5"], **{f"test_{m}": tm.get(m) for m in METRICS},
                             "params": json.dumps(params, default=str)})

    with open(os.path.join(args.out_dir, "rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"Decoder study on {', '.join(args.runs)} (test; every setting tuned on val by boundary F1@5).",
             "Columns: boundary F1 at +-2/+-5/+-10, recall of short / long-pause boundaries, predicted/gold count,",
             "IoU, val F1@5. 'seeds' = mean +- SD over the runs; ENS = the same decoder on the seed ensemble.",
             "FrameF1 / F1_B = frame-level macro F1 and B-class F1 of the DECODED labels (argmax row = the usual Frame F1",
             "on the raw argmax labels; other rows: B at each decoded start, I to the segment end, O elsewhere).", ""]
    for vname in variants:
        lines.append(f"=== variant: {vname} ===")
        lines.append(f"{'decoder':15s} {'idea':4s} {'FrameF1':>7s} {'F1_B':>6s} {'F1@2':>12s} {'F1@5':>13s} {'F1@10':>12s} {'short':>6s} {'long':>6s} "
                     f"{'K/gold':>6s} {'IoU':>6s} {'valF1@5':>7s} | {'ENS F1@5':>8s}  vs peak_thr (seeds, paired): diff [95% CI] P<=0")
        ref_counts = [results[(vname, r)]["peak_thr"][2] for r in args.runs]
        for dec in ORDER:
            per = [results[(vname, r)][dec] for r in args.runs if dec in results[(vname, r)]]
            if not per:
                continue
            def ms(m):
                x = np.array([p[1][m] for p in per], dtype=float)
                return x.mean(), x.std()
            f2, f5, f10 = ms("bnd_f1_2"), ms("bnd_f1_5"), ms("bnd_f1_10")
            ens = results[(vname, "ENSEMBLE(seeds)")].get(dec)
            boot = ""
            if dec != "peak_thr":
                b = paired_bootstrap([p[2] for p in per], ref_counts, n_boot=args.n_boot)
                boot = f"{b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] {b['p_le_0']:.3f}"
            lines.append(f"{dec:15s} {IDEA.get(dec, 'ref'):4s} {ms('frame_f1')[0]:7.3f} {ms('f1_B')[0]:6.3f} {f2[0]:.3f}+-{f2[1]:.3f} {f5[0]:.3f}+-{f5[1]:.3f} "
                         f"{f10[0]:.3f}+-{f10[1]:.3f} {ms('recall_short_pause')[0]:6.3f} {ms('recall_long_pause')[0]:6.3f} "
                         f"{ms('k_ratio')[0]:6.2f} {ms('iou')[0]:6.3f} {np.mean([p[0]['bnd_f1_5'] for p in per]):7.3f} | "
                         f"{(ens[1]['bnd_f1_5'] if ens else float('nan')):8.3f}  {boot}")
        if "ENSEMBLE(+extra)" in {k[1] for k in results if k[0] == vname}:
            ex = results[(vname, "ENSEMBLE(+extra)")]
            lines.append("  cross-config ensemble F1@5: " + ", ".join(f"{d} {ex[d][1]['bnd_f1_5']:.3f}" for d in ORDER if d in ex))
        c = results[(vname, args.runs[0])].get("rescore")
        if c:
            lines.append(f"  rescore coefficients (run {args.runs[0]}): {c[3].get('coef')}")
        lines.append("")
    if "tta" in variants:
        b = paired_bootstrap([results[("tta", r)]["peak_thr"][2] for r in args.runs],
                             [results[("base", r)]["peak_thr"][2] for r in args.runs], n_boot=args.n_boot)
        lines.append(f"idea 6 (TTA predictions vs base), peak_thr F1@5: {b['diff']:+.3f} "
                     f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] P<=0 {b['p_le_0']:.3f}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {args.out_dir}/rows.csv and summary.txt")


# ============================================================================ selftest
def _synthetic(rng, n_vid, prefix, noise):
    probs, gold = {}, {}
    for i in range(n_vid):
        T = int(rng.integers(3000, 5000))
        g = np.zeros(T, dtype=np.int64)
        ons, t = [], 30
        while t < T - 60:
            d = int(rng.integers(8, 30))
            ons.append((t, t + d - 1))
            t += d + int(rng.choice([0, 2, 5, 10, 30, 60]))
        ons = np.array(ons)
        starts = ons[::4, 0]
        for a, s in enumerate(starts):
            e = starts[a + 1] - 1 if a + 1 < len(starts) else ons[-1, 1]
            g[s:e + 1] = I_TAG
            g[s] = B_TAG
        p = np.full((3, T), 0.02)
        p[O_TAG, g == O_TAG] = 0.9
        p[I_TAG, g != O_TAG] = 0.9
        for s in starts:                               # blurred, shifted B with stutter
            for off, amp in ((int(rng.integers(-3, 4)), 0.7), (int(rng.integers(5, 12)), 0.4)):
                if 0 <= s + off < T:
                    p[B_TAG, s + off] += amp
        p[B_TAG] = gaussian_filter1d(p[B_TAG], 2) + rng.random(T) * noise
        p = p / p.sum(0, keepdims=True)
        v = f"{prefix}{i}_A"
        probs[v], gold[v] = p.astype(np.float32), g
    return probs, gold


def selftest():
    rng = np.random.default_rng(0)
    val, test = Split(*_synthetic(rng, 6, "V", 0.08)), Split(*_synthetic(rng, 5, "S", 0.08))
    assert np.allclose(prev_max(np.array([0, 3, 0, 0, 0, 1, 0.]), 2), [0, 3, 3, 3, 0, 1, 1])
    res = analyse(val, test)
    for d in ORDER:
        assert d in res, d
    f = {d: res[d][1]["bnd_f1_5"] for d in ORDER}
    assert f["peak_thr"] > f["argmax"], f
    assert f["oracleK_video"] >= f["peak_thr"] - 0.05, f
    print("decoder F1@5 on synthetic test:", {d: round(x, 3) for d, x in f.items()})
    print("selftest passed")


if __name__ == "__main__":
    main()