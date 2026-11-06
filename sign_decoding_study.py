"""
sign_decoding_study.py -- phrase decoders that use the SIGN head's predictions (no retraining).

Needs, for every run, saved phrase predictions (evaluate_phrase.py --save-predictions, val + test)
and sign predictions (evaluate_sign_head.py, val + test). Every setting is tuned on VAL (pooled
boundary F1 within +-5 frames) and applied once to TEST.

  peak_thr          reference: peaks of smoothed phrase P(B) above a threshold (current best decoder)
  snap_product      idea 2a: peaks of  P_phrase(B) x max_{+-r} P_sign(B) ^ alpha   (alpha, r tuned)
  snap_move         idea 2b: peak_thr starts moved to the nearest predicted sign start within r frames
  snap_move_drop    idea 2c: as snap_move, but a start with no sign start within r frames is dropped
  posthoc_lr        idea 4 (post hoc): every predicted sign start is a candidate; a classifier on
  posthoc_gbm         phrase-head evidence + sign timing decides. Timing is relative to the VIDEO
                      (pause vs this signer's median pause, biggest pause within +-1 s, sign
                      durations vs the median), which a 512-frame window cannot see. Fitted on val
                      (leave-one-video-out scores tune the threshold and a minimum distance), applied
                      to test. lr = logistic regression, gbm = gradient-boosted trees
  posthoc_gbm_timing  the same with sign timing features only (no phrase head): how far predicted
                      sign timing alone goes (compare: gold sign timing, thesis_error_stats.py)

Also reported: the share of gold phrase starts within +-5 frames of a predicted sign start (the
recall ceiling of every candidate-based decoder) and sign-start F1. Paired bootstrap over videos
(seeds averaged) of each decoder vs peak_thr.

  python sign_decoding_study.py --runs multistream_bilstm-360 multistream_bilstm-361 multistream_bilstm-362
  python sign_decoding_study.py --selftest
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from count_oracle_decoding import score, starts_to_segments, tune_peaks_on_val, peaks_with_heights
from phrase_stats import (PRED_ROOT, boundary_counts, f1_from_counts, gold_starts_of, load_predictions,
                          paired_bootstrap)
from src.metrics import B_TAG, O_TAG, extract_segments, merge_close_starts

SIGN_ROOT = os.path.join("evaluation_phrase", "predictions_sign")
OUT_DIR = os.path.join("evaluation_phrase", "sign_decoding")
K = 5
ALPHA_GRID = (0.5, 1.0, 2.0)
R_GRID = (0, 2, 4)
MOVE_GRID = (2, 4, 8)
MD_GRID = (0, 5, 10)
TIMING = ["log_pause", "rel_pause", "local_max_pause", "prev_dur_rel", "cur_dur_rel", "signs_since_long_pause",
          "sign_b"]
PHRASE = ["ph_b_max", "ph_b_smooth", "ph_o_before", "ph_o_after"]


class Run:
    def __init__(self, phrase, sign, gold):
        self.phrase, self.sign, self.gold = phrase, sign, gold
        self.vids = sorted(set(phrase) & set(sign))
        self.gs = {v: gold_starts_of(gold[v]) for v in self.vids}


def load_run(pred_root, sign_root, run, split):
    phrase, _, gold = load_predictions(pred_root, run, split)
    z = np.load(os.path.join(sign_root, run, f"{split}.npz"))
    sign = {}
    for v in z.files:
        if v in phrase:
            T = min(phrase[v].shape[1], z[v].shape[1])
            phrase[v], gold[v], sign[v] = phrase[v][:, :T], gold[v][:T], z[v][:, :T].astype(np.float32)
    return Run(phrase, sign, gold)


# ------------------------------------------------------------------------- helpers
def f1_of(starts, data, k=K):
    tp = npred = ngold = 0
    for v in data.vids:
        a, b, c = boundary_counts(data.gs[v], starts[v], k)
        tp, npred, ngold = tp + a, npred + b, ngold + c
    return f1_from_counts(tp, npred, ngold)


def sign_cands(sign_p):
    segs = merge_close_starts(extract_segments(sign_p.argmax(axis=0)), sign_p[B_TAG], 3)
    return (np.array([s for s, _ in segs], dtype=np.int64), np.array([e for _, e in segs], dtype=np.int64))


def peak_starts(probs_b, p):
    idx, h = peaks_with_heights(probs_b, p["min_dist"], p["sigma"])
    return idx[h >= p["thr"]]


# ------------------------------------------------------------------------- decoders
def fit_peak(val):
    return tune_peaks_on_val(({v: val.phrase[v] for v in val.vids}, None, val.gold))


def dec_peak(data, p):
    return {v: peak_starts(data.phrase[v][B_TAG], p) for v in data.vids}


def product_probs(data, alpha, r):
    out = {}
    for v in data.vids:
        p = data.phrase[v].copy()
        sb = maximum_filter1d(data.sign[v][B_TAG], 2 * r + 1) if r else data.sign[v][B_TAG]
        p[B_TAG] = p[B_TAG] * np.clip(sb, 0, 1) ** alpha
        out[v] = p
    return out


def fit_product(val):
    best = None
    for a in ALPHA_GRID:
        for r in R_GRID:
            t = tune_peaks_on_val((product_probs(val, a, r), None, val.gold))
            if best is None or t["val_bnd_f1_5"] > best["val_bnd_f1_5"]:
                best = dict(t, alpha=a, r=r)
    return best


def dec_product(data, p):
    pp = product_probs(data, p["alpha"], p["r"])
    return {v: peak_starts(pp[v][B_TAG], p) for v in data.vids}


def move_to_signs(starts, cands, r, drop):
    if len(starts) == 0:
        return starts
    if len(cands) == 0:
        return starts[:0] if drop else starts
    d = np.abs(starts[:, None] - cands[None, :])
    j = d.argmin(axis=1)
    near = d[np.arange(len(starts)), j] <= r
    moved = np.where(near, cands[j], starts)
    moved = moved[near] if drop else moved
    return np.unique(moved)


def fit_move(val, base, drop):
    st = dec_peak(val, base)
    cands = {v: sign_cands(val.sign[v])[0] for v in val.vids}
    best = max(MOVE_GRID, key=lambda r: f1_of({v: move_to_signs(st[v], cands[v], r, drop) for v in val.vids}, val))
    return {**{k: v for k, v in base.items() if k != "val_bnd_f1_5"}, "r": best}


def dec_move(data, p, drop):
    st = dec_peak(data, p)
    return {v: move_to_signs(st[v], sign_cands(data.sign[v])[0], p["r"], drop) for v in data.vids}


# ------------------------------------------------------------------------- post-hoc sign-level classifier
def candidate_table(phrase_p, sign_p):
    """Predicted sign starts of one video and their features (dict name -> (K,))."""
    s, e = sign_cands(sign_p)
    T = phrase_p.shape[1]
    if len(s) == 0:
        return s, {}
    prev_end = np.r_[-1, e[:-1]]
    pause = np.clip(s - prev_end - 1, 0, 200).astype(np.float64)
    dur = (e - s + 1).astype(np.float64)
    lp = np.log1p(pause)
    lmax = np.array([lp[np.abs(s - t) <= 50].max() for t in s])
    since, n = [], 0
    for x in pause:
        n = 0 if x > 25 else n + 1
        since.append(n)
    smooth_b = gaussian_filter1d(phrase_p[B_TAG].astype(np.float64), 2)
    pb, po, sbp = phrase_p[B_TAG], phrase_p[O_TAG], sign_p[B_TAG]
    f = {
        "log_pause": lp,
        "rel_pause": lp - np.log1p(np.median(pause)),
        "local_max_pause": lp - lmax,
        "prev_dur_rel": np.log(np.r_[np.median(dur), dur[:-1]] / np.median(dur)),
        "cur_dur_rel": np.log(dur / np.median(dur)),
        "signs_since_long_pause": np.log1p(np.array(since, dtype=np.float64)),
        "sign_b": np.array([sbp[max(0, t - 2):t + 3].max() for t in s]),
        "ph_b_max": np.array([pb[max(0, t - 5):t + 6].max() for t in s]),
        "ph_b_smooth": smooth_b[s],
        "ph_o_before": np.array([po[max(0, t - 10):t].mean() if t > 0 else 1.0 for t in s]),
        "ph_o_after": np.array([po[t:min(T, t + 10)].mean() for t in s]),
    }
    return s, f


def candidate_labels(cands, gold_starts, k=K):
    """1 for the candidate nearest to each gold phrase start within +-k (one per gold start)."""
    y = np.zeros(len(cands), dtype=np.int64)
    for g in gold_starts:
        if len(cands):
            j = int(np.argmin(np.abs(cands - g)))
            if abs(int(cands[j]) - int(g)) <= k:
                y[j] = 1
    return y


def build_tables(data, feats):
    tabs = {}
    for v in data.vids:
        s, f = candidate_table(data.phrase[v], data.sign[v])
        X = np.stack([f[n] for n in feats], axis=1) if len(s) else np.zeros((0, len(feats)))
        tabs[v] = (s, X, candidate_labels(s, data.gs[v]))
    return tabs


def make_clf(kind):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, random_state=0)


def thin(idx, sc, md):
    if md <= 0 or len(idx) <= 1:
        return idx
    order = np.argsort(-sc, kind="stable")
    keep = []
    for j in order:
        if all(abs(int(idx[j]) - int(idx[i])) > md for i in keep):
            keep.append(j)
    return np.sort(idx[keep])


def select(tabs, scores, thr, md):
    return {v: thin(tabs[v][0][scores[v] >= thr], scores[v][scores[v] >= thr], md) for v in tabs}


def fit_posthoc(val, kind, feats):
    tabs = build_tables(val, feats)
    vids = [v for v in val.vids if len(tabs[v][0])]
    oof = {v: np.zeros(len(tabs[v][0])) for v in val.vids}
    for v in vids:                                               # leave-one-video-out scores
        tr = [u for u in vids if u != v]
        X = np.concatenate([tabs[u][1] for u in tr])
        y = np.concatenate([tabs[u][2] for u in tr])
        clf = make_clf(kind).fit(X, y)
        oof[v] = clf.predict_proba(tabs[v][1])[:, 1]
    allsc = np.concatenate([oof[v] for v in vids])
    grid = np.unique(np.round(np.quantile(allsc, np.linspace(0.3, 0.99, 70)), 4))
    best = (-1.0, None, None)
    for md in MD_GRID:
        for thr in grid:
            f = f1_of(select(tabs, oof, thr, md), val)
            if f > best[0]:
                best = (f, float(thr), md)
    X = np.concatenate([tabs[v][1] for v in vids])
    y = np.concatenate([tabs[v][2] for v in vids])
    return {"kind": kind, "feats": feats, "thr": best[1], "md": best[2], "val_oof_f1": round(best[0], 4),
            "clf": make_clf(kind).fit(X, y)}


def dec_posthoc(data, p):
    tabs = build_tables(data, p["feats"])
    sc = {v: (p["clf"].predict_proba(tabs[v][1])[:, 1] if len(tabs[v][0]) else np.zeros(0)) for v in data.vids}
    return select(tabs, sc, p["thr"], p["md"])


DECODERS = {
    "peak_thr": (fit_peak, dec_peak),
    "snap_product": (lambda val, base: fit_product(val), dec_product),
    "snap_move": (lambda val, base: fit_move(val, base, False), lambda d, p: dec_move(d, p, False)),
    "snap_move_drop": (lambda val, base: fit_move(val, base, True), lambda d, p: dec_move(d, p, True)),
    "posthoc_lr": (lambda val, base: fit_posthoc(val, "lr", TIMING + PHRASE), dec_posthoc),
    "posthoc_gbm": (lambda val, base: fit_posthoc(val, "gbm", TIMING + PHRASE), dec_posthoc),
    "posthoc_gbm_timing": (lambda val, base: fit_posthoc(val, "gbm", TIMING), dec_posthoc),
}
ORDER = list(DECODERS)


def evaluate(starts, data):
    per = {v: (np.asarray(starts[v], dtype=np.int64), starts_to_segments(starts[v], data.phrase[v]), data.gold[v])
           for v in data.vids}
    m = score(per)
    counts = {v: boundary_counts(data.gs[v], np.asarray(starts[v], dtype=np.int64), K) for v in data.vids}
    return m, counts


def coverage(data):
    hit = tot = 0
    sf = [0, 0, 0]
    for v in data.vids:
        c = sign_cands(data.sign[v])[0]
        g = data.gs[v]
        if len(g):
            hit += int((np.abs(g[:, None] - c[None, :]).min(axis=1) <= K).sum()) if len(c) else 0
            tot += len(g)
    return hit / max(1, tot)


def run_study(runs, load, out_dir, n_boot=2000):
    rows, counts = [], defaultdict(list)
    for run in runs:
        val, test = load(run, "val"), load(run, "test")
        base = fit_peak(val)
        cov = (coverage(val), coverage(test))
        print(f"{run}: gold phrase starts within +-{K} of a predicted sign start -- val {cov[0]:.1%}, test {cov[1]:.1%}")
        for name in ORDER:
            fit, dec = DECODERS[name]
            p = base if name == "peak_thr" else fit(val, base)
            mv, _ = evaluate(dec(val, p), val)
            mt, ct = evaluate(dec(test, p), test)
            counts[name].append(ct)
            params = {k: v for k, v in p.items() if k not in ("clf", "feats")}
            rows.append({"run": run, "decoder": name, "params": json.dumps(params), "val_bnd_f1_5": mv["bnd_f1_5"],
                         "phrase_cover_test": round(cov[1], 4),
                         **{f"test_{k}": mt[k] for k in ("bnd_f1_2", "bnd_f1_5", "bnd_f1_10", "bnd_p_5", "bnd_r_5",
                                                         "iou", "pct", "k_ratio")}})
            print(f"  {name:20s} val {mv['bnd_f1_5']:.3f} | test F1@2 {mt['bnd_f1_2']:.3f} F1@5 {mt['bnd_f1_5']:.3f} "
                  f"P {mt['bnd_p_5']:.3f} R {mt['bnd_r_5']:.3f} IoU {mt['iou']:.3f} K/gold {mt['k_ratio']:.2f}  {params}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Sign-head decoders on {', '.join(runs)} (test; tuned on val by boundary F1@{K}).",
             "mean over runs; diff vs peak_thr = paired bootstrap over videos (runs averaged).", ""]
    for name in ORDER:
        rr = [r for r in rows if r["decoder"] == name]
        mean = {k: np.mean([r[k] for r in rr]) for k in ("test_bnd_f1_2", "test_bnd_f1_5", "test_bnd_f1_10", "test_iou",
                                                          "test_k_ratio")}
        s = (f"{name:20s} F1@2 {mean['test_bnd_f1_2']:.3f}  F1@5 {mean['test_bnd_f1_5']:.3f}  F1@10 "
             f"{mean['test_bnd_f1_10']:.3f}  IoU {mean['test_iou']:.3f}  K/gold {mean['test_k_ratio']:.2f}")
        if name != "peak_thr":
            b = paired_bootstrap(counts[name], counts["peak_thr"], n_boot=n_boot)
            s += f"  | vs peak_thr {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] P<=0 {b['p_le_0']:.3f}"
        lines.append(s)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"-> {out_dir}/rows.csv, summary.txt")
    return rows


# ------------------------------------------------------------------------- selftest
def _synthetic(rng, n_vid, jitter):
    """Videos with sign starts every ~25 frames, phrase starts after long pauses; noisy predictions."""
    out = {}
    for i in range(n_vid):
        T = 3000
        starts, t = [], 20
        while t < T - 60:
            starts.append(t)
            t += int(rng.integers(15, 35)) + (int(rng.integers(20, 60)) if rng.random() < 0.2 else 0)
        starts = np.array(starts)
        pause = np.r_[20, np.diff(starts)]
        is_phrase = (pause > 45) | (rng.random(len(starts)) < 0.03)
        is_phrase[0] = True
        gold = np.zeros(T, dtype=np.int64)
        ph = starts[is_phrase]
        for a, b in zip(ph, np.r_[ph[1:], T - 10]):
            gold[a] = 2
            gold[a + 1:b - 8] = 1
        sign = np.full((3, T), 0.05, dtype=np.float32)
        sign[1] = 0.9
        for s in starts:                                             # sign head: accurate, small jitter
            q = int(np.clip(s + rng.integers(-1, 2), 0, T - 1))
            sign[:, q] = (0.05, 0.15, 0.8)
        phrase = np.zeros((3, T), dtype=np.float32)
        phrase[0] = (gold == 0) * 0.8 + 0.1
        phrase[1] = (gold > 0) * 0.8 + 0.1
        phrase[2] = 0.02
        for s in ph:                                                 # phrase head: jittered, smeared peaks
            q = int(np.clip(s + rng.integers(-jitter, jitter + 1), 0, T - 1))
            phrase[2, max(0, q - 2):q + 3] += np.array([0.1, 0.25, 0.45, 0.25, 0.1])[:len(phrase[2, max(0, q - 2):q + 3])]
        for q in rng.integers(0, T, 8):                              # false peaks
            phrase[2, q] += 0.35
        phrase /= phrase.sum(0, keepdims=True)
        sign /= sign.sum(0, keepdims=True)
        out[f"v{i}"] = (phrase, sign, gold)
    return Run({v: x[0] for v, x in out.items()}, {v: x[1] for v, x in out.items()}, {v: x[2] for v, x in out.items()})


def selftest():
    rng = np.random.default_rng(0)
    data = {(r, s): _synthetic(rng, 5 if s == "val" else 6, 6) for r in ("a", "b") for s in ("val", "test")}
    rows = run_study(["a", "b"], lambda r, s: data[(r, s)], os.path.join(OUT_DIR, "_selftest"), n_boot=200)
    f = {r["decoder"]: r["test_bnd_f1_2"] for r in rows if r["run"] == "a"}
    ok = f["snap_move"] > f["peak_thr"] and f["snap_product"] >= f["peak_thr"] - 0.02
    print(f"\nselftest: jittered phrase peaks, accurate sign starts -> snapping should help at +-2: "
          f"peak_thr {f['peak_thr']:.3f}, snap_move {f['snap_move']:.3f} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--sign-root", default=SIGN_ROOT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.runs:
        ap.error("--runs is required (or --selftest)")
    run_study(a.runs, lambda r, s: load_run(a.pred_root, a.sign_root, r, s), a.out_dir, a.n_boot)


if __name__ == "__main__":
    main()