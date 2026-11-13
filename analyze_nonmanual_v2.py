"""
analyze_nonmanual_v2.py -- does the face stream help at phrase starts marked by non-manual signals?

Version 2 of analyze_nonmanual_starts.py. v1 had two problems: its markers fired almost everywhere
(head movement at 94% of starts AND 93% of random moments), and it compared RECALL of two models whose
peak-picking thresholds were tuned separately, so the face model's higher precision showed up as a loss.

DESIGN (many variants, kept honest)
  1. Marker library. ~200 marker specifications from the cue files (data_processing/extract_cue_features.py):
     eyes   blink onset / blink anywhere / eye-openness dip / eye-openness step / eyes-narrowed onset
     brows  brow-height step / inner-brow step / furrow step / raise, lower, any brow velocity
     mouth  mouthing-activity step / mouth-opening step / mouth-opening velocity / mouth still
     face   face deformation energy high / step
     head   (CONTROL family: the head is in the body stream, not the face stream)
            pitch step / pitch velocity / vertical velocity / rotation speed / yaw step / roll step
     each with thresholds k in {1.5, 2, 2.5, 3} (robust z) where it has one, and windows of +-5 / 10 / 15
     frames around the start. "step" = |mean over the next 0.3 s - mean over the previous 0.3 s| >= k.
  2. Marker validity on the TRAINING videos (gold labels + cue files only, no model): rate at phrase
     starts vs. at random moments inside phrases (>= 0.5 s from any start) -> enrichment ratio. A marker
     is "enriched" when the ratio >= --min-enrichment (default 1.2) and it occurs at >= 5% of starts.
     Enrichment itself is a finding: which non-manual signals mark DGS phrase starts.
  3. Threshold-free model scores (no decoder, so the two models' different thresholds drop out):
       score  = max smoothed phrase-start probability within +-tol of the moment
       rank   = that score as a percentile among all frames of the video (calibration-free; PRIMARY)
     plus a count-matched recall: both models keep their top-K peaks over the split, K = number of gold
     starts, so they work at the SAME number of predictions.
  4. Effects, face minus no face (each group's runs averaged):
       E_marked    effect at marked starts
       E_unmarked  effect at unmarked starts
       interaction E_marked - E_unmarked    ( > 0: the face helps MORE where the face marks the start)
       FA side     the same at marked vs unmarked in-phrase moments ( < 0: the face suppresses false
                   starts at facial events inside phrases)
       AUC         start vs in-phrase moment discrimination, at marked starts, per model
     paired bootstrap over videos (95% intervals); p = two-sided bootstrap p.
  5. Which results are confirmatory:
       PRIMARY      per family, the enriched spec with the highest TRAINING enrichment (chosen before any
                    model result), rank metric; Holm correction over the families
       VAL-SELECTED the enriched spec with the largest VALIDATION interaction, reported on test
       SPEC CURVE   every enriched spec (and every spec) on test: how many favour the face, sorted plot
       REGRESSION   per start, rank(face) - rank(no face) ~ primary markers + short pause, cluster
                    bootstrap over videos (controls for markers occurring together and for pauses)
     Report the primary results and the curve; exploratory variants are labelled as such.

    python analyze_nonmanual_v2.py                                    # 115-117 vs 112-114
    python analyze_nonmanual_v2.py --with-face 373-375 --without-face <B+H on that machine>
    python analyze_nonmanual_v2.py --selftest
Outputs (--out-dir, default evaluation_phrase/nonmanual_v2):
    summary.txt, marker_enrichment.csv, spec_results.csv, starts_test.csv,
    enrichment.png, primary.png, spec_curve.png
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

FPS = 50
FACE_DETECTED = "face detected"


# ============================================================================= marker library
def _z(x):
    """Robust z per video (median / 1.4826 MAD)."""
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 1e-9 else (np.nanstd(x) + 1e-9))


def _step(x, L):
    """|mean(x[t:t+L]) - mean(x[t-L:t])| at every t (0 near the ends)."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    c = np.concatenate([[0.0], np.cumsum(x)])
    T = len(x)
    out = np.zeros(T)
    t = np.arange(L, T - L + 1)
    if len(t):
        after = (c[t + L] - c[t]) / L
        before = (c[t] - c[t - L]) / L
        out[t] = np.abs(after - before)
    return out


def _deriv_z(x):
    return _z(np.gradient(np.nan_to_num(np.asarray(x, dtype=np.float64))))


# (name, family, channel, kind, has_k). kind: onset | level | high | low | abs | step | dabs | stepz
BASE_MARKERS = [
    ("blink onset", "eyes", "blink", "onset", False),
    ("blink anywhere", "eyes", "blink", "level", False),
    ("eye-openness dip", "eyes", "EAR (eye openness)", "low", True),
    ("eye-openness step", "eyes", "EAR (eye openness)", "step", True),
    ("eyes-narrowed onset", "eyes", "eyes narrowed (long)", "onset", False),
    ("brow-height step", "brows", "brow height", "step", True),
    ("inner-brow step", "brows", "inner brow height", "step", True),
    ("brow-furrow step", "brows", "brow furrow (inner distance)", "step", True),
    ("brow raise", "brows", "brow vertical velocity", "high", True),
    ("brow lower", "brows", "brow vertical velocity", "low", True),
    ("brow movement", "brows", "brow vertical velocity", "abs", True),
    ("mouthing-activity step", "mouth", "mouth motion energy", "step", True),
    ("mouth-opening step", "mouth", "MAR (mouth aspect ratio)", "step", True),
    ("mouth-opening velocity", "mouth", "mouth opening velocity", "abs", True),
    ("mouth still", "mouth", "mouth motion energy", "low", True),
    ("face deformation", "face", "face deformation energy", "high", True),
    ("face-deformation step", "face", "face deformation energy", "step", True),
    ("head-pitch step", "head", "head pitch (nod)", "step", True),
    ("head-pitch velocity", "head", "head pitch (nod)", "dabs", True),
    ("head vertical velocity", "head", "head vertical velocity", "abs", True),
    ("head rotation speed", "head", "head rotation speed", "high", True),
    ("head-yaw step", "head", "head yaw deviation", "step", True),
    ("head-roll step", "head", "head roll deviation", "step", True),
]
FAMILIES = ["eyes", "brows", "mouth", "face", "head"]
FACE_FAMILIES = ["eyes", "brows", "mouth", "face"]           # what the face stream sees; head = control
KS = (1.5, 2.0, 2.5, 3.0)
WINDOWS = (5, 10, 15)
STEP_L = 15                                                   # 0.3 s


def marker_specs():
    specs = []
    for name, fam, ch, kind, has_k in BASE_MARKERS:
        for k in (KS if has_k else (None,)):
            for w in WINDOWS:
                label = f"{name}" + (f" k={k}" if k is not None else "") + f" w={w}"
                specs.append({"id": label, "name": name, "family": fam, "channel": ch, "kind": kind, "k": k, "w": w})
    return specs


def event_array(cues, spec, cache):
    """Per-frame boolean event for a marker spec (before widening by the window)."""
    key = (spec["channel"], spec["kind"], spec["k"])
    if key in cache:
        return cache[key]
    x = cues[spec["channel"]]
    kind, k = spec["kind"], spec["k"]
    if kind == "onset":
        b = x > 0.5
        ev = b & ~np.concatenate([[False], b[:-1]])
    elif kind == "level":
        ev = x > 0.5
    elif kind == "high":
        ev = x >= k
    elif kind == "low":
        ev = x <= -k
    elif kind == "abs":
        ev = np.abs(x) >= k
    elif kind == "step":
        sk = ("_step", spec["channel"])
        if sk not in cache:
            cache[sk] = _z(_step(x, STEP_L))
        ev = cache[sk] >= k
    elif kind == "dabs":
        dk = ("_dz", spec["channel"])
        if dk not in cache:
            cache[dk] = _deriv_z(x)
        ev = np.abs(cache[dk]) >= k
    else:
        raise ValueError(kind)
    ev = np.nan_to_num(ev).astype(bool)
    cache[key] = ev
    return ev


def any_in_window(ev, idx, w):
    """For every index in idx: any event within [i - w, i + w]."""
    c = np.concatenate([[0], np.cumsum(ev.astype(np.int64))])
    T = len(ev)
    lo = np.clip(idx - w, 0, T)
    hi = np.clip(idx + w + 1, 0, T)
    return (c[hi] - c[lo]) > 0


def face_ok(cues, idx, w=10):
    if FACE_DETECTED not in cues:
        return np.ones(len(idx), dtype=bool)
    fd = np.nan_to_num(cues[FACE_DETECTED])
    c = np.concatenate([[0.0], np.cumsum(fd)])
    T = len(fd)
    lo, hi = np.clip(idx - w, 0, T), np.clip(idx + w + 1, 0, T)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1) >= 0.5


# ============================================================================= moments
def starts_of(gold, O=0):
    """Phrase starts = first frame of every non-O run."""
    g = np.asarray(gold)
    inside = g != O
    return np.flatnonzero(inside & ~np.concatenate([[False], inside[:-1]]))


def negatives_of(gold, starts, rng, min_gap=25, n_max=200, O=0):
    inside = np.flatnonzero(np.asarray(gold) != O)
    if len(starts) and len(inside):
        d = np.min(np.abs(inside[:, None] - starts[None, :]), axis=1)
        inside = inside[d >= min_gap]
    if len(inside) > n_max:
        inside = np.sort(rng.choice(inside, n_max, replace=False))
    return inside


def pause_before(gold, s, O=0):
    n = 0
    while s - n - 1 >= 0 and gold[s - n - 1] == O:
        n += 1
    return n


# ============================================================================= model scores
def window_max(x, idx, tol):
    T = len(x)
    return np.array([x[max(0, i - tol):min(T, i + tol + 1)].max() for i in idx])


def run_scores(p_b, starts, negs, tol, sigma=2.0):
    """Raw and rank scores at starts and negatives for one video of one run."""
    s = gaussian_filter1d(np.asarray(p_b, dtype=np.float64), sigma) if sigma else np.asarray(p_b, np.float64)
    rank = (rankdata(s) - 0.5) / len(s)
    return {"raw_s": window_max(s, starts, tol), "rank_s": window_max(rank, starts, tol),
            "raw_n": window_max(s, negs, tol), "rank_n": window_max(rank, negs, tol), "smooth": s}


def matched_found(per_video_smooth, starts_by_vid, tol, ratio=1.0, min_dist=10):
    """Top-K peaks over the whole split, K = ratio x number of gold starts -> found flags per start."""
    from scipy.signal import find_peaks
    cand = []
    for v, s in per_video_smooth.items():
        idx, _ = find_peaks(s, distance=min_dist)
        cand += [(float(s[i]), v, int(i)) for i in idx]
    K = int(round(ratio * sum(len(x) for x in starts_by_vid.values())))
    cand.sort(reverse=True)
    keep = {}
    for _, v, i in cand[:K]:
        keep.setdefault(v, []).append(i)
    out = {}
    for v, st in starts_by_vid.items():
        pred = np.sort(np.array(keep.get(v, []), dtype=np.int64))
        found = np.zeros(len(st), dtype=bool)
        if len(pred):
            used = np.zeros(len(pred), dtype=bool)
            for j in range(len(st)):
                d = np.abs(pred - st[j])
                d[used] = 10 ** 9
                m = int(d.argmin())
                if d[m] <= tol:
                    found[j], used[m] = True, True
        out[v] = found
    return out


# ============================================================================= bootstrap engine
class Boot:
    """Resample videos with replacement; weights W (n_boot, V) shared by every statistic."""

    def __init__(self, vids, n_boot, seed=0):
        self.vids = list(vids)
        self.vi = {v: i for i, v in enumerate(self.vids)}
        rng = np.random.default_rng(seed)
        V = len(self.vids)
        self.W = np.stack([np.bincount(rng.integers(0, V, V), minlength=V) for _ in range(n_boot)]).astype(float)

    def per_video(self, vid_of, values, mask):
        S, N = np.zeros(len(self.vids)), np.zeros(len(self.vids))
        np.add.at(S, [self.vi[v] for v in vid_of[mask]], values[mask])
        np.add.at(N, [self.vi[v] for v in vid_of[mask]], 1.0)
        return S, N

    def mean_diff(self, vid_of, d, m):
        """mean(d | m) - mean(d | ~m), mean(d | m), mean(d | ~m) with bootstrap draws of each."""
        S1, N1 = self.per_video(vid_of, d, m)
        S0, N0 = self.per_video(vid_of, d, ~m)
        with np.errstate(invalid="ignore", divide="ignore"):
            e1, e0 = S1.sum() / N1.sum(), S0.sum() / N0.sum()
            b1, b0 = (self.W @ S1) / (self.W @ N1), (self.W @ S0) / (self.W @ N0)
        return (e1 - e0, b1 - b0), (e1, b1), (e0, b0)


def ci(draws):
    d = draws[np.isfinite(draws)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (np.nan, np.nan)


def pval(draws):
    d = draws[np.isfinite(draws)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan


def holm(ps):
    order = np.argsort(ps)
    adj, prev = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        prev = max(prev, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = prev
    return adj


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fmt(e, c, p=None):
    s = f"{e:+.4f} (95% CI: {c[0]:+.4f} to {c[1]:+.4f})"
    return s + (f", p = {p:.3f}" if p is not None else "")


# ============================================================================= data assembly
def load_cues(cue_dir, vid, names):
    path = os.path.join(cue_dir, f"{vid}.npy")
    if not os.path.exists(path):
        return None
    arr = np.load(path).astype(np.float64)
    return {n: arr[:, i] for i, n in enumerate(names)}


def mark_moments(cues, idx, specs):
    """(len(specs), len(idx)) bool matrix of markers at the given moments."""
    cache, out = {}, np.zeros((len(specs), len(idx)), dtype=bool)
    for j, sp in enumerate(specs):
        out[j] = any_in_window(event_array(cues, sp, cache), idx, sp["w"])
    return out


def enrichment(train_vids, labels_dir, cue_dir, names, specs, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    vids, M_s, M_n, vs, vn = [], [], [], [], []
    for v in train_vids:
        lp = os.path.join(labels_dir, f"{v}.npy")
        cues = load_cues(cue_dir, v, names)
        if cues is None or not os.path.exists(lp):
            continue
        gold = np.load(lp)
        T = min(len(gold), len(next(iter(cues.values()))))
        gold, cues = gold[:T], {k: a[:T] for k, a in cues.items()}
        st = starts_of(gold)
        ng = negatives_of(gold, st, rng)
        st, ng = st[face_ok(cues, st)], ng[face_ok(cues, ng)]
        if not len(st) or not len(ng):
            continue
        vids.append(v)
        M_s.append(mark_moments(cues, st, specs))
        M_n.append(mark_moments(cues, ng, specs))
        vs += [v] * len(st)
        vn += [v] * len(ng)
    Ms, Mn = np.concatenate(M_s, axis=1), np.concatenate(M_n, axis=1)
    vs, vn = np.array(vs), np.array(vn)
    boot = Boot(vids, n_boot, seed)
    rows = []
    for j, sp in enumerate(specs):
        Ss, Ns = boot.per_video(vs, Ms[j].astype(float), np.ones(len(vs), bool))
        Sn, Nn = boot.per_video(vn, Mn[j].astype(float), np.ones(len(vn), bool))
        rs, rn = Ss.sum() / Ns.sum(), Sn.sum() / Nn.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            br = ((boot.W @ Ss) / (boot.W @ Ns)) / ((boot.W @ Sn) / (boot.W @ Nn))
        rows.append({**{k: sp[k] for k in ("id", "name", "family", "k", "w")}, "rate_starts": rs, "rate_inphrase": rn,
                     "enrichment": rs / rn if rn > 0 else np.inf, "enr_lo": ci(br)[0], "enr_hi": ci(br)[1]})
    return rows, len(vids), len(vs)


def collect_split(groups, split, pred_root, cue_dir, names, specs, tol, seed=0):
    """Per-start / per-negative table for one split: markers + both groups' scores."""
    from phrase_stats import load_predictions
    rng = np.random.default_rng(seed)
    loaded = {g: [load_predictions(pred_root, r, split) for r in runs] for g, runs in groups.items()}
    vids = sorted(set.intersection(*[set(p[0]) for runs in loaded.values() for p in runs]))
    gold = loaded["face"][0][2]
    rec = {"vid_s": [], "vid_n": [], "M_s": [], "M_n": [], "pause": [], "start": []}
    sc = {g: {"rank_s": [], "rank_n": [], "raw_s": [], "raw_n": []} for g in groups}
    smooth = {g: [dict() for _ in loaded[g]] for g in groups}
    starts_by_vid, skipped = {}, []
    for v in vids:
        cues = load_cues(cue_dir, v, names)
        if cues is None:
            skipped.append(v)
            continue
        T = min(len(gold[v]), len(next(iter(cues.values()))))
        g = gold[v][:T]
        cues = {k: a[:T] for k, a in cues.items()}
        st = starts_of(g)
        ng = negatives_of(g, st, rng)
        st, ng = st[face_ok(cues, st)], ng[face_ok(cues, ng)]
        starts_by_vid[v] = st
        rec["vid_s"] += [v] * len(st)
        rec["vid_n"] += [v] * len(ng)
        rec["start"] += list(st)
        rec["pause"] += [pause_before(g, int(s)) for s in st]
        rec["M_s"].append(mark_moments(cues, st, specs))
        rec["M_n"].append(mark_moments(cues, ng, specs))
        for gname in groups:
            per = [run_scores(p[0][v][2][:T], st, ng, tol) for p in loaded[gname]]
            for key in sc[gname]:
                sc[gname][key].append(np.mean([q[key] for q in per], axis=0))
            for r, q in enumerate(per):
                smooth[gname][r][v] = q["smooth"]
    out = {"vid_s": np.array(rec["vid_s"]), "vid_n": np.array(rec["vid_n"]), "start": np.array(rec["start"]),
           "pause": np.array(rec["pause"]), "M_s": np.concatenate(rec["M_s"], axis=1),
           "M_n": np.concatenate(rec["M_n"], axis=1), "skipped": skipped, "vids": sorted(starts_by_vid)}
    for gname in groups:
        for key in sc[gname]:
            out[f"{gname}_{key}"] = np.concatenate(sc[gname][key])
        found = [matched_found(sm, starts_by_vid, tol) for sm in smooth[gname]]
        out[f"{gname}_matched"] = np.concatenate([np.mean([f[v] for f in found], axis=0) for v in out["vids"]])
    return out


# ============================================================================= effects
def spec_effects(D, specs, boot, short_only=False):
    """Per spec: rank / raw / matched-recall interaction, marked & unmarked effects, FA side."""
    rows = []
    d_rank = D["face_rank_s"] - D["noface_rank_s"]
    d_raw = D["face_raw_s"] - D["noface_raw_s"]
    d_match = D["face_matched"] - D["noface_matched"]
    d_fa = D["face_rank_n"] - D["noface_rank_n"]
    sel = D["pause"] <= 12 if short_only else np.ones(len(d_rank), bool)
    for j, sp in enumerate(specs):
        m = D["M_s"][j] & sel
        row = {**{k: sp[k] for k in ("id", "name", "family", "k", "w")}, "n_marked": int(m.sum()),
               "n_unmarked": int((~D["M_s"][j] & sel).sum())}
        for tag, d in (("rank", d_rank), ("raw", d_raw), ("matched", d_match)):
            vid = D["vid_s"][sel]
            (it, itb), (e1, b1), (e0, b0) = boot.mean_diff(vid, d[sel], D["M_s"][j][sel])
            row.update({f"{tag}_inter": it, f"{tag}_inter_lo": ci(itb)[0], f"{tag}_inter_hi": ci(itb)[1],
                        f"{tag}_inter_p": pval(itb), f"{tag}_marked": e1, f"{tag}_marked_lo": ci(b1)[0],
                        f"{tag}_marked_hi": ci(b1)[1], f"{tag}_unmarked": e0})
        if not short_only:
            (it, itb), (e1, _), (e0, _) = boot.mean_diff(D["vid_n"], d_fa, D["M_n"][j])
            row.update({"fa_inter": it, "fa_inter_lo": ci(itb)[0], "fa_inter_hi": ci(itb)[1], "fa_marked": e1})
        rows.append(row)
    return rows


def auc_effect(D, j, boot_vids, n_boot=500, seed=0):
    """AUC (marked starts vs all in-phrase moments), face minus no face, bootstrap over videos."""
    m = D["M_s"][j]
    rng = np.random.default_rng(seed)
    vids = list(boot_vids)
    idx_s = {v: np.flatnonzero((D["vid_s"] == v) & m) for v in vids}
    idx_n = {v: np.flatnonzero(D["vid_n"] == v) for v in vids}

    def stat(sample):
        ps = np.concatenate([idx_s[v] for v in sample])
        pn = np.concatenate([idx_n[v] for v in sample])
        return (auc(D["face_rank_s"][ps], D["face_rank_n"][pn]), auc(D["noface_rank_s"][ps], D["noface_rank_n"][pn]))

    a_f, a_g = stat(vids)
    draws = []
    for _ in range(n_boot):
        f, g = stat([vids[i] for i in rng.integers(0, len(vids), len(vids))])
        draws.append(f - g)
    return a_f, a_g, a_f - a_g, ci(np.array(draws))


def regression(D, cols, names, boot):
    """OLS of rank(face) - rank(no face) on marker flags + short pause; cluster bootstrap over videos."""
    y = D["face_rank_s"] - D["noface_rank_s"]
    X = np.column_stack([np.ones(len(y))] + [D["M_s"][j].astype(float) for j in cols]
                        + [(D["pause"] <= 12).astype(float)])
    labels = ["intercept"] + names + ["short pause"]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    vid_idx = np.array([boot.vi[v] for v in D["vid_s"]])
    draws = []
    for w in boot.W[:500]:
        ww = w[vid_idx]
        Xw = X * np.sqrt(ww)[:, None]
        yw = y * np.sqrt(ww)
        draws.append(np.linalg.lstsq(Xw, yw, rcond=None)[0])
    draws = np.array(draws)
    return [(labels[i], beta[i], ci(draws[:, i]), pval(draws[:, i])) for i in range(len(labels))]


# ============================================================================= figures
FAM_COLORS = {"eyes": "#1F5FA8", "brows": "#2A9D8F", "mouth": "#D9822B", "face": "#7A5195", "head": "#8A8F98"}


def fig_enrichment(enr_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = {}
    for r in enr_rows:
        if r["rate_starts"] >= 0.05 and (r["name"] not in best or r["enrichment"] > best[r["name"]]["enrichment"]):
            best[r["name"]] = r
    rows = sorted(best.values(), key=lambda r: r["enrichment"])
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.2), dpi=200)
    y = np.arange(len(rows))
    ax.hlines(y, [r["enr_lo"] for r in rows], [r["enr_hi"] for r in rows], color="#BBBBBB", lw=2)
    ax.scatter([r["enrichment"] for r in rows], y, c=[FAM_COLORS[r["family"]] for r in rows], s=28, zorder=3)
    ax.axvline(1.0, color="#444444", lw=0.8, ls="--")
    ax.set_yticks(y, [f"{r['id']}  ({r['rate_starts'] * 100:.0f}% of starts)" for r in rows], fontsize=7)
    ax.set_xlabel("Enrichment at phrase starts (rate at starts / rate inside phrases), training videos", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAM_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)


def fig_primary(prim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    x = np.arange(len(prim))
    for off, key, col, lab in ((-0.19, "rank_marked", "#1F5FA8", "marked starts"),
                               (0.19, "rank_unmarked", "#9AA5B1", "unmarked starts")):
        ax.bar(x + off, [p[key] * 100 for p in prim], 0.36, color=col, label=lab)
    for i, p in enumerate(prim):
        ax.errorbar(i - 0.19, p["rank_marked"] * 100, yerr=[[(p["rank_marked"] - p["rank_marked_lo"]) * 100],
                    [(p["rank_marked_hi"] - p["rank_marked"]) * 100]], color="#222222", capsize=3, lw=1)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(x, [f"{p['family']}\n{p['name']}" for p in prim], fontsize=7)
    ax.set_ylabel("Face effect on the start score\n(percentile points, face - no face)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def fig_curve(rows, path, key="rank_inter"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted([r for r in rows if np.isfinite(r[key])], key=lambda r: r[key])
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200)
    x = np.arange(len(rows))
    ax.vlines(x, [r[f"{key}_lo"] * 100 for r in rows], [r[f"{key}_hi"] * 100 for r in rows], color="#CCCCCC", lw=1)
    ax.scatter(x, [r[key] * 100 for r in rows], c=[FAM_COLORS[r["family"]] for r in rows], s=8, zorder=3)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xlabel(f"Marker specifications (sorted), n = {len(rows)}", fontsize=8)
    ax.set_ylabel("Interaction (percentile points)\nface effect at marked - unmarked starts", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAM_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


# ============================================================================= main
def run(args):
    from phrase_stats import PRED_ROOT, expand_prefixes            # noqa: F401
    from src.cue_sets import cue_names_path
    groups = {"face": expand_prefixes(args.model, args.with_face), "noface": expand_prefixes(args.model, args.without_face)}
    with open(cue_names_path(args.cue_dir)) as f:
        names = json.load(f)
    specs = [s for s in marker_specs() if s["channel"] in names]
    miss = sorted({s[2] for s in BASE_MARKERS} - set(names))
    with open(args.splits) as f:
        splits = json.load(f)
    train = [v.replace(".npy", "") for v in splits["train"]][: args.enrich_max_videos]
    os.makedirs(args.out_dir, exist_ok=True)
    L = []

    print(f"1/4 marker enrichment on {len(train)} training videos ({len(specs)} specs) ...")
    enr, n_tv, n_ts = enrichment(train, args.labels_dir, args.cue_dir, names, specs, args.n_boot)
    for s, r in zip(specs, enr):
        s.update(enrichment=r["enrichment"], rate_starts=r["rate_starts"], enr_lo=r["enr_lo"])
    enriched = [j for j, s in enumerate(specs) if s["enrichment"] >= args.min_enrichment and s["rate_starts"] >= 0.05
                and s["enr_lo"] > 1.0]
    primary = []
    for fam in FAMILIES:
        cand = [j for j in enriched if specs[j]["family"] == fam]
        if cand:
            primary.append(max(cand, key=lambda j: specs[j]["enrichment"]))

    print("2/4 model scores on val and test ...")
    D = {sp: collect_split(groups, sp, args.pred_root, args.cue_dir, names, specs, args.tol) for sp in ("val", "test")}
    boots = {sp: Boot(D[sp]["vids"], args.n_boot) for sp in D}

    print("3/4 effects for every spec ...")
    res = {sp: spec_effects(D[sp], specs, boots[sp]) for sp in D}
    res_short = spec_effects(D["test"], specs, boots["test"], short_only=True)

    # ---------------------------------------------------------------- report
    T = D["test"]
    d_rank = T["face_rank_s"] - T["noface_rank_s"]
    L += ["Face stream and non-manually marked phrase starts -- v2 (threshold-free scores, train-validated markers)",
          f"with face: {', '.join(groups['face'])} | without face: {', '.join(groups['noface'])}",
          f"tol +-{args.tol} frames; score = max smoothed P(start) within tol; rank = its percentile in the video",
          f"test: {len(T['start'])} starts with face data, {len(T['vids'])} videos"
          + (f" (no cue file: {', '.join(T['skipped'])})" if T["skipped"] else ""), ""]
    if miss:
        L.append(f"(cue channels not in {args.cue_dir}: {', '.join(miss)} -- those markers are skipped)")
    ov_f, ov_g = T["face_rank_s"].mean(), T["noface_rank_s"].mean()
    L += ["OVERALL (all test starts)",
          f"  mean rank score: with face {ov_f * 100:.1f}, without {ov_g * 100:.1f}",
          f"  count-matched recall (K = gold count): with face {T['face_matched'].mean() * 100:.1f}%, "
          f"without {T['noface_matched'].mean() * 100:.1f}%",
          f"  AUC start vs in-phrase moment: with face "
          f"{auc(T['face_rank_s'], T['face_rank_n']):.3f}, without {auc(T['noface_rank_s'], T['noface_rank_n']):.3f}", ""]

    L += [f"1. MARKER ENRICHMENT (training: {n_tv} videos, {n_ts} starts); enriched = ratio >= {args.min_enrichment}, "
          f">= 5% of starts, lower CI > 1: {len(enriched)} of {len(specs)} specs"]
    best_by_name = {}
    for r in enr:
        if r["rate_starts"] >= 0.05 and (r["name"] not in best_by_name or r["enrichment"] > best_by_name[r["name"]]["enrichment"]):
            best_by_name[r["name"]] = r
    for r in sorted(best_by_name.values(), key=lambda r: -r["enrichment"]):
        L.append(f"  {r['family']:<6s} {r['id']:<36s} starts {r['rate_starts'] * 100:5.1f}%  inside {r['rate_inphrase'] * 100:5.1f}%  "
                 f"ratio {r['enrichment']:.2f} (95% CI: {r['enr_lo']:.2f} to {r['enr_hi']:.2f})")

    L += ["", "2. PRIMARY (per family: the most enriched spec on TRAINING, chosen before any model result; test; "
              "rank metric; Holm over families)"]
    prim_rows = [res["test"][j] for j in primary]
    adj = holm(np.array([r["rank_inter_p"] for r in prim_rows])) if prim_rows else []
    for r, pa in zip(prim_rows, adj):
        L += [f"  [{r['family']}] {r['id']}  (n marked {r['n_marked']}, unmarked {r['n_unmarked']})",
              f"     face effect at marked starts   {fmt(r['rank_marked'], (r['rank_marked_lo'], r['rank_marked_hi']))}",
              f"     face effect at unmarked starts {r['rank_unmarked']:+.4f}",
              f"     interaction                    {fmt(r['rank_inter'], (r['rank_inter_lo'], r['rank_inter_hi']), r['rank_inter_p'])}, "
              f"Holm p = {pa:.3f}",
              f"     count-matched recall interaction {fmt(r['matched_inter'], (r['matched_inter_lo'], r['matched_inter_hi']))}",
              f"     false-start side (in-phrase moments, marked - unmarked; < 0 = fewer false starts at facial events) "
              f"{fmt(r['fa_inter'], (r['fa_inter_lo'], r['fa_inter_hi']))}"]
        a_f, a_g, da, dci = auc_effect(T, specs.index(next(s for s in specs if s["id"] == r["id"])), T["vids"])
        L.append(f"     AUC at marked starts: with face {a_f:.3f}, without {a_g:.3f}, difference {fmt(da, dci)}")
        rs = next(x for x in res_short if x["id"] == r["id"])
        L.append(f"     short pauses only (<= 0.24 s): interaction {fmt(rs['rank_inter'], (rs['rank_inter_lo'], rs['rank_inter_hi']))}, "
                 f"n marked {rs['n_marked']}")

    L += ["", "3. VALIDATION-SELECTED (enriched spec with the largest validation interaction, rank metric) -> test"]
    if enriched:
        jv = max(enriched, key=lambda j: res["val"][j]["rank_inter"] if np.isfinite(res["val"][j]["rank_inter"]) else -9)
        rv, rt = res["val"][jv], res["test"][jv]
        L += [f"  {rt['id']} [{rt['family']}]: validation {rv['rank_inter']:+.4f} -> test "
              f"{fmt(rt['rank_inter'], (rt['rank_inter_lo'], rt['rank_inter_hi']), rt['rank_inter_p'])}"]

    L += ["", "4. SPECIFICATION CURVE (test, rank-metric interaction)"]
    for label, idx in (("enriched specs", enriched), ("all specs", list(range(len(specs))))):
        for fams, fl in ((FACE_FAMILIES, "face families"), (["head"], "head (control)")):
            rr = [res["test"][j] for j in idx if specs[j]["family"] in fams and np.isfinite(res["test"][j]["rank_inter"])]
            if rr:
                pos = sum(r["rank_inter"] > 0 for r in rr)
                sig = sum(r["rank_inter_lo"] > 0 for r in rr)
                neg = sum(r["rank_inter_hi"] < 0 for r in rr)
                L.append(f"  {label:<15s} {fl:<15s} n = {len(rr):3d}: interaction > 0 in {pos} ({pos / len(rr):.0%}); "
                         f"CI above 0 in {sig}; CI below 0 in {neg}; median {np.median([r['rank_inter'] for r in rr]) * 100:+.2f} pp")

    L += ["", "5. REGRESSION (test): rank(face) - rank(no face) per start ~ primary markers + short pause; "
              "cluster bootstrap over videos"]
    if primary:
        for lab, b, c, p in regression(T, primary, [f"{specs[j]['family']}: {specs[j]['id']}" for j in primary], boots["test"]):
            L.append(f"  {lab:<48s} {fmt(b, c, p)}")

    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out_dir, "marker_enrichment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(enr[0]))
        w.writeheader()
        w.writerows(enr)
    with open(os.path.join(args.out_dir, "spec_results.csv"), "w", newline="") as f:
        allrows = [{"split": sp, "enriched": int(j in enriched), "primary": int(j in primary), **r}
                   for sp in res for j, r in enumerate(res[sp])]
        allrows += [{"split": "test_short_pause", "enriched": int(j in enriched), "primary": int(j in primary), **r}
                    for j, r in enumerate(res_short)]
        keys = list(dict.fromkeys(k for r in allrows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)
    with open(os.path.join(args.out_dir, "starts_test.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vid", "start", "pause_frames", "rank_face", "rank_noface", "matched_face", "matched_noface"]
                   + [specs[j]["id"] for j in primary])
        for i in range(len(T["start"])):
            w.writerow([T["vid_s"][i], int(T["start"][i]), int(T["pause"][i]), round(T["face_rank_s"][i], 4),
                        round(T["noface_rank_s"][i], 4), round(T["face_matched"][i], 3), round(T["noface_matched"][i], 3)]
                       + [int(T["M_s"][j][i]) for j in primary])
    fig_enrichment(enr, os.path.join(args.out_dir, "enrichment.png"))
    if prim_rows:
        fig_primary(prim_rows, os.path.join(args.out_dir, "primary.png"))
    fig_curve([res["test"][j] for j in enriched] or res["test"], os.path.join(args.out_dir, "spec_curve.png"))
    print(f"\nwrote {args.out_dir}/ summary.txt, marker_enrichment.csv, spec_results.csv, starts_test.csv, "
          f"enrichment.png, primary.png, spec_curve.png")


# ============================================================================= selftest
def selftest():
    ok = []
    rng = np.random.default_rng(0)
    T = 600
    x = rng.normal(size=T)
    x[300:] += 4.0                                               # a level step at t = 300
    st = _z(_step(x, 15))
    ok.append(int(np.argmax(st)) in range(295, 306))
    b = np.zeros(T)
    b[100:104] = 1
    ev = event_array({"blink": b}, {"channel": "blink", "kind": "onset", "k": None}, {})
    ok.append(np.flatnonzero(ev).tolist() == [100])
    ok.append(any_in_window(ev, np.array([95, 120]), 5).tolist() == [True, False])
    g = np.zeros(50, int)
    g[10:20] = 1
    g[25:40] = 1
    ok.append(starts_of(g).tolist() == [10, 25] and pause_before(g, 25) == 5)
    ok.append(abs(auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) - 1.0) < 1e-9)
    vids = np.array(["a"] * 50 + ["b"] * 50)
    d = np.r_[np.ones(25), np.zeros(25), np.ones(25), np.zeros(25)]
    m = np.r_[np.ones(25, bool), np.zeros(25, bool), np.ones(25, bool), np.zeros(25, bool)]
    (it, itb), _, _ = Boot(["a", "b"], 200).mean_diff(vids, d, m)
    ok.append(abs(it - 1.0) < 1e-9 and np.allclose(itb[np.isfinite(itb)], 1.0))
    ok.append(np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06]))
    sm = {"v": np.r_[np.zeros(10), 1.0, np.zeros(10), 0.5, np.zeros(10)]}
    f = matched_found(sm, {"v": np.array([10, 30])}, 2, ratio=1.0)
    ok.append(f["v"].tolist() == [True, False])
    ok.append(len(marker_specs()) == sum((len(KS) if hk else 1) * len(WINDOWS) for *_, hk in BASE_MARKERS))
    print(f"selftest: {sum(ok)}/{len(ok)} passed")
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="multistream_bilstm")
    ap.add_argument("--with-face", nargs="+", default=["115-117"])
    ap.add_argument("--without-face", nargs="+", default=["112-114"])
    ap.add_argument("--pred-root", default=None, help="default: phrase_stats.PRED_ROOT")
    ap.add_argument("--cue-dir", default="processed_data/cue_features")
    ap.add_argument("--labels-dir", default="processed_data/BIO_tags_phrase_v2")
    ap.add_argument("--splits", default="dataset_splits.json")
    ap.add_argument("--enrich-max-videos", type=int, default=500)
    ap.add_argument("--min-enrichment", type=float, default=1.2)
    ap.add_argument("--tol", type=int, default=5, help="frames (5 = +-0.1 s)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "nonmanual_v2"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.pred_root is None:
        from phrase_stats import PRED_ROOT
        args.pred_root = PRED_ROOT
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
