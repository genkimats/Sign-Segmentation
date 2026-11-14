"""
analyze_nonmanual_direct.py -- DIRECT comparison of the models with and without the face at phrase
starts with and without a facial cue, plus a sensitivity check (does the face model simply predict more
phrase starts?). Exploratory follow-up to the pre-registered test (analyze_nonmanual_pooled.py).

Why: the interaction (face effect at cued starts - face effect at uncued starts) removes anything the face
does everywhere, but it hides the two effects it is made of. Here each is reported on its own:
  cued starts    face vs no face at starts marked by the eyes / brows      (does the face help there?)
  uncued starts  face vs no face at starts without that marker             (does it also help elsewhere?)
and, to rule out "the face model just fires more often":
  sensitivity    predicted phrase starts per gold start, boundary precision, false starts per minute

Starts, markers and negatives are exactly those of analyze_nonmanual_v2.py / _pooled.py (same markers
fixed on the training videos, same face-visibility filter, same in-phrase negatives).

MEASURES (all on the test videos; tol = +-5 frames)
  Decoded (the run's own threshold decoder of Moryossef et al. (2023), thresholds tuned on validation, read
  from predictions/<run>/test_meta.json; 0.5 / 0.5 if the run was saved without --tune-on):
    hit rate            % of starts with a predicted phrase start within +-tol (one-to-one matching)
    starts per gold     number of predicted phrase starts / number of gold phrase starts (all gold starts,
                        B-tags included) -> > 1 means the model predicts more starts than exist
    precision           % of predicted starts within +-tol of a gold start
    false starts / min  predicted starts not matched to any gold start, per minute of video (50 fps)
  Threshold-free:
    start score         percentile (0-100) of the max smoothed P(start) within +-tol, among all frames of
                        the video (the rank metric of the earlier analyses)
    in-phrase score     the same score at moments inside phrases (>= 0.5 s from any start): lower is
                        better; shows whether the face lowers the score where there is NO start
    matched recall      % of starts found when both models keep the same number of peaks (top-K over the
                        split, K = number of gold starts) -> sensitivity cannot differ by construction

HOW TO READ
  hit rate at cued starts up AND starts per gold / false starts / min not up -> the face finds the cued
  starts without simply predicting more starts. If starts per gold goes up, look at matched recall
  (same number of predictions for both models) and precision.

STATISTICS
  Group value = mean over the runs of a group (per-video sums averaged over runs, then pooled). Face - no
  face with a 95% CI from a bootstrap over test videos (same draws for every measure and pair) and a
  two-sided bootstrap p. Seed level: mean +- SD per group, number of same-seed differences > 0, and
  whether every face run beats every no-face run (exact one-sided permutation p, 3 vs 3 -> .05).

    python analyze_nonmanual_direct.py
    python analyze_nonmanual_direct.py --pair "MS z790" multistream_bilstm 373-375 428-430
Output: evaluation_phrase/nonmanual_direct/{summary.txt, measures.csv, runs.csv}
"""
import argparse
import csv
import json
import os
import sys
from math import comb

import numpy as np

import analyze_nonmanual_v2 as v2
from analyze_nonmanual_pooled import DEFAULT_PAIRS, PRIMARY_IDS
from src.metrics import decode_threshold_2023, extract_segments, match_boundaries

FPS = 50
SHORT_PAUSE = 12
PAIRS = [
    ("MS B+H(+Fc) z790 [confirmatory]", "multistream_bilstm", ["373-375"], ["428-430"]),
    ("MS face-regions z790 [confirmatory]", "multistream_bilstm", ["411-413"], ["428-430"]),
] + DEFAULT_PAIRS[:3]          # the plain early-fusion pairs have broken saved predictions -> left out
MARKERS = ("eyes", "brows")

# measure -> (label, unit, higher is better)
MEASURES = {
    "starts_per_gold": ("predicted starts per gold start", "ratio", None),
    "precision": ("boundary precision", "%", True),
    "false_per_min": ("false starts per minute", "per min", False),
    "hit_all": ("hit rate, all starts", "%", True),
    "score_all": ("start score, all starts", "pp", True),
    "score_neg": ("in-phrase score (no start)", "pp", False),
}
for _mk in MARKERS:
    for _side, _txt in (("cued", "cued"), ("uncued", "uncued")):
        MEASURES[f"hit_{_mk}_{_side}"] = (f"hit rate, {_mk}-{_txt} starts", "%", True)
        MEASURES[f"score_{_mk}_{_side}"] = (f"start score, {_mk}-{_txt} starts", "pp", True)
        MEASURES[f"matched_{_mk}_{_side}"] = (f"matched recall, {_mk}-{_txt} starts", "%", True)
        MEASURES[f"hitshort_{_mk}_{_side}"] = (f"hit rate, {_mk}-{_txt}, short pauses", "%", True)


def thresholds(meta):
    for k in ("tuned_thresholds", "fixed_thresholds"):
        if meta.get(k):
            return tuple(meta[k]), k.split("_")[0]
    return (0.5, 0.5), "default"


def run_tables(run, pred_root, cue_dir, names, specs, tol):
    """Per-video (sum, count) of every measure for one run, plus its seed and thresholds."""
    from phrase_stats import load_predictions
    probs, meta, gold = load_predictions(pred_root, run, "test")
    (b_t, o_t), thr_kind = thresholds(meta)
    rng = np.random.default_rng(0)                        # same negatives as analyze_nonmanual_v2.collect_split
    per_vid, smooth, starts_by_vid, rows = {}, {}, {}, {}
    for v in sorted(probs):
        cues = v2.load_cues(cue_dir, v, names)
        if cues is None:
            continue
        T = min(len(gold[v]), len(next(iter(cues.values()))))
        g = gold[v][:T]
        cues = {k: a[:T] for k, a in cues.items()}
        p = probs[v][:, :T]
        st = v2.starts_of(g)
        ng = v2.negatives_of(g, st, rng)
        st, ng = st[v2.face_ok(cues, st)], ng[v2.face_ok(cues, ng)]
        M = v2.mark_moments(cues, st, specs)               # (2, n_starts)
        short = np.array([v2.pause_before(g, int(s)) <= SHORT_PAUSE for s in st], dtype=bool)
        sc = v2.run_scores(p[2], st, ng, tol)
        smooth[v], starts_by_vid[v] = sc["smooth"], st
        pred = np.array([s for s, _ in decode_threshold_2023(p, b_t, o_t)], dtype=np.int64)
        gold_all = np.array([s for s, _ in extract_segments(g)], dtype=np.int64)
        g_ok_all, p_ok, _ = match_boundaries(gold_all, pred, tol)
        hit, _, _ = match_boundaries(st, pred, tol)
        rows[v] = {"M": M, "short": short, "hit": hit, "rank": sc["rank_s"] * 100}
        per_vid[v] = {"starts_per_gold": (len(pred), len(gold_all)),
                      "precision": (100 * p_ok.sum(), len(pred)),
                      "false_per_min": ((~p_ok).sum(), T / (FPS * 60)),
                      "hit_all": (100 * hit.sum(), len(st)),
                      "score_all": (sc["rank_s"].sum() * 100, len(st)),
                      "score_neg": (sc["rank_n"].sum() * 100, len(ng))}
    found = v2.matched_found(smooth, starts_by_vid, tol)
    for v, r in rows.items():
        for j, mk in enumerate(MARKERS):
            for side, sel in (("cued", r["M"][j]), ("uncued", ~r["M"][j])):
                per_vid[v][f"hit_{mk}_{side}"] = (100 * r["hit"][sel].sum(), sel.sum())
                per_vid[v][f"score_{mk}_{side}"] = (r["rank"][sel].sum(), sel.sum())
                per_vid[v][f"matched_{mk}_{side}"] = (100 * found[v][sel].sum(), sel.sum())
                s2 = sel & r["short"]
                per_vid[v][f"hitshort_{mk}_{side}"] = (100 * r["hit"][s2].sum(), s2.sum())
    return per_vid, str(meta.get("seed", "")), (b_t, o_t), thr_kind


def ratio(S, N):
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / N


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", action="append", default=None, metavar="X",
                    help="NAME MODEL WITH_FACE WITHOUT_FACE (prefix ranges); repeatable; replaces the defaults")
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--cue-dir", default="processed_data/cue_features")
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "nonmanual_direct"))
    args = ap.parse_args()
    from phrase_stats import PRED_ROOT, expand_prefixes
    from src.cue_sets import cue_names_path
    pred_root = args.pred_root or PRED_ROOT
    pairs = [(p[0], p[1], [p[2]], [p[3]]) for p in args.pair] if args.pair else PAIRS
    with open(cue_names_path(args.cue_dir)) as f:
        names = json.load(f)
    allspecs = {s["id"]: s for s in v2.marker_specs()}
    specs = [allspecs[PRIMARY_IDS["eyes"]], allspecs[PRIMARY_IDS["brows"]]]

    cache, boot, vids = {}, None, None
    L = ["Direct face vs no-face comparison at cued / uncued phrase starts + sensitivity check (test; EXPLORATORY)",
         f"markers: eyes = {PRIMARY_IDS['eyes']}, brows = {PRIMARY_IDS['brows']}; tol = +-{args.tol} frames; "
         f"short pause <= {SHORT_PAUSE} frames; decoder = threshold decoder (Moryossef et al., 2023), each run's "
         "own thresholds",
         "units: % = percent of starts (hit rate, matched recall) or of predicted starts (precision); "
         "pp = percentile points (0-100); diff = face - no face",
         "face / no face = group value +- SD over seeds; 95% CI: bootstrap over test videos; seeds: same-seed pairs "
         "where the face run is better ('higher' for starts per gold), 'sep' = every face run beats every no-face "
         "run (exact permutation p)", ""]
    m_rows, r_rows = [], []
    for name, model, wf, nf in pairs:
        groups = {"face": expand_prefixes(model, wf), "noface": expand_prefixes(model, nf)}
        missing = [r for g in groups.values() for r in g if not os.path.exists(os.path.join(pred_root, r, "test.npz"))]
        if missing:
            L += [f"[{name}] skipped -- no test predictions for {', '.join(missing)}", ""]
            continue
        print(f"{name} ...")
        for runs in groups.values():
            for r in runs:
                if r not in cache:
                    cache[r] = run_tables(r, pred_root, args.cue_dir, names, specs, args.tol)
        pv = sorted(set.intersection(*[set(cache[r][0]) for rs in groups.values() for r in rs]))
        if boot is None:
            vids, boot = pv, v2.Boot(pv, args.n_boot)
        elif pv != vids:
            L += [f"[{name}] skipped -- different test videos", ""]
            continue
        thr_txt = "; ".join(f"{r.split('-')[-1]}: b={cache[r][2][0]} o={cache[r][2][1]} ({cache[r][3]})"
                            for rs in groups.values() for r in rs)
        L += [f"[{name}]  face: {' '.join(groups['face'])}   no face: {' '.join(groups['noface'])}",
              f"  thresholds  {thr_txt}",
              f"  {'measure':<40s} {'unit':>7s} {'face':>16s} {'no face':>16s}  {'diff (95% CI)':<34s} {'p':>6s}  seeds"]
        for key, (label, unit, better) in MEASURES.items():
            # per-video sums, averaged over the runs of each group
            SN = {}
            for gname, runs in groups.items():
                S = np.mean([[cache[r][0][v][key][0] for v in vids] for r in runs], axis=0)
                N = np.mean([[cache[r][0][v][key][1] for v in vids] for r in runs], axis=0)
                SN[gname] = (S, N)
            est = {g: ratio(S.sum(), N.sum()) for g, (S, N) in SN.items()}
            draws = ratio(boot.W @ SN["face"][0], boot.W @ SN["face"][1]) - \
                ratio(boot.W @ SN["noface"][0], boot.W @ SN["noface"][1])
            diff = est["face"] - est["noface"]
            lo, hi = v2.ci(draws)
            p = v2.pval(draws)
            # seed level
            per_run = {g: [(cache[r][1], ratio(sum(cache[r][0][v][key][0] for v in vids),
                                                sum(cache[r][0][v][key][1] for v in vids))) for r in runs]
                       for g, runs in groups.items()}
            f = np.array([x for _, x in per_run["face"]])
            n = np.array([x for _, x in per_run["noface"]])
            sf, sn = dict(per_run["face"]), dict(per_run["noface"])
            common = [s for s in sf if s and s in sn]
            same = [sf[s] - sn[s] for s in common]
            if better is False:                            # lower is better: "beats" = lower
                sep = bool(f.max() < n.min())
            else:
                sep = bool(f.min() > n.max())
            perm = 1 / comb(len(f) + len(n), len(f))
            sd = lambda a: a.std(ddof=1) if len(a) > 1 else np.nan  # noqa: E731
            sign = -1 if better is False else 1           # count "face better" (or "face higher" if neutral)
            word = "face higher" if better is None else "face better"
            n_good = sum(sign * d > 0 for d in same)
            seeds_txt = f"{word} {n_good}/{len(same)}" + (f", sep (p={perm:.2f})" if sep else "")
            L.append(f"  {label:<40s} {unit:>7s} {est['face']:>8.2f} ± {sd(f):<5.2f} {est['noface']:>8.2f} ± {sd(n):<5.2f}"
                     f"  {diff:+7.2f} ({lo:+.2f} to {hi:+.2f}){'':<10s} {p:>6.3f}  {seeds_txt}")
            m_rows.append({"pair": name, "measure": key, "label": label, "unit": unit,
                           "face": est["face"], "face_sd_seeds": sd(f), "noface": est["noface"],
                           "noface_sd_seeds": sd(n), "diff": diff, "ci_lo": lo, "ci_hi": hi, "p": p,
                           "same_seed_face_better": f"{n_good}/{len(same)}",
                           "face_runs_beat_all_noface": sep, "perm_p_if_sep": perm if sep else np.nan})
            for g, lst in per_run.items():
                for (seed, x), r in zip(lst, groups[g]):
                    r_rows.append({"pair": name, "run": r, "face": int(g == "face"), "seed": seed,
                                   "measure": key, "value": x})
        L.append("")
    text = "\n".join(L)
    print("\n" + text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    for fname, rows in (("measures.csv", m_rows), ("runs.csv", r_rows)):
        if rows:
            with open(os.path.join(args.out_dir, fname), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"\nwrote {args.out_dir}/summary.txt, measures.csv, runs.csv")


if __name__ == "__main__":
    sys.exit(main())
