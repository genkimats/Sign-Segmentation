"""
analyze_nonmanual_pooled.py -- replication of the non-manual-start analysis across every model pair
that differs ONLY by the face stream.

analyze_nonmanual_v2.py tested one pair (multi-stream B+H+Fc 115-117 vs B+H 112-114). If the face stream
really helps more at phrase starts marked by the eyes / brows, the same interaction should appear in
other architectures and recipes trained with and without the face. This script runs the identical,
pre-fixed analysis on each pair and pools them.

FIXED IN ADVANCE (nothing here is chosen from these results):
  markers   the two primary markers analyze_nonmanual_v2.py selected on the TRAINING videos:
              eyes   "eyes-narrowed onset w=5"
              brows  "inner-brow step k=3.0 w=5"
            plus, labelled exploratory, their union ("eyes or brows").
  metrics   the v2 rank metric (percentile of the max smoothed P(start) within +-tol) -> interaction
            = face effect at marked starts - face effect at unmarked starts; count-matched recall
            interaction; AUC interaction (gain at marked - gain at unmarked).
  pairs     DEFAULT_PAIRS below (every face / no-face pair in the experiment log with the same recipe).

STATISTICS: all pairs are evaluated on the same 17 test videos, so they are NOT independent samples of
videos. The pooled effect is the mean of the pair effects, and its interval comes from ONE joint
bootstrap: the same resampled set of videos is applied to every pair in each draw. Also reported: how
many pairs point the same way, and a sign test over pairs (p = probability of at least that many
positive pairs if each were a coin flip; a descriptive replication count).

Only TEST predictions are needed: evaluation_phrase/predictions/<run>/test.npz (evaluate_phrase.py
--split test --save-predictions). A pair whose predictions are missing is skipped and listed.

    python analyze_nonmanual_pooled.py
    python analyze_nonmanual_pooled.py --pair "MS (z790)" multistream_bilstm 373-375 <B+H prefixes>
Outputs (--out-dir, default evaluation_phrase/nonmanual_pooled): summary.txt, pairs.csv, forest.png
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import analyze_nonmanual_v2 as v2

PRIMARY_IDS = {"eyes": "eyes-narrowed onset w=5", "brows": "inner-brow step k=3.0 w=5"}

# (name, model, WITH-face prefixes, WITHOUT-face prefixes): same architecture and recipe, face is the
# only difference
DEFAULT_PAIRS = [
    ("Multi-stream B+H(+Fc)", "multistream_bilstm", ["115-117"], ["112-114"]),
    ("EF-joint B+H(+F)", "bilstm_2023", ["302-303"], ["299-301"]),
    ("EF-norm B+H(+F)", "bilstm_2023", ["296-298"], ["293-295"]),
    ("Early fusion B+H(+F)", "bilstm_2023", ["268-270"], ["259-261"]),
    ("Early fusion + TC B+H(+F)", "bilstm_2023", ["289-291"], ["280-282"]),
    ("Early fusion B(+F)", "bilstm_2023", ["262-264"], ["250-252"]),
    ("Early fusion + TC B(+F)", "bilstm_2023", ["283-285"], ["271-273"]),
]


def sign_test_p(k, n):
    """P(X >= k), X ~ Binomial(n, 0.5)."""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def pair_effects(D, specs_used, boot, n_boot):
    """Per marker: rank / matched interaction (point + draws), AUC interaction (point + draws)."""
    d_rank = D["face_rank_s"] - D["noface_rank_s"]
    d_match = D["face_matched"] - D["noface_matched"]
    out = {}
    for j, label in enumerate(specs_used):
        m = D["M_s"][j]
        (r_it, r_b), (r_m, _), (r_u, _) = boot.mean_diff(D["vid_s"], d_rank, m)
        (m_it, m_b), _, _ = boot.mean_diff(D["vid_s"], d_match, m)
        A = auc_draws(D, j, boot)
        out[label] = {"n_marked": int(m.sum()), "rank_inter": r_it, "rank_draws": r_b, "rank_marked": r_m,
                      "rank_unmarked": r_u, "matched_inter": m_it, "matched_draws": m_b,
                      "auc_inter": A[0], "auc_draws": A[1], "auc_gain_marked": A[2]}
    return out


def auc_draws(D, j, boot):
    """AUC interaction with the SAME video resamples as boot.W (joint across pairs)."""
    m = D["M_s"][j]
    vids = boot.vids
    idx_m = {v: np.flatnonzero((D["vid_s"] == v) & m) for v in vids}
    idx_u = {v: np.flatnonzero((D["vid_s"] == v) & ~m) for v in vids}
    idx_n = {v: np.flatnonzero(D["vid_n"] == v) for v in vids}

    def stat(counts):
        rep = lambda idx: np.concatenate([np.repeat(idx[v], int(c)) for v, c in zip(vids, counts) if c > 0]  # noqa: E731
                                         ) if counts.sum() else np.array([], int)
        pm, pu, pn = rep(idx_m), rep(idx_u), rep(idx_n)
        g = {}
        for grp in ("face", "noface"):
            neg = D[f"{grp}_rank_n"][pn]
            g[grp] = (v2.auc(D[f"{grp}_rank_s"][pm], neg), v2.auc(D[f"{grp}_rank_s"][pu], neg))
        gain_m, gain_u = g["face"][0] - g["noface"][0], g["face"][1] - g["noface"][1]
        return gain_m - gain_u, gain_m

    it, gm = stat(np.ones(len(vids)))
    draws = np.array([stat(w)[0] for w in boot.W[: AUC_BOOT]])
    return it, draws, gm


AUC_BOOT = 500


def forest(rows, pooled, path, metric_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    markers = list(pooled)
    fig, axes = plt.subplots(1, len(markers), figsize=(5.2 * len(markers), 0.42 * (len(rows) // len(markers)) + 2.0),
                             dpi=200, sharey=True, squeeze=False)
    colors = {"eyes": "#1F5FA8", "brows": "#2A9D8F", "eyes or brows": "#7A5195"}
    for ax, mk in zip(axes[0], markers):
        rr = [r for r in rows if r["marker"] == mk]
        ys = list(range(len(rr), 0, -1))
        for yi, r in zip(ys, rr):
            ax.hlines(yi, r["rank_lo"] * 100, r["rank_hi"] * 100, color="#BBBBBB", lw=2)
            ax.scatter(r["rank_inter"] * 100, yi, color=colors.get(mk, "#444444"), s=26, zorder=3)
        p = pooled[mk]
        ax.hlines(0, p["lo"] * 100, p["hi"] * 100, color="#222222", lw=3)
        ax.scatter(p["est"] * 100, 0, marker="D", color="#222222", s=40, zorder=4)
        ax.axvline(0, color="#444444", lw=0.8, ls="--")
        ax.set_yticks(ys + [0], [r["pair"] for r in rr] + ["Pooled (joint bootstrap)"], fontsize=7)
        ax.set_title(f"{mk}: {p['pos']}/{p['n']} pairs > 0", fontsize=9)
        ax.set_xlabel(metric_label, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", action="append", default=None, metavar="X",
                    help='NAME MODEL WITH_FACE WITHOUT_FACE (prefix ranges like 115-117); repeatable; replaces the defaults')
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--cue-dir", default="processed_data/cue_features")
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "nonmanual_pooled"))
    args = ap.parse_args()
    from phrase_stats import PRED_ROOT, expand_prefixes
    from src.cue_sets import cue_names_path
    pred_root = args.pred_root or PRED_ROOT
    pairs = ([(p[0], p[1], [p[2]], [p[3]]) for p in args.pair] if args.pair else DEFAULT_PAIRS)
    with open(cue_names_path(args.cue_dir)) as f:
        names = json.load(f)
    allspecs = {s["id"]: s for s in v2.marker_specs()}
    specs = [allspecs[PRIMARY_IDS["eyes"]], allspecs[PRIMARY_IDS["brows"]]]
    labels = ["eyes", "brows", "eyes or brows"]

    results, skipped, boot = [], [], None
    for name, model, wf, nf in pairs:
        groups = {"face": expand_prefixes(model, wf), "noface": expand_prefixes(model, nf)}
        missing = [r for g in groups.values() for r in g
                   if not os.path.exists(os.path.join(pred_root, r, "test.npz"))]
        if missing:
            skipped.append(f"{name}: no test predictions for {', '.join(missing)}")
            continue
        print(f"{name}: {', '.join(groups['face'])} vs {', '.join(groups['noface'])} ...")
        D = v2.collect_split(groups, "test", pred_root, args.cue_dir, names, specs, args.tol)
        D["M_s"] = np.vstack([D["M_s"], D["M_s"][0] | D["M_s"][1]])          # + union (exploratory)
        D["M_n"] = np.vstack([D["M_n"], D["M_n"][0] | D["M_n"][1]])
        if boot is None:
            boot = v2.Boot(D["vids"], args.n_boot)
        elif D["vids"] != boot.vids:
            skipped.append(f"{name}: different test videos ({len(D['vids'])}) -- left out of the joint bootstrap")
            continue
        results.append((name, groups, pair_effects(D, labels, boot, args.n_boot)))
    if not results:
        raise SystemExit("No pair had test predictions.\n" + "\n".join(skipped))

    os.makedirs(args.out_dir, exist_ok=True)
    L = ["Face effect at non-manually marked phrase starts -- replication over model pairs (test)",
         "Markers fixed in advance (training enrichment, analyze_nonmanual_v2.py): "
         f"eyes = {PRIMARY_IDS['eyes']}, brows = {PRIMARY_IDS['brows']}; 'eyes or brows' is exploratory.",
         "Interaction = face effect at marked starts - at unmarked starts (rank metric: percentile points / 100).",
         "Pooled = mean over pairs; 95% CI from one joint bootstrap over the 17 test videos.", ""]
    rows, pooled = [], {}
    for mk in labels:
        L.append(f"[{mk}]")
        L.append(f"  {'pair':<28s} {'n marked':>8s}  {'rank interaction':<40s} {'matched recall':>16s} {'AUC interaction':>16s}")
        draws_r, draws_m, draws_a, est_r, est_m, est_a = [], [], [], [], [], []
        for name, groups, eff in results:
            e = eff[mk]
            lo, hi = v2.ci(e["rank_draws"])
            rows.append({"pair": name, "marker": mk, "n_marked": e["n_marked"], "rank_inter": e["rank_inter"],
                         "rank_lo": lo, "rank_hi": hi, "rank_p": v2.pval(e["rank_draws"]),
                         "rank_marked": e["rank_marked"], "rank_unmarked": e["rank_unmarked"],
                         "matched_inter": e["matched_inter"], "auc_inter": e["auc_inter"],
                         "auc_gain_marked": e["auc_gain_marked"],
                         "face_runs": " ".join(groups["face"]), "noface_runs": " ".join(groups["noface"])})
            L.append(f"  {name:<28s} {e['n_marked']:>8d}  {v2.fmt(e['rank_inter'], (lo, hi)):<40s} "
                     f"{e['matched_inter']:>+16.4f} {e['auc_inter']:>+16.4f}")
            draws_r.append(e["rank_draws"]); draws_m.append(e["matched_draws"]); draws_a.append(e["auc_draws"])
            est_r.append(e["rank_inter"]); est_m.append(e["matched_inter"]); est_a.append(e["auc_inter"])
        n = len(est_r)
        pr, pm, pa = np.nanmean(draws_r, axis=0), np.nanmean(draws_m, axis=0), np.nanmean(draws_a, axis=0)
        pos = int(sum(x > 0 for x in est_r))
        pooled[mk] = {"est": float(np.mean(est_r)), "lo": v2.ci(pr)[0], "hi": v2.ci(pr)[1], "pos": pos, "n": n}
        L += [f"  {'POOLED':<28s} {'':>8s}  {v2.fmt(np.mean(est_r), v2.ci(pr), v2.pval(pr))}",
              f"  {'':<28s} {'':>8s}  matched recall {v2.fmt(np.mean(est_m), v2.ci(pm), v2.pval(pm))}",
              f"  {'':<28s} {'':>8s}  AUC            {v2.fmt(np.mean(est_a), v2.ci(pa), v2.pval(pa))}",
              f"  pairs with a positive rank interaction: {pos}/{n} (sign test p = {sign_test_p(pos, n):.3f}); "
              f"AUC: {int(sum(x > 0 for x in est_a))}/{n}; matched recall: {int(sum(x > 0 for x in est_m))}/{n}", ""]
    if skipped:
        L += ["Skipped:"] + [f"  {s}" for s in skipped]
    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out_dir, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    forest(rows, pooled, os.path.join(args.out_dir, "forest.png"),
           "Interaction (percentile points)\nface effect at marked - unmarked starts")
    print(f"\nwrote {args.out_dir}/summary.txt, pairs.csv, forest.png")


if __name__ == "__main__":
    sys.exit(main())
