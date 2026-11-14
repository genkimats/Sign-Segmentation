"""
analyze_nonmanual_seeds.py -- seed-level view of the non-manual-start result: is the face effect at
eye- / brow-marked phrase starts larger than the seed-to-seed noise?

The pooled analysis (analyze_nonmanual_pooled.py) averages the seeds of each model before computing the
interaction, so it shows video-level uncertainty but not seed-level spread. Here every RUN is scored on
its own, with the same fixed markers, starts, negatives and metric:

  gap(run, marker) = mean start score at MARKED starts - mean start score at UNMARKED starts
                     (rank metric: percentile of the max smoothed P(start) within +-tol, in pp)

The interaction in analyze_nonmanual_pooled.py is exactly  mean(gap of face runs) - mean(gap of no-face
runs), so this table splits that number into its seeds. Per pair it reports:
  face runs    gap mean +- SD (one value per seed)
  no-face runs gap mean +- SD
  difference   mean(face) - mean(no face) (= the pooled interaction)
  seed pairs   face seed s - no-face seed s for seeds trained with the same seed (42 / 123 / 2024),
               and how many are > 0
  separation   whether every face run is above every no-face run, and the exact permutation p of that
               ordering (one-sided: 1 / C(n_face + n_noface, n_face); 3 vs 3 -> 0.05, 2 vs 3 -> 0.10)
Also, per run, the start score at ALL starts and after short pauses only (<= 12 frames), marked vs unmarked.

Seeds are a small sample (2-3 per arm): the SD is a rough guide, the permutation p is exact but can
never be below 0.05 with 3 vs 3.

    python analyze_nonmanual_seeds.py
    python analyze_nonmanual_seeds.py --pair "MS z790" multistream_bilstm 373-375 <B+H prefixes>
Output: evaluation_phrase/nonmanual_seeds/{seeds.csv, runs.csv, summary.txt}
"""
import argparse
import csv
import json
import os
import re
import sys
from math import comb

import numpy as np

import analyze_nonmanual_v2 as v2
from analyze_nonmanual_pooled import DEFAULT_PAIRS, PRIMARY_IDS

# the plain early-fusion face models have broken saved predictions (start AUC ~ .55); left out
PAIRS = DEFAULT_PAIRS[:3]
SHORT_PAUSE = 12


def run_gaps(run, pred_root, cue_dir, names, specs, tol):
    D = v2.collect_split({"face": [run]}, "test", pred_root, cue_dir, names, specs, tol)
    r = D["face_rank_s"] * 100
    short = D["pause"] <= SHORT_PAUSE
    out = {"run": run, "all_starts": r.mean()}
    for j, mk in enumerate(("eyes", "brows")):
        m = D["M_s"][j]
        out[f"{mk}_marked"] = r[m].mean()
        out[f"{mk}_unmarked"] = r[~m].mean()
        out[f"{mk}_gap"] = r[m].mean() - r[~m].mean()
        out[f"{mk}_gap_short"] = r[m & short].mean() - r[~m & short].mean()
    return out


def seed_of(run, desc):
    m = re.search(r"seed=(\d+)", desc.get(run, ""))
    return m.group(1) if m else None


def descriptions():
    """run -> description from the test metric CSVs (for the seed of each run)."""
    out = {}
    for f in ("multistream_bilstm_test_metrics.csv", "bilstm_2023_test_metrics.csv"):
        p = os.path.join("evaluation_phrase", f)
        if os.path.exists(p):
            with open(p) as fh:
                for row in csv.DictReader(fh):
                    out.setdefault(row["run"], row.get("description", ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", action="append", default=None, metavar="X",
                    help="NAME MODEL WITH_FACE WITHOUT_FACE (prefix ranges); repeatable; replaces the defaults")
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--cue-dir", default="processed_data/cue_features")
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "nonmanual_seeds"))
    args = ap.parse_args()
    from phrase_stats import PRED_ROOT, expand_prefixes
    from src.cue_sets import cue_names_path
    pred_root = args.pred_root or PRED_ROOT
    pairs = [(p[0], p[1], [p[2]], [p[3]]) for p in args.pair] if args.pair else PAIRS
    with open(cue_names_path(args.cue_dir)) as f:
        names = json.load(f)
    allspecs = {s["id"]: s for s in v2.marker_specs()}
    specs = [allspecs[PRIMARY_IDS["eyes"]], allspecs[PRIMARY_IDS["brows"]]]
    desc = descriptions()

    run_rows, seed_rows, L = [], [], []
    L += ["Seed-level face effect at non-manually marked phrase starts (test, rank metric, percentile points)",
          f"gap = start score at marked - unmarked starts; markers: eyes = {PRIMARY_IDS['eyes']}, "
          f"brows = {PRIMARY_IDS['brows']}; short pause <= {SHORT_PAUSE} frames", ""]
    for name, model, wf, nf in pairs:
        groups = {"face": expand_prefixes(model, wf), "noface": expand_prefixes(model, nf)}
        missing = [r for g in groups.values() for r in g if not os.path.exists(os.path.join(pred_root, r, "test.npz"))]
        if missing:
            L += [f"{name}: skipped, no test predictions for {', '.join(missing)}", ""]
            continue
        print(f"{name} ...")
        res = {g: [run_gaps(r, pred_root, args.cue_dir, names, specs, args.tol) for r in runs]
               for g, runs in groups.items()}
        for g, rows in res.items():
            for row in rows:
                run_rows.append({"pair": name, "face": int(g == "face"), "seed": seed_of(row["run"], desc), **row})
        L.append(f"[{name}]  face: {' '.join(groups['face'])}   no face: {' '.join(groups['noface'])}")
        for key, label in (("all_starts", "start score, all starts"),
                           ("eyes_gap", "eyes gap"), ("brows_gap", "brows gap"),
                           ("eyes_gap_short", "eyes gap, short pauses"), ("brows_gap_short", "brows gap, short pauses")):
            f = np.array([r[key] for r in res["face"]])
            n = np.array([r[key] for r in res["noface"]])
            sdf = f.std(ddof=1) if len(f) > 1 else np.nan
            sdn = n.std(ddof=1) if len(n) > 1 else np.nan
            sep = bool(f.min() > n.max())
            p_perm = 1 / comb(len(f) + len(n), len(f))
            # same-seed differences
            sf = {seed_of(r["run"], desc): r[key] for r in res["face"]}
            sn = {seed_of(r["run"], desc): r[key] for r in res["noface"]}
            common = sorted(s for s in sf if s is not None and s in sn)
            paired = [sf[s] - sn[s] for s in common]
            if not common:                       # no seed in the descriptions: pair by prefix order
                common = [f"#{i + 1}" for i in range(min(len(f), len(n)))]
                paired = list(f[: len(common)] - n[: len(common)])
            seed_rows.append({"pair": name, "measure": key, "face_mean": f.mean(), "face_sd": sdf,
                              "face_values": " ".join(f"{x:.3f}" for x in f), "noface_mean": n.mean(),
                              "noface_sd": sdn, "noface_values": " ".join(f"{x:.3f}" for x in n),
                              "diff": f.mean() - n.mean(), "all_face_above_all_noface": sep,
                              "perm_p_if_separated": p_perm if sep else np.nan,
                              "same_seed_diffs": " ".join(f"{s}:{d:+.3f}" for s, d in zip(common, paired)),
                              "same_seed_positive": f"{sum(d > 0 for d in paired)}/{len(paired)}"})
            L.append(f"  {label:<26s} face {f.mean():6.2f} ± {sdf:4.2f}   no face {n.mean():6.2f} ± {sdn:4.2f}   "
                     f"diff {f.mean() - n.mean():+5.2f}   same-seed > 0: {sum(d > 0 for d in paired)}/{len(paired)}   "
                     + (f"all face > all no-face (perm p = {p_perm:.2f})" if sep else "overlap"))
        L.append("")
    text = "\n".join(L)
    print("\n" + text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    for fname, rows in (("runs.csv", run_rows), ("seeds.csv", seed_rows)):
        if rows:
            with open(os.path.join(args.out_dir, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"\nwrote {args.out_dir}/summary.txt, runs.csv, seeds.csv")


if __name__ == "__main__":
    sys.exit(main())
