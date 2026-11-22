"""
count_nonmanual_groups.py -- how many test phrase starts are in each group of the face slide.

Uses exactly the starts, markers and pause lengths of analyze_nonmanual_seeds.py (same collect_split call,
same primary markers, short pause <= 12 frames). The counts depend only on the data and the markers, not on
the model; the run only tells collect_split which predictions to load. The script checks that two runs give
the same counts.

    python count_nonmanual_groups.py
    python count_nonmanual_groups.py --runs multistream_bilstm-373 multistream_bilstm-428
"""
import argparse
import json

import numpy as np

import analyze_nonmanual_v2 as v2
from analyze_nonmanual_pooled import PRIMARY_IDS

SHORT_PAUSE = 12


def counts(run, pred_root, cue_dir, names, specs, tol):
    D = v2.collect_split({"face": [run]}, "test", pred_root, cue_dir, names, specs, tol)
    pause = np.asarray(D["pause"], dtype=float)
    known = np.isfinite(pause)
    short = known & (pause <= SHORT_PAUSE)
    out = {"all starts": len(pause), "pause unknown": int((~known).sum()), "short pause (<= 12 frames)": int(short.sum())}
    for j, mk in enumerate(("eyes", "brows")):
        m = np.asarray(D["M_s"][j], dtype=bool)
        out[f"{mk}: marked"] = int(m.sum())
        out[f"{mk}: unmarked"] = int((~m).sum())
        out[f"{mk}: marked, short pause"] = int((m & short).sum())
        out[f"{mk}: unmarked, short pause"] = int((~m & short).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["multistream_bilstm-373", "multistream_bilstm-428"])
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--cue-dir", default="processed_data/cue_features")
    ap.add_argument("--tol", type=int, default=5)
    args = ap.parse_args()
    from phrase_stats import PRED_ROOT
    from src.cue_sets import cue_names_path
    pred_root = args.pred_root or PRED_ROOT
    with open(cue_names_path(args.cue_dir)) as f:
        names = json.load(f)
    allspecs = {s["id"]: s for s in v2.marker_specs()}
    specs = [allspecs[PRIMARY_IDS["eyes"]], allspecs[PRIMARY_IDS["brows"]]]

    res = [counts(r, pred_root, args.cue_dir, names, specs, args.tol) for r in args.runs]
    print(f"Test phrase starts per group (markers: eyes = {PRIMARY_IDS['eyes']}, brows = {PRIMARY_IDS['brows']})")
    for k in res[0]:
        print(f"  {k:<30s} {res[0][k]:>6d}")
    same = all(r == res[0] for r in res[1:])
    print(f"\nsame counts for {', '.join(args.runs)}: {'yes' if same else 'NO -- check which run the slide used'}")
    if not same:
        for r, c in zip(args.runs, res):
            print(f"  {r}: {c}")


if __name__ == "__main__":
    main()