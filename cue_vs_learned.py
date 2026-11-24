"""
cue_vs_learned.py -- engineered cue features compared with learned features (midterm question: "was any
feature engineering done?").

The engineered features are the 48 per-frame cues of data_processing/extract_cue_features.py (eye openness,
blinks, brow height, mouth opening, head nod / height / speed, wrist height and speed, holds, hand openness,
HaMeR finger flexion and articulation speed, ...), robust-z normalized per video. They were trained in
CUE_STUDY (multistream_bilstm 157-177, 3 seeds each, same recipe as 112-117). No new training is needed:
this script evaluates those runs with the main-result decoder and compares them with the learned streams.

Questions and the arms that answer them (defaults; change with --arm NAME=run,run,run):
  Q1  How far do engineered cues get on their own?        cues36 / cues36_tc / cues48  vs  learned (B+H+Fc)
  Q2  Can engineered face cues replace the learned face?  bh_facecues (B+H + 12 face cues)  vs  learned
      (and vs B+H without any face: do the face cues add anything?)
  Q3  Do engineered cues add on top of learned features?  learned_plus_cues (B+H+Fc + 36 cues)  vs  learned

Metrics, per run, then averaged over the 3 seeds:
  boundary F1 (+-5 frames), peak picking + threshold tuned on validation per run (tune_peaks_on_val) -- the
  decoder of the main results; boundary precision / recall from the same decoder;
  frame-level F1 (macro over O / I / B, argmax) -- the selection guard.
Intervals: paired bootstrap over videos (phrase_stats.paired_bootstrap, runs averaged within each arm).

Selection rule (thesis section 6): validation boundary F1, with frame-level F1 not significantly worse. Run
on --split val for the decision; --split test only for reporting, as for every other comparison.

The script also prints the training configs of the first run of each arm and the keys that differ, so a
difference in recipe (class weights, epoch selection metric, machine) is not mistaken for a feature effect.

Needs saved predictions (val for tuning; plus the split you evaluate):
    python evaluate_phrase.py --model multistream_bilstm --prefixes 112-117 157-177 --split val --save-predictions
    python evaluate_phrase.py --model multistream_bilstm --prefixes 112-117 157-177 --split test --save-predictions
Then:
    python cue_vs_learned.py --split val
    python cue_vs_learned.py --split test
Output: evaluation_phrase/cue_vs_learned/{summary_<split>.txt, per_run_<split>.csv}
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

M = "multistream_bilstm"
DEFAULT_ARMS = {
    "learned":           ([f"{M}-{i}" for i in (115, 116, 117)], "Learned: B+H+Fc (proposed)"),
    "learned_bh":        ([f"{M}-{i}" for i in (112, 113, 114)], "Learned: B+H (no face)"),
    "cues36":            ([f"{M}-{i}" for i in (157, 158, 159)], "Engineered only: 36 selected cues"),
    "cues36_tc":         ([f"{M}-{i}" for i in (160, 161, 162)], "Engineered only: 36 cues + temporal conv"),
    "cues48":            ([f"{M}-{i}" for i in (163, 164, 165)], "Engineered only: all 48 cues"),
    "bh_facecues":       ([f"{M}-{i}" for i in (166, 167, 168)], "B+H + 12 face cues (instead of learned face)"),
    "bh_nonmanualcues":  ([f"{M}-{i}" for i in (169, 170, 171)], "B+H + 19 non-manual cues (face + head)"),
    "bh_cues36":         ([f"{M}-{i}" for i in (172, 173, 174)], "B+H + 36 selected cues"),
    "learned_plus_cues": ([f"{M}-{i}" for i in (175, 176, 177)], "B+H+Fc + 36 selected cues"),
}
# (arm A, arm B, question): reported as A - B
COMPARISONS = [
    ("cues36", "learned", "Q1 engineered only (36) vs learned"),
    ("cues36_tc", "learned", "Q1 engineered only (36 + temporal conv) vs learned"),
    ("cues48", "learned", "Q1 engineered only (48) vs learned"),
    ("bh_facecues", "learned", "Q2 face: 12 engineered cues vs learned face encoder"),
    ("bh_nonmanualcues", "learned", "Q2 face+head: 19 engineered cues vs learned face encoder"),
    ("bh_facecues", "learned_bh", "Q2 do engineered face cues add to B+H?"),
    ("learned", "learned_bh", "Q2 reference: does the learned face add to B+H?"),
    ("bh_cues36", "learned_bh", "Q3 do 36 cues add to B+H?"),
    ("learned_plus_cues", "learned", "Q3 do 36 cues add to B+H+Fc?"),
]
K = 5


# ----------------------------------------------------------------------------- counts
def frame_counts(probs, gold):
    """(3, T) probs, (T,) gold -> (tp_O, fp_O, fn_O, tp_I, fp_I, fn_I, tp_B, fp_B, fn_B)."""
    pred = probs.argmax(axis=0)
    out = []
    for c in range(3):
        p, g = pred == c, gold == c
        out += [int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())]
    return tuple(out)


def macro_f1(*c):
    f = []
    for i in range(3):
        tp, fp, fn = c[3 * i: 3 * i + 3]
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


def bnd_precision(tp, n_pred, n_gold):
    return tp / n_pred if n_pred else 0.0


def bnd_recall(tp, n_pred, n_gold):
    return tp / n_gold if n_gold else 0.0


def run_counts(run, split, pred_root):
    from count_oracle_decoding import peaks_with_heights, tune_peaks_on_val
    from phrase_stats import boundary_counts, gold_starts_of, load_predictions
    from src.metrics import B_TAG
    tuned = tune_peaks_on_val(load_predictions(pred_root, run, "val"))
    probs, meta, gold = load_predictions(pred_root, run, split)
    bnd, frm = {}, {}
    for v, p in probs.items():
        idx, h = peaks_with_heights(p[B_TAG], tuned["min_dist"], tuned["sigma"])
        starts = np.sort(idx[h >= tuned["thr"]])
        bnd[v] = boundary_counts(gold_starts_of(gold[v]), starts, K)
        frm[v] = frame_counts(p, gold[v])
    return bnd, frm, tuned, meta


def pooled(counts, metric):
    return metric(*np.sum([counts[v] for v in counts], axis=0))


# ----------------------------------------------------------------------------- configs
CONFIG_KEYS = ("config", "hyperparameters", "hparams", "args", "train_config")


def config_of(run, pred_root, meta):
    """Training config of a run: from the prediction meta, a config.json next to the checkpoint, or the
    checkpoint itself. Returns {} when none is found."""
    for k in CONFIG_KEYS:
        if isinstance(meta.get(k), dict):
            return meta[k]
    for pat in (f"saved_models_phrase/{run}*.json", f"saved_models_phrase/{run}/*.json",
                f"saved_models_phrase/{run}*config*"):
        for p in sorted(glob.glob(pat)):
            try:
                with open(p) as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    return d
            except Exception:                                   # noqa: BLE001
                pass
    for p in sorted(glob.glob(f"saved_models_phrase/{run}.pt*")) + sorted(glob.glob(f"saved_models_phrase/{run}/*.pt*")):
        try:
            import torch
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                for k in CONFIG_KEYS:
                    if isinstance(ck.get(k), dict):
                        return ck[k]
        except Exception:                                       # noqa: BLE001
            pass
    return {}


IGNORE = {"description", "seed", "prefix", "run", "timestamp", "date", "best_epoch", "save_path", "model_path"}
STREAM_KEYS = ("use_stream", "cue_set", "cue_temporal_kernel", "face_subset", "basename")


def config_report(cfgs):
    """cfgs: {arm: config} -> lines listing keys that differ between arms (stream switches marked)."""
    have = {a: c for a, c in cfgs.items() if c}
    if len(have) < 2:
        return ["  (configs not found -- compare saved_models_phrase/<run> configs by hand: class_weights,",
                "   force_selection_metric / selection metric, epochs, learning_rate, window_size)"]
    keys = sorted(set().union(*[set(c) for c in have.values()]) - IGNORE)
    L = []
    for k in keys:
        vals = {a: json.dumps(c.get(k, "<absent>"), sort_keys=True, default=str) for a, c in have.items()}
        if len(set(vals.values())) > 1:
            tag = "stream" if k.startswith(STREAM_KEYS) else "RECIPE"
            L.append(f"  [{tag}] {k}: " + "; ".join(f"{a}={v}" for a, v in vals.items()))
    if not any("[RECIPE]" in l for l in L):
        L.insert(0, "  only input/stream switches differ (good: the comparison is about the features)")
    else:
        L.insert(0, "  WARNING: recipe keys differ -- the comparison mixes features and training recipe")
    missing = [a for a, c in cfgs.items() if not c]
    if missing:
        L.append(f"  configs not found for: {', '.join(missing)}")
    return L


# ----------------------------------------------------------------------------- main
def fmt_ci(res):
    lo, hi = res["ci95"]
    return f"{res['diff']:+.3f} (95% CI: {lo:+.3f} to {hi:+.3f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--arm", action="append", default=[], help="override an arm: NAME=run,run,run")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "cue_vs_learned"))
    args = ap.parse_args()
    from phrase_stats import PRED_ROOT, f1_from_counts, paired_bootstrap
    pred_root = args.pred_root or PRED_ROOT

    arms = {k: list(v[0]) for k, v in DEFAULT_ARMS.items()}
    names = {k: v[1] for k, v in DEFAULT_ARMS.items()}
    for spec in args.arm:
        k, runs = spec.split("=", 1)
        arms[k] = [r.strip() for r in runs.split(",") if r.strip()]
        names.setdefault(k, k)

    data, rows, cfgs, skipped = {}, [], {}, []
    for arm, runs in arms.items():
        per = []
        for r in runs:
            try:
                bnd, frm, tuned, meta = run_counts(r, args.split, pred_root)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"skip {r}: {e}")
                skipped.append(r)
                continue
            per.append((r, bnd, frm, tuned))
            if arm not in cfgs:
                cfgs[arm] = config_of(r, pred_root, meta)
            rows.append({"arm": arm, "run": r, "videos": len(bnd),
                         "bnd_f1": pooled(bnd, f1_from_counts), "bnd_p": pooled(bnd, bnd_precision),
                         "bnd_r": pooled(bnd, bnd_recall), "frame_f1": pooled(frm, macro_f1),
                         "thr": tuned["thr"], "min_dist": tuned["min_dist"], "sigma": tuned["sigma"]})
        if per:
            data[arm] = per

    L = [f"Engineered cues vs learned features ({args.split}; peak picking + threshold tuned on val per run;",
         f"boundary F1 +-{K} frames; frame-level F1 = macro over O/I/B, argmax; mean of seeds; paired bootstrap",
         f"over videos, {args.n_boot} draws)", ""]
    L.append(f"{'arm':<20s} {'n':>2s} {'bnd F1':>7s} {'bnd P':>6s} {'bnd R':>6s} {'frame F1':>8s}  description")
    for arm in data:
        rs = [x for x in rows if x["arm"] == arm]
        mean = {k: np.mean([x[k] for x in rs]) for k in ("bnd_f1", "bnd_p", "bnd_r", "frame_f1")}
        L.append(f"{arm:<20s} {len(rs):>2d} {mean['bnd_f1']:7.3f} {mean['bnd_p']:6.3f} {mean['bnd_r']:6.3f} "
                 f"{mean['frame_f1']:8.3f}  {names[arm]}")
    L += ["", "Paired differences (A - B):"]
    for a, b, q in COMPARISONS:
        if a not in data or b not in data:
            L.append(f"  {q}: skipped (missing predictions)")
            continue
        rb = paired_bootstrap([x[1] for x in data[a]], [x[1] for x in data[b]], n_boot=args.n_boot,
                              metric=f1_from_counts)
        rf = paired_bootstrap([x[2] for x in data[a]], [x[2] for x in data[b]], n_boot=args.n_boot,
                              metric=macro_f1)
        L.append(f"  {q}  [{a} - {b}]")
        L.append(f"      boundary F1    {fmt_ci(rb)}")
        L.append(f"      frame-level F1 {fmt_ci(rf)}")
    L += ["", "Training configs (first run of each arm), keys that differ:"] + config_report(cfgs)
    if skipped:
        L += ["", "Skipped (no predictions): " + ", ".join(skipped)]

    os.makedirs(args.out_dir, exist_ok=True)
    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, f"summary_{args.split}.txt"), "w") as f:
        f.write(txt + "\n")
    if rows:
        with open(os.path.join(args.out_dir, f"per_run_{args.split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {args.out_dir}/summary_{args.split}.txt, per_run_{args.split}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())