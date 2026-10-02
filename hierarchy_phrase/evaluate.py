"""
hierarchy_phrase/evaluate.py -- end-to-end evaluation: flat E1s-style head vs hierarchy, predicted vs gold signs.

    python evaluate.py --stage-a sa_s42 --tag main --seed 42          # writes results/sa_s42__main_s42.json
    python aggregate.py results/*__main_s*.json                       # mean +- std over seeds

Every number comes from metrics.evaluate_videos on RAW gold at the working frame rate (25 fps by default). Thresholds are
tuned on the VALIDATION split only and applied unchanged to test; both default (0.5) and tuned numbers are reported
because IoU/ratio depend heavily on decoding thresholds.

Rows:
  flat          Stage-A phrase head (frame-level BIO), greedy thresholded decoding            -- the E1s-style baseline
  hier          sign head -> predicted signs -> Stage C P(B) per sign -> phrases               -- the method
  hier_oracle   GOLD signs -> Stage C P(B) per sign -> phrases                                 -- ceiling / cascade loss (ablation 2)
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from common import RUNS_DIR, HERE
from metrics import evaluate_videos
from sign_tokens import greedy_decode
from stage_c import Bundle, load_items, phrases_from_pB, stage_c_dir
from segments import set_end_rule

GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def score(res, key):
    return res["mF1S(0.1-0.5)"] if key == "mF1S" else res[key]


def eval_segs(P, items):
    return evaluate_videos(P, [it["gold_phrase"] for it in items], [it["T"] for it in items],
                           tols=(2, 5, 10), iou_thrs=(0.1, 0.3, 0.5, 0.7))


def tune_flat(items, key):
    best, arg = -1, (0.5, 0.5)
    for b, o in itertools.product(GRID, GRID):
        r = eval_segs([greedy_decode(it["phrase_probs"], b, o) for it in items], items)
        if score(r, key) > best:
            best, arg = score(r, key), (b, o)
    return arg


def tune_sign_decoder(items):
    """Sign decoding thresholds tuned on validation SIGN quality (mean of segment F1@0.5 and start F1@5)."""
    gold = [it["gold_sign"] for it in items]
    L = [it["T"] for it in items]
    best, arg = -1, (0.5, 0.5)
    for b, o in itertools.product(GRID, GRID):
        r = evaluate_videos([greedy_decode(it["sign_probs"], b, o) for it in items], gold, L, tols=(5,), iou_thrs=(0.5,))
        v = 0.5 * (r["seg_f1@0.5"] + r["start_f1@5"])
        if v > best:
            best, arg = v, (b, o)
    return arg


def tune_thr(bundle, items, signs_of, key):
    pBs = [(signs_of(it), bundle.pB(it, signs_of(it))) for it in items]
    best, thr = -1, 0.5
    for t in np.arange(0.1, 0.91, 0.05):
        r = eval_segs([phrases_from_pB(s, p, t) for s, p in pBs], items)
        if score(r, key) > best:
            best, thr = score(r, key), float(t)
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune-metric", default="frame_f1", choices=["frame_f1", "start_f1@5", "seg_f1@0.5", "mF1S"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--end-rule", choices=["last_sign_end", "next_start"], default="last_sign_end")
    a = ap.parse_args()
    set_end_rule(a.end_rule)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = os.path.join(RUNS_DIR, a.stage_a, "cache")
    val, test = load_items([os.path.join(base, "val")], a.limit), load_items([os.path.join(base, "test")], a.limit)
    bundle = Bundle(stage_c_dir(a.stage_a, a.tag, a.seed), device)
    print(f"Stage C: {bundle.cfg['arch']}  groups {bundle.groups}  source {bundle.cfg['source']}  params {bundle.cfg['n_params']:,}")

    # thresholds: validation only ------------------------------------------------------------------
    has_flat = all(it["has_phrase_head"] for it in val + test)
    fb, fo = tune_flat(val, a.tune_metric) if has_flat else (None, None)
    if not has_flat:
        print("cache has no phrase head (imported sign model): flat_* rows are omitted")
    sb, so = tune_sign_decoder(val)
    pred_default = lambda it: greedy_decode(it["sign_probs"], 0.5, 0.5)           # noqa: E731
    pred_tuned = lambda it: greedy_decode(it["sign_probs"], sb, so)               # noqa: E731
    gold_signs = lambda it: it["gold_sign"]                                       # noqa: E731
    thr_default = 0.5
    thr_tuned = tune_thr(bundle, val, pred_tuned, a.tune_metric)
    thr_oracle = tune_thr(bundle, val, gold_signs, a.tune_metric)
    print(f"tuned on val ({a.tune_metric}): flat b/o = {fb}/{fo} | sign decoder b/o = {sb}/{so} | phrase thr = {thr_tuned:.2f} (oracle {thr_oracle:.2f})")

    def run(items):
        rows = {} if not has_flat else {
            "flat_default": eval_segs([greedy_decode(it["phrase_probs"], 0.5, 0.5) for it in items], items),
            "flat_tuned": eval_segs([greedy_decode(it["phrase_probs"], fb, fo) for it in items], items),
        }
        rows.update({
            "hier_default": eval_segs([phrases_from_pB(pred_default(it), bundle.pB(it, pred_default(it)), thr_default) for it in items], items),
            "hier_tuned": eval_segs([phrases_from_pB(pred_tuned(it), bundle.pB(it, pred_tuned(it)), thr_tuned) for it in items], items),
            "hier_oracle_default": eval_segs([phrases_from_pB(it["gold_sign"], bundle.pB(it, it["gold_sign"]), thr_default) for it in items], items),
            "hier_oracle_tuned": eval_segs([phrases_from_pB(it["gold_sign"], bundle.pB(it, it["gold_sign"]), thr_oracle) for it in items], items),
        })
        # sign-segmentation quality of the cascade's first stage (context for the cascade loss)
        sg = evaluate_videos([pred_tuned(it) for it in items], [it["gold_sign"] for it in items], [it["T"] for it in items], tols=(2, 5, 10), iou_thrs=(0.5,))
        rows["sign_stage_tuned"] = sg
        return rows

    out = {"stage_a": a.stage_a, "tag": a.tag, "seed": a.seed, "tune_metric": a.tune_metric, "end_rule": a.end_rule,
           "thresholds": {"flat": [fb, fo], "sign_decoder": [sb, so], "phrase_thr": thr_tuned, "phrase_thr_oracle": thr_oracle},
           "stage_c_config": {k: bundle.cfg[k] for k in ("arch", "source", "groups", "n_params")},
           "val": run(val), "test": run(test)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{a.stage_a}__{a.tag}_s{a.seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    cols = ["frame_f1", "frame_f1_B", "mask_iou", "ratio", "start_f1@2", "start_f1@5", "start_f1@10", "seg_f1@0.5", "mF1S(0.1-0.5)"]
    print(f"\nTEST  (thresholds from validation)\n{'row':<22}" + "".join(f"{c:>14}" for c in cols))
    for k, r in out["test"].items():
        print(f"{k:<22}" + "".join(f"{r[c]:>14.3f}" for c in cols))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()