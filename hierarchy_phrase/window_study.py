"""
hierarchy_phrase/window_study.py -- does the inference window / overlap matter? NO retraining: re-runs a trained Stage A on the validation
(and optionally test) videos under many inference modes and reports sign and flat-phrase quality for each.

    python window_study.py --run al_sa_s42                      # val, whole-video vs windows 64..1024 x keep 1.0 / 0.75 / 0.5
    python window_study.py --run al_sa_s42 --windows 64 256 --keeps 1.0 0.75 --splits val test

Rows: window 0 = the whole video in one pass (what stage_a.py does by default). keep = fraction of each window that is kept: 1.0 is plain
concatenation of non-overlapping windows (what the project did with 64-frame windows), 0.75 keeps each window's central 75% (overlap 25%),
0.5 its central half. Columns per head (sign / phrase): argmax frame macro-F1 and #pred/#gold ratio (the 2023 primary metric), plus the
greedy-decoded segment F1@0.5 and start-boundary F1@5 at the default 0.5/0.5 thresholds. Choose the mode on VAL; only then look at test.
A trained window shorter than the phrase context (a phrase is ~100 frames at 25 fps) is expected to hurt the phrase head most.
"""
import argparse
import json
import os

import numpy as np
import torch

from common import HERE, RUNS_DIR
from metrics import evaluate_videos
from segments import bio_to_segments
from sign_tokens import greedy_decode, logits_to_probs
from stage_a import Store, infer_video, load_model


def score(sg, ph, store):
    rows = {}
    for head, outs, key in (("sign", sg, "sign"), ("phrase", ph, "phrase")):
        if outs is None:
            continue
        gold = [g[key] for g in store.gold]
        lens = [g["T"] for g in store.gold]
        am = evaluate_videos([bio_to_segments(o.argmax(1)) for o in outs], gold, lens, tols=(5,), iou_thrs=(0.5,))
        gd = evaluate_videos([greedy_decode(logits_to_probs(o)) for o in outs], gold, lens, tols=(5,), iou_thrs=(0.5,))
        rows[head] = {"argmax_F1": am["frame_f1"], "argmax_ratio": am["ratio"], "greedy_segF1@.5": gd["seg_f1@0.5"],
                      "greedy_startF1@5": gd["start_f1@5"], "greedy_ratio": gd["ratio"]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--windows", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    ap.add_argument("--keeps", nargs="+", type=float, default=[1.0, 0.75, 0.5])
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.run, device)
    stride = cfg["stride"]
    modes = [(0, 1.0)] + [(w, k) for w in a.windows for k in a.keeps]
    out = {}
    for split in a.splits:
        store = Store(split, stride)
        print(f"\n=== {split}: {len(store)} videos, run {a.run} (trained on crops of {cfg['crop']} frames, arch {cfg.get('arch', 'bilstm')}) ===")
        print(f"{'window':>7} {'keep':>5} | {'sign F1':>8} {'ratio':>6} {'segF1':>6} {'start5':>6} | {'phr F1':>8} {'ratio':>6} {'segF1':>6} {'start5':>6}")
        out[split] = {}
        for w, k in modes:
            if w == 0 and k != 1.0:
                continue
            sg, ph = [], []
            for i in range(len(store)):
                s_, p_, _ = infer_video(model, store.x[i], device, w, k)
                sg.append(s_); ph.append(p_)
            r = score(sg, None if ph[0] is None else ph, store)
            out[split][f"{w}x{k}"] = r
            f = lambda d, key: d[key] if d else float("nan")                    # noqa: E731
            print(f"{('full' if w == 0 else w):>7} {k:>5.2f} | {f(r['sign'], 'argmax_F1'):>8.4f} {f(r['sign'], 'argmax_ratio'):>6.2f} "
                  f"{f(r['sign'], 'greedy_segF1@.5'):>6.3f} {f(r['sign'], 'greedy_startF1@5'):>6.3f} | "
                  + (f"{r['phrase']['argmax_F1']:>8.4f} {r['phrase']['argmax_ratio']:>6.2f} {r['phrase']['greedy_segF1@.5']:>6.3f} {r['phrase']['greedy_startF1@5']:>6.3f}" if 'phrase' in r else ""))
    path = os.path.join(HERE, "results", f"window_study_{a.run}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}\nNote: val has few videos, so differences below ~0.01 F1 are within noise.")


if __name__ == "__main__":
    main()