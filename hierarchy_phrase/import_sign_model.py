"""
hierarchy_phrase/import_sign_model.py -- use ANY sign segmenter from the main project as the hierarchy's first stage.

The decisive question after the first results is whether a BETTER sign segmenter closes the cascade gap (on test: Stage C with
gold signs 0.83 frame F1 / 0.62 start F1@5 vs predicted signs 0.60 / 0.38). The project's own HaMeR models (e.g. stgcn_bilstm-10)
are candidates. This script turns their per-frame sign logits -- exported with decoder_study/export_logits.py -- into a hierarchy
cache, so Stage C and evaluate.py run on them unchanged:

    python ../decoder_study/export_logits.py --run stgcn_bilstm-10 --splits train val test       # from the project, once
    HP_PHRASE_DIR=BIO_tags_phrase_signaligned python import_sign_model.py --export-run stgcn_bilstm-10 --as al_imp_sb10
    python stage_c.py train --stage-a al_imp_sb10 --tag main --seed 42 --source mix --groups sign_probs prosody
    python evaluate.py --stage-a al_imp_sb10 --tag main --seed 42 --tune-metric mF1S

Notes
  * Logits are sub-sampled exactly like the keypoints (x[::stride]) to the working rate.
  * The imported model has no hidden features here and no phrase head: the cache stores a dummy 1-d `h`, so Stage C must use
    --groups sign_probs prosody (stage_c.py refuses h_pool on such a cache), and evaluate.py omits the flat_* rows.
  * Train-split logits from export_logits.py are IN-SAMPLE for that model (too clean). Use --source mix or jitter for Stage C.
"""
import argparse
import os

import numpy as np

from common import PROJECT_ROOT, RUNS_DIR, DEFAULT_STRIDE, load_gold_segments, load_keypoints, save_cache_video, segs_to_arr

EXPORTS_DIR = os.path.join(PROJECT_ROOT, "decoder_study", "exports")


def import_split(export_run, split, target, stride, exports_dir=EXPORTS_DIR):
    path = os.path.join(exports_dir, export_run, f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run decoder_study/export_logits.py --run {export_run} --splits {split}")
    data = np.load(path)
    vids = sorted({k.split("__", 1)[1] for k in data.files if k.startswith("logits__")})
    n, skipped = 0, {}
    for vid in vids:
        logits = data[f"logits__{vid}"]
        g = load_gold_segments(vid, stride)
        if g is None or not g["sign"]:
            skipped["no_labels"] = skipped.get("no_labels", 0) + 1
            continue
        if len(logits) != g["T_native"]:
            skipped["length_mismatch"] = skipped.get("length_mismatch", 0) + 1
            continue
        kp = load_keypoints(vid, stride)
        lg = logits[::stride]
        T = min(len(kp), g["T"], len(lg))
        save_cache_video(os.path.join(RUNS_DIR, target, "cache", split, f"{vid}.npz"),
                         xyz=kp[:T].astype(np.float16), h=np.zeros((T, 1), np.float16), sign_logits=lg[:T].astype(np.float32),
                         phrase_logits=np.zeros((T, 3), np.float32),
                         gold_sign=segs_to_arr([s for s in g["sign"] if s[1] <= T]),
                         gold_phrase=segs_to_arr([s for s in g["phrase"] if s[1] <= T]),
                         T=np.int64(T), stride=np.int64(stride), has_phrase_head=np.int64(0))
        n += 1
    print(f"[{split}] imported {n} videos from '{export_run}' -> {os.path.join(RUNS_DIR, target, 'cache', split)}  skipped: {skipped or 'none'}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-run", required=True, help="run name under decoder_study/exports/")
    ap.add_argument("--as", dest="target", required=True, help="hierarchy run name to create (e.g. al_imp_sb10)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    a = ap.parse_args()
    for sp in a.splits:
        import_split(a.export_run, sp, a.target, a.stride)
    print(f"next: python stage_c.py train --stage-a {a.target} --tag main --seed 42 --source mix --groups sign_probs prosody")


if __name__ == "__main__":
    main()