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
  * Logits reach the working rate by averaging the softmax probabilities of each frame pair (--pool mean, default); plain
    sub-sampling (--pool sub, the first version) can drop the 1-2-frame-wide Begin peaks of a model trained with tolerance 1.
  * The imported model has no hidden features here: the cache stores a dummy 1-d `h`, so Stage C must use
    --groups sign_probs prosody (stage_c.py refuses h_pool on such a cache).
  * --phrase-export-run adds a FLAT phrase model's logits (exported the same way) as the flat_* rows of evaluate.py, so the hierarchy is
    compared with your own phrase model on identical metrics. Without it, evaluate.py omits the flat_* rows.
  * Two different runs can share a name in different directories: export them with  export_logits.py --root <dir> --as <unique name>.
  * Train-split logits from export_logits.py are IN-SAMPLE for that model (too clean). Use --source mix or jitter for Stage C.
"""
import argparse
import os

import numpy as np

from common import PROJECT_ROOT, RUNS_DIR, DEFAULT_STRIDE, load_gold_segments, load_keypoints, save_cache_video, segs_to_arr

EXPORTS_DIR = os.path.join(PROJECT_ROOT, "decoder_study", "exports")


def downsample_logits(logits, stride, pool="mean"):
    """(T, 3) native-rate logits -> (ceil(T/stride), 3) at the working rate.
    'sub'  keeps every stride-th frame (what x[::stride] does to the keypoints) -- can drop narrow B peaks.
    'mean' averages the softmax probabilities over each block of `stride` frames (block j = frames [j*stride, (j+1)*stride),
           aligned with the sub-sampled keypoint frame j*stride) and returns their log: a 1-frame B peak keeps half its mass."""
    if stride == 1 or pool == "sub":
        return logits[::stride]
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    T = len(p); Tw = -(-T // stride)
    pad = Tw * stride - T
    if pad:
        p = np.concatenate([p, np.repeat(p[-1:], pad, axis=0)])
    q = p.reshape(Tw, stride, p.shape[1]).mean(1)
    return np.log(np.clip(q, 1e-8, 1.0)).astype(np.float32)


def _load_export(exports_dir, run, split):
    path = os.path.join(exports_dir, run, f"{split}.npz")
    return np.load(path) if os.path.exists(path) else None


def import_split(export_run, split, target, stride, exports_dir=EXPORTS_DIR, phrase_run=None, pool="mean"):
    path = os.path.join(exports_dir, export_run, f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run decoder_study/export_logits.py --run {export_run} --splits {split}")
    data = np.load(path)
    vids = sorted({k.split("__", 1)[1] for k in data.files if k.startswith("logits__")})
    n, skipped = 0, {}
    ph = _load_export(exports_dir, phrase_run, split) if phrase_run else None
    if phrase_run and ph is None:
        if split != "train":
            raise FileNotFoundError(f"phrase export {phrase_run}/{split}.npz not found -- export the phrase model for '{split}'")
        print(f"[{split}] no phrase-model export for train: fine, Stage C never uses phrase logits (flat rows use val/test only)")
    for vid in vids:
        logits = data[f"logits__{vid}"]
        g = load_gold_segments(vid, stride)
        if g is None or not g["sign"]:
            skipped["no_labels"] = skipped.get("no_labels", 0) + 1
            continue
        if len(logits) != g["T_native"]:
            skipped["length_mismatch"] = skipped.get("length_mismatch", 0) + 1
            continue
        phr = None
        if ph is not None:
            key = f"logits__{vid}"
            if key not in ph.files or len(ph[key]) != g["T_native"]:
                skipped["phrase_export_missing_or_length"] = skipped.get("phrase_export_missing_or_length", 0) + 1
                continue
            phr = downsample_logits(ph[key], stride, pool)
        kp = load_keypoints(vid, stride)
        lg = downsample_logits(logits, stride, pool)
        T = min(len(kp), g["T"], len(lg))
        save_cache_video(os.path.join(RUNS_DIR, target, "cache", split, f"{vid}.npz"),
                         xyz=kp[:T].astype(np.float16), h=np.zeros((T, 1), np.float16), sign_logits=lg[:T].astype(np.float32),
                         phrase_logits=(phr[:T] if phr is not None else np.zeros((T, 3))).astype(np.float32),
                         gold_sign=segs_to_arr([s for s in g["sign"] if s[1] <= T]),
                         gold_phrase=segs_to_arr([s for s in g["phrase"] if s[1] <= T]),
                         T=np.int64(T), stride=np.int64(stride), has_phrase_head=np.int64(phr is not None))
        n += 1
    print(f"[{split}] imported {n} videos from '{export_run}' -> {os.path.join(RUNS_DIR, target, 'cache', split)}  skipped: {skipped or 'none'}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-run", required=True, help="run name under decoder_study/exports/")
    ap.add_argument("--as", dest="target", required=True, help="hierarchy run name to create (e.g. al_imp_sb10)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--pool", choices=["mean", "sub"], default="mean",
                    help="how 50 fps logits become 25 fps: mean of probabilities per frame pair (default) or plain sub-sampling")
    ap.add_argument("--phrase-export-run", default=None,
                    help="optional: export of a FLAT phrase model (e.g. your stgcn_bilstm phrase run); its logits become the flat_* baseline rows")
    a = ap.parse_args()
    for sp in a.splits:
        import_split(a.export_run, sp, a.target, a.stride, phrase_run=a.phrase_export_run, pool=a.pool)
    print(f"next: python stage_c.py train --stage-a {a.target} --tag main --seed 42 --source mix --groups sign_probs prosody")


if __name__ == "__main__":
    main()