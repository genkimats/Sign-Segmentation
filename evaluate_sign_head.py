"""
evaluate_sign_head.py -- saves the SIGN head's predictions of sign-head runs and scores sign starts.

For every run (val and test, the same windows as training / evaluate_phrase.py):
  evaluation_phrase/predictions_sign/<run>/<split>.npz   {video: (3, T) float16 sign O / I / B probabilities}
  evaluation_phrase/predictions_sign/<run>/<split>_meta.json
and prints sign-start boundary F1 at +-2 / +-5 frames against processed_data/BIO_tags
(argmax starts, and argmax starts with duplicates closer than 3 frames merged), plus the share of
gold PHRASE starts that have a predicted sign start within +-5 frames -- the most a decoder that
only puts phrase starts at predicted sign starts (sign_decoding_study.py) can recall.

The phrase predictions come from evaluate_phrase.py as usual (--save-predictions); this script does
not change them.

  python evaluate_sign_head.py --model multistream_bilstm --prefixes 360 361 --splits val test
"""
import argparse
import json
import os

import numpy as np
import torch

import evaluate_phrase as ep
from phrase_stats import boundary_counts, f1_from_counts, gold_starts_of
from src.dataset import SIGN_LABELS_DEFAULT_DIR
from src.evaluation import predict_video_probs
from src.metrics import B_TAG, extract_segments, merge_close_starts
from src.model_factory import build_model_kwargs

OUT_ROOT = os.path.join("evaluation_phrase", "predictions_sign")


def sign_starts(probs, merge=0):
    segs = extract_segments(probs.argmax(axis=0))
    if merge:
        segs = merge_close_starts(segs, probs[B_TAG], merge)
    return np.array([s for s, _ in segs], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="multistream_bilstm")
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for pfx in args.prefixes:
        run = f"{args.model}-{pfx}"
        with open(os.path.join(ep.EXP_DIR, run, "hyperparameters.json")) as f:
            config = json.load(f)
        if not config.get("sign_head"):
            print(f"{run}: no sign head -- skipped")
            continue
        sign_dir = config.get("sign_labels_dir", SIGN_LABELS_DEFAULT_DIR)
        model = None
        for split in args.splits:
            ds = ep.get_dataset(config, split)
            if model is None:
                cls, kw = build_model_kwargs(config, detected_hamer_dim=ds.detected_hamer_dim,
                                             detected_dinov2_dim=ds.detected_dinov2_dim,
                                             detected_cue_dim=getattr(ds, "detected_cue_dim", None))
                model = cls(**kw).to(device)
                model.load_state_dict(torch.load(os.path.join(ep.MODEL_DIR, f"{run}.pth"), map_location=device),
                                      strict=True)
                model.eval()
            bs = args.batch_size or config.get("batch_size", 16)
            out = {}
            c = {"argmax": [0, 0, 0], "merge3": [0, 0, 0], "merge3@2": [0, 0, 0], "phrase_cover": [0, 0]}
            for vid, cached in ds.video_cache.items():
                _, sp = predict_video_probs(model, config, cached, device, bs, with_sign=True)
                out[vid] = sp.astype(np.float16)
                gp = os.path.join(sign_dir, f"{vid}.npy")
                if not os.path.exists(gp):
                    continue
                g = np.load(gp).astype(np.int64)
                T = min(len(g), sp.shape[1])
                gs = gold_starts_of(g[:T])
                for name, st, k in (("argmax", sign_starts(sp[:, :T]), 5), ("merge3", sign_starts(sp[:, :T], 3), 5),
                                    ("merge3@2", sign_starts(sp[:, :T], 3), 2)):
                    tp, npred, ngold = boundary_counts(gs, st, k)
                    c[name] = [c[name][0] + tp, c[name][1] + npred, c[name][2] + ngold]
                ph = gold_starts_of(np.asarray(cached["hard_labels"])[:T])          # gold PHRASE starts
                st = sign_starts(sp[:, :T], 3)
                if len(ph):
                    near = np.abs(ph[:, None] - st[None, :]).min(axis=1) <= 5 if len(st) else np.zeros(len(ph), bool)
                    c["phrase_cover"][0] += int(near.sum())
                    c["phrase_cover"][1] += len(ph)
            d = os.path.join(args.out_root, run)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{split}.npz"), **out)
            with open(os.path.join(d, f"{split}_meta.json"), "w") as f:
                json.dump({"run": run, "split": split, "sign_labels_dir": sign_dir,
                           "labels_dir": ep.labels_dir_for(config), "description": config.get("description", "")},
                          f, indent=2)
            r = {"run": run, "split": split,
                 "sign_f1_5_argmax": f1_from_counts(*c["argmax"]), "sign_f1_5_merge3": f1_from_counts(*c["merge3"]),
                 "sign_f1_2_merge3": f1_from_counts(*c["merge3@2"]),
                 "phrase_starts_covered_5": c["phrase_cover"][0] / max(1, c["phrase_cover"][1])}
            rows.append(r)
            print(f"[{split}] {run}: sign-start F1@5 argmax {r['sign_f1_5_argmax']:.3f} | merge3 "
                  f"{r['sign_f1_5_merge3']:.3f} (F1@2 {r['sign_f1_2_merge3']:.3f}) | gold phrase starts with a "
                  f"predicted sign start within +-5: {r['phrase_starts_covered_5']:.1%}  -> {d}/{split}.npz")
    if rows:
        path = os.path.join(args.out_root, "sign_head_summary.csv")
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write(",".join(rows[0]) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
        print(f"summary appended to {path}")


if __name__ == "__main__":
    main()