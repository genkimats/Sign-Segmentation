"""
export_phrase_logits.py -- export a run trained with train_phrase.py (e.g. a multistream model trained on
SIGN labels, queue_train_phrase.py SIGN_MS_STUDY) in the decoder-study export format, so it can be the sign
source of the A2 tempo features.

    python export_phrase_logits.py --run multistream_bilstm-340 --splits train val test
    python data_processing/extract_tempo_features.py --source pred \
           --sign-logits decoder_study/exports/multistream_bilstm-340 --out-dir processed_data/tempo_features_pred_ms

Output: decoder_study/exports/<run>/<split>.npz with logits__<vid> (T, 3) and labels__<vid> (T,), plus
<split>_meta.json. The "logits" are log-probabilities of the stitched window predictions (the same windows
as training, via src.evaluation.predict_split); the tempo extractor applies softmax / Viterbi to them, which
is invariant to that choice. Labels are the run's own labels_dir (processed_data/BIO_tags for sign models).

Train-split outputs are IN-SAMPLE for the exported model; the extractor's mismatch check reports how much
cleaner they are than val.
"""
import argparse
import json
import os

import numpy as np
import torch

from evaluate_phrase import EXP_DIR, MODEL_DIR, get_dataset, labels_dir_for
from src.evaluation import predict_split
from src.model_factory import build_model_kwargs

EXPORTS_DIR = os.path.join("decoder_study", "exports")


def main():
    ap = argparse.ArgumentParser(description="Export a train_phrase.py run's per-frame log-probs (decoder-study format).")
    ap.add_argument("--run", required=True, help="e.g. multistream_bilstm-340")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    with open(os.path.join(EXP_DIR, args.run, "hyperparameters.json")) as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(EXPORTS_DIR, args.run)
    os.makedirs(out_dir, exist_ok=True)
    model = None
    for split in args.splits:
        ds = get_dataset(config, split)
        if model is None:
            model_class, kw = build_model_kwargs(config, detected_hamer_dim=ds.detected_hamer_dim,
                                                 detected_dinov2_dim=ds.detected_dinov2_dim,
                                                 detected_cue_dim=getattr(ds, "detected_cue_dim", None))
            model = model_class(**kw).to(device)
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{args.run}.pth"), map_location=device),
                                  strict=True)
            model.eval()
        probs, gold = predict_split(model, config, ds, device,
                                    batch_size=args.batch_size or config.get("batch_size", 16))
        arrays = {}
        for vid, p in probs.items():
            T = min(p.shape[1], len(gold[vid]))
            arrays[f"logits__{vid}"] = np.log(np.clip(p[:, :T].T, 1e-8, 1.0)).astype(np.float32)   # (T, 3)
            arrays[f"labels__{vid}"] = gold[vid][:T].astype(np.int8)
        np.savez_compressed(os.path.join(out_dir, f"{split}.npz"), **arrays)
        with open(os.path.join(out_dir, f"{split}_meta.json"), "w") as f:
            json.dump({"run": args.run, "source": "train_phrase.py run (export_phrase_logits.py)",
                       "labels_dir": labels_dir_for(config), "window": config.get("window_size"),
                       "n_videos": len(probs), "logits_are": "log-probabilities", "hyperparameters": config},
                      f, indent=2)
        print(f"[{split}] saved {len(probs)} videos to {out_dir}/{split}.npz")


if __name__ == "__main__":
    main()