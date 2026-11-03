"""
export_features.py -- freeze a trained multi-stream phrase run and export, per video, the features
its ENCODER receives (the fused streams after fusion / change-point / FiLM, i.e. the input of the
BiLSTM or Transformer), so FACT and the frozen-BiLSTM reference can replace exactly that encoder.

    python fact_phrase/export_features.py --run multistream_bilstm-310               # R0, seed 42
    python fact_phrase/export_features.py --run multistream_bilstm-208               # CE reference

Output (fact_phrase/features/<run>/):
    <split>/<video>.npy        (T, d_model) float16 at the raw 50 fps frame rate
    <split>_probs.npz          the run's own stitched (3, T) probabilities (sanity check / reference)
    meta.json                  source config, labels_dir, feature dim, video list and lengths per split

How it works: the windows, batching and stitching are those of src/evaluation.py (same windows as
training, overlapping windows averaged). A forward pre-hook on the encoder captures its input for
each window; the run's own probabilities are recomputed in the same pass and, for the first video
of every split, checked against src.evaluation.predict_video_probs -- if the two differ the export
stops, because then the features would not be what the encoder saw.

Train-split features are IN-SAMPLE for the frozen front end (it was trained on those videos). The
frozen-BiLSTM reference gets exactly the same features, so the FACT-vs-BiLSTM comparison is fair;
just do not compare train-split numbers with val/test.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from evaluate_phrase import EXP_DIR, MODEL_DIR, get_dataset, labels_dir_for  # noqa: E402
from src.evaluation import _prepare_window, predict_video_probs, video_windows  # noqa: E402
from src.metrics import evaluate_videos  # noqa: E402
from src.model_factory import build_model_kwargs, call_model  # noqa: E402
from src.models import MultiStreamSegmenter  # noqa: E402

FEATURES_ROOT = os.path.join("fact_phrase", "features")


def encoder_entry(model):
    """The module whose INPUT is the fused representation the encoder consumes."""
    if model.encoder_type in ("bilstm", "bilstm_attn"):
        return model.lstm
    if getattr(model, "legacy_encoder", False):
        return model.pos_encoder
    return model.rel_encoder


@torch.no_grad()
def probs_and_features(model, config, cached, device, batch_size, captured):
    """src.evaluation.predict_video_probs, extended to also stitch the captured encoder input."""
    window_size = config["window_size"]
    overlap = config.get("overlap", 0)
    k = max(1, int(config.get("temporal_downsample_factor", 1)))
    use_hamer = config.get("use_hamer_features", False)
    use_dinov2 = config.get("use_dinov2_features", False)
    use_cues = config.get("use_cue_features", False)
    features = cached["features"]
    T = features.shape[0]
    target_T = window_size // k
    windows = video_windows(T, window_size, overlap)
    if config.get("use_full_length", False) and len(windows) == 1:
        target_T = -(-T // k)
    prob_sum = np.zeros((3, T), dtype=np.float64)
    feat_sum = None
    counts = np.zeros(T, dtype=np.float64)

    def side(name, ws):
        x = torch.stack([_prepare_window(cached[name], s, e, k, target_T, False) for s, e in ws])
        return torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)

    for b0 in range(0, len(windows), batch_size):
        ws = windows[b0:b0 + batch_size]
        feats = torch.stack([_prepare_window(features, s, e, k, target_T, True) for s, e in ws])
        feats = torch.nan_to_num(feats.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        hamer = side("hamer_features", ws) if use_hamer else None
        dinov2 = side("dinov2_features", ws) if use_dinov2 else None
        cues = side("cue_features", ws) if use_cues else None
        captured.clear()
        logits = call_model(model, config, feats, hamer, dinov2, cues)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        enc_in = captured["x"].float().cpu().numpy()                     # (B, T_model, D)
        if feat_sum is None:
            feat_sum = np.zeros((T, enc_in.shape[-1]), dtype=np.float64)
        for (s, e), p, f in zip(ws, probs, enc_in):
            raw_len = e - s
            if k > 1:
                p = np.repeat(p, k, axis=1)
                f = np.repeat(f, k, axis=0)
            if p.shape[1] < raw_len:
                p = np.concatenate([p, np.repeat(p[:, -1:], raw_len - p.shape[1], axis=1)], axis=1)
                f = np.concatenate([f, np.repeat(f[-1:], raw_len - f.shape[0], axis=0)], axis=0)
            prob_sum[:, s:e] += p[:, :raw_len]
            feat_sum[s:e] += f[:raw_len]
            counts[s:e] += 1.0
    counts[counts == 0] = 1.0
    return (prob_sum / counts).astype(np.float32), (feat_sum / counts[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Export a phrase run's encoder-input features for the FACT pilot.")
    ap.add_argument("--run", required=True, help="e.g. multistream_bilstm-310 (R0, seed 42)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out-root", default=FEATURES_ROOT)
    ap.add_argument("--overwrite", action="store_true", help="re-export videos that already exist")
    args = ap.parse_args()

    with open(os.path.join(EXP_DIR, args.run, "hyperparameters.json")) as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_root, args.run)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"source_run": args.run, "labels_dir": labels_dir_for(config), "config": config,
                 "feature_point": "encoder input (fused streams after fusion / change-point / FiLM)",
                 "fps": 50})
    meta.setdefault("splits", {})

    model = None
    captured = {}                                                        # filled by the encoder pre-hook
    for split in args.splits:
        t0 = time.time()
        dataset = get_dataset(config, split)
        if model is None:
            model_class, kwargs = build_model_kwargs(
                config, detected_hamer_dim=dataset.detected_hamer_dim,
                detected_dinov2_dim=dataset.detected_dinov2_dim,
                detected_cue_dim=getattr(dataset, "detected_cue_dim", None))
            model = model_class(**kwargs).to(device)
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{args.run}.pth"), map_location=device),
                                  strict=True)
            model.eval()
            if not isinstance(model, MultiStreamSegmenter):
                raise SystemExit(f"{args.run} is a {type(model).__name__}; only multi-stream runs are supported.")
            def grab(_module, inputs):
                captured["x"] = inputs[0].detach()
            encoder_entry(model).register_forward_pre_hook(grab)
        bs = args.batch_size or config.get("batch_size", 16)
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        probs_all, frames, gold = {}, {}, {}
        n_bytes = 0
        for i, (vid, cached) in enumerate(dataset.video_cache.items()):
            path = os.path.join(split_dir, f"{vid}.npy")
            probs, feats = probs_and_features(model, config, cached, device, bs, captured)
            if i == 0:                                                   # the export must reproduce the run
                ref = predict_video_probs(model, config, cached, device, bs)
                err = float(np.abs(ref - probs).max())
                if err > 1e-4:
                    raise SystemExit(f"stitched probabilities differ from src.evaluation by {err:.2e} -- not exporting")
                print(f"  [{split}] check vs src.evaluation.predict_video_probs: max |diff| = {err:.1e}  ok")
            if not np.isfinite(feats).all():
                raise SystemExit(f"non-finite features in {vid}")
            if args.overwrite or not os.path.exists(path):
                np.save(path, feats.astype(np.float16))
            n_bytes += feats.shape[0] * feats.shape[1] * 2
            probs_all[vid] = probs.astype(np.float16)
            frames[vid] = int(feats.shape[0])
            gold[vid] = np.asarray(cached["hard_labels"]).astype(np.int64)
            meta["dim"] = int(feats.shape[1])
        np.savez_compressed(os.path.join(out_dir, f"{split}_probs.npz"), **probs_all)
        meta["splits"][split] = {"videos": sorted(frames), "frames": frames}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        m = evaluate_videos({v: p.astype(np.float32) for v, p in probs_all.items()}, gold, light=True)
        print(f"  [{split}] {len(frames)} videos, {n_bytes / 1e9:.2f} GB of features, "
              f"{time.time() - t0:.0f}s | the run's own argmax on this split: BndF1@5 {m['BndF1_5']:.4f}, "
              f"IoU {m['IoU']:.4f}, % {m['Pct']:.3f}  (should match evaluate_phrase.py's argmax row)")
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
