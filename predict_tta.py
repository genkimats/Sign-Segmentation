"""
predict_tta.py -- test-time inference variants for decoder idea 6 (phrase_decoder_study.py, variant "tta").

The saved predictions (evaluate_phrase.py --save-predictions) come from NON-overlapping windows (overlap 0),
so frames near a window edge are predicted with context on one side only. This script re-predicts with
  * overlapping windows (stride = --stride-frac x window, default 0.5) whose outputs are averaged with a
    centre-weighted (Hann) window, so every frame is mostly predicted from the middle of some window;
  * optionally (--mirror) the average with the prediction on the left-right mirrored input
    (src/frontend.py mirror_graph_tv / mirror_hamer; the face subset is mirrored too).
Output has the same format as evaluate_phrase.py --save-predictions:
  <out-root>/<run>/<split>.npz  {video: (3, T) float16}   and   <split>_meta.json

  python predict_tta.py --model multistream_bilstm --prefixes 115 116 117 --splits val test
  python predict_tta.py --model multistream_bilstm --prefixes 115 116 117 --splits val test --mirror \
         --out-root evaluation_phrase/predictions_tta_mirror
  python phrase_decoder_study.py --runs multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117 \
         --tta-root evaluation_phrase/predictions_tta
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_phrase as ep
from src.evaluation import _prepare_window
from src.frontend import face_mirror_perm, mirror_graph_tv, mirror_hamer
from src.model_factory import build_model_kwargs, call_model


def strided_windows(T, W, stride):
    if T <= W:
        return [(0, T)]
    starts = list(range(0, T - W + 1, stride))
    if starts[-1] + W < T:
        starts.append(T - W)
    return [(s, s + W) for s in starts]


@torch.no_grad()
def predict_weighted(model, config, cached, device, batch_size, stride_frac):
    W = int(config["window_size"])
    k = max(1, int(config.get("temporal_downsample_factor", 1)))
    target_T = W // k
    feats_tv = cached["features"]
    T = feats_tv.shape[0]
    stride = max(1, int(round(W * stride_frac)))
    wins = strided_windows(T, W, stride)
    if config.get("use_full_length", False) and len(wins) == 1:
        target_T = -(-T // k)
    pos_w = np.sin(np.pi * (np.arange(W) + 0.5) / W) ** 2 + 1e-3        # centre-weighted
    acc = np.zeros((3, T))
    wsum = np.zeros(T)
    use = {n: config.get(f"use_{n}_features", False) for n in ("hamer", "dinov2", "cue")}
    for b0 in range(0, len(wins), batch_size):
        bw = wins[b0:b0 + batch_size]
        x = torch.stack([_prepare_window(feats_tv, s, e, k, target_T, True) for s, e in bw])
        x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        side = {}
        for name, key in (("hamer", "hamer_features"), ("dinov2", "dinov2_features"), ("cue", "cue_features")):
            if use[name]:
                t = torch.stack([_prepare_window(cached[key], s, e, k, target_T, False) for s, e in bw])
                side[name] = torch.nan_to_num(t.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        logits = call_model(model, config, x, side.get("hamer"), side.get("dinov2"), side.get("cue"))
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        for (s, e), p in zip(bw, probs):
            n = e - s
            if k > 1:
                p = np.repeat(p, k, axis=1)
            if p.shape[1] < n:
                p = np.concatenate([p, np.repeat(p[:, -1:], n - p.shape[1], axis=1)], axis=1)
            w = pos_w[:n] if n == W else np.ones(n)
            acc[:, s:e] += p[:, :n] * w
            wsum[s:e] += w
    wsum[wsum == 0] = 1.0
    return (acc / wsum).astype(np.float32)


def mirrored(cached, ds, config):
    face_perm = (face_mirror_perm(config.get("face_subset", "full"))
                 if config.get("use_face_keypoints", False) and not config.get("face_only", False) else None)
    m = dict(cached)
    m["features"] = mirror_graph_tv(cached["features"].to(torch.float32), ds._mirror_x_channels,
                                    face_perm).to(cached["features"].dtype)
    if config.get("use_hamer_features", False):
        m["hamer_features"] = mirror_hamer(cached["hamer_features"].to(torch.float32)).to(cached["hamer_features"].dtype)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="multistream_bilstm")
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--stride-frac", type=float, default=0.5)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out-root", default=os.path.join("evaluation_phrase", "predictions_tta"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for pfx in args.prefixes:
        run = f"{args.model}-{pfx}"
        with open(os.path.join(ep.EXP_DIR, run, "hyperparameters.json")) as f:
            config = json.load(f)
        if args.mirror and config.get("use_cue_features", False):
            raise SystemExit(f"{run}: --mirror is not defined for the cue stream")
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
            for vid, cached in ds.video_cache.items():
                p = predict_weighted(model, config, cached, device, bs, args.stride_frac)
                if args.mirror:
                    p = 0.5 * (p + predict_weighted(model, config, mirrored(cached, ds, config), device, bs,
                                                    args.stride_frac))
                out[vid] = p.astype(np.float16)
            d = os.path.join(args.out_root, run)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{split}.npz"), **out)
            with open(os.path.join(d, f"{split}_meta.json"), "w") as f:
                json.dump({"run": run, "model": args.model, "prefix": pfx, "split": split,
                           "labels_dir": ep.labels_dir_for(config), "description": config.get("description", ""),
                           "seed": config.get("seed", ""), "tta": {"stride_frac": args.stride_frac,
                                                                   "mirror": args.mirror, "weights": "hann"}},
                          f, indent=2)
            print(f"[{split}] {run}: {len(out)} videos -> {d}/{split}.npz")


if __name__ == "__main__":
    main()