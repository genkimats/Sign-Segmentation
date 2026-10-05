"""
check_stream_balance.py -- how much does each input stream (ST-GCN graph features,
HaMeR, DINOv2) actually contribute to the model, measured at the projection layer
where the streams are concatenated?

For every run of a model it reports, per stream:
  dims          number of input dimensions the stream has at the projection
  in RMS        typical size of the stream's input values
  W col norm    average size of the projection weights per input dimension
                (if the model compensated for a small stream, this is larger for it)
  norm share    share of the projection output's size coming from this stream
                (|W_s x_s| / sum over streams), averaged over frames
  var share     share of the frame-to-frame VARIATION coming from this stream --
                the part that can carry information (a constant offset can't)
and an ablation test (unless --no-ablation): the stream is replaced by its average
value over the split (so the input stays realistic but carries no information), and
the change in predictions and metrics is measured:
  changed       % of frames whose argmax prediction changes
  dF1 / dIoU / d%   change in Frame F1, IoU and % (ratio) vs. the normal model

Reading the result:
  - small var share AND few changed frames when ablated -> the model largely ignores
    that stream (imbalance; balancing the streams is worth trying)
  - small var share but many changed frames -> the stream is small but used
  - large shares and large changes -> the stream is used

Usage (repo root):
  python check_stream_balance.py --model stgcn_bilstm
  python check_stream_balance.py --model stgcn_bilstm --prefixes 10 --split val
Only works for models that concatenate streams before a `projection` layer
(stgcn_bilstm, stgcn_transformer, ...) -- other models are skipped with a message.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from evaluate_phrase import discover_runs, choose_model_interactively, get_dataset, EXP_DIR, MODEL_DIR
from src.model_factory import build_model_kwargs
from src.evaluation import predict_split
from src.metrics import evaluate_videos


# ==============================================================================
# Locating the projection and the stream slices
# ==============================================================================
def find_projection_and_streams(model):
    """
    Returns (linear, [(name, start, end), ...]) where `linear` is the first Linear layer
    whose input is the concatenated bridge (model.bridge_dim), and the slices follow the
    concatenation order used in models.py: graph features, then HaMeR, then DINOv2.
    """
    bridge_dim = getattr(model, "bridge_dim", None)
    if bridge_dim is None:
        raise RuntimeError("model has no `bridge_dim` (streams are not concatenated before a projection).")
    linear = None
    for attr in ("projection", "feature_proj"):  # stgcn_bilstm / stgcn_transformer naming
        proj = getattr(model, attr, None)
        if proj is None:
            continue
        for m in proj.modules():
            if isinstance(m, nn.Linear) and m.in_features == bridge_dim:
                linear = m
                break
        if linear is not None:
            break
    if linear is None:
        raise RuntimeError("could not find a Linear layer with in_features == bridge_dim in "
                           "`projection` / `feature_proj`.")

    hamer_dim = model.hamer_encoder[0].out_features if getattr(model, "hamer_dim", None) else 0
    dinov2_dim = model.dinov2_encoder[0].out_features if getattr(model, "dinov2_dim", None) else 0
    graph_dim = bridge_dim - hamer_dim - dinov2_dim  # = stgcn_proj_dim when the graph branch is projected

    streams = [("ST-GCN", 0, graph_dim)]
    if hamer_dim:
        streams.append(("HaMeR", graph_dim, graph_dim + hamer_dim))
    if dinov2_dim:
        streams.append(("DINOv2", graph_dim + hamer_dim, bridge_dim))
    return linear, streams


# ==============================================================================
# Streaming statistics collected with a forward hook (nothing large is stored)
# ==============================================================================
class StreamStats:
    def __init__(self, linear, streams):
        self.W = linear.weight.detach()          # (d_model, bridge_dim)
        self.streams = streams
        self.n = 0
        self.sum_x = {s: None for s, _, _ in streams}     # per input dim (for the ablation mean)
        self.sum_x2 = {s: 0.0 for s, _, _ in streams}     # for input RMS
        self.sum_c = {s: None for s, _, _ in streams}     # per output dim
        self.sum_c2 = {s: None for s, _, _ in streams}
        self.sum_norm = {s: 0.0 for s, _, _ in streams}   # sum over frames of |c_s|
        self.sum_share = {s: 0.0 for s, _, _ in streams}  # sum over frames of |c_s| / sum_k |c_k|

    @torch.no_grad()
    def hook(self, module, inputs, output):
        x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])  # (frames, bridge_dim)
        self.n += x.shape[0]
        norms = {}
        for name, a, b in self.streams:
            xs = x[:, a:b]
            c = xs @ self.W[:, a:b].T.float()                            # (frames, d_model)
            sx = xs.sum(0)
            self.sum_x[name] = sx if self.sum_x[name] is None else self.sum_x[name] + sx
            self.sum_x2[name] += float((xs ** 2).sum())
            sc, sc2 = c.sum(0), (c ** 2).sum(0)
            self.sum_c[name] = sc if self.sum_c[name] is None else self.sum_c[name] + sc
            self.sum_c2[name] = sc2 if self.sum_c2[name] is None else self.sum_c2[name] + sc2
            norms[name] = c.norm(dim=1)
            self.sum_norm[name] += float(norms[name].sum())
        total = sum(norms.values()) + 1e-12
        for name in norms:
            self.sum_share[name] += float((norms[name] / total).sum())

    def summary(self):
        rows = {}
        variances = {}
        for name, a, b in self.streams:
            mean_c = self.sum_c[name] / self.n
            var = float((self.sum_c2[name] / self.n - mean_c ** 2).clamp(min=0).sum())
            variances[name] = var
        total_var = sum(variances.values()) + 1e-12
        for name, a, b in self.streams:
            dims = b - a
            rows[name] = {
                "dims": dims,
                "in_rms": (self.sum_x2[name] / (self.n * dims)) ** 0.5,
                "w_col_norm": float(self.W[:, a:b].norm(dim=0).mean()),
                "norm_share": self.sum_share[name] / self.n,
                "var_share": variances[name] / total_var,
            }
        return rows

    def stream_mean(self, name):
        return self.sum_x[name] / self.n


class MeanAblation:
    """Forward pre-hook replacing one stream's slice of the projection input with its split mean."""

    def __init__(self, start, end, mean):
        self.start, self.end, self.mean = start, end, mean

    def __call__(self, module, inputs):
        x = inputs[0].clone()
        x[..., self.start:self.end] = self.mean.to(x.dtype)
        return (x,)


# ==============================================================================
# One run
# ==============================================================================
def check_run(run_name, split, device, do_ablation, batch_size_override):
    with open(os.path.join(EXP_DIR, run_name, "hyperparameters.json")) as f:
        config = json.load(f)
    dataset = get_dataset(config, split)
    model_class, model_kwargs = build_model_kwargs(
        config, detected_hamer_dim=dataset.detected_hamer_dim, detected_dinov2_dim=dataset.detected_dinov2_dim)
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{run_name}.pth"), map_location=device), strict=True)
    model.eval()

    linear, streams = find_projection_and_streams(model)
    batch_size = batch_size_override or config.get("batch_size", 16)

    # Pass 1: normal predictions + streaming statistics at the projection input
    stats = StreamStats(linear, streams)
    handle = linear.register_forward_hook(stats.hook)
    try:
        probs, gold = predict_split(model, config, dataset, device, batch_size=batch_size)
    finally:
        handle.remove()
    base = evaluate_videos(probs, gold, decoder="argmax")
    base_pred = {v: p.argmax(0) for v, p in probs.items()}

    rows = stats.summary()

    # Pass 2..k: replace one stream at a time by its mean
    if do_ablation and len(streams) > 1:
        for name, a, b in streams:
            ablation = MeanAblation(a, b, stats.stream_mean(name))
            handle = linear.register_forward_pre_hook(ablation)
            try:
                probs_ab, _ = predict_split(model, config, dataset, device, batch_size=batch_size)
            finally:
                handle.remove()
            ab = evaluate_videos(probs_ab, gold, decoder="argmax")
            changed = sum(int((base_pred[v] != probs_ab[v].argmax(0)).sum()) for v in probs_ab)
            total = sum(len(base_pred[v]) for v in probs_ab)
            rows[name].update({
                "changed": changed / max(1, total),
                "dF1": ab["Frame_F1"] - base["Frame_F1"],
                "dIoU": ab["IoU"] - base["IoU"],
                "dPct": ab["Pct"] - base["Pct"],
            })
    return config, base, rows


def print_table(rows, base, has_ablation):
    print(f"  normal model (argmax): Frame F1 {base['Frame_F1']:.4f} | IoU {base['IoU']:.4f} | "
          f"% (ratio) {base['Pct']:.4f}")
    header = f"  {'stream':<8} {'dims':>6} {'in RMS':>8} {'W col norm':>11} {'norm share':>11} {'var share':>10}"
    if has_ablation:
        header += f" | {'changed':>8} {'dF1':>8} {'dIoU':>8} {'d%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, r in rows.items():
        line = (f"  {name:<8} {r['dims']:>6} {r['in_rms']:>8.3f} {r['w_col_norm']:>11.4f} "
                f"{r['norm_share']:>10.1%} {r['var_share']:>10.1%}")
        if has_ablation and "changed" in r:
            line += (f" | {r['changed']:>7.2%} {r['dF1']:>+8.4f} {r['dIoU']:>+8.4f} {r['dPct']:>+8.3f}")
        print(line)


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Measure how much each input stream (ST-GCN / HaMeR / DINOv2) contributes at the "
                    "projection layer, and what happens when it is removed (mean ablation).")
    parser.add_argument("--model", help="Model basename, e.g. stgcn_bilstm (asks if omitted).")
    parser.add_argument("--prefixes", nargs="*", help="Only these prefixes (default: all).")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Split (default: val).")
    parser.add_argument("--no-ablation", action="store_true", help="Skip the ablation passes (faster).")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size override.")
    args = parser.parse_args()

    runs = discover_runs(EXP_DIR, MODEL_DIR)
    if not runs:
        print("No runs found.")
        return
    model_name = args.model or choose_model_interactively(runs)
    if model_name not in runs:
        print(f"No saved runs for '{model_name}'. Available: {sorted(runs)}")
        return
    selected = runs[model_name]
    if args.prefixes:
        wanted = {int(p) for p in args.prefixes}
        selected = [(p, r) for p, r in selected if int(p) in wanted]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _, run_name in selected:
        print(f"\n--- {run_name} ---")
        try:
            config, base, rows = check_run(run_name, args.split, device, not args.no_ablation, args.batch_size)
        except Exception as e:
            print(f"  ⚠️  Skipped: {type(e).__name__}: {e}")
            continue
        print(f"  {config.get('description', '')}")
        if len(rows) == 1:
            print("  (only one stream -- this run uses no HaMeR/DINOv2 branch, nothing to compare)")
        print_table(rows, base, has_ablation=not args.no_ablation and len(rows) > 1)


if __name__ == "__main__":
    main()