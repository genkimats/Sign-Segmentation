"""
count_params.py -- parameter counts for stgcn_bimamba, stgcn_bilstm, stgcn_transformer
with HaMeR as an extra input branch (the setup in src/models.py / train.py).

Run from the project root (the folder that contains src/):

    python count_params.py                       # analytic table, plus real-model check if torch+mamba_ssm import
    python count_params.py --hamer-dim 288       # set your HaMeR feature size (train.py auto-detects it from the data)
    python count_params.py --no-hamer            # same models without the HaMeR branch
    python count_params.py --analytic-only       # never touch torch / src.models

Two numbers are reported per model:
  trainable : parameters with requires_grad=True (what the optimizer updates)
  total     : trainable + the frozen adjacency matrices inside SpatialGraphConv
              (nn.Parameter(..., requires_grad=False), 2 blocks x V*V entries)
Positional-encoding tables are buffers, not parameters, and are not counted.

The analytic path needs only the standard library. If torch and your src.models
import, the script ALSO builds the real classes with the same kwargs train.py uses and
prints any disagreement with the analytic count, so a wrong assumption here shows up
instead of silently producing a wrong table.
"""
import argparse
import math
import os
import sys


# ----------------------------------------------------------------- primitives
def linear(i, o, bias=True):
    return i * o + (o if bias else 0)


def layernorm(d):
    return 2 * d


def batchnorm(c):
    return 2 * c


def conv2d(i, o, kh, kw, bias=True):
    return i * o * kh * kw + (o if bias else 0)


def lstm(input_size, hidden, layers, bidirectional=True):
    """torch.nn.LSTM: per layer and direction W_ih (4h x in), W_hh (4h x h), b_ih, b_hh (4h each)."""
    dirs = 2 if bidirectional else 1
    total = 0
    for layer in range(layers):
        in_sz = input_size if layer == 0 else hidden * dirs
        total += dirs * (4 * hidden * in_sz + 4 * hidden * hidden + 2 * 4 * hidden)
    return total


def transformer_encoder_layer(d, ff):
    """torch.nn.TransformerEncoderLayer (post-norm): MHA(in_proj + out_proj) + 2 linears + 2 LayerNorms."""
    mha = (3 * d * d + 3 * d) + (d * d + d)
    return mha + linear(d, ff) + linear(ff, d) + 2 * layernorm(d)


def mamba_block(d_model, d_state=16, d_conv=4, expand=2):
    """mamba_ssm.Mamba (v1 block): in_proj, depthwise conv1d(+bias), x_proj, dt_proj(+bias), A_log, D, out_proj."""
    d_inner = expand * d_model
    dt_rank = math.ceil(d_model / 16)
    return (linear(d_model, 2 * d_inner, bias=False)
            + d_inner * d_conv + d_inner                     # depthwise conv weight (groups=d_inner) + bias
            + linear(d_inner, dt_rank + 2 * d_state, bias=False)
            + linear(dt_rank, d_inner)                       # dt_proj, bias=True
            + d_inner * d_state                              # A_log
            + d_inner                                        # D
            + linear(d_inner, d_model, bias=False))


def stgcn_block(i, o, V, temporal_kernel=9):
    """src/stgcn.py STGCNBlock. Returns (trainable, frozen_adjacency)."""
    gcn = conv2d(i, o, 1, 1)
    tcn = batchnorm(o) + conv2d(o, o, temporal_kernel, 1) + batchnorm(o)
    res = conv2d(i, o, 1, 1) if i != o else 0
    return gcn + tcn + res, V * V


# ------------------------------------------------------------------- analytic
def analytic(model, a):
    """Returns (ordered {component: trainable params}, frozen adjacency params)."""
    V, d = a.num_vertices, a.d_model
    parts, frozen = {}, 0

    b1, f1 = stgcn_block(a.in_channels, a.stgcn_channels, V)
    b2, f2 = stgcn_block(a.stgcn_channels, a.stgcn_channels, V)
    parts["ST-GCN (2 blocks)"] = b1 + b2
    frozen = f1 + f2

    bridge = V * a.stgcn_channels
    if a.hamer_dim:
        parts["HaMeR encoder"] = linear(a.hamer_dim, a.hamer_proj_dim) + layernorm(a.hamer_proj_dim)
        bridge += a.hamer_proj_dim
    parts["input projection"] = linear(bridge, d) + layernorm(d)

    if model == "stgcn_bimamba":
        parts["temporal backbone (BiMamba)"] = 2 * a.n_layers * mamba_block(d, a.mamba_d_state, a.mamba_d_conv, a.mamba_expand)
        parts["classifier"] = linear(2 * d, 3)
    elif model == "stgcn_bilstm":
        parts["temporal backbone (BiLSTM)"] = lstm(d, d, a.n_layers, bidirectional=True)
        parts["classifier"] = linear(2 * d, 3)
    elif model == "stgcn_transformer":
        parts["temporal backbone (Transformer)"] = a.n_layers * transformer_encoder_layer(d, a.dim_feedforward)
        parts["classifier"] = linear(d, 3)
    else:
        raise ValueError(model)
    return parts, frozen


# ------------------------------------------------------------- real-model check
def real_counts(model, a):
    """Build the real class the way train.py does. Returns (trainable, total, per-top-level-module) or None."""
    sys.path.insert(0, os.getcwd())
    try:
        import torch  # noqa: F401
        from src import models as M
    except Exception as e:  # torch / mamba_ssm / src missing
        return None, f"{type(e).__name__}: {e}"
    cls = {"stgcn_bimamba": M.STGCN_BiMamba, "stgcn_bilstm": M.STGCN_BiLSTM,
           "stgcn_transformer": M.STGCN_Transformer}[model]
    kw = dict(in_channels=a.in_channels, num_vertices=a.num_vertices, num_classes=3,
              d_model=a.d_model, n_layers=a.n_layers)
    if model == "stgcn_transformer":
        kw.update(nhead=a.nhead, dim_feedforward=a.dim_feedforward)
    if model in ("stgcn_bimamba",):
        kw.update(mamba_d_state=a.mamba_d_state, mamba_d_conv=a.mamba_d_conv, mamba_expand=a.mamba_expand)
    if a.hamer_dim:
        kw.update(hamer_dim=a.hamer_dim, hamer_proj_dim=a.hamer_proj_dim)
    try:
        net = cls(**kw)
    except Exception as e:
        return None, f"could not build {cls.__name__}: {type(e).__name__}: {e}"
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    return (train, total), None


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hamer-dim", type=int, default=288, help="HaMeR feature size per frame (default 288)")
    ap.add_argument("--hamer-proj-dim", type=int, default=64, help="models.py default is 64")
    ap.add_argument("--no-hamer", action="store_true")
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-vertices", type=int, default=65)
    ap.add_argument("--stgcn-channels", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim-feedforward", type=int, default=None, help="default 4*d_model, as train.py")
    ap.add_argument("--mamba-d-state", type=int, default=16)
    ap.add_argument("--mamba-d-conv", type=int, default=4)
    ap.add_argument("--mamba-expand", type=int, default=2)
    ap.add_argument("--analytic-only", action="store_true")
    a = ap.parse_args()
    if a.no_hamer:
        a.hamer_dim = None
    if a.dim_feedforward is None:
        a.dim_feedforward = 4 * a.d_model

    models = ["stgcn_bimamba", "stgcn_bilstm", "stgcn_transformer"]
    results = {m: analytic(m, a) for m in models}

    comps = []
    for m in models:
        for k in results[m][0]:
            if k not in comps:
                comps.append(k)
    # fixed, readable row order
    order = ["ST-GCN (2 blocks)", "HaMeR encoder", "input projection"]
    backbones = [c for c in comps if c.startswith("temporal")]
    rows = [c for c in order if c in comps] + backbones + ["classifier"]

    w = 34
    print(f"\nHaMeR: {'off' if not a.hamer_dim else f'on (dim {a.hamer_dim} -> {a.hamer_proj_dim})'}   "
          f"in_channels={a.in_channels}  V={a.num_vertices}  d_model={a.d_model}  layers={a.n_layers}\n")
    print(f"{'component':<{w}}" + "".join(f"{m:>20}" for m in models))
    print("-" * (w + 20 * len(models)))
    for r in rows:
        line = f"{r:<{w}}"
        for m in models:
            v = results[m][0].get(r)
            line += f"{v:>20,}" if v is not None else f"{'-':>20}"
        print(line)
    print("-" * (w + 20 * len(models)))
    train = {m: sum(results[m][0].values()) for m in models}
    froz = {m: results[m][1] for m in models}
    print(f"{'trainable':<{w}}" + "".join(f"{train[m]:>20,}" for m in models))
    print(f"{'+ frozen adjacency (A)':<{w}}" + "".join(f"{froz[m]:>20,}" for m in models))
    print(f"{'total (incl. frozen)':<{w}}" + "".join(f"{train[m] + froz[m]:>20,}" for m in models))
    print(f"{'trainable, millions':<{w}}" + "".join(f"{train[m] / 1e6:>19.2f}M" for m in models))
    backbone_only = {m: results[m][0][[k for k in results[m][0] if k.startswith('temporal')][0]] for m in models}
    print(f"{'temporal backbone only, millions':<{w}}" + "".join(f"{backbone_only[m] / 1e6:>19.2f}M" for m in models))

    if a.analytic_only:
        return
    print("\nCross-check against the real classes in src/models.py:")
    all_ok = True
    for m in models:
        got, err = real_counts(m, a)
        if got is None:
            print(f"  {m:<18} skipped ({err})")
            all_ok = None
            continue
        tr, tot = got
        ok = (tr == train[m]) and (tot == train[m] + froz[m])
        all_ok = all_ok and ok if all_ok is not None else all_ok
        print(f"  {m:<18} real trainable={tr:,} total={tot:,}  "
              f"{'MATCHES analytic' if ok else f'MISMATCH (analytic {train[m]:,} / {train[m] + froz[m]:,})'}")
    if all_ok is None:
        print("  (run from the project root in the training environment to enable this check; "
              "the analytic table above does not depend on it)")


if __name__ == "__main__":
    main()