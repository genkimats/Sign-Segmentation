"""
hands_on_2025.py -- the architecture of "Hands-On: Segmenting Individual Signs from
Continuous Sequences" (2025), as an nn.Module, plus a parameter counter.

WHAT THE PAPER SPECIFIES (Sec. III-C, Fig. 2)
  * frozen features: HaMeR 288-d (2x15x3x3 hand pose + 2x3x3 global orient), 3D skeleton angles 104-d
  * two feature-specific "auxiliary" modules, each a THREE-layer MLP up-projecting to 512
  * temporal downsampling by 2, then concat -> 1024
  * a "multi-modal mixer": also a THREE-layer MLP
  * a Transformer encoder classifier -> per-frame BIO (3 classes)
  * loss: CE on BIO + gloss-level CTC   (CTC head is training-only)

WHAT THE PAPER DOES NOT SPECIFY (so they are arguments here, not facts)
  * hidden widths inside the three-layer MLPs, and the mixer's output width
  * Transformer: number of layers, d_model, heads, feed-forward size
  * how the x2 downsampling is done (assumed parameter-free: take every 2nd frame)
  * how predictions return to full frame rate (assumed parameter-free: repeat)
  * normalisation inside the MLPs, positional encoding type (assumed none / sinusoidal)
  * the CTC head's gloss vocabulary size
So the parameter count of THE PAPER'S MODEL IS NOT DETERMINABLE from the paper. What this
script gives is the exact count of this implementation under stated choices, and how much
the unstated choices matter (sensitivity grid).

USAGE (project root, next to count_params.py):
    python hands_on_2025.py                       # scenario table + sensitivity + comparison with your 3 models
    python hands_on_2025.py --d-model 512 --n-layers 6 --nhead 8 --dim-feedforward 2048
    python hands_on_2025.py --gloss-vocab 1500    # also report the training-only CTC head
    python hands_on_2025.py --analytic-only       # skip the real-model build / cross-check
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_params as C  # primitives verified against the real torch modules (matched exactly on your machine)


# ============================================================ analytic counter
def count_hands_on(hamer_dim=288, angle_dim=104, adapter_dim=512, adapter_hidden=None,
                   mixer_hidden=512, d_model=256, n_layers=4, dim_feedforward=None,
                   num_classes=3, gloss_vocab=0, use_norm=False, downsample_params=0):
    """Returns ordered {component: parameter count}. Dropout / activations / sinusoidal PE have none."""
    ah = adapter_hidden or adapter_dim
    ff = dim_feedforward or 4 * d_model

    def mlp3(i, h, o):
        n = C.linear(i, h) + C.linear(h, h) + C.linear(h, o)
        if use_norm:                       # LayerNorm after the two hidden layers
            n += 2 * C.layernorm(h)
        return n

    parts = {
        "HaMeR adapter (3-layer MLP)": mlp3(hamer_dim, ah, adapter_dim),
        "Angle adapter (3-layer MLP)": mlp3(angle_dim, ah, adapter_dim),
        "temporal downsample": downsample_params,
        "mixer (3-layer MLP)": mlp3(2 * adapter_dim, mixer_hidden, d_model),
        "Transformer encoder": n_layers * C.transformer_encoder_layer(d_model, ff),
        "BIO head": C.linear(d_model, num_classes),
    }
    if gloss_vocab:
        parts["CTC head (training only)"] = C.linear(d_model, gloss_vocab + 1)   # +1 blank
    return parts


# ================================================================== the model
def build_model(hamer_dim=288, angle_dim=104, adapter_dim=512, adapter_hidden=None, mixer_hidden=512,
                d_model=256, n_layers=4, nhead=8, dim_feedforward=None, dropout=0.1,
                downsample=2, num_classes=3, gloss_vocab=0, use_norm=False):
    import torch
    import torch.nn as nn

    ah = adapter_hidden or adapter_dim
    ff = dim_feedforward or 4 * d_model

    def mlp3(i, h, o):
        layers = [nn.Linear(i, h)]
        if use_norm:
            layers.append(nn.LayerNorm(h))
        layers += [nn.GELU(), nn.Dropout(dropout), nn.Linear(h, h)]
        if use_norm:
            layers.append(nn.LayerNorm(h))
        layers += [nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o)]
        return nn.Sequential(*layers)

    class HandsOn2025(nn.Module):
        """forward(hamer (B,T,hamer_dim), angles (B,T,angle_dim)) -> (logits (B,3,T), embeddings (B,T/ds,d_model))."""

        def __init__(self):
            super().__init__()
            self.downsample = downsample
            self.d_model = d_model
            self.hamer_adapter = mlp3(hamer_dim, ah, adapter_dim)
            self.angle_adapter = mlp3(angle_dim, ah, adapter_dim)
            self.mixer = mlp3(2 * adapter_dim, mixer_hidden, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff,
                                               dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.bio_head = nn.Linear(d_model, num_classes)
            self.ctc_head = nn.Linear(d_model, gloss_vocab + 1) if gloss_vocab else None

        def _positional(self, T, device, dtype):
            pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
                            * (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(T, self.d_model, device=device)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)[:, : self.d_model // 2]
            return pe.to(dtype).unsqueeze(0)

        def forward(self, hamer, angles):
            B, T, _ = hamer.shape
            h = self.hamer_adapter(hamer)                       # (B,T,512)
            a = self.angle_adapter(angles)                      # (B,T,512)
            if self.downsample > 1:                             # parameter-free temporal downsample
                h, a = h[:, :: self.downsample], a[:, :: self.downsample]
            x = self.mixer(torch.cat([h, a], dim=-1))           # (B,T',d_model)
            x = x + self._positional(x.shape[1], x.device, x.dtype)
            emb = self.encoder(x)                               # (B,T',d_model)
            logits = self.bio_head(emb)                         # (B,T',3)
            if self.downsample > 1:                             # back to full frame rate
                logits = logits.repeat_interleave(self.downsample, dim=1)[:, :T]
            return logits.permute(0, 2, 1), emb

        def gloss_logits(self, emb):                            # CTC branch (training only)
            return self.ctc_head(emb) if self.ctc_head is not None else None

    return HandsOn2025()


# ====================================================================== report
def user_models_trainable(d_model, n_layers, ff, hamer_dim, hamer_proj_dim=64):
    """Your three models at the same temporal settings (uses count_params' verified formulas)."""
    ns = argparse.Namespace(in_channels=3, num_vertices=65, stgcn_channels=64, d_model=d_model,
                            n_layers=n_layers, nhead=8, dim_feedforward=ff, mamba_d_state=16,
                            mamba_d_conv=4, mamba_expand=2, hamer_dim=hamer_dim, hamer_proj_dim=hamer_proj_dim)
    out = {}
    for m in ("stgcn_bimamba", "stgcn_bilstm", "stgcn_transformer"):
        out[m] = sum(C.analytic(m, ns)[0].values())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hamer-dim", type=int, default=288)
    ap.add_argument("--angle-dim", type=int, default=104)
    ap.add_argument("--adapter-dim", type=int, default=512)
    ap.add_argument("--adapter-hidden", type=int, default=None, help="default = adapter-dim")
    ap.add_argument("--mixer-hidden", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=None, help="if set, only this one configuration is reported")
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim-feedforward", type=int, default=None, help="default 4*d_model")
    ap.add_argument("--gloss-vocab", type=int, default=0)
    ap.add_argument("--use-norm", action="store_true", help="LayerNorm inside the MLPs")
    ap.add_argument("--analytic-only", action="store_true")
    a = ap.parse_args()

    base = dict(hamer_dim=a.hamer_dim, angle_dim=a.angle_dim, adapter_dim=a.adapter_dim,
                adapter_hidden=a.adapter_hidden, mixer_hidden=a.mixer_hidden, use_norm=a.use_norm)

    # ---- 1. headline scenarios -------------------------------------------------
    scen = [
        ("A: same temporal backbone as your encoders (d=256, 4 layers, ff=1024)",
         dict(d_model=256, n_layers=4, dim_feedforward=1024)),
        ("B: wider/deeper guess (d=512, 6 layers, ff=2048)",
         dict(d_model=512, n_layers=6, dim_feedforward=2048)),
    ]
    if a.d_model:
        scen = [(f"custom (d={a.d_model}, {a.n_layers} layers, ff={a.dim_feedforward or 4 * a.d_model})",
                 dict(d_model=a.d_model, n_layers=a.n_layers, dim_feedforward=a.dim_feedforward))]

    results = []
    for name, cfg in scen:
        parts = count_hands_on(**base, **cfg, gloss_vocab=a.gloss_vocab)
        results.append((name, cfg, parts))

    w = 36
    print("\nHands-On (2025) architecture -- parameter count under stated assumptions")
    print(f"adapters {a.hamer_dim}/{a.angle_dim} -> {a.adapter_dim}, mixer hidden {a.mixer_hidden}, "
          f"LayerNorm in MLPs: {a.use_norm}, downsample/upsample parameter-free\n")
    rows = list(results[0][2].keys())
    print(f"{'component':<{w}}" + "".join(f"{'scenario ' + (n.split(':')[0] if ':' in n else 'custom'):>16}" for n, _, _ in results))
    print("-" * (w + 16 * len(results)))
    for r in rows:
        print(f"{r:<{w}}" + "".join(f"{p.get(r, 0):>16,}" for _, _, p in results))
    print("-" * (w + 16 * len(results)))
    infer = [sum(v for k, v in p.items() if "CTC" not in k) for _, _, p in results]
    print(f"{'total (without CTC head)':<{w}}" + "".join(f"{t:>16,}" for t in infer))
    print(f"{'in millions':<{w}}" + "".join(f"{t / 1e6:>15.2f}M" for t in infer))
    for n, _, _ in results:
        print(f"  {n}")
    if not a.gloss_vocab:
        d = results[0][1]["d_model"]
        print(f"\nCTC head is training-only and depends on the gloss vocabulary V: (d_model+1)*(V+1) "
              f"= {d + 1:,} per vocabulary entry (e.g. V=1000 -> {(d + 1) * 1001:,} at d={d}). Use --gloss-vocab V.")

    # ---- 2. sensitivity to the unstated choices ---------------------------------
    if not a.d_model:
        print("\nSensitivity: total (no CTC) in millions vs the choices the paper does not state")
        ds = (256, 512, 768, 1024)
        print(f"{'layers \\ d_model':<18}" + "".join(f"{d:>10}" for d in ds))
        for L in (2, 4, 6, 8):
            print(f"{L:<18}" + "".join(
                f"{sum(count_hands_on(**base, d_model=d, n_layers=L).values()) / 1e6:>9.2f}M" for d in ds))
        print("(ff = 4*d_model; mixer hidden as given)")
        for mh in (512, 1024):
            b = dict(base, mixer_hidden=mh)
            t = sum(count_hands_on(**b, d_model=256, n_layers=4, dim_feedforward=1024).values())
            print(f"  mixer hidden {mh:>4}, scenario-A backbone: {t / 1e6:.2f}M")
        b = dict(base, use_norm=True)
        t = sum(count_hands_on(**b, d_model=256, n_layers=4, dim_feedforward=1024).values())
        print(f"  LayerNorm inside MLPs, scenario-A backbone: {t / 1e6:.2f}M")

    # ---- 3. comparison with your models at the same temporal backbone ---------
    d, L, ff = 256, 4, 1024
    um = user_models_trainable(d, L, ff, a.hamer_dim)
    ho = sum(v for k, v in count_hands_on(**base, d_model=d, n_layers=L, dim_feedforward=ff).items() if "CTC" not in k)
    print(f"\nWhere the parameters are, scenario-A backbone (d={d}, {L} layers):")
    print(f"  {'model':<22}{'trainable':>14}  front end / backbone")
    tb = L * C.transformer_encoder_layer(d, ff)
    print(f"  {'hands-on 2025':<22}{ho:>14,}  front end {ho - tb - C.linear(d, 3):,} (adapters + mixer) / Transformer {tb:,}")
    for m, v in um.items():
        print(f"  {m:<22}{v:>14,}")

    # ---- 4. real-model cross-check ---------------------------------------------
    if a.analytic_only:
        return
    print("\nCross-check against the real nn.Module built here:")
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"  skipped ({type(e).__name__}: {e}); the analytic table does not depend on it")
        return
    for name, cfg, parts in results:
        net = build_model(**base, **cfg, nhead=a.nhead, gloss_vocab=a.gloss_vocab)
        real = sum(p.numel() for p in net.parameters())
        ana = sum(parts.values())
        print(f"  {name.split(':')[0]} real={real:,}  analytic={ana:,}  {'MATCHES' if real == ana else 'MISMATCH'}")
        if cfg is results[0][1]:
            import torch
            h = torch.randn(2, 64, a.hamer_dim)
            g = torch.randn(2, 64, a.angle_dim)
            logits, emb = net(h, g)
            ok = tuple(logits.shape) == (2, 3, 64)
            print(f"  forward shape check: logits {tuple(logits.shape)} emb {tuple(emb.shape)}  "
                  f"{'OK' if ok else 'UNEXPECTED (want (2,3,64))'}")


if __name__ == "__main__":
    main()