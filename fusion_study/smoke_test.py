"""
smoke_test.py -- checks every fusion mode WITHOUT data, on random inputs shaped like R0's.
Run from the repo root on the training machine (needs the project's environment, incl. mamba_ssm,
because src/models.py imports it; CPU is fine, ~1 min):

    python fusion_study/smoke_test.py

  1. trace_core self-tests (numpy).
  2. Every arm of FUSION_STUDY + FUSION_OPTIONAL is built through src/model_factory.py from R0's
     config (experiments_phrase/multistream_bilstm-310/hyperparameters.json, or --config), exactly
     as train_phrase.py would build it: forward shapes, finite outputs, backward, gradients reach
     the new fusion modules, the traces have the documented shapes, extra_loss is > 0 only for
     late fusion, the capture / ablation hooks used by check_stream_balance.py work, and the
     state dict reloads strictly. Parameter counts are printed next to R0's.
  3. Initialisation: the frame gate starts uniform (= plain sum), FiLM starts as the identity.
  4. LocalLagAttention equals a brute-force per-frame / per-offset implementation.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fusion_study.fusion_experiments import fusion_optional_arms, fusion_study_arms  # noqa: E402
from fusion_study.fusion_layers import FaceFiLM, LocalLagAttention  # noqa: E402
from src.boundary_losses import boundary_aware_loss  # noqa: E402
from src.model_factory import build_model_kwargs  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(cfg):
    cls, kw = build_model_kwargs(cfg, detected_hamer_dim=288)
    return cls(**kw).to(DEV)


def inputs(cfg, B=2, T=128):
    x = torch.randn(B, cfg["in_channels"], T, cfg["num_vertices"], device=DEV)
    hamer = torch.randn(B, 288, T, device=DEV) if cfg.get("use_hamer_features") else None
    y = torch.randint(0, 3, (B, T), device=DEV)
    onehot = F.one_hot(y, 3).permute(0, 2, 1).float()
    return x, hamer, onehot


def check_arm(cfg, ref_params):
    name = cfg["description"].split(":")[0]
    model = build(cfg)
    model.train()
    x, hamer, tgt = inputs(cfg)
    kwargs = {"hamer": hamer} if hamer is not None else {}
    logits, emb = model(x, **kwargs)
    B, T = x.shape[0], x.shape[2]
    assert logits.shape == (B, 3, T) and torch.isfinite(logits).all(), f"{name}: bad logits {logits.shape}"
    w = torch.tensor(cfg["class_weights"], device=DEV)

    def same_loss(lg):
        return boundary_aware_loss(lg, tgt, w, soft=False, count_weight=2.0, radius=5)

    loss = same_loss(logits)
    extra = model.extra_loss(same_loss) if hasattr(model, "extra_loss") else torch.zeros(())
    mode = getattr(model, "fusion_mode", cfg.get("fusion"))
    assert (float(extra) > 0) == (mode == "late"), f"{name}: extra_loss {float(extra)} for mode {mode}"
    (loss + extra).backward()
    new_mods = {"frame_gate": "frame_gate", "lag_face": "lag", "film_face": "film_face", "late": "stream_heads"}
    if mode in new_mods:
        mod = getattr(model, new_mods[mode])
        grads = [p.grad for p in mod.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), f"{name}: missing grads"
        assert any(g.abs().sum() > 0 for g in grads), f"{name}: all-zero grads in the fusion module"
        tr = model._trace
        expect = {"frame_gate": {"gate": (B, T, len(model.streams))},
                  "lag_face": {"lag_attn": (B, T, 2 * cfg["lag_max_frames"] + 1), "lag_mean": (B, T)},
                  "film_face": {"film_gamma": (B, T), "film_beta": (B, T)},
                  "late": {"stream_probs": (B, len(model.streams), 3, T), "late_weight": (B, T, len(model.streams))}}[mode]
        for k, shape in expect.items():
            assert k in tr and tuple(tr[k].shape) == shape, f"{name}: trace {k} {tuple(tr[k].shape)} != {shape}"

    # capture / ablation hooks (check_stream_balance.py)
    model.eval()
    got = {}
    model.set_capture(lambda z, c: got.update(z=set(z), c=set(c)))
    with torch.no_grad():
        model(x, **kwargs)
    model.set_capture(None)
    assert got and got["c"] == set(model.streams), f"{name}: capture reports {got.get('c')} vs streams {model.streams}"
    s0 = model.streams[-1]
    model.set_ablation(s0, torch.zeros(model.stream_dims[s0], device=DEV))
    with torch.no_grad():
        model(x, **kwargs)
    model.set_ablation(None)

    clone = build(cfg)
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    p = n_params(model)
    print(f"   {name:4s} {mode:10s} ok   params {p / 1e6:5.2f} M ({p / ref_params:4.2f} x R0)")
    return model


def check_init(cfg_by_mode):
    m = build(cfg_by_mode["frame_gate"]).eval()
    x, hamer, _ = inputs(cfg_by_mode["frame_gate"])
    with torch.no_grad():
        m(x, hamer=hamer)
    g = m._trace["gate"]
    assert torch.allclose(g, torch.full_like(g, 1.0 / len(m.streams)), atol=1e-6), "frame gate is not uniform at init"
    film = FaceFiLM(16, 24).to(DEV)
    face, tgt = torch.randn(2, 30, 16, device=DEV), torch.randn(2, 30, 24, device=DEV)
    assert torch.allclose(film(face, tgt), tgt), "FiLM is not the identity at init"
    print("   frame gate uniform at init, FiLM identity at init -- ok")


def brute_lag(mod, q_src, kv_src):
    B, T, _ = kv_src.shape
    H, dh, K = mod.H, mod.dh, mod.K
    v = mod.v(kv_src).view(B, T, H, dh)
    if mod.attention == "content":
        q = mod.q(q_src).view(B, T, H, dh)
        k = mod.k(kv_src).view(B, T, H, dh)
    out = torch.zeros(B, T, H, dh, device=kv_src.device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                logits, vals = [], []
                for j, d in enumerate(range(-K, K + 1)):
                    if 0 <= t + d < T:
                        lg = mod.bias[h, j].clone()
                        if mod.attention == "content":
                            lg = lg + (q[b, t, h] * k[b, t + d, h]).sum() / math.sqrt(dh)
                        logits.append(lg)
                        vals.append(v[b, t + d, h])
                a = torch.softmax(torch.stack(logits), 0)
                out[b, t, h] = (a[:, None] * torch.stack(vals)).sum(0)
    return mod.norm(mod.o(out.reshape(B, T, H * dh)))


def check_lag():
    for att in ("content", "fixed"):
        mod = LocalLagAttention(12, 10, 10, max_lag=3, heads=2, head_dim=4, attention=att).to(DEV).eval()
        with torch.no_grad():
            mod.bias.normal_()
            q, kv = torch.randn(2, 9, 12, device=DEV), torch.randn(2, 9, 10, device=DEV)
            a, b = mod(q, kv), brute_lag(mod, q, kv)
        assert torch.allclose(a, b, atol=1e-5), f"lag attention ({att}) differs from brute force: {(a - b).abs().max()}"
        assert torch.allclose(mod.trace["lag_attn"].sum(-1), torch.ones(2, 9, device=DEV), atol=1e-5)
    print("   LocalLagAttention == brute force (content and fixed), attention sums to 1 -- ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("experiments_phrase", "multistream_bilstm-310", "hyperparameters.json"),
                    help="an R0 run config (the arms are derived from it exactly as in the queue)")
    args = ap.parse_args()

    print("== 1. trace_core self-tests")
    r = subprocess.run([sys.executable, os.path.join("fusion_study", "test_trace_core.py")], capture_output=True, text=True)
    print("   " + (r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr))
    if r.returncode != 0:
        raise SystemExit(r.stdout + r.stderr)

    with open(args.config) as f:
        r0 = json.load(f)
    r0.pop("prefix", None)
    ref = build(r0)
    print(f"\n== 2. every arm through src/model_factory.py (R0 = {args.config}, {n_params(ref) / 1e6:.2f} M params)")
    cfg_by_mode = {}
    for arm in fusion_study_arms(r0) + fusion_optional_arms(r0):
        cfg = {**r0, **arm}
        check_arm(cfg, n_params(ref))
        cfg_by_mode.setdefault(cfg.get("fusion_mode", cfg.get("fusion")), cfg)

    print("\n== 3. initialisation")
    check_init(cfg_by_mode)
    print("\n== 4. lag attention vs brute force")
    check_lag()
    print("\nFusion smoke test passed.")


if __name__ == "__main__":
    main()
