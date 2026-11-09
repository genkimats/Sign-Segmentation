"""
Smoke test for EARLY_FUSION_MECHANISM_STUDY (queue_train_phrase.py). Run from the repo root:
    python test_early_fusion_mechanism.py
  1. every arm builds from its queue config, trains one step (all parameters get a gradient), gives valid outputs
  2. EF-norm has one LayerNorm per modality; EF-joint runs the identity-graph ST-GCN blocks
  3. old bilstm_2023 configs (plain / + temporal conv) get no new arguments and no new parameters
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue_train_phrase as q                                   # noqa: E402
from src.model_factory import build_model_kwargs                 # noqa: E402

OK = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    OK.append(bool(cond))


def cfg(exp):
    c = {**q.MAMBA_DEFAULTS, **exp}
    c["in_channels"] = q.calculate_in_channels(c)
    c["num_vertices"] = q.calculate_num_vertices(c)
    return c


print("1-2. arms")
T = 96
for exp in q.EARLY_FUSION_MECHANISM_STUDY:
    torch.manual_seed(0)
    c = cfg(exp)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    m = cls(**kw).train()
    x, h = torch.randn(2, c["in_channels"], T, c["num_vertices"]), torch.randn(2, 288, T)
    y = torch.randint(0, 3, (2, T))
    logits = m(x, hamer=h)[0]
    F.cross_entropy(logits, y).backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    m.eval()
    with torch.no_grad():
        out = m(x, hamer=h)[0]
    extra = (f"{len(m.part_norms)} LayerNorms {m.parts_dims}" if getattr(m, "modality_norm", False)
             else f"joint TCN {sum(p.numel() for p in m.joint_tcn.parameters()) / 1e3:.0f}k params" if m.joint_tcn is not None
             else "")
    n_mod = 1 + (c["num_vertices"] > 65) + 1
    ok = (out.shape == (2, 3, T) and bool(torch.isfinite(out).all()) and not no_grad
          and (not kw.get("modality_norm") or len(m.part_norms) == n_mod)
          and (not kw.get("joint_temporal_layers") or m.joint_tcn is not None))
    check(exp["description"], ok, f"V={c['num_vertices']}, {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params, "
          f"{extra}" + (f", NO GRAD {no_grad[:3]}" if no_grad else ""))

print("3. old configs")
for exp in (q.BL2023_COMBO_STUDY[2], q.BL2023_TCN_STUDY[3]):
    c = cfg(exp)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    keys = set(cls(**kw).state_dict())
    check(exp["description"], not any(k in kw for k in ("modality_norm", "joint_temporal_layers"))
          and not any(k.startswith(("joint_tcn", "part_norms")) for k in keys))

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
