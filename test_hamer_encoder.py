"""
Smoke test for HAMER_ENCODER_STUDY (queue_train_phrase.py). Run from the repo root:
    python test_hamer_encoder.py
  1. MANO adjacency: 16 nodes, 15 tree edges, symmetric; "joint" = identity
  2. HandTreeEncoder: shape, both hands share weights (swapping the hand slots swaps the outputs),
     a missing hand gives a finite output
  3. every arm builds from its queue config, trains one step (every parameter gets a gradient), valid outputs
  4. old configs (B+H+Fc, HaMeR front-end) get no new arguments / parameters
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue_train_phrase as q                                       # noqa: E402
from src.frontend import HandTreeEncoder, mano_adjacency             # noqa: E402
from src.model_factory import build_model_kwargs                     # noqa: E402

OK = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    OK.append(bool(cond))


def cfg(exp):
    c = {**q.MAMBA_DEFAULTS, **exp}
    c["use_face_keypoints"] = bool(c.get("use_stream_face"))
    c["use_hamer_features"] = bool(c.get("use_stream_hamer"))
    c["in_channels"] = q.calculate_in_channels(c)
    c["num_vertices"] = q.calculate_num_vertices(c)
    return c


def random_hamer(B, T):
    """Valid rotation matrices (QR of random matrices) in the 288 layout."""
    R = torch.linalg.qr(torch.randn(B, T, 32, 3, 3))[0]
    H = R[:, :, :30].reshape(B, T, 270)          # 2 x 15 joints
    G = R[:, :, 30:].reshape(B, T, 18)           # 2 global orientations
    return torch.cat([H, G], -1).permute(0, 2, 1).contiguous()


print("1. MANO adjacency")
A = mano_adjacency("tree")
check("16 nodes, 15 edges, symmetric", A.shape == (16, 16) and int(((A > 0).sum() - 16) / 2) == 15 and np.allclose(A, A.T))
check("'joint' = identity", np.allclose(mano_adjacency("joint"), np.eye(16)))

print("2. HandTreeEncoder")
torch.manual_seed(0)
enc = HandTreeEncoder("tree").eval()
h = random_hamer(2, 40)
with torch.no_grad():
    y = enc(h)
    hs = h.clone()                                # swap hand slots: pose and global blocks
    P = hs[:, :270].reshape(2, 2, 135, 40).flip(1).reshape(2, 270, 40)
    Gs = hs[:, 270:].reshape(2, 2, 9, 40).flip(1).reshape(2, 18, 40)
    ys = enc(torch.cat([P, Gs], 1))
check("output (B, T, 2 x 16 x 64)", y.shape == (2, 40, 2048), str(tuple(y.shape)))
check("weights shared by both hands (slot swap swaps outputs)",
      torch.allclose(ys.reshape(2, 40, 2, 1024).flip(2).reshape(2, 40, 2048), y, atol=1e-5))
hm = h.clone()
hm[:, 135:270] = 0
hm[:, 279:288] = 0                               # right hand missing
with torch.no_grad():
    check("missing hand -> finite output", bool(torch.isfinite(enc(hm)).all()))

print("3. arms")
T = 96
for exp in q.HAMER_ENCODER_STUDY:
    torch.manual_seed(0)
    c = cfg(exp)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    m = cls(**kw).train()
    x = torch.randn(2, c["in_channels"], T, c["num_vertices"])
    logits = m(x, hamer=random_hamer(2, T))[0]
    F.cross_entropy(logits, torch.randint(0, 3, (2, T))).backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    m.eval()
    with torch.no_grad():
        out = m(x, hamer=random_hamer(2, T))[0]
    check(exp["description"], out.shape == (2, 3, T) and bool(torch.isfinite(out).all()) and not no_grad
          and hasattr(m, "hamer_tree") and c.get("hand_canonicalize", False) == ("canonical" in exp["description"]),
          f"{kw['hamer_encoder']}, {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params"
          + (f", NO GRAD {no_grad[:3]}" if no_grad else ""))

print("4. old configs")
for exp in (q._MS_BEST, {**q._MS_BEST, **q._HAMER_FRONTEND}):
    c = cfg(exp)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    keys = set(cls(**kw).state_dict())
    check(exp.get("description", "B+H+Fc") if "hamer_rep" not in exp else "B+H+Fc + HaMeR front-end",
          "hamer_encoder" not in kw and not any(k.startswith("hamer_tree") for k in keys))

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
