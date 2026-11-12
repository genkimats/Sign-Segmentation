"""
Smoke test for STRUCTURE_STUDY (queue_train_phrase.py). Run from the repo root (needs torch; reads no data):
    python test_structure_study.py
  1. face subsets mouth / mouth_full / mouth_eyes: sizes 11 / 33 / 21, unique, graphs build
  2. BS-split: pose + left + right encoders; the fusion input size equals the reference (pose 128 + 2 x 64);
     the hand keypoints still change the output; left and right have their own weights
  3. BS-shared: one hand encoder; a left hand that is the x-mirror of the right hand gives the same features
  4. EF-gap arms: separate body / face filters, HaMeR branch; the right parts sizes
  5. MLP arms: 3-layer branches (and fusion), the original single layers otherwise
  6. every arm builds from its queue config, trains one step (every parameter gets a gradient), valid output
  7. old configs (B+H+Fc, EF-joint B+F+H) get no new arguments and no new parameters
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue_train_phrase as q                                                     # noqa: E402
from src.face_subsets import face_subset_size                                      # noqa: E402
from src.graph import SkeletonGraph                                                # noqa: E402
from src.model_factory import build_model_kwargs                                   # noqa: E402

OK = []
T = 96


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    OK.append(bool(cond))


def cfg(exp):
    c = {**q.MAMBA_DEFAULTS, **exp}
    if c["basename"].startswith("multistream"):
        c["use_face_keypoints"] = bool(c.get("use_stream_face"))
        c["use_hamer_features"] = bool(c.get("use_stream_hamer"))
    c["in_channels"] = q.calculate_in_channels(c)
    c["num_vertices"] = q.calculate_num_vertices(c)
    return c


def build(exp, seed=0):
    torch.manual_seed(seed)
    c = cfg(exp)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288, detected_cue_dim=110)
    return c, kw, cls(**kw)


def inputs(c, B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, c["in_channels"], T, c["num_vertices"], generator=g)
    h = torch.randn(B, 288, T, generator=g)
    cues = torch.randn(B, 110, T, generator=g)
    return x, h, cues


def fwd(m, x, h, cues):
    return m(x, hamer=h, cues=cues)[0] if getattr(m, "streams", None) and "cues" in m.streams else m(x, hamer=h)[0]


print("1. face subsets")
sizes = {s: face_subset_size(s) for s in ("mouth", "mouth_full", "mouth_eyes")}
check("sizes 11 / 33 / 21", sizes == {"mouth": 11, "mouth_full": 33, "mouth_eyes": 21}, str(sizes))
for s, n in sizes.items():
    check(f"  {s}: face-only and full graphs build",
          SkeletonGraph(n, face_only=True).A.shape == (n, n) and SkeletonGraph(65 + n).A.shape == (65 + n,) * 2)

print("2. BS-split")
ref_c, ref_kw, ref = build(q._MS_BEST)
c, kw, m = build(q.BODY_SPLIT_STUDY[0])
m.eval()
check("pose 23 + left / right hand encoders", m.n_body_vertices == 23 and hasattr(m, "lhand_stgcn")
      and hasattr(m, "rhand_stgcn") and not hasattr(m, "hand_stgcn"))
check("fusion input size = reference", m.concat_dim == ref.concat_dim and m.stream_dims["body_hands"] == 256,
      f"{m.concat_dim} vs {ref.concat_dim}")
x, h, cues = inputs(c)
with torch.no_grad():
    y0 = m(x, hamer=h)[0]
    xp = x.clone()
    xp[..., 23:44] += 1.0
    y1 = m(xp, hamer=h)[0]
check("left-hand keypoints change the output", not torch.allclose(y0, y1))
shared = {p.data_ptr() for p in m.lhand_stgcn.parameters()} & {p.data_ptr() for p in m.rhand_stgcn.parameters()}
check("left / right encoders have their own weights", not shared)

print("3. BS-shared")
c, kw, m = build(q.BODY_SPLIT_STUDY[1])
m.eval()
check("one shared hand encoder + branch", hasattr(m, "hand_stgcn") and hasattr(m, "hand_branch")
      and not hasattr(m, "lhand_stgcn") and kw.get("hand_mirror_channels") == [0])
x, h, cues = inputs(c, B=1)
x[..., 23:44] = x[..., 44:65]
x[:, 0, :, 23:44] = -x[:, 0, :, 44:65]                         # left hand = x-mirrored right hand
outs = []
hook = m.hand_branch.register_forward_hook(lambda mod, i, o: outs.append(o))
with torch.no_grad():
    m(x, hamer=h)
hook.remove()
check("mirrored left hand -> the same features as the right hand",
      len(outs) == 2 and torch.allclose(outs[0], outs[1], atol=1e-5))

print("4. EF-gap arms")
for exp, want_sep, want_h in zip(q.EF_GAP_STUDY, (True, False, True), (False, True, True)):
    c, kw, m = build(exp)
    ok = ((m.joint_tcn_body is not None) == want_sep and (m.joint_tcn is None) == want_sep
          and (m.hamer_proj is not None) == want_h and m.parts_dims[-1] == (256 if want_h else 288))
    check(exp["description"].split(":")[0], ok, f"parts {m.parts_dims}")

print("5. MLP arms")
n_lin = lambda s: sum(isinstance(mod, torch.nn.Linear) for mod in s)                # noqa: E731
for exp, nb, nf in zip(q.MLP_STUDY, (3, 3), (1, 3)):
    c, kw, m = build(exp)
    check(exp["description"].split(":")[0], all(n_lin(b) == nb for b in m.branches.values())
          and n_lin(m.fusion_proj) == nf, f"branch {[n_lin(b) for b in m.branches.values()]}, fusion {n_lin(m.fusion_proj)}")
check("reference: single-layer branches and fusion",
      all(n_lin(b) == 1 for b in ref.branches.values()) and n_lin(ref.fusion_proj) == 1)

print("6. arms")
for exp in q.STRUCTURE_STUDY:
    c, kw, m = build(exp)
    m.train()
    x, h, cues = inputs(c)
    logits = fwd(m, x, h, cues)
    F.cross_entropy(logits, torch.randint(0, 3, (2, T))).backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    m.eval()
    with torch.no_grad():
        out = fwd(m, x, h, cues)
    check(exp["description"].split(":")[0], out.shape == (2, 3, T) and bool(torch.isfinite(out).all()) and not no_grad,
          f"V={c['num_vertices']}, {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params"
          + (f", NO GRAD {no_grad[:3]}" if no_grad else ""))

print("7. old configs")
new_keys = {"body_split", "hand_proj_dim", "hand_mirror_channels", "branch_layers", "fusion_layers",
            "joint_temporal_per_modality", "hamer_branch", "hamer_branch_dim"}
for name, exp in (("B+H+Fc", q._MS_BEST), ("EF-joint B+F+H", q._EF_JOINT_BFH)):
    c, kw, m = build(exp)
    keys = set(m.state_dict())
    check(f"{name}: no new arguments, no new parameters", not (new_keys & set(kw))
          and not any(k.startswith(("lhand_", "rhand_", "hand_", "joint_tcn_body", "joint_tcn_face", "hamer_proj"))
                      for k in keys))

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
