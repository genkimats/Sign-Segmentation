"""
Smoke test for TAILORED_ENCODER_STUDY (queue_train_phrase.py). Run from the repo root (needs torch; reads
no data -- the prototype file is a temporary random one):
    python test_tailored_encoders.py
  1. face articulators: compact face -> eyes 12 / brows 10 / mouth 9, every vertex once, no graph edge
     between articulators
  2. FaceRegionEncoder: shape; an impulse on one eye vertex reaches only the eyes, within +-8 frames;
     on a mouth vertex +-16, on a brow vertex +-32 (multirate); +-8 everywhere with all dilations 1
  3. HandShapePrototypeEncoder: shape; both hands share weights (slot swap swaps outputs); assignments sum
     to 1 for a detected hand, 0 for a missing one; the k-means file is loaded exactly; wrong k / missing
     file raise; the Linear control has the same input / output sizes
  4. body stream without hands: the 42 hand keypoints no longer change the output; fewer parameters
  5. every arm builds from its queue config, trains one step (every parameter gets a gradient), valid output
  6. old configs (B+H+Fc, HE1) get no new arguments and the same parameter names
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue_train_phrase as q                                                     # noqa: E402
from src.frontend import (FaceRegionEncoder, HandShapePrototypeEncoder, face_articulator_regions,  # noqa: E402
                          hamer_hand_parts)
from src.graph import SkeletonGraph                                                # noqa: E402
from src.model_factory import build_model_kwargs                                   # noqa: E402

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


def random_hamer(B, T, gen=None):
    """Valid rotation matrices (QR of random matrices) in the 288 layout."""
    R = torch.linalg.qr(torch.randn(B, T, 32, 3, 3, generator=gen))[0]
    return torch.cat([R[:, :, :30].reshape(B, T, 270), R[:, :, 30:].reshape(B, T, 18)], -1).permute(0, 2, 1).contiguous()


def swap_hands(h):
    B, _, T = h.shape
    P = h[:, :270].reshape(B, 2, 135, T).flip(1).reshape(B, 270, T)
    G = h[:, 270:].reshape(B, 2, 9, T).flip(1).reshape(B, 18, T)
    return torch.cat([P, G], 1)


def reach(diff):
    """diff: (T,) -> (first, last) frame where the output changed."""
    nz = torch.nonzero(diff > 1e-6).flatten()
    return (int(nz.min()), int(nz.max())) if len(nz) else (None, None)


tmp = tempfile.mkdtemp()
proto_path = os.path.join(tmp, "hamer_prototypes_k32.npz")
rng = np.random.default_rng(0)
np.savez(proto_path, centroids=rng.normal(size=(32, 90)).astype(np.float32), tau=np.float32(0.5))

print("1. face articulators")
regions = dict(face_articulator_regions("compact"))
check("eyes 12 / brows 10 / mouth 9", {k: len(v) for k, v in regions.items()} == {"eyes": 12, "brows": 10, "mouth": 9},
      str({k: len(v) for k, v in regions.items()}))
check("every face vertex exactly once", sorted(i for v in regions.values() for i in v) == list(range(31)))
which = {i: a for a, idx in regions.items() for i in idx}
edges = SkeletonGraph(num_vertices=31, face_only=True)._get_face_edges(0)
check("no face-graph edge between articulators", all(which[i] == which[j] for i, j in edges), f"{len(edges)} edges")

print("2. FaceRegionEncoder")
torch.manual_seed(0)
T, t0 = 120, 60
for dil, expect in (({"eyes": 1, "mouth": 2, "brows": 4}, {"eyes": 8, "mouth": 16, "brows": 32}),
                    ({"eyes": 1, "mouth": 1, "brows": 1}, {"eyes": 8, "mouth": 8, "brows": 8})):
    enc = FaceRegionEncoder(2, 64, 31, dilations=dil).eval()
    x = torch.randn(1, 2, T, 31)
    with torch.no_grad():
        y = enc(x)
        res = []
        for art in ("eyes", "mouth", "brows"):
            v = regions[art][0]
            xp = x.clone()
            xp[0, :, t0, v] += 5.0
            d = (enc(xp) - y).abs()                                   # (1, C, T, V)
            per_v = d.sum((0, 1, 2))
            lo, hi = reach(d.sum((0, 1, 3)))
            others = [i for i in range(31) if which[i] != art]
            res.append((art, lo is not None and t0 - lo == expect[art] and hi - t0 == expect[art]
                        and float(per_v[others].max()) == 0.0, (lo, hi)))
    check(f"output (1, 64, T, 31), dilations {dil}", y.shape == (1, 64, T, 31))
    for art, good, span in res:
        check(f"  {art}: reach +-{expect[art]} frames, no leak to other articulators", good,
              f"changed frames {span[0]}-{span[1]}")

print("3. HandShapePrototypeEncoder")
torch.manual_seed(0)
enc = HandShapePrototypeEncoder("proto", 32, 64, 2, init_path=proto_path).eval()
c = np.load(proto_path)
check("k-means file loaded exactly", torch.allclose(enc.prototypes, torch.from_numpy(c["centroids"]))
      and abs(float(enc.log_tau.exp()) - 0.5) < 1e-5)
h = random_hamer(2, 40)
with torch.no_grad():
    y = enc(h)
    ys = enc(swap_hands(h))
    check("output (B, T, 2 x 64)", y.shape == (2, 40, 128), str(tuple(y.shape)))
    check("weights shared by both hands (slot swap swaps outputs)",
          torch.allclose(ys.reshape(2, 40, 2, 64).flip(2).reshape(2, 40, 128), y, atol=1e-5))
    hm = h.clone()
    hm[:, 135:270] = 0
    hm[:, 279:288] = 0                                                # right hand missing
    out = enc(hm)
    a = enc._last_assign                                              # (B, T, 2, K)
    check("missing hand -> finite output", bool(torch.isfinite(out).all()))
    check("assignment sums to 1 (detected) / 0 (missing)",
          torch.allclose(a[..., 0, :].sum(-1), torch.ones(2, 40), atol=1e-5) and float(a[..., 1, :].abs().max()) == 0.0)
    shape, orient, det = hamer_hand_parts(h)
    check("6D layout = first two matrix columns", torch.allclose(
        shape[0, 0, 0, :6], h[0, :9, 0].reshape(3, 3)[:, :2].reshape(6)) and shape.shape == (2, 40, 2, 90)
        and orient.shape == (2, 40, 2, 6) and bool((det == 1).all()))
try:
    HandShapePrototypeEncoder("proto", 16, init_path=proto_path)
    check("wrong k raises", False)
except ValueError:
    check("wrong k raises", True)
try:
    HandShapePrototypeEncoder("proto", 32, init_path=os.path.join(tmp, "missing.npz"))
    check("missing prototype file raises", False)
except FileNotFoundError:
    check("missing prototype file raises", True)
lin = HandShapePrototypeEncoder("linear", 32, 64, 2).eval()
n_p = sum(p.numel() for p in enc.parameters())
n_l = sum(p.numel() for p in lin.parameters())
with torch.no_grad():
    check("Linear control: same output size, same parameter count (+-K biases)",
          lin(h).shape == y.shape and abs(n_p - n_l) <= 32, f"proto {n_p}, linear {n_l}")

print("4. body stream without hands")
base, body = cfg(q._MS_BEST), cfg(q.TAILORED_ENCODER_STUDY[0])
torch.manual_seed(0)
cls, kw = build_model_kwargs(body, detected_hamer_dim=288)
m = cls(**kw).eval()
x = torch.randn(1, body["in_channels"], 64, body["num_vertices"])
hh = random_hamer(1, 64)
with torch.no_grad():
    y0 = m(x, hamer=hh)[0]
    xp = x.clone()
    xp[..., 23:65] += torch.randn_like(xp[..., 23:65])
    y1 = m(x=xp, hamer=hh)[0]
    xb = x.clone()
    xb[..., 15] += 1.0                                                # a pose wrist still matters
    y2 = m(xb, hamer=hh)[0]
cls0, kw0 = build_model_kwargs(base, detected_hamer_dim=288)
n0 = sum(p.numel() for p in cls0(**kw0).parameters())
n1 = sum(p.numel() for p in m.parameters())
check("hand keypoints (23-64) do not change the output", torch.allclose(y0, y1), "")
check("pose wrist (15) still changes the output", not torch.allclose(y0, y2))
check("fewer parameters (branch 23 x 64 instead of 65 x 64 inputs)", n1 < n0, f"{n0 / 1e6:.2f}M -> {n1 / 1e6:.2f}M")

print("5. arms")
T = 96
for exp in q.TAILORED_ENCODER_STUDY:
    torch.manual_seed(0)
    c = cfg({**exp, "hamer_proto_init": proto_path} if exp.get("hamer_encoder") == "proto" else exp)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    m = cls(**kw).train()
    x = torch.randn(2, c["in_channels"], T, c["num_vertices"])
    logits = m(x, hamer=random_hamer(2, T))[0]
    F.cross_entropy(logits, torch.randint(0, 3, (2, T))).backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    m.eval()
    with torch.no_grad():
        out = m(x, hamer=random_hamer(2, T))[0]
    kind = ("face" if "face_encoder" in kw else "hamer" if "hamer_encoder" in kw else "body")
    want = {"face": isinstance(getattr(m, "face_stgcn", None), FaceRegionEncoder),
            "hamer": isinstance(getattr(m, "hamer_shape", None), HandShapePrototypeEncoder),
            "body": m.n_body_vertices == 23}[kind]
    check(exp["description"], out.shape == (2, 3, T) and bool(torch.isfinite(out).all()) and not no_grad and want,
          f"{kind}, {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params"
          + (f", NO GRAD {no_grad[:3]}" if no_grad else ""))

print("6. old configs")
new_keys = {"body_stream_hands", "face_encoder", "face_region_dilations", "hamer_proto_k", "hamer_proto_init"}
for name, exp in (("B+H+Fc", q._MS_BEST), ("HE1", q.HAMER_ENCODER_STUDY[0])):
    c = cfg(exp)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    keys = set(cls(**kw).state_dict())
    check(f"{name}: no new arguments, no new parameters", not (new_keys & set(kw))
          and not any(k.startswith(("hamer_shape", "face_stgcn.encoders")) for k in keys))

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
