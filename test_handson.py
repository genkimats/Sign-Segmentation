"""test_handson.py -- numpy-only tests for skeleton_angles.py and handson_ctc_core.py (no torch needed)."""
import itertools
import math
import sys

import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeleton_angles as SA
import handson_ctc_core as CC

rng = np.random.default_rng(0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def random_rotation():
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))          # proper rotation


def random_skeleton(n=None):
    shape = (65, 3) if n is None else (n, 65, 3)
    return rng.normal(size=shape).astype(np.float64)


# ------------------------------------------------------------------ feature layout
check("feature dimension is 70 and names are unique", SA.ANGLE_FEATURE_DIM == 70 == len(set(SA.ANGLE_FEATURE_NAMES)),
      f"dim={SA.ANGLE_FEATURE_DIM}")
check("invariant + equivariant indices partition the features",
      sorted(SA.ANGLE_INDICES + SA.VECTOR_INDICES) == list(range(70)))

# ------------------------------------------------------------------ exact geometry
S = np.zeros((65, 3))
off = 23
# left index finger: wrist(0,0,0) -> MCP(1,0,0) -> PIP(2,0,0) straight; PIP -> DIP bent 90 deg; DIP -> tip straight on
S[off + 0] = [0, 0, 0]; S[off + 5] = [1, 0, 0]; S[off + 6] = [2, 0, 0]; S[off + 7] = [2, 1, 0]; S[off + 8] = [2, 2, 0]
f = SA.skeleton_angle_features(S)
nm = SA.ANGLE_FEATURE_NAMES.index
check("straight joint -> 1.0 (angle pi/pi)", abs(f[nm("L.index.flex_MCP/CMC")] - 1.0) < 1e-9,
      f"{f[nm('L.index.flex_MCP/CMC')]:.4f}")
check("right-angle bend -> 0.5", abs(f[nm("L.index.flex_PIP/MCP")] - 0.5) < 1e-9, f"{f[nm('L.index.flex_PIP/MCP')]:.4f}")
check("straight again at the next joint -> 1.0", abs(f[nm("L.index.flex_DIP/IP")] - 1.0) < 1e-9)

# brute-force check of all 32 interior angles against an independent formula on a random skeleton
P = random_skeleton()
F_ = SA.skeleton_angle_features(P)
ok = True
for k, (a, b, c) in enumerate(SA._TRIP):
    u, v = P[a] - P[b], P[c] - P[b]
    ref = math.acos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)) / math.pi
    ok &= abs(F_[k] - ref) < 1e-9
check("32 interior angles match independent formula", ok)
ok = True
for k, (p0, p1, q0, q1) in enumerate(SA._PAIR):
    u, v = P[p1] - P[p0], P[q1] - P[q0]
    ref = math.acos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)) / math.pi
    ok &= abs(F_[len(SA._TRIP) + k] - ref) < 1e-9
check("13 bone-pair angles match independent formula", ok)
u = P[SA.L_EL] - P[SA.L_SH]
i0 = SA.ANGLE_FEATURE_NAMES.index("L.upper_arm_dir.x")
check("unit direction vector = normalised bone", np.allclose(F_[i0:i0 + 3], u / np.linalg.norm(u)))
w, i5, p17 = P[23], P[28], P[40]
n = np.cross(i5 - w, p17 - w); n /= np.linalg.norm(n)
j0 = SA.ANGLE_FEATURE_NAMES.index("L.palm_normal.x")
check("palm normal = normalised cross product", np.allclose(F_[j0:j0 + 3], n))

# ------------------------------------------------------------------ invariances
R, t, s = random_rotation(), rng.normal(size=3), 3.7
P2 = (P @ R.T) * s + t
F2 = SA.skeleton_angle_features(P2)
check("angle features invariant to rotation + translation + uniform scale",
      np.allclose(F_[SA.ANGLE_INDICES], F2[SA.ANGLE_INDICES], atol=1e-9),
      f"max diff {np.abs(F_[SA.ANGLE_INDICES] - F2[SA.ANGLE_INDICES]).max():.2e}")
V1 = F_[SA.VECTOR_INDICES].reshape(-1, 3)
V2 = F2[SA.VECTOR_INDICES].reshape(-1, 3)
check("direction/normal features rotate with the skeleton (equivariant)", np.allclose(V1 @ R.T, V2, atol=1e-9))
check("all outputs finite and within [-1, 1]", np.isfinite(F_).all() and np.abs(F_).max() <= 1 + 1e-12)

# ------------------------------------------------------------------ robustness
Z = SA.skeleton_angle_features(np.zeros((65, 3)))
check("all-zero skeleton (undetected) -> all zeros, no NaN", np.all(Z == 0) and np.isfinite(Z).all())
P3 = P.copy(); P3[44:65] = 0                       # right hand never detected
F3 = SA.skeleton_angle_features(P3)
rh = [k for k, nme in enumerate(SA.ANGLE_FEATURE_NAMES) if nme.startswith("R.") and ("finger" in nme or "flex" in nme
      or "spread" in nme or "hand_dir" in nme or "palm" in nme)]
check("missing right hand zeroes only right-hand features, left hand unchanged",
      np.all(F3[rh] == 0) and np.allclose(F3[:15], F_[:15]) and np.isfinite(F3).all())
P4 = P.copy(); P4[23 + 6] = P4[23 + 5]             # zero-length bone
check("zero-length bone gives 0, not NaN", np.isfinite(SA.skeleton_angle_features(P4)).all())

# ------------------------------------------------------------------ batching and y_scale
B = random_skeleton(6).reshape(2, 3, 65, 3)
Fb = SA.skeleton_angle_features(B)
check("batched shape (B,T,V,3)->(B,T,70)", Fb.shape == (2, 3, 70), str(Fb.shape))
check("batched == frame-by-frame", all(np.allclose(Fb[b, t_], SA.skeleton_angle_features(B[b, t_]))
                                       for b in range(2) for t_ in range(3)))
ys = 0.5625
check("y_scale equals pre-scaling the y coordinate",
      np.allclose(SA.skeleton_angle_features(P, y_scale=ys), SA.skeleton_angle_features(P * np.array([1, ys, 1]))))
check("y_scale actually changes the angles", not np.allclose(SA.skeleton_angle_features(P, y_scale=ys), F_))
try:
    SA.skeleton_angle_features(np.zeros((40, 3)))
    check("rejects fewer than 65 vertices", False)
except ValueError:
    check("rejects fewer than 65 vertices", True)

# ------------------------------------------------------------------ CTC target derivation
def brute_signs(bio):
    n, i = 0, 0
    bio = list(bio)
    while i < len(bio):
        starts = (bio[i] == 2 and (i == 0 or bio[i - 1] != 2)) or (bio[i] == 1 and (i == 0 or bio[i - 1] == 0))
        n += starts
        i += 1
    return n


cases = {"empty": [0] * 10, "one sign BII": [0, 2, 1, 1, 0], "two adjacent": [2, 1, 1, 2, 1, 0],
         "begin run (dilated)": [0, 2, 2, 2, 1, 1, 0], "no begin tag": [0, 1, 1, 0, 1, 0], "window-cut sign": [1, 1, 1, 0, 2, 1],
         "begin only": [2, 0, 2, 0, 2]}
expect = {"empty": 0, "one sign BII": 1, "two adjacent": 2, "begin run (dilated)": 1, "no begin tag": 2,
          "window-cut sign": 2, "begin only": 3}
check("hand-written sign counts", all(CC.count_signs(v) == expect[k] for k, v in cases.items()),
      str({k: CC.count_signs(v) for k, v in cases.items()}))
rand = [rng.integers(0, 3, size=64) for _ in range(500)]
check("sign count matches brute-force scan on 500 random tag sequences",
      all(CC.count_signs(r) == brute_signs(r) for r in rand))

# ------------------------------------------------------------------ CTC feasibility vs a reference CTC implementation
def rand_logprobs(T, K=2):
    x = rng.normal(size=(T, K)) * 2
    return x - np.log(np.exp(x).sum(1, keepdims=True))


agree = True
for T, L in itertools.product(range(1, 9), range(0, 6)):
    nll = CC.ctc_nll_numpy(rand_logprobs(T), L)
    feasible = np.isfinite(nll)
    agree &= (feasible == CC.ctc_feasible(L, T))
check("feasibility rule T >= 2L-1 agrees with the reference CTC forward algorithm for T<=8, L<=5", agree)

lp = rand_logprobs(7)
check("L=0 target: NLL equals -sum(log p_blank)", abs(CC.ctc_nll_numpy(lp, 0) - (-lp[:, 0].sum())) < 1e-9)
# brute-force enumeration over all 2^T frame labellings for a small case
def brute_nll(lp, L):
    T = lp.shape[0]
    tot = 0.0
    for path in itertools.product([0, 1], repeat=T):
        collapsed, prev = [], None
        for p in path:
            if p != prev and p != 0:
                collapsed.append(p)
            prev = p
        if len(collapsed) == L:
            tot += np.exp(sum(lp[t, p] for t, p in enumerate(path)))
    return -np.log(tot) if tot > 0 else np.inf


ok = True
for T, L in [(5, 1), (5, 2), (6, 3), (7, 2), (4, 2)]:
    lpx = rand_logprobs(T)
    ok &= abs(CC.ctc_nll_numpy(lpx, L) - brute_nll(lpx, L)) < 1e-9
check("reference CTC forward == brute-force sum over all alignments", ok)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)