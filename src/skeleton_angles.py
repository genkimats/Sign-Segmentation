"""
src/skeleton_angles.py -- 3D joint-angle features computed from the 65-vertex MediaPipe skeleton.

WHY: the 2025 Hands-On paper feeds a 104-d "3D skeleton angle" vector (joints + angles from a separate pose
model). This project does not have that model, but it has MediaPipe 3D landmarks, from which joint angles can
be computed directly and deterministically. This is NOT the paper's feature -- same kind of information (how
the arms/fingers are articulated), different source and dimensionality (ANGLE_FEATURE_DIM, not 104).

Works on numpy arrays AND torch tensors with identical code (no learnable parameters, no torch import at
module level), so it is unit-tested here with numpy and runs on-GPU inside HandsOn2025.

INPUT   xyz  (..., V>=65, 3)   vertex layout of src/graph.py:
            0-22  MediaPipe pose landmarks (11/12 shoulders, 13/14 elbows, 15/16 wrists, 17-22 hand points)
            23-43 left hand  (21 MediaPipe hand landmarks: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
                              13-16 ring, 17-20 pinky)
            44-64 right hand (same order)
OUTPUT  (..., ANGLE_FEATURE_DIM) float features, all in [-1, 1]:
   * 32  interior joint angles / pi: 15 finger-joint flexion angles per hand (wrist-MCP-PIP, MCP-PIP-DIP,
         PIP-DIP-tip; thumb analogue) + left/right elbow           (1 = straight, 0.5 = right angle)
   * 13  angles between two bones / pi: 4 finger-spread angles per hand (adjacent proximal finger bones),
         shoulder (upper arm vs shoulder line) x2, wrist (forearm vs hand direction) x2, left-vs-right forearm
   * 18  unit direction vectors (camera frame): upper arm x2, forearm x2, hand direction (wrist -> middle MCP) x2
   *  6  palm normals (cross(wrist->index MCP, wrist->pinky MCP)) per hand, unit length
   *  1  angle between the two palm normals / pi
   The 32 + 13 + 1 angle features are invariant to translation, rotation and uniform scale; the 24 vector
   features are rotation-EQUIVARIANT (they say where limbs point in the image frame).

CAVEATS
   * MediaPipe coordinates are normalised image coordinates: x in units of image width, y in units of image
     height, z roughly in width units. For non-square video this distorts 3D angles. y_scale = height/width
     (0.5625 for 16:9) corrects it; the default 1.0 means "no correction". Your shoulder normalisation is a
     uniform scale, so it does not change angles either way.
   * Zero-length bones (a hand that was never detected becomes all zeros after nan_to_num) give feature 0, never NaN.
   * Depth (z) is the noisiest MediaPipe axis; finger-flexion angles use it.
"""
import math

import numpy as np

EPS_LEN = 1e-4      # bones shorter than this (in shoulder-width units) are treated as missing

# --- landmark indices (src/graph.py layout) ----------------------------------------------------
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
HAND_OFF = {"L": 23, "R": 44}
FINGERS = [("thumb", (1, 2, 3, 4)), ("index", (5, 6, 7, 8)), ("middle", (9, 10, 11, 12)),
           ("ring", (13, 14, 15, 16)), ("pinky", (17, 18, 19, 20))]
ARM = {"L": (L_SH, L_EL, L_WR, R_SH), "R": (R_SH, R_EL, R_WR, L_SH)}   # shoulder, elbow, wrist, other shoulder


def _build_spec():
    trip, trip_n = [], []          # (a, b, c): angle at b between (a-b) and (c-b)
    pair, pair_n = [], []          # (p0, p1, q0, q1): angle between bone p0->p1 and bone q0->q1
    unit, unit_n = [], []          # (p0, p1): unit vector of bone p0->p1
    norm, norm_n = [], []          # (w, i, p): unit normal of cross(i-w, p-w)

    for side, off in HAND_OFF.items():
        for fname, (a, b, c, d) in FINGERS:
            chain = [0, a, b, c, d]
            for k, jn in zip((1, 2, 3), ("MCP/CMC", "PIP/MCP", "DIP/IP")):
                trip.append((off + chain[k - 1], off + chain[k], off + chain[k + 1]))
                trip_n.append(f"{side}.{fname}.flex_{jn}")
    for side in ("L", "R"):
        sh, el, wr, _ = ARM[side]
        trip.append((sh, el, wr))
        trip_n.append(f"{side}.elbow_angle")

    for side, off in HAND_OFF.items():
        prox = [(off + a, off + b) for _, (a, b, _, _) in FINGERS]       # proximal bone of each finger
        for i in range(4):
            pair.append(prox[i] + prox[i + 1])
            pair_n.append(f"{side}.spread_{FINGERS[i][0]}-{FINGERS[i + 1][0]}")
    for side in ("L", "R"):
        sh, el, wr, osh = ARM[side]
        pair.append((sh, el, sh, osh))
        pair_n.append(f"{side}.shoulder_angle")
    for side, off in HAND_OFF.items():
        sh, el, wr, _ = ARM[side]
        pair.append((el, wr, off, off + 9))
        pair_n.append(f"{side}.wrist_angle")
    pair.append((L_EL, L_WR, R_EL, R_WR))
    pair_n.append("forearm_L-vs-R")

    for side in ("L", "R"):
        sh, el, wr, _ = ARM[side]
        unit.append((sh, el)); unit_n += [f"{side}.upper_arm_dir.{ax}" for ax in "xyz"]
        unit.append((el, wr)); unit_n += [f"{side}.forearm_dir.{ax}" for ax in "xyz"]
    for side, off in HAND_OFF.items():
        unit.append((off, off + 9)); unit_n += [f"{side}.hand_dir.{ax}" for ax in "xyz"]

    for side, off in HAND_OFF.items():
        norm.append((off, off + 5, off + 17)); norm_n += [f"{side}.palm_normal.{ax}" for ax in "xyz"]

    return trip, trip_n, pair, pair_n, unit, unit_n, norm, norm_n


_TRIP, _TRIP_N, _PAIR, _PAIR_N, _UNIT, _UNIT_N, _NORM, _NORM_N = _build_spec()
ANGLE_FEATURE_NAMES = _TRIP_N + _PAIR_N + _UNIT_N + _NORM_N + ["palm_normals_L-vs-R"]
ANGLE_FEATURE_DIM = len(ANGLE_FEATURE_NAMES)
N_SCALAR_ANGLES = len(_TRIP) + len(_PAIR)                 # angle features come first, then the vector features
ANGLE_INDICES = list(range(N_SCALAR_ANGLES)) + [ANGLE_FEATURE_DIM - 1]     # the rotation-invariant ones
VECTOR_INDICES = list(range(N_SCALAR_ANGLES, ANGLE_FEATURE_DIM - 1))       # the rotation-equivariant ones

_IDX = lambda spec, k: [s[k] for s in spec]               # noqa: E731


# --- backend adapter ----------------------------------------------------------------------------
class _NpOps:
    sqrt = staticmethod(np.sqrt)
    acos = staticmethod(np.arccos)

    @staticmethod
    def clip(x, lo, hi): return np.clip(x, lo, hi)

    @staticmethod
    def stack(xs, axis=-1): return np.stack(xs, axis=axis)

    @staticmethod
    def cat(xs, axis=-1): return np.concatenate(xs, axis=axis)

    @staticmethod
    def mask(x, cond): return np.where(cond, x, np.zeros_like(x))

    @staticmethod
    def scale(x, s): return x * np.asarray(s, dtype=x.dtype)


def _torch_ops():
    import torch

    class _TorchOps:
        sqrt = staticmethod(torch.sqrt)
        acos = staticmethod(torch.acos)

        @staticmethod
        def clip(x, lo, hi): return torch.clamp(x, lo, hi)

        @staticmethod
        def stack(xs, axis=-1): return torch.stack(xs, dim=axis)

        @staticmethod
        def cat(xs, axis=-1): return torch.cat(xs, dim=axis)

        @staticmethod
        def mask(x, cond): return torch.where(cond, x, torch.zeros_like(x))

        @staticmethod
        def scale(x, s): return x * x.new_tensor(list(s))

    return _TorchOps


def _ops_for(x):
    return _torch_ops() if type(x).__module__.split(".")[0] == "torch" else _NpOps


# --- geometry -----------------------------------------------------------------------------------
def _norm(v, ops):
    return ops.sqrt((v * v).sum(-1))


def _angle_between(u, v, ops):
    """(..., n, 3) x2 -> (..., n): angle / pi in [0, 1]; 0 where either vector is (near) zero."""
    nu, nv = _norm(u, ops), _norm(v, ops)
    ok = (nu > EPS_LEN) & (nv > EPS_LEN)
    cos = (u * v).sum(-1) / ops.clip(nu * nv, 1e-12, 1e12)       # clamp, not add: no bias on real bones
    return ops.mask(ops.acos(ops.clip(cos, -1.0, 1.0)) / math.pi, ok)


def _unit(v, ops):
    n = _norm(v, ops)[..., None]
    return ops.mask(v / ops.clip(n, 1e-12, 1e12), n > EPS_LEN)


def _cross(a, b, ops):
    return ops.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                      a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                      a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _flat(t, n):
    return t.reshape(tuple(t.shape[:-2]) + (n * 3,))


def skeleton_angle_features(xyz, y_scale=1.0):
    """xyz: (..., V>=65, 3) numpy array or torch tensor -> (..., ANGLE_FEATURE_DIM)."""
    if xyz.shape[-2] < 65 or xyz.shape[-1] != 3:
        raise ValueError(f"expected (..., >=65, 3) vertices, got {tuple(xyz.shape)}")
    ops = _ops_for(xyz)
    P = ops.scale(xyz, (1.0, float(y_scale), 1.0)) if y_scale != 1.0 else xyz
    pt = lambda idx: P[..., idx, :]                                                  # noqa: E731

    a, b, c = pt(_IDX(_TRIP, 0)), pt(_IDX(_TRIP, 1)), pt(_IDX(_TRIP, 2))
    flex = _angle_between(a - b, c - b, ops)                                         # (..., 32)
    pang = _angle_between(pt(_IDX(_PAIR, 1)) - pt(_IDX(_PAIR, 0)),
                          pt(_IDX(_PAIR, 3)) - pt(_IDX(_PAIR, 2)), ops)              # (..., 13)
    units = _unit(pt(_IDX(_UNIT, 1)) - pt(_IDX(_UNIT, 0)), ops)                      # (..., 6, 3)
    nrm = _unit(_cross(pt(_IDX(_NORM, 1)) - pt(_IDX(_NORM, 0)),
                       pt(_IDX(_NORM, 2)) - pt(_IDX(_NORM, 0)), ops), ops)           # (..., 2, 3)
    nang = _angle_between(nrm[..., 0:1, :], nrm[..., 1:2, :], ops)                   # (..., 1)
    return ops.cat([flex, pang, _flat(units, len(_UNIT)), _flat(nrm, len(_NORM)), nang], -1)