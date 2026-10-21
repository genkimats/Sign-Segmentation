"""
Front-end building blocks for the multi-stream model -- everything between the raw input
features and the encoder that is not the ST-GCN itself.

1. HANDEDNESS (data level, used by src/dataset.py)
   dominant_wrist(...)     which pose wrist moves more in a video (15 = MediaPipe "left",
                           16 = MediaPipe "right")
   mirror_graph_tv(...)    left/right mirror of the (T, V, K) keypoint tensor: x is reflected
                           about the shoulder midline (body) / the face midline (face-local
                           face), left/right landmarks and the two hand blocks are swapped
   mirror_hamer(...)       the same mirror for the 288-dim HaMeR vector: the two hand slots
                           are swapped and every rotation is conjugated with M = diag(-1, 1, 1)
                           (the reflection extract_hamer_features.py itself uses for left hands)
   Used for canonicalization (dominant hand always on the same side) and for random
   mirror augmentation during training.

2. HaMeRFrontEnd (model level)
   Turns the raw 288 values into: rotations as raw matrices or 6D (first two columns), a
   per-hand "detected" mask (missing hands are all-zero matrices, not valid rotations),
   and per-joint rotation speed (geodesic chord distance between frame t and t - lag).

3. VertexPooling (model level)
   Replaces "flatten all vertices x channels -> one big Linear" with body-part tokens:
   mean or attention pooling of the ST-GCN output inside each part (head, left/right arm,
   left/right hand; eyes / brows / mouth for the face).
"""
import numpy as np
import torch
import torch.nn as nn

from src.face_subsets import (SAVED_FACE_INDICES, LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
                              LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES, LIPS_INDICES,
                              face_subset_raw_indices, CONTOURS_2023, IRIS_A, IRIS_B)

# ==============================================================================
# Left/right correspondences
# ==============================================================================
# MediaPipe Pose 0-22: 0 nose | 1-3 left eye (inner, eye, outer) | 4-6 right eye | 7/8 ears |
# 9/10 mouth corners | 11/12 shoulders | 13/14 elbows | 15/16 wrists | 17/18 pinky |
# 19/20 index | 21/22 thumb.  23-43 = one hand block, 44-64 = the other (same point order).
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22)]
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


def _body_mirror_perm():
    perm = list(range(65))
    for a, b in _POSE_PAIRS:
        perm[a], perm[b] = b, a
    for i in range(21):
        perm[23 + i], perm[44 + i] = 44 + i, 23 + i
    return perm


BODY_MIRROR_PERM = _body_mirror_perm()

# MediaPipe Face Mesh: the eye / brow contour lists are in corresponding order; lips are
# symmetric about the centre points 13, 14, 17.
_LIP_PAIRS = [(61, 291), (146, 375), (91, 321), (181, 405), (84, 314), (308, 78), (324, 95),
              (318, 88), (402, 178), (317, 87), (191, 415), (80, 310), (81, 311), (82, 312)]
_LIP_CENTRE = [13, 14, 17]


def _face_mirror_map():
    m = {}
    for a, b in list(zip(LEFT_EYE_INDICES, RIGHT_EYE_INDICES)) + \
            list(zip(LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES)) + _LIP_PAIRS:
        m[a], m[b] = b, a
    for c in _LIP_CENTRE:
        m[c] = c
    missing = sorted(set(SAVED_FACE_INDICES) - set(m))
    extra = sorted(set(m) - set(SAVED_FACE_INDICES))
    if missing or extra or any(m[m[k]] != k for k in m):
        raise RuntimeError(f"Face mirror map inconsistent (missing={missing}, extra={extra}).")
    return m


FACE_MIRROR_MAP = _face_mirror_map()


def face_mirror_perm(face_subset):
    """perm[i] = position (inside the subset) of the mirror partner of subset vertex i."""
    raw = face_subset_raw_indices(face_subset)
    pos = {r: i for i, r in enumerate(raw)}
    try:
        return [pos[FACE_MIRROR_MAP[r]] for r in raw]
    except KeyError as e:
        raise ValueError(f"Face subset '{face_subset}' is not left/right symmetric "
                         f"(point {e} has no mirror partner in it).") from None


# ==============================================================================
# Handedness: dominance + mirroring
# ==============================================================================
def dominant_wrist(base_xy):
    """
    base_xy: (T, 65, 2) body + hand x, y (any normalization). Returns 15 or 16: the pose
    wrist with the longer path (total frame-to-frame movement) over the video.
    """
    w = np.asarray(base_xy, dtype=np.float64)[:, [L_WRIST, R_WRIST]]   # (T, 2, 2)
    ok = np.all(np.isfinite(w), axis=(1, 2)) & (np.abs(w).sum(axis=(1, 2)) > 0)
    w = w[ok]
    if len(w) < 2:
        return R_WRIST
    path = np.linalg.norm(np.diff(w, axis=0), axis=-1).sum(axis=0)    # (2,)
    return L_WRIST if path[0] > path[1] else R_WRIST


def mirror_graph_tv(x, x_channels, face_perm=None):
    """
    Mirror of a (T, V, K) keypoint tensor (V = 65 body + hands [+ face]).
    x_channels: channel indices that carry x (positions AND their time derivatives).
      body : x' = (x_lshoulder + x_rshoulder) - x   (reflection about the shoulder midline;
             exact for positions and, being linear, for velocity / acceleration / jerk)
      face : x' = -x                                 (face-local v2 coords, origin on the midline)
    Vertices that are exactly zero in all channels (missing detections) stay zero.
    Then left/right landmarks and the hand blocks are swapped (BODY_MIRROR_PERM, face_perm).
    """
    out = x.clone()
    body = out[:, :65]
    zero = body.abs().sum(dim=-1) == 0                                  # (T, 65)
    for c in x_channels:
        s = body[:, L_SHOULDER, c] + body[:, R_SHOULDER, c]             # (T,)
        body[:, :, c] = s[:, None] - body[:, :, c]
    body[zero] = 0
    out[:, :65] = body[:, BODY_MIRROR_PERM]
    if out.shape[1] > 65:
        if face_perm is None:
            raise ValueError("Face vertices present but no face_perm given.")
        face = out[:, 65:]
        for c in x_channels:
            face[:, :, c] = -face[:, :, c]
        out[:, 65:] = face[:, face_perm]
    return out


_M_SIGN = torch.tensor([[1., -1., -1.], [-1., 1., 1.], [-1., 1., 1.]])   # (M R M)_ij = m_i m_j R_ij


def mirror_hamer(h):
    """h: (..., 288) = [hand_pose (2 x 15 x 3 x 3) | global_orient (2 x 3 x 3)], slot 0 = left."""
    if h.shape[-1] != 288:
        raise ValueError(f"mirror_hamer expects 288-dim HaMeR features, got {h.shape[-1]}.")
    lead = h.shape[:-1]
    sign = _M_SIGN.to(h.dtype).to(h.device)
    H = h[..., :270].reshape(*lead, 2, 15, 3, 3).flip(dims=[-4]) * sign
    G = h[..., 270:].reshape(*lead, 2, 3, 3).flip(dims=[-3]) * sign
    return torch.cat([H.reshape(*lead, 270), G.reshape(*lead, 18)], dim=-1)


# ==============================================================================
# HaMeR front-end
# ==============================================================================
class HaMeRFrontEnd(nn.Module):
    """
    (B, 288, T) raw HaMeR -> (B, T, out_dim) per-frame features:
      rep="matrix" : the 288 raw rotation-matrix values (original input)
      rep="rot6d"  : first two columns of all 32 rotations (2 hands x (global + 15 joints))
                     = 192 values; continuous, no redundant third column
      mask=True    : + 2 channels, 1 where that hand was detected (non-zero global_orient)
      motion=True  : + 32 channels, per-joint rotation speed between t and t - lag:
                     chord distance ||R_t - R_{t-lag}||_F / sqrt(8) ~= theta / 2 for small
                     angles (smooth everywhere, no arccos); 0 where either frame is missing.
                     Batch-normalized (rotation speeds are tiny numbers per frame).
    """

    def __init__(self, hamer_dim=288, rep="matrix", mask=False, motion=False, motion_lag=1):
        super().__init__()
        if rep not in ("matrix", "rot6d"):
            raise ValueError(f"hamer_rep must be 'matrix' or 'rot6d', got '{rep}'.")
        if hamer_dim != 288 and (rep != "matrix" or mask or motion):
            raise ValueError(f"HaMeR front-end options need the 288-dim HaMeR layout (got {hamer_dim}).")
        self.rep, self.use_mask, self.use_motion = rep, bool(mask), bool(motion)
        self.motion_lag = max(1, int(motion_lag))
        self.out_dim = (288 if rep == "matrix" else 192) + (2 if mask else 0) + (32 if motion else 0)
        if self.use_motion:
            self.motion_bn = nn.BatchNorm1d(32)

    def forward(self, h):
        B, D, T = h.shape
        x = h.permute(0, 2, 1)                                              # (B, T, 288)
        H = x[..., :270].reshape(B, T, 2, 15, 3, 3)
        G = x[..., 270:].reshape(B, T, 2, 3, 3)
        R = torch.cat([G.unsqueeze(3), H], dim=3)                          # (B, T, 2, 16, 3, 3)
        detected = (G.abs().flatten(-2).sum(-1) > 1e-6).to(x.dtype)        # (B, T, 2)
        feats = [x if self.rep == "matrix" else R[..., :, :2].reshape(B, T, 192)]
        if self.use_mask:
            feats.append(detected)
        if self.use_motion:
            L = self.motion_lag
            diff = torch.zeros(B, T, 2, 16, device=x.device, dtype=x.dtype)
            valid = torch.zeros(B, T, 2, device=x.device, dtype=x.dtype)
            if T > L:
                d = (R[:, L:] - R[:, :-L]).pow(2).flatten(-2).sum(-1)       # (B, T-L, 2, 16)
                diff[:, L:] = torch.sqrt(d / 8.0 + 1e-8)
                valid[:, L:] = detected[:, L:] * detected[:, :-L]
            motion = (diff * valid.unsqueeze(-1)).reshape(B, T, 32)
            motion = self.motion_bn(motion.permute(0, 2, 1)).permute(0, 2, 1)
            feats.append(motion)
        return torch.cat(feats, dim=-1)


class TemporalResidual(nn.Module):
    """h + Dropout(GELU(Conv1d_k(LN(h)))) over time; (B, T, D) -> (B, T, D)."""

    def __init__(self, dim, kernel, dropout):
        super().__init__()
        k = int(kernel) | 1
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, k, padding=k // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        y = self.conv(self.norm(h).permute(0, 2, 1)).permute(0, 2, 1)
        return h + self.drop(self.act(y))


# ==============================================================================
# Vertex pooling (body parts instead of flatten)
# ==============================================================================
BODY_PARTS = [
    ("head", list(range(0, 11))),
    ("left arm", [11, 13, 15, 17, 19, 21]),
    ("right arm", [12, 14, 16, 18, 20, 22]),
    ("left hand", list(range(23, 44))),
    ("right hand", list(range(44, 65))),
]


def _face_regions():
    """Part definitions; with the full-mesh layout each part also takes the matching
    MediaPipe contour region (found by a shared point) and the iris of that eye."""
    regions = {"left eye": set(LEFT_EYE_INDICES) | set(IRIS_A), "right eye": set(RIGHT_EYE_INDICES) | set(IRIS_B),
               "left brow": set(LEFT_EYEBROW_INDICES), "right brow": set(RIGHT_EYEBROW_INDICES),
               "mouth": set(LIPS_INDICES)}
    if CONTOURS_2023 is not None:
        for key, pts in CONTOURS_2023["regions"].items():
            pts = set(pts)
            if key == "face_oval":
                regions["face oval"] = pts
                continue
            for name in ("left eye", "right eye", "left brow", "right brow", "mouth"):
                if pts & regions[name]:
                    regions[name] |= pts
                    break
    return regions


def face_parts(face_subset):
    raw = face_subset_raw_indices(face_subset)
    pos = {r: i for i, r in enumerate(raw)}
    parts = []
    for name, region in _face_regions().items():
        idx = sorted({pos[r] for r in region if r in pos})
        if idx:
            parts.append((name, idx))
    covered = {i for _, idx in parts for i in idx}
    if len(covered) != len(raw):
        raise RuntimeError(f"Face parts do not cover subset '{face_subset}'.")
    return parts


class VertexPooling(nn.Module):
    """
    (B, C, T, V) ST-GCN output -> (B, T, P * C) part tokens.
      mode="part_mean": plain mean over the vertices of each part
      mode="part_attn": softmax-weighted mean; the weight of a vertex = a learned per-vertex
                        bias + a data-dependent score (1x1 conv on its features), softmax
                        taken inside each part per frame -> the model can pick the informative
                        fingertips / landmarks frame by frame.
    """

    def __init__(self, channels, parts, mode="part_mean"):
        super().__init__()
        if mode not in ("part_mean", "part_attn"):
            raise ValueError(f"vertex_pooling must be 'flatten', 'part_mean' or 'part_attn', got '{mode}'.")
        self.mode = mode
        self.part_names = [n for n, _ in parts]
        self.num_parts = len(parts)
        self.out_dim = self.num_parts * channels
        V = max(i for _, idx in parts for i in idx) + 1
        member = torch.zeros(self.num_parts, V)
        for p, (_, idx) in enumerate(parts):
            member[p, idx] = 1.0
        self.register_buffer("member", member)                              # (P, V)
        self.register_buffer("outside", (member == 0).float() * -1e9)       # additive mask
        if mode == "part_attn":
            self.score = nn.Conv2d(channels, 1, kernel_size=1)
            self.vertex_bias = nn.Parameter(torch.zeros(V))

    def forward(self, h):
        B, C, T, V = h.shape
        if self.mode == "part_mean":
            w = self.member / self.member.sum(dim=1, keepdim=True)          # (P, V)
            pooled = torch.einsum("bctv,pv->btpc", h, w)
        else:
            s = self.score(h).squeeze(1) + self.vertex_bias                 # (B, T, V)
            s = s.unsqueeze(2) + self.outside                               # (B, T, P, V)
            a = torch.softmax(s, dim=-1)
            pooled = torch.einsum("bctv,btpv->btpc", h, a)
        return pooled.reshape(B, T, self.num_parts * C)