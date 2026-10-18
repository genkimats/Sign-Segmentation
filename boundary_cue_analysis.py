"""
boundary_cue_analysis.py -- which visible cues mark phrase boundaries? (model-free)

No trained model is involved: the script measures interpretable cues (eye blinks,
eyebrow height, mouth opening, head nods, hand height / speed / holds, handshape
change, HaMeR finger articulation ...) straight from the keypoint files and asks how
they behave around GOLD phrase boundaries compared with matched reference frames.

------------------------------------------------------------------------------------
Events (all from the gold labels, never from a model)
  start_continuous / start_short / start_long
        phrase starts (first frame of the first sign) typed by what precedes them
        (same rule as evaluate_phrase.py --boundary-types: no gap / <= short_gap O
        frames / longer pause). start_all = all of them (video starts excluded).
  end_continuous / end_short / end_long
        phrase ends (last frame of the last sign) typed by what follows. end_all.
  sign_start / sign_end
        sign starts / ends INSIDE a phrase that are not phrase boundaries (from the
        gloss labels, processed_data/BIO_tags), at least --exclusion frames away from
        any phrase boundary. THE KEY CONTROL: a cue that is equally strong here only
        marks "a new sign", not "a new phrase".
  sign_start_gap
        the PAUSE-MATCHED control: within-phrase sign starts preceded by a gloss gap
        of 1..short_gap frames (like start_short). start_short vs sign_start_gap shows
        what marks a phrase start BEYOND the short pause itself (hands relaxing during
        the gap appear in both).
  control
        random frames inside phrases, at least --margin away from any phrase boundary.
  rest
        random Outside frames (no signing), at least --margin away from any boundary.

Cues (per frame; computed per video, then robust-z-scored per video = (x - median) /
(1.4826 MAD), so signers with different faces / signing styles can be pooled; "rate"
cues are 0/1 and kept as rates)
  eyes   : eye aspect ratio (EAR), blink (EAR < --blink-ratio x video median), eye
           openness asymmetry
  brows  : brow height above the eye (outer upper edge), inner-brow height, brow
           furrow (distance between the inner brow ends), brow vertical velocity
  mouth  : mouth aspect ratio (MAR, inner lips / mouth width), mouth opening, mouth
           width (spread / smile), lower-lip (jaw) position, mouth motion energy
           (mouthing activity), mouth opening velocity
  face   : whole-face deformation energy (the v2 face is face-local, so this is
           expression change, not head movement)
  head   : head height, pitch (nod), yaw / roll deviation from the video median, head
           speed, vertical head velocity (signed: nods), head rotation speed
  torso  : shoulder tilt, shoulder-width change (lean / turn) -- skipped automatically
           if the body files are already shoulder-normalized (no variance)
  hands  : dominant / non-dominant wrist height, wrist speed, dominant acceleration,
           dominant hold (speed in the video's lowest 20 %), distance between the hands,
           dominant hand-to-face distance, signing-space reach, two-handedness
           (non-dominant share of hand speed), mirrored velocity symmetry
  handshape (MediaPipe hands): detection rate, dominant hand openness (fingertips
           to wrist / palm size), finger spread, handshape(+orientation) change speed
  hamer  : detection rate, dominant finger flexion (mean joint angle), finger
           articulation speed (geodesic change of the 15 joint rotations), wrist
           rotation speed (global_orient), non-dominant articulation speed
  body   : whole-body motion energy (mean speed of all body + hand points)
  "Dominant" = per video, the hand (pose wrist / MediaPipe hand block / HaMeR slot)
  that moves more, so left- and right-handed signers are pooled correctly and the
  MediaPipe / HaMeR left-right naming conventions do not matter.

Outputs (evaluation_phrase/boundary_cues/<split>/)
  locked_curves.csv      cue x event x offset: mean / SE of z (and raw units)
  curves_phrase_starts.png / curves_phrase_ends.png
                         boundary-locked averages, one panel per cue
  cue_summary.csv        per cue x comparison x window statistic (pre / post / delta):
                         AUC (video-cluster bootstrap 95 % CI), Cohen's d, means, counts
  cue_peaks.csv          when (offset) each cue deviates most from the control level
  ranking.png            cues ranked by how well they separate phrase boundaries from
                         within-phrase sign boundaries
  probe.csv / probe.png  (unless --no-probe) cross-validated logistic-regression probe
                         per cue family (alone and leave-one-family-out), grouped by
                         video: a model-free estimate of how much boundary information
                         each family carries and how much is unique to it
  run_info.json          settings, event counts, cue availability

Usage (repo root):
  python boundary_cue_analysis.py                      # train split, all defaults
  python boundary_cue_analysis.py --split val
  python boundary_cue_analysis.py --split all --fps 50 --window 0.5 --radius 1.5
  python boundary_cue_analysis.py --families eyes brows mouth --no-probe
  python boundary_cue_analysis.py --max-videos 20      # quick check

Reading the results
  AUC 0.5 = the cue has no information; > 0.5 = the cue is HIGHER at the first event
  than at the reference, < 0.5 = lower (direction matters: e.g. EAR < 0.5 = more
  closed eyes at phrase boundaries). The bootstrap resamples whole videos, so the CI
  reflects signer-to-signer variation, not only frame noise.
  Caveats: correlation, not cause -- a cue can co-occur with boundaries without the
  model (or the signer) using it; pause-related cues (hands down, hold) dominate
  start_long by construction, which is why start_short / start_continuous vs
  sign_start is the informative comparison; 2D face landmarks cannot see gaze (no
  iris points were saved) or lip protrusion.
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.face_subsets import SAVED_FACE_INDICES, LIPS_INDICES  # noqa: E402
from src.cue_sets import FAMILIES, CUES, CUE_NAMES, CUE_INFO  # noqa: E402,F401
from src.metrics import extract_segments, O_TAG  # noqa: E402

OUT_ROOT = os.path.join("evaluation_phrase", "boundary_cues")
SPLIT_FILE = "dataset_splits.json"
DEFAULT_DIRS = {
    "labels": "processed_data/BIO_tags_phrase_v2",
    "gloss": "processed_data/BIO_tags",
    "kinematic": "processed_data/kinematic_features",
    "face": "processed_data/face_keypoints_normalized_v2",
    "hamer": "processed_data/hamer_features",
}

FAMILY_COLORS = {"eyes": "#1f77b4", "brows": "#17becf", "mouth": "#e377c2", "face": "#9467bd",
                 "head": "#8c564b", "torso": "#7f7f7f", "hands": "#d62728", "handshape": "#ff7f0e",
                 "hamer": "#2ca02c", "body": "#bcbd22"}

START_TYPES = ["start_continuous", "start_short", "start_long"]
END_TYPES = ["end_continuous", "end_short", "end_long"]
EVENT_TYPES = START_TYPES + END_TYPES + ["sign_start", "sign_start_gap", "sign_end", "control", "rest"]
GROUPS = {"start_all": START_TYPES, "end_all": END_TYPES}

# (first, reference): AUC > 0.5 means the cue is higher at `first`
COMPARISONS = [
    ("start_all", "sign_start"),
    ("start_short", "sign_start"),
    ("start_short", "sign_start_gap"),
    ("start_continuous", "sign_start"),
    ("start_all", "control"),
    ("end_all", "sign_end"),
    ("end_short", "sign_end"),
    ("end_all", "control"),
]
STATS = ["pre", "post", "delta"]

# MediaPipe face points used (raw indices; must be in SAVED_FACE_INDICES)
EYE_L = [33, 160, 158, 133, 153, 144]     # p1..p6 for EAR (outer, top, top, inner, bottom, bottom)
EYE_R = [263, 387, 385, 362, 380, 373]
BROW_L_UPPER = [70, 63, 105, 66, 107]
BROW_R_UPPER = [300, 293, 334, 296, 336]
BROW_INNER = (107, 336)
EYE_INNER = (133, 362)
MOUTH_CORNERS = (61, 291)
INNER_LIP_PAIRS = [(13, 14), (81, 87), (311, 317)]
LOWER_LIP_CENTRE = 17

# pose landmarks (MediaPipe Pose, vertices 0-22 of the body files)
NOSE, L_EAR, R_EAR, L_SH, R_SH, L_WR, R_WR = 0, 7, 8, 11, 12, 15, 16
HAND_BLOCKS = (23, 44)          # MediaPipe hand blocks: 21 points each
FINGERTIPS = [4, 8, 12, 16, 20]
PALM_MID_MCP = 9


# ==============================================================================
# Small helpers
# ==============================================================================
def interp_nan(x):
    """Linear interpolation over NaN along axis 0 (nearest value at the ends). x: (T, ...)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    flat = x.reshape(T, -1).copy()
    t = np.arange(T)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = np.isfinite(col)
        if ok.all() or not ok.any():
            continue
        flat[:, c] = np.interp(t, t[ok], col[ok])
    return flat.reshape(x.shape)


def smooth(x, k):
    """Centered moving average along axis 0 (edge-padded). Assumes no NaN."""
    if k is None or k <= 1:
        return x
    k = int(k) | 1  # odd
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], pad, axis=0)], axis=0)
    c = np.cumsum(xp, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    return (c[k:] - c[:-k]) / k


def deriv(x, fps):
    """Central-difference time derivative (per second)."""
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, axis=0) * fps


def norm(v, axis=-1):
    return np.linalg.norm(v, axis=axis)


def masked(x, valid):
    out = np.asarray(x, dtype=np.float64).copy()
    if valid is not None:
        out[~valid] = np.nan
    return out


def robust_z(x):
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.full_like(x, np.nan), None
    med = np.median(x[ok])
    sd = np.std(x[ok])
    # Robust scale with a floor: for speed-like cues that are ~0 in most frames (face /
    # mouth motion at rest) the MAD collapses towards 0 and z explodes (values of 1e4).
    # The floor keeps z interpretable without giving up robustness for normal cues.
    scale = max(1.4826 * np.median(np.abs(x[ok] - med)), 0.1 * sd)
    if not np.isfinite(scale) or scale < 1e-9:
        return np.full_like(x, np.nan), None  # no variation in this video
    return (x - med) / scale, (med, scale)


def angle_dev(a):
    """|angle - video median|, angles in radians, wrapped."""
    ok = np.isfinite(a)
    if not ok.any():
        return a
    med = np.median(a[ok])
    return np.abs(np.angle(np.exp(1j * (a - med))))


def load_pt(path):
    import torch  # only needed for .pt files
    return torch.load(path, weights_only=False, map_location="cpu")


def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ==============================================================================
# Cue computation
# ==============================================================================
def face_cues(face, detected, fps, k):
    """face: (T, 83, 2) face-local v2 coordinates (inter-ocular units, y down)."""
    pos = {r: i for i, r in enumerate(SAVED_FACE_INDICES)}
    T = face.shape[0]
    valid = np.ones(T, dtype=bool) if detected is None else detected.astype(bool)[:T]
    f = face.astype(np.float64).copy()
    f[~valid] = np.nan
    f = smooth(interp_nan(f), k)

    def P(raw):
        return f[:, pos[raw]]

    def ear(ids):
        p = [P(r) for r in ids]
        width = norm(p[0] - p[3])
        return (norm(p[1] - p[5]) + norm(p[2] - p[4])) / (2.0 * np.maximum(width, 1e-6))

    ear_l, ear_r = ear(EYE_L), ear(EYE_R)
    ear_m = (ear_l + ear_r) / 2.0
    med_ear = np.nanmedian(ear_m[valid]) if valid.any() else np.nan
    out = {}
    out["EAR (eye openness)"] = ear_m
    out["eye asymmetry"] = np.abs(ear_l - ear_r)

    eye_c_l = np.mean([P(r) for r in EYE_L], axis=0)
    eye_c_r = np.mean([P(r) for r in EYE_R], axis=0)
    brow_l = np.mean([P(r) for r in BROW_L_UPPER], axis=0)
    brow_r = np.mean([P(r) for r in BROW_R_UPPER], axis=0)
    brow_h = ((eye_c_l[:, 1] - brow_l[:, 1]) + (eye_c_r[:, 1] - brow_r[:, 1])) / 2.0  # y down -> above = +
    out["brow height"] = brow_h
    out["inner brow height"] = ((P(EYE_INNER[0])[:, 1] - P(BROW_INNER[0])[:, 1]) +
                                (P(EYE_INNER[1])[:, 1] - P(BROW_INNER[1])[:, 1])) / 2.0
    out["brow furrow (inner distance)"] = norm(P(BROW_INNER[0]) - P(BROW_INNER[1]))
    out["brow vertical velocity"] = deriv(brow_h, fps)

    mw = norm(P(MOUTH_CORNERS[0]) - P(MOUTH_CORNERS[1]))
    opening = np.mean([norm(P(a) - P(b)) for a, b in INNER_LIP_PAIRS], axis=0)
    mar = opening / np.maximum(mw, 1e-6)
    out["MAR (mouth aspect ratio)"] = mar
    out["mouth opening"] = norm(P(13) - P(14))
    out["mouth width"] = mw
    eye_y = (eye_c_l[:, 1] + eye_c_r[:, 1]) / 2.0
    out["lower lip / jaw"] = P(LOWER_LIP_CENTRE)[:, 1] - eye_y
    lips = sorted({pos[r] for r in LIPS_INDICES})
    out["mouth motion energy"] = smooth(norm(deriv(f[:, lips], fps)).mean(axis=1), k)
    out["mouth opening velocity"] = deriv(mar, fps)
    out["face deformation energy"] = smooth(norm(deriv(f, fps)).mean(axis=1), k)

    out = {n: masked(v, valid) for n, v in out.items()}
    # blink = a SHORT closure: EAR below blink_ratio x median for 2 frames .. 0.4 s.
    # Longer low-EAR stretches are looking down / squinting, not blinks.
    low = np.nan_to_num(ear_m, nan=np.inf) < BLINK_RATIO * med_ear
    blink = np.zeros(T)
    max_len = max(2, int(round(0.4 * fps)))
    d = np.diff(np.concatenate([[0], low.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if 2 <= b - a <= max_len:
            blink[a:b] = 1.0
    out["blink"] = masked(blink, valid)
    out["eyes narrowed (long)"] = masked((low & (blink == 0)).astype(np.float64), valid)
    return out


def body_hand_cues(base, fps, k):
    """base: (T, 65, C) MediaPipe body (0-22) + left hand (23-43) + right hand (44-64)."""
    xy = base[..., :2].astype(np.float64)
    T = xy.shape[0]
    out, notes = {}, []

    # --- hand detection masks (missing hands are zero / constant / forward-filled) ---
    hand_valid = []
    for off in HAND_BLOCKS:
        h = xy[:, off:off + 21]
        spread = h.reshape(T, -1).std(axis=1)
        same = np.zeros(T, dtype=bool)
        same[1:] = np.all(h[1:] == h[:-1], axis=(1, 2))
        hand_valid.append((spread > 1e-6) & ~same & np.all(np.isfinite(h), axis=(1, 2)))

    pose = xy[:, :23].copy()
    pose_ok = np.all(np.isfinite(pose), axis=(1, 2)) & (pose.reshape(T, -1).std(axis=1) > 1e-9)
    if pose_ok.sum() < 10:
        return {}, ["no usable pose frames"]
    pose[~pose_ok] = np.nan
    pose = interp_nan(pose)

    sh_mid = (pose[:, L_SH] + pose[:, R_SH]) / 2.0
    sh_vec = pose[:, R_SH] - pose[:, L_SH]
    sh_w = norm(sh_vec)
    scale = np.median(sh_w[pose_ok])
    if not np.isfinite(scale) or scale < 1e-9:
        return {}, ["zero shoulder width"]

    def to_body(p):  # per-frame shoulder-centred, per-video shoulder-width units
        return (p - sh_mid[:, None, :] if p.ndim == 3 else p - sh_mid) / scale

    pb = smooth(to_body(pose), k)
    # y direction: the nose must be ABOVE the shoulders -> decide the sign of "up"
    up_sign = -1.0 if np.median(pb[pose_ok, NOSE, 1]) < 0 else 1.0
    if up_sign > 0:
        notes.append("y axis points up in the body files")

    def up(p):
        return up_sign * p[..., 1]

    # --- head ---
    nose, ear_l, ear_r = pb[:, NOSE], pb[:, L_EAR], pb[:, R_EAR]
    ear_mid = (ear_l + ear_r) / 2.0
    ear_d = np.maximum(norm(ear_r - ear_l), 1e-6)
    yaw = (nose[:, 0] - ear_mid[:, 0]) / ear_d
    pitch = (up(nose) - up(ear_mid)) / ear_d
    roll = np.arctan2(ear_r[:, 1] - ear_l[:, 1], ear_r[:, 0] - ear_l[:, 0])
    out["head height"] = up(nose)
    out["head pitch (nod)"] = pitch
    out["head yaw deviation"] = np.abs(yaw - np.median(yaw[pose_ok]))
    out["head roll deviation"] = angle_dev(roll)
    out["head speed"] = smooth(norm(deriv(nose, fps)), k)
    out["head vertical velocity"] = deriv(up(nose), fps)
    out["head rotation speed"] = smooth(np.sqrt(deriv(yaw, fps) ** 2 + deriv(pitch, fps) ** 2), k)

    # --- torso (only meaningful if the files are not already shoulder-normalized) ---
    tilt = np.arctan2(sh_vec[:, 1], sh_vec[:, 0])
    out["shoulder tilt"] = angle_dev(smooth(tilt, k))
    out["shoulder width change"] = smooth(sh_w / scale, k)

    # --- wrists: dominant = the one that moves more ---
    wr = {L_WR: pb[:, L_WR], R_WR: pb[:, R_WR]}
    spd = {j: smooth(norm(deriv(w, fps)), k) for j, w in wr.items()}
    dom = L_WR if np.median(spd[L_WR][pose_ok]) > np.median(spd[R_WR][pose_ok]) else R_WR
    ndom = R_WR if dom == L_WR else L_WR
    dw, nw = wr[dom], wr[ndom]
    out["dom wrist height"] = up(dw)
    out["nondom wrist height"] = up(nw)
    out["dom wrist speed"] = spd[dom]
    out["nondom wrist speed"] = spd[ndom]
    out["dom wrist acceleration"] = smooth(norm(deriv(deriv(dw, fps), fps)), k)
    thr = np.percentile(spd[dom][pose_ok], 20)
    out["dom hold"] = (spd[dom] <= thr).astype(np.float64)
    out["hand distance"] = norm(dw - nw)
    out["dom hand-to-face"] = norm(dw - nose)
    out["dom reach (signing space)"] = norm(dw)
    out["two-handedness"] = spd[ndom] / np.maximum(spd[dom] + spd[ndom], 1e-6)
    vd, vn = deriv(dw, fps), deriv(nw, fps)
    vn_m = vn * np.array([-1.0, 1.0])
    cos = (vd * vn_m).sum(-1) / np.maximum(norm(vd) * norm(vn_m), 1e-6)
    moving = (norm(vd) > np.percentile(norm(vd)[pose_ok], 20)) & (norm(vn) > np.percentile(norm(vn)[pose_ok], 20))
    out["velocity symmetry (mirrored)"] = np.where(moving, smooth(cos, k), np.nan)
    out = {n: masked(v, pose_ok) for n, v in out.items()}

    # --- MediaPipe hand shape: match each block to a pose wrist, then dom / nondom ---
    blocks = {}
    for off, hv in zip(HAND_BLOCKS, hand_valid):
        if hv.sum() < 10:
            continue
        h0 = to_body(xy[:, off])
        d_l = np.median(norm(h0[hv] - pb[hv, L_WR]))
        d_r = np.median(norm(h0[hv] - pb[hv, R_WR]))
        blocks[L_WR if d_l < d_r else R_WR] = (off, hv)
    for role, joint in (("dom", dom), ("nondom", ndom)):
        name_det = f"MP {role} hand detected"
        if joint not in blocks:
            out[name_det] = masked(np.zeros(T), pose_ok)
            continue
        off, hv = blocks[joint]
        out[name_det] = masked(hv.astype(np.float64), pose_ok)
        h = xy[:, off:off + 21].copy()
        h[~hv] = np.nan
        h = smooth(interp_nan(h), k)
        rel = h - h[:, :1]
        palm = np.maximum(norm(rel[:, PALM_MID_MCP]), 1e-6)
        shape = rel / palm[:, None, None]
        change = smooth(norm(deriv(shape, fps)).mean(axis=1), k)
        out[f"{role} handshape change"] = masked(change, hv)
        if role == "dom":
            out["dom hand openness"] = masked(norm(shape[:, FINGERTIPS]).mean(axis=1), hv)
            out["dom finger spread"] = masked(norm(shape[:, 8] - shape[:, 20]), hv)

    # --- whole-body motion energy (pose + detected hands) ---
    all_pts = to_body(xy)
    sp = norm(deriv(smooth(interp_nan(np.where(np.isfinite(all_pts), all_pts, np.nan)), k), fps))
    valid_pts = np.ones((T, 65), dtype=bool)
    for off, hv in zip(HAND_BLOCKS, hand_valid):
        valid_pts[~hv, off:off + 21] = False
    sp = np.where(valid_pts, sp, np.nan)
    with np.errstate(invalid="ignore"):
        energy = np.nanmean(sp, axis=1)
    out["whole-body motion energy"] = masked(smooth(interp_nan(energy), k), pose_ok)
    return out, notes


def rot_angle(R):
    """Rotation angle of (..., 3, 3) matrices, radians."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def geodesic(Ra, Rb):
    return rot_angle(np.einsum("...ji,...jk->...ik", Ra, Rb))


def hamer_cues(hp, go, ds, T, fps, k):
    """hp: (T', 2, 15, 3, 3), go: (T', 2, 3, 3) at 1/ds of the frame rate; zeros = no hand."""
    hp = hp.astype(np.float64)
    go = go.astype(np.float64)
    Tp = hp.shape[0]
    det = np.abs(go).reshape(Tp, 2, -1).sum(-1) > 1e-6      # (T', 2)
    eff_fps = fps / max(ds, 1)
    per_slot = []
    for s in range(2):
        d = det[:, s]
        flex = rot_angle(hp[:, s]).mean(axis=1)                       # (T',)
        art = np.full(Tp, np.nan)
        wrot = np.full(Tp, np.nan)
        both = d[1:] & d[:-1]
        art[1:][both] = geodesic(hp[1:, s][both], hp[:-1, s][both]).mean(axis=1) * eff_fps
        wrot[1:][both] = geodesic(go[1:, s][both], go[:-1, s][both]) * eff_fps
        flex[~d] = np.nan
        per_slot.append({"det": d, "flex": flex, "art": art, "wrot": wrot})
    med = [np.nanmedian(p["wrot"]) if np.isfinite(p["wrot"]).any() else -1 for p in per_slot]
    dom = int(np.argmax(med))
    idx = (np.arange(T) // max(ds, 1)).clip(max=Tp - 1)

    def up_sample(x):
        v = x[idx].astype(np.float64)
        ok = np.isfinite(v)
        if ok.sum() >= 2:
            v = smooth(interp_nan(v), k)
            v[~ok] = np.nan
        return v

    d_dom = per_slot[dom]["det"][idx]
    out = {
        "HaMeR dom detected": per_slot[dom]["det"][idx].astype(np.float64),
        "HaMeR dom finger flexion": up_sample(per_slot[dom]["flex"]),
        "HaMeR dom articulation speed": up_sample(per_slot[dom]["art"]),
        "HaMeR dom wrist rotation speed": up_sample(per_slot[dom]["wrot"]),
        "HaMeR nondom articulation speed": up_sample(per_slot[1 - dom]["art"]),
    }
    for n in ("HaMeR dom finger flexion", "HaMeR dom articulation speed", "HaMeR dom wrist rotation speed"):
        out[n][~d_dom] = np.nan
    return out


# ==============================================================================
# Events
# ==============================================================================
def sign_segments(gloss):
    segs, start = [], -1
    for i, tag in enumerate(gloss):
        if tag == 2:
            if start != -1:
                segs.append((start, i - 1))
            start = i
        elif tag == 0:
            if start != -1:
                segs.append((start, i - 1))
                start = -1
        elif start == -1:
            start = i
    if start != -1:
        segs.append((start, len(gloss) - 1))
    return segs


def find_events(labels, gloss, short_gap, margin, exclusion, max_per_video, rng):
    """Returns (frames, type_index) arrays."""
    T = len(labels)
    phrases = extract_segments(labels)
    ev = []
    bounds = []
    for i, (s, e) in enumerate(phrases):
        bounds += [s, e]
        if s > 0:
            if labels[s - 1] != O_TAG:
                ev.append((s, "start_continuous"))
            else:
                k = s - 1
                while k >= 0 and labels[k] == O_TAG:
                    k -= 1
                ev.append((s, "start_short" if s - 1 - k <= short_gap else "start_long"))
        if e < T - 1:
            if labels[e + 1] != O_TAG:
                ev.append((e, "end_continuous"))
            else:
                k = e + 1
                while k < T and labels[k] == O_TAG:
                    k += 1
                if k >= T:
                    continue  # rest until the video ends: no following phrase
                ev.append((e, "end_short" if k - e - 1 <= short_gap else "end_long"))
    bounds = np.array(sorted(bounds)) if bounds else np.array([], dtype=int)

    def dist_to_bound(frames):
        if len(bounds) == 0:
            return np.full(len(frames), np.inf)
        j = np.searchsorted(bounds, frames)
        lo = np.abs(frames - bounds[np.clip(j - 1, 0, len(bounds) - 1)])
        hi = np.abs(bounds[np.clip(j, 0, len(bounds) - 1)] - frames)
        return np.minimum(lo, hi)

    def capped(frames, name):
        frames = np.asarray(frames, dtype=np.int64)
        if len(frames) > max_per_video:
            frames = rng.choice(frames, max_per_video, replace=False)
        ev.extend((int(f), name) for f in frames)

    inside = labels != O_TAG
    if gloss is not None:
        ss, sg, se = [], [], []
        prev_end = None
        for s, e in sign_segments(gloss):
            for f, lst in ((s, ss), (e, se)):
                if 0 <= f < T and inside[f]:
                    lst.append(f)
            if prev_end is not None and 0 <= s < T and inside[s] and 1 <= s - prev_end - 1 <= short_gap:
                sg.append(s)
            prev_end = e
        for lst, name in ((ss, "sign_start"), (sg, "sign_start_gap"), (se, "sign_end")):
            arr = np.array(lst, dtype=np.int64)
            if len(arr):
                arr = arr[dist_to_bound(arr) >= exclusion]
            capped(arr, name)
    frames = np.arange(T)
    far = dist_to_bound(frames) >= margin
    capped(frames[inside & far], "control")
    capped(frames[~inside & far], "rest")

    if not ev:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    f = np.array([e[0] for e in ev], dtype=np.int64)
    t = np.array([EVENT_TYPES.index(e[1]) for e in ev], dtype=np.int64)
    return f, t


# ==============================================================================
# Statistics
# ==============================================================================
def placements(a, b):
    """Mann-Whitney placement values. AUC = mean(v10) = mean(v01). Ties count 1/2."""
    bs, as_ = np.sort(b), np.sort(a)
    v10 = (np.searchsorted(bs, a, "left") + np.searchsorted(bs, a, "right")) / (2.0 * len(b))
    v01 = 1.0 - (np.searchsorted(as_, b, "left") + np.searchsorted(as_, b, "right")) / (2.0 * len(a))
    return v10, v01


def auc_with_ci(a, va, b, vb, n_videos, boot_w):
    """AUC of a vs b with a video-cluster bootstrap CI (structural-components placements)."""
    ok_a, ok_b = np.isfinite(a), np.isfinite(b)
    a, va, b, vb = a[ok_a], va[ok_a], b[ok_b], vb[ok_b]
    if len(a) < 5 or len(b) < 5:
        return None
    v10, v01 = placements(a, b)
    auc = float(v10.mean())
    s10 = np.bincount(va, weights=v10, minlength=n_videos)
    c10 = np.bincount(va, minlength=n_videos)
    s01 = np.bincount(vb, weights=v01, minlength=n_videos)
    c01 = np.bincount(vb, minlength=n_videos)
    lo = hi = np.nan
    if boot_w is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = 0.5 * ((boot_w @ s10) / (boot_w @ c10) + (boot_w @ s01) / (boot_w @ c01))
        boots = boots[np.isfinite(boots)]
        if len(boots):
            lo, hi = np.percentile(boots, [2.5, 97.5])
    pooled = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / max(len(a) + len(b) - 2, 1))
    d = (a.mean() - b.mean()) / pooled if pooled > 1e-12 else np.nan
    return {"auc": auc, "ci_lo": float(lo), "ci_hi": float(hi), "cohens_d": float(d),
            "mean_first": float(a.mean()), "mean_ref": float(b.mean()),
            "n_first": int(len(a)), "n_ref": int(len(b)),
            "videos_first": int((c10 > 0).sum()), "videos_ref": int((c01 > 0).sum())}


def roc_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(placements(pos, neg)[0].mean())


def fit_logreg(X, y, w, lam=1.0, iters=50):
    """L2-regularized logistic regression by DAMPED Newton (step halving until the
    penalized loss decreases), so it cannot diverge on correlated features.
    Returns (coef, intercept)."""
    n, p = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(p + 1)
    reg = np.full(p + 1, lam)
    reg[-1] = 0.0

    def loss(b):
        z = Xa @ b
        return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(reg * b * b))

    cur = loss(beta)
    for _ in range(iters):
        z = np.clip(Xa @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-z))
        g = Xa.T @ (w * (mu - y)) + reg * beta
        H = (Xa * (w * mu * (1 - mu))[:, None]).T @ Xa + np.diag(reg + 1e-6)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4:
            cand = beta - t * step
            new = loss(cand)
            if new <= cur:
                break
            t *= 0.5
        else:
            break
        beta, gain, cur = cand, cur - new, new
        if gain < 1e-7 * max(abs(cur), 1.0):
            break
    return beta[:-1], beta[-1]


def cv_auc(X, y, groups, folds, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    fold_of = {g: i % folds for i, g in enumerate(perm)}
    fid = np.array([fold_of[g] for g in groups])
    scores = np.full(len(y), np.nan)
    for f in range(folds):
        tr, te = fid != f, fid == f
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        # robust standardization on the training fold, then clipping: heavy-tailed cues
        # (speeds) must not produce extreme scores that wreck the pooled AUC
        med = np.median(X[tr], axis=0)
        q75, q25 = np.percentile(X[tr], [75, 25], axis=0)
        sd = (q75 - q25) / 1.349
        sd_fallback = X[tr].std(0)
        sd = np.where(sd < 1e-9, sd_fallback, sd)
        sd[sd < 1e-9] = 1.0
        Xtr = np.clip((X[tr] - med) / sd, -5, 5)
        Xte = np.clip((X[te] - med) / sd, -5, 5)
        pos = y[tr].mean()
        w = np.where(y[tr] == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))
        coef, b0 = fit_logreg(Xtr, y[tr].astype(np.float64), w / w.mean(), lam=PROBE_L2)
        scores[te] = Xte @ coef + b0
    ok = np.isfinite(scores)
    return roc_auc(y[ok], scores[ok])


# ==============================================================================
# Main
# ==============================================================================
BLINK_RATIO = 0.75
PROBE_L2 = 10.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Model-free analysis of visible cues (blinks, brows, mouth, head, hands, "
                    "handshape, HaMeR) around gold phrase boundaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python boundary_cue_analysis.py\n"
               "  python boundary_cue_analysis.py --split val --radius 2.0\n"
               "  python boundary_cue_analysis.py --families eyes brows mouth head --no-probe\n"
               "  python boundary_cue_analysis.py --max-videos 20   # quick test")
    p.add_argument("--split", default="train", help="train / val / test / all (default: train -- "
                   "more data, and val/test stay untouched).")
    p.add_argument("--split-file", default=SPLIT_FILE)
    p.add_argument("--labels-dir", default=DEFAULT_DIRS["labels"], help="phrase BIO labels (default: v2)")
    p.add_argument("--gloss-dir", default=DEFAULT_DIRS["gloss"], help="sign-level BIO labels (for sign_start / sign_end)")
    p.add_argument("--kinematic-dir", default=DEFAULT_DIRS["kinematic"])
    p.add_argument("--face-dir", default=DEFAULT_DIRS["face"], help="face v2 files (+ _detection/)")
    p.add_argument("--hamer-dir", default=DEFAULT_DIRS["hamer"])
    p.add_argument("--fps", type=float, default=50.0, help="frame rate of the label files (Public DGS Corpus: 50)")
    p.add_argument("--radius", type=float, default=1.5, help="curve half-width in seconds (default 1.5)")
    p.add_argument("--window", type=float, default=0.5, help="pre / post window for the statistics, seconds (default 0.5)")
    p.add_argument("--short-gap", type=int, default=12, help="max Outside frames of a short pause (same as evaluate_phrase.py)")
    p.add_argument("--margin", type=float, default=1.0, help="control / rest frames: min distance to a phrase boundary, s")
    p.add_argument("--exclusion", type=float, default=None,
                   help="sign_start / sign_end: min distance to a phrase boundary, s (default: --window)")
    p.add_argument("--max-per-video", type=int, default=60, help="cap for sign / control / rest events per video")
    p.add_argument("--smooth", type=int, default=3, help="moving-average length in frames before derivatives")
    p.add_argument("--blink-ratio", type=float, default=0.75, help="blink = EAR below this x video median")
    p.add_argument("--families", nargs="+", default=None, choices=FAMILIES, help="only these cue families")
    p.add_argument("--bootstrap", type=int, default=1000, help="video bootstrap samples for the AUC CI (0 = off)")
    p.add_argument("--no-probe", action="store_true", help="skip the logistic-regression family probe")
    p.add_argument("--probe-window", type=float, default=1.0, help="probe features: +- seconds around the event")
    p.add_argument("--probe-bins", type=int, default=6, help="probe features: time bins per cue")
    p.add_argument("--probe-max", type=int, default=20000, help="probe: max events per class")
    p.add_argument("--probe-folds", type=int, default=5)
    p.add_argument("--max-videos", type=int, default=None, help="only the first N videos (quick test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None, help=f"default: {OUT_ROOT}/<split>")
    return p.parse_args()


def main():
    global BLINK_RATIO
    args = parse_args()
    BLINK_RATIO = args.blink_ratio
    fps = args.fps
    R = int(round(args.radius * fps))
    W = max(1, int(round(args.window * fps)))
    margin = int(round(args.margin * fps))
    exclusion = int(round((args.exclusion if args.exclusion is not None else args.window) * fps))
    PW = int(round(args.probe_window * fps))
    nb = args.probe_bins
    offsets = np.arange(-R, R + 1)
    rng = np.random.default_rng(args.seed)
    cue_names = [c for c in CUE_NAMES if args.families is None or CUE_INFO[c]["family"] in args.families]
    nC, nE = len(cue_names), len(EVENT_TYPES)

    with open(args.split_file) as f:
        splits = json.load(f)
    if args.split == "all":
        vids = sorted({v for s in splits.values() for v in s})
    elif args.split in splits:
        vids = list(splits[args.split])
    else:
        raise SystemExit(f"Split '{args.split}' not in {args.split_file} (have: {list(splits)}).")
    vids = [v.replace(".npy", "").replace(".pt", "") for v in vids]
    if args.max_videos:
        vids = vids[:args.max_videos]
    out_dir = args.out_dir or os.path.join(OUT_ROOT, args.split)
    os.makedirs(out_dir, exist_ok=True)

    # accumulators: [cue, event_type, offset]
    acc = {k: np.zeros((nC, nE, 2 * R + 1)) for k in ("z_sum", "z_sq", "z_n", "r_sum", "r_sq")}
    ev_type_all, ev_vid_all = [], []
    stats_all = []   # per video: (n_ev, nC, 3)
    probe_all = []   # per video: (n_ev, nC, nb)
    counts = {"videos_used": 0, "no_labels": 0, "no_gloss": 0, "no_body": 0, "no_face": 0,
              "no_hamer": 0, "length_mismatch_cropped": 0, "no_events": 0}
    cue_videos = {c: 0 for c in cue_names}
    notes_seen = set()
    raw_scale = {c: [] for c in cue_names}  # per-video (median, scale) for unit reference
    t0 = time.time()

    pre_sl = slice(R - W, R)
    post_sl = slice(R, R + W)
    edges = np.linspace(-PW, PW, nb + 1).round().astype(int) + R

    for vi, vid in enumerate(vids):
        lp = os.path.join(args.labels_dir, f"{vid}.npy")
        if not os.path.exists(lp):
            counts["no_labels"] += 1
            continue
        labels = np.load(lp).astype(np.int64)
        T = len(labels)
        gp = os.path.join(args.gloss_dir, f"{vid}.npy")
        gloss = np.load(gp).astype(np.int64) if os.path.exists(gp) else None
        if gloss is None:
            counts["no_gloss"] += 1
        elif len(gloss) != T:
            counts["length_mismatch_cropped"] += 1
            gloss = gloss[:T] if len(gloss) >= T else np.concatenate([gloss, np.zeros(T - len(gloss), int)])

        cues = {}
        # body + hands
        kp = os.path.join(args.kinematic_dir, f"{vid}.pt")
        if os.path.exists(kp):
            try:
                kin = load_pt(kp)
                if isinstance(kin, dict) and "mediapipe" in kin:
                    kin = kin["mediapipe"]
                base = to_np(kin["base"]).astype(np.float64)
                if base.shape[0] != T:
                    counts["length_mismatch_cropped"] += 1
                    base = base[:T] if base.shape[0] >= T else np.concatenate(
                        [base, np.repeat(base[-1:], T - base.shape[0], 0)])
                c, notes = body_hand_cues(base, fps, args.smooth)
                cues.update(c)
                notes_seen.update(notes)
            except Exception as e:  # noqa: BLE001
                notes_seen.add(f"body load error: {type(e).__name__}")
                counts["no_body"] += 1
        else:
            counts["no_body"] += 1
        # face
        fp = os.path.join(args.face_dir, f"{vid}.npy")
        if os.path.exists(fp):
            face = np.load(fp)
            if face.ndim == 3 and face.shape[1] == len(SAVED_FACE_INDICES):
                dp = os.path.join(args.face_dir, "_detection", f"{vid}.npy")
                det = np.load(dp) if os.path.exists(dp) else None
                if face.shape[0] != T:
                    counts["length_mismatch_cropped"] += 1
                    n = min(T, face.shape[0])
                    face2 = np.repeat(face[-1:], T, 0)
                    face2[:n] = face[:n]
                    det2 = np.zeros(T, dtype=bool)
                    det2[:n] = det[:n] if det is not None else True
                    face, det = face2, det2
                cues.update(face_cues(face[..., :2], det, fps, args.smooth))
            else:
                counts["no_face"] += 1
        else:
            counts["no_face"] += 1
        # HaMeR
        hp_path = os.path.join(args.hamer_dir, f"{vid}_hamer.pt")
        if os.path.exists(hp_path):
            try:
                hd = load_pt(hp_path)
                hpose, gor = to_np(hd["hand_pose"]), to_np(hd["global_orient"])
                if len(hpose):
                    cues.update(hamer_cues(hpose, gor, int(hd.get("temporal_downsample_factor", 1)), T, fps, args.smooth))
                else:
                    counts["no_hamer"] += 1
            except Exception as e:  # noqa: BLE001
                notes_seen.add(f"hamer load error: {type(e).__name__}")
                counts["no_hamer"] += 1
        else:
            counts["no_hamer"] += 1

        frames, types = find_events(labels, gloss, args.short_gap, margin, exclusion, args.max_per_video, rng)
        if len(frames) == 0:
            counts["no_events"] += 1
            continue
        vix = counts["videos_used"]
        counts["videos_used"] += 1
        idx = frames[:, None] + offsets[None, :] + R  # into the R-padded series
        n_ev = len(frames)
        st = np.full((n_ev, nC, 3), np.nan, dtype=np.float32)
        pr = np.full((n_ev, nC, nb), np.nan, dtype=np.float32)
        for ci, cname in enumerate(cue_names):
            x = cues.get(cname)
            if x is None:
                continue
            x = np.asarray(x, dtype=np.float64)[:T]
            if len(x) < T:
                x = np.concatenate([x, np.full(T - len(x), np.nan)])
            if CUE_INFO[cname]["kind"] == "rate":
                z = x
                if not np.isfinite(x).any():
                    continue
            else:
                z, ms = robust_z(x)
                if ms is None:
                    continue
                raw_scale[cname].append(ms)
            cue_videos[cname] += 1
            pad = np.full(R, np.nan)
            wz = np.concatenate([pad, z, pad])[idx]
            wr = np.concatenate([pad, x, pad])[idx]
            for e in range(nE):
                m = types == e
                if not m.any():
                    continue
                a, b = wz[m], wr[m]
                ok = np.isfinite(a)
                acc["z_sum"][ci, e] += np.where(ok, a, 0).sum(0)
                acc["z_sq"][ci, e] += np.where(ok, a * a, 0).sum(0)
                acc["z_n"][ci, e] += ok.sum(0)
                acc["r_sum"][ci, e] += np.where(ok, b, 0).sum(0)
                acc["r_sq"][ci, e] += np.where(ok, b * b, 0).sum(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN windows -> NaN
                pre = np.nanmean(wz[:, pre_sl], axis=1)
                post = np.nanmean(wz[:, post_sl], axis=1)
                st[:, ci, 0], st[:, ci, 1], st[:, ci, 2] = pre, post, post - pre
                for bi in range(nb):
                    pr[:, ci, bi] = np.nanmean(wz[:, edges[bi]:max(edges[bi + 1], edges[bi] + 1)], axis=1)
        ev_type_all.append(types)
        ev_vid_all.append(np.full(n_ev, vix, dtype=np.int64))
        stats_all.append(st)
        probe_all.append(pr)
        if (vi + 1) % 25 == 0 or vi + 1 == len(vids):
            print(f"  {vi + 1}/{len(vids)} videos ({time.time() - t0:.0f}s)", flush=True)

    if counts["videos_used"] == 0:
        raise SystemExit(f"No usable videos. Counts: {counts}")
    types = np.concatenate(ev_type_all)
    vidx = np.concatenate(ev_vid_all)
    stats = np.concatenate(stats_all)
    probe = np.concatenate(probe_all)
    n_vid = counts["videos_used"]
    ev_counts = {t: int((types == i).sum()) for i, t in enumerate(EVENT_TYPES)}
    print(f"\nVideos used: {n_vid}/{len(vids)} | events: {ev_counts}")
    print(f"Missing inputs (videos): body {counts['no_body']}, face {counts['no_face']}, "
          f"HaMeR {counts['no_hamer']}, gloss {counts['no_gloss']}"
          + (f", length mismatches cropped {counts['length_mismatch_cropped']}" if counts["length_mismatch_cropped"] else ""))
    unavailable = [c for c in cue_names if cue_videos[c] == 0]
    if unavailable:
        print(f"Cues without data / without variation (skipped): {unavailable}")
    for n in sorted(notes_seen):
        print(f"Note: {n}")

    def group_mask(name):
        members = GROUPS.get(name, [name])
        return np.isin(types, [EVENT_TYPES.index(m) for m in members])

    # ---------------- locked curves ----------------
    def curve(ci, names, key="z"):
        es = [EVENT_TYPES.index(n) for n in names]
        s = acc[f"{key}_sum"][ci, es].sum(0)
        q = acc[f"{key}_sq"][ci, es].sum(0)
        n = acc["z_n"][ci, es].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / n
            var = q / n - mean ** 2
            se = np.sqrt(np.maximum(var, 0) / np.maximum(n - 1, 1))
        return mean, se, n

    curve_sets = START_TYPES + ["start_all"] + END_TYPES + ["end_all", "sign_start", "sign_start_gap", "sign_end",
                                                         "control", "rest"]
    with open(os.path.join(out_dir, "locked_curves.csv"), "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["cue", "family", "event", "offset_frames", "offset_s", "mean_z", "se_z", "mean_raw", "se_raw", "n"])
        for ci, cname in enumerate(cue_names):
            if cue_videos[cname] == 0:
                continue
            for ev in curve_sets:
                names = GROUPS.get(ev, [ev])
                mz, sz, n = curve(ci, names, "z")
                mr, sr, _ = curve(ci, names, "r")
                for j, o in enumerate(offsets):
                    if n[j] == 0:
                        continue
                    wri.writerow([cname, CUE_INFO[cname]["family"], ev, int(o), round(o / fps, 4),
                                  f"{mz[j]:.5f}", f"{sz[j]:.5f}", f"{mr[j]:.6g}", f"{sr[j]:.6g}", int(n[j])])

    # peaks: offset of the largest deviation from the control level
    peak_rows = []
    for ci, cname in enumerate(cue_names):
        if cue_videos[cname] == 0:
            continue
        cm, _, cn = curve(ci, ["control"])
        ctrl = np.nanmean(cm) if np.isfinite(cm).any() else 0.0
        for ev in ["start_all", "start_short", "start_long", "end_all", "sign_start", "sign_start_gap"]:
            m, se, n = curve(ci, GROUPS.get(ev, [ev]))
            if not np.isfinite(m).any() or n.max() < 20:
                continue
            dev = m - ctrl
            j = int(np.nanargmax(np.abs(dev)))
            peak_rows.append({"cue": cname, "family": CUE_INFO[cname]["family"], "event": ev,
                              "peak_offset_frames": int(offsets[j]), "peak_offset_s": round(offsets[j] / fps, 3),
                              "peak_deviation_from_control": round(float(dev[j]), 4),
                              "value_at_boundary": round(float(m[R]), 4), "control_level": round(float(ctrl), 4),
                              "n_events": int(n[R])})
    write_csv(os.path.join(out_dir, "cue_peaks.csv"), peak_rows)

    # ---------------- AUC summary ----------------
    boot_w = None
    if args.bootstrap > 0:
        boot_w = rng.multinomial(n_vid, np.full(n_vid, 1.0 / n_vid), size=args.bootstrap).astype(np.float64)
    rows = []
    for first, ref in COMPARISONS:
        ma, mb = group_mask(first), group_mask(ref)
        if ma.sum() < 5 or mb.sum() < 5:
            continue
        for ci, cname in enumerate(cue_names):
            if cue_videos[cname] == 0:
                continue
            for si, sname in enumerate(STATS):
                r = auc_with_ci(stats[ma, ci, si].astype(np.float64), vidx[ma],
                                stats[mb, ci, si].astype(np.float64), vidx[mb], n_vid, boot_w)
                if r is None:
                    continue
                rows.append({"comparison": f"{first} vs {ref}", "cue": cname, "family": CUE_INFO[cname]["family"],
                             "kind": CUE_INFO[cname]["kind"], "stat": sname,
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                             "auc_minus_half": round(r["auc"] - 0.5, 4),
                             "ci_excludes_half": bool(np.isfinite(r["ci_lo"]) and (r["ci_lo"] > 0.5 or r["ci_hi"] < 0.5))})
    write_csv(os.path.join(out_dir, "cue_summary.csv"), rows)
    print_top(rows, "start_all vs sign_start", 12)
    print_top(rows, "start_short vs sign_start", 8)
    print_top(rows, "start_short vs sign_start_gap", 8)
    print_top(rows, "end_all vs sign_end", 8)

    # ---------------- probe ----------------
    probe_rows = []
    if not args.no_probe:
        probe_rows = run_probe(probe, types, vidx, cue_names, cue_videos, group_mask, args, rng)
        write_csv(os.path.join(out_dir, "probe.csv"), probe_rows)

    # ---------------- plots ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        plot_curves(curve, cue_names, cue_videos, offsets / fps, args.window,
                    [("start_short", "phrase start, short pause", "#d62728", "-"),
                     ("start_long", "phrase start, long pause", "#ff9896", "-"),
                     ("start_continuous", "phrase start, continuous", "#8c1a1a", ":"),
                     ("sign_start", "sign start inside a phrase", "#1f77b4", "-"),
                     ("sign_start_gap", "sign start after a short gloss gap", "#9edae5", "--"),
                     ("rest", "rest (Outside)", "#999999", ":")],
                    "Cues around PHRASE STARTS (t = 0: first frame of the first sign; z per video)",
                    os.path.join(out_dir, "curves_phrase_starts.png"))
        plot_curves(curve, cue_names, cue_videos, offsets / fps, args.window,
                    [("end_short", "phrase end, short pause follows", "#2ca02c", "-"),
                     ("end_long", "phrase end, long pause follows", "#98df8a", "-"),
                     ("end_continuous", "phrase end, continuous", "#145214", ":"),
                     ("sign_end", "sign end inside a phrase", "#1f77b4", "-"),
                     ("rest", "rest (Outside)", "#999999", ":")],
                    "Cues around PHRASE ENDS (t = 0: last frame of the last sign; z per video)",
                    os.path.join(out_dir, "curves_phrase_ends.png"))
        plot_ranking(rows, os.path.join(out_dir, "ranking.png"))
        if probe_rows:
            plot_probe(probe_rows, os.path.join(out_dir, "probe.png"))
    except ImportError:
        print("matplotlib not available -- plots skipped.")

    info = {"split": args.split, "videos_requested": len(vids), "fps": fps, "radius_s": args.radius,
            "window_s": args.window, "short_gap_frames": args.short_gap, "margin_s": args.margin,
            "exclusion_frames": exclusion, "max_per_video": args.max_per_video, "smooth": args.smooth,
            "blink_ratio": args.blink_ratio, "labels_dir": args.labels_dir, "gloss_dir": args.gloss_dir,
            "face_dir": args.face_dir, "hamer_dir": args.hamer_dir, "kinematic_dir": args.kinematic_dir,
            "counts": counts, "events": ev_counts, "cue_videos": cue_videos, "notes": sorted(notes_seen),
            "bootstrap": args.bootstrap, "seed": args.seed}
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nResults in {out_dir}/ ({time.time() - t0:.0f}s)")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def print_top(rows, comparison, n):
    sel = [r for r in rows if r["comparison"] == comparison]
    if not sel:
        return
    best = {}
    for r in sel:
        if r["cue"] not in best or abs(r["auc_minus_half"]) > abs(best[r["cue"]]["auc_minus_half"]):
            best[r["cue"]] = r
    top = sorted(best.values(), key=lambda r: -abs(r["auc_minus_half"]))[:n]
    print(f"\n=== {comparison}: strongest cues (best of pre/post/delta; AUC > .5 = higher at the phrase boundary) ===")
    print(f"{'cue':34s} {'fam':9s} {'stat':5s} {'AUC':>6s} {'95% CI':>15s} {'d':>6s}")
    for r in top:
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if np.isfinite(r["ci_lo"]) else ""
        mark = " *" if r["ci_excludes_half"] else ""
        print(f"{r['cue']:34s} {r['family']:9s} {r['stat']:5s} {r['auc']:6.3f} {ci:>15s} {r['cohens_d']:6.2f}{mark}")
    print(f"(n = {top[0]['n_first']} vs {top[0]['n_ref']} events; * = CI excludes 0.5)")


def run_probe(probe, types, vidx, cue_names, cue_videos, group_mask, args, rng):
    tasks = [("start_all", "sign_start"), ("start_short", "sign_start_gap"), ("start_all", "control"),
             ("end_all", "sign_end")]
    avail = [i for i, c in enumerate(cue_names) if cue_videos[c] > 0]
    fams = [f for f in FAMILIES if any(CUE_INFO[cue_names[i]]["family"] == f for i in avail)]
    rows = []
    print("\n=== Probe: cross-validated AUC of a logistic regression on boundary-locked cues "
          f"(+-{args.probe_window}s, {args.probe_bins} bins, {args.probe_folds}-fold by video) ===")
    for first, ref in tasks:
        ma, mb = np.flatnonzero(group_mask(first)), np.flatnonzero(group_mask(ref))
        if len(ma) < 50 or len(mb) < 50:
            continue
        if len(ma) > args.probe_max:
            ma = rng.choice(ma, args.probe_max, replace=False)
        if len(mb) > args.probe_max:
            mb = rng.choice(mb, args.probe_max, replace=False)
        sel = np.concatenate([ma, mb])
        y = np.concatenate([np.ones(len(ma)), np.zeros(len(mb))]).astype(int)
        g = vidx[sel]
        Xfull = probe[sel].astype(np.float64)   # (n, nC, nb)
        Xfull = np.nan_to_num(Xfull, nan=0.0)   # z = 0 is the video median
        name = f"{first} vs {ref}"

        def feats(cidx):
            return Xfull[:, cidx, :].reshape(len(sel), -1)

        all_auc = cv_auc(feats(avail), y, g, args.probe_folds, args.seed)
        rows.append({"task": name, "features": "ALL", "mode": "all", "auc": round(all_auc, 4), "n_cues": len(avail),
                     "n_first": len(ma), "n_ref": len(mb)})
        line = [f"{name}:  ALL {all_auc:.3f}"]
        alone = {fam: cv_auc(feats([i for i in avail if CUE_INFO[cue_names[i]]["family"] == fam]),
                             y, g, args.probe_folds, args.seed) for fam in fams}
        for fam in fams:
            cidx = [i for i in avail if CUE_INFO[cue_names[i]]["family"] == fam]
            rest = [i for i in avail if CUE_INFO[cue_names[i]]["family"] != fam]
            a_alone = alone[fam]
            a_drop = cv_auc(feats(rest), y, g, args.probe_folds, args.seed) if rest else np.nan
            best_other = max([v for f2, v in alone.items() if f2 != fam] or [0.5])
            if np.isfinite(a_drop) and a_drop < best_other - 0.02:
                print(f"  WARNING ({name}): ALL without {fam} = {a_drop:.3f} is below a single other family "
                      f"({best_other:.3f}) -- the fit is unreliable; treat this 'unique' value with care.")
            rows.append({"task": name, "features": fam, "mode": "alone", "auc": round(a_alone, 4),
                         "n_cues": len(cidx), "n_first": len(ma), "n_ref": len(mb)})
            rows.append({"task": name, "features": fam, "mode": "all_without", "auc": round(a_drop, 4),
                         "n_cues": len(rest), "n_first": len(ma), "n_ref": len(mb),
                         "unique_contribution": round(all_auc - a_drop, 4)})
            line.append(f"{fam} {a_alone:.3f} (drop {all_auc - a_drop:+.3f})")
        print("  " + " | ".join(line))
    print("  (alone = family by itself; drop = AUC lost when the family is removed from ALL = its unique information)")
    return rows


def plot_curves(curve, cue_names, cue_videos, t, window, series, title, path):
    import matplotlib.pyplot as plt
    names = [c for c in cue_names if cue_videos[c] > 0]
    if not names:
        return
    ncol = 6
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.4 * nrow), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    ci_map = {c: i for i, c in enumerate(cue_names)}
    for ax, cname in zip(axes.flat, names):
        ci = ci_map[cname]
        cm, cs, cn = curve(ci, ["control"])
        if np.isfinite(cm).any():
            lvl = np.nanmean(cm)
            band = np.nanmean(cs) * 1.96
            ax.axhspan(lvl - band, lvl + band, color="#dddddd", zorder=0)
            ax.axhline(lvl, color="#888888", lw=0.8, ls="--", zorder=1)
        for ev, label, color, ls in series:
            m, se, n = curve(ci, [ev])
            if n.max() < 20:
                continue
            ax.plot(t, m, color=color, lw=1.3, ls=ls, label=label)
            ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15, lw=0)
        ax.axvline(0, color="black", lw=0.7)
        ax.axvspan(-window, window, color="#fff3c4", alpha=0.35, zorder=0)
        fam = CUE_INFO[cname]["family"]
        ax.set_title(f"{cname}", fontsize=8.5, color=FAMILY_COLORS[fam])
        ax.tick_params(labelsize=7)
        if CUE_INFO[cname]["kind"] == "rate":
            ax.set_ylabel("rate", fontsize=7)
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    import matplotlib.patches as mpatches
    handles.append(mpatches.Patch(color="#dddddd", label="in-phrase control (mean +- 95% CI)"))
    labels.append("in-phrase control (mean +- 95% CI)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.15 / nrow))
    fig.suptitle(title + "   |   x: seconds   |   title colour = cue family", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / nrow))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(rows, path):
    import matplotlib.pyplot as plt
    panels = [("start_all vs sign_start", "Phrase START vs within-phrase sign start"),
              ("start_short vs sign_start_gap", "Phrase START after a short pause\nvs sign start after an equally short gap"),
              ("end_all vs sign_end", "Phrase END vs within-phrase sign end")]
    panels = [p for p in panels if any(r["comparison"] == p[0] for r in rows)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 10), squeeze=False)
    for ax, (comp, title) in zip(axes[0], panels):
        best = {}
        for r in rows:
            if r["comparison"] != comp:
                continue
            if r["cue"] not in best or abs(r["auc_minus_half"]) > abs(best[r["cue"]]["auc_minus_half"]):
                best[r["cue"]] = r
        items = sorted(best.values(), key=lambda r: abs(r["auc_minus_half"]))
        y = np.arange(len(items))
        vals = [r["auc"] for r in items]
        cols = [FAMILY_COLORS[r["family"]] for r in items]
        ax.barh(y, [v - 0.5 for v in vals], left=0.5, color=cols, alpha=0.85)
        lo = [r["auc"] - r["ci_lo"] if np.isfinite(r["ci_lo"]) else 0 for r in items]
        hi = [r["ci_hi"] - r["auc"] if np.isfinite(r["ci_hi"]) else 0 for r in items]
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor="black", lw=0.8, capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['cue']} ({r['stat']})" for r in items], fontsize=8)
        ax.axvline(0.5, color="black", lw=0.8)
        ax.set_xlabel("AUC  (> 0.5: higher at the phrase boundary; < 0.5: lower)")
        ax.set_title(title + f"\n(n = {items[-1]['n_first']} vs {items[-1]['n_ref']}; 95% video-bootstrap CI)", fontsize=10)
    import matplotlib.patches as mpatches
    fams = sorted({r["family"] for r in rows}, key=FAMILIES.index)
    fig.legend([mpatches.Patch(color=FAMILY_COLORS[f]) for f in fams], fams, loc="lower center",
               ncol=len(fams), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_probe(rows, path):
    import matplotlib.pyplot as plt
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sel = [r for r in rows if r["task"] == task]
        all_auc = next(r["auc"] for r in sel if r["mode"] == "all")
        fams = [r["features"] for r in sel if r["mode"] == "alone"]
        alone = [r["auc"] for r in sel if r["mode"] == "alone"]
        uniq = [r["unique_contribution"] for r in sel if r["mode"] == "all_without"]
        x = np.arange(len(fams))
        ax.bar(x - 0.2, alone, 0.4, color=[FAMILY_COLORS[f] for f in fams], label="family alone")
        ax.bar(x + 0.2, [0.5 + u for u in uniq], 0.4, color="#444444", alpha=0.6,
               label="0.5 + unique (ALL - ALL without family)")
        ax.axhline(all_auc, color="black", ls="--", lw=0.9, label=f"ALL = {all_auc:.3f}")
        ax.axhline(0.5, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.45, max(1.0, all_auc + 0.02))
        ax.set_title(task, fontsize=10)
        ax.set_ylabel("cross-validated AUC")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()