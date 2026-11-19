"""
viz_keypoints.py -- keypoint-only figures (no video frames) for the slides.

  keypoints_frame.png   slide 5: one frame of the model input -- body, both hands, the face points
  keypoints_strip.png   slide 2: skeletons sampled along a stretch of signing, above a timeline cut into
                        the annotated phrases ("what segmenting into phrases means")
  bio_strip.png         slide 5: the B / I / O label of every frame over a short stretch

Data (defaults follow the repo layout; override with the flags):
  body + hands  processed_data/kinematic_features/<video>.pt   ["mediapipe"]["base"]: (T, 65, 3), shoulder-normalized
                (0-22 MediaPipe Pose upper body, 23-43 left hand, 44-64 right hand; graph.py's layout)
  face          the first of processed_data/*face*normalized*/<video>.npy found, or --face-dir. Must be in the
                same shoulder-normalized frame as the body (face_keypoints_normalized etc.); with 83 points the
                lip / eye / brow contours are drawn as lines, with any other count (e.g. the compact 31) as dots.
  labels        processed_data/BIO_tags_phrase_v2/<video>.npy  (0 = O, 1 = I, 2 = B)

  python viz_keypoints.py --video 1209077_B                    # picks a stretch with 3 phrase starts
  python viz_keypoints.py --video 1209077_B --start 12000 --seconds 8 --frames 7
  python viz_keypoints.py --selftest
Output: evaluation_phrase/figures/
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

FPS = 50
BODY_C, LEFT_C, RIGHT_C, FACE_C, INK, MUTED = "#6B7785", "#2F6DB5", "#E07B39", "#2E8B57", "#1F2933", "#5F6B78"
PHRASE_COLORS = ["#4C7FBF", "#E39B54", "#59A26B", "#B56CB0", "#C9A227", "#4AA5A8"]

POSE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15),
              (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
# graph.py's 83-point face: raw MediaPipe indices per contour (lines are drawn only for this layout)
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
         78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
_LEYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_REYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
_LBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_FACE83 = sorted(set(_LIPS + _LEYE + _REYE + _LBROW + _RBROW))


# ----------------------------------------------------------------------------- loading
def load_any(path):
    if path.endswith(".npy"):
        x = np.load(path, allow_pickle=True)
        return x.item() if x.dtype == object else x
    import torch
    return torch.load(path, weights_only=False, map_location="cpu")


def load_body(kin_dir, video):
    d = load_any(os.path.join(kin_dir, f"{video}.pt") if os.path.exists(os.path.join(kin_dir, f"{video}.pt"))
                 else os.path.join(kin_dir, f"{video}.npy"))
    if isinstance(d, dict):
        d = d.get("mediapipe", d)["base"]
    return np.asarray(d, dtype=np.float64)[:, :, :2]


def find_face(face_dir, video):
    if face_dir:
        cands = [face_dir]
    else:
        cands = sorted(glob.glob(os.path.join("processed_data", "*face*normalized*")))
        cands = [c for c in cands if os.path.isdir(c)]
    for c in cands:
        p = os.path.join(c, f"{video}.npy")
        if os.path.exists(p):
            print(f"  face points from {p}")
            return np.asarray(np.load(p), dtype=np.float64)[:, :, :2]
    print("  no shoulder-normalized face file found -- drawing body and hands only (pass --face-dir)")
    return None


def valid(pts):
    return pts is not None and np.all(np.isfinite(pts)) and not np.allclose(pts, 0)


# ----------------------------------------------------------------------------- drawing
def draw_skeleton(ax, body, face, lw=2.0, ms=10, dx=0.0):
    """body: (65, 2), face: (F, 2) or None; x right, y down (image convention)."""
    def seg(a, b, c, w):
        ax.plot([a[0] + dx, b[0] + dx], [a[1], b[1]], color=c, lw=w, solid_capstyle="round")
    pose = body[:23]
    if valid(pose[11:17]):
        for i, j in POSE_EDGES:
            if i >= 11 and j >= 11 and max(i, j) <= 16:   # shoulders and arms only (the hands come from the hand model)
                seg(pose[i], pose[j], BODY_C, lw * 1.4)
    for off, c, wrist in ((23, LEFT_C, 15), (44, RIGHT_C, 16)):
        h = body[off:off + 21]
        if valid(h):
            for i, j in HAND_EDGES:
                seg(h[i], h[j], c, lw * 0.8)
    if face is not None and valid(face):
        if len(face) == len(_FACE83):
            pos = {raw: k for k, raw in enumerate(_FACE83)}
            for chain, closed in ((_LIPS, True), (_LEYE, True), (_REYE, True), (_LBROW, False), (_RBROW, False)):
                idx = [pos[r] for r in chain] + ([pos[chain[0]]] if closed else [])
                ax.plot(face[idx, 0] + dx, face[idx, 1], color=FACE_C, lw=lw * 0.6)
        else:
            ax.scatter(face[:, 0] + dx, face[:, 1], s=ms, color=FACE_C, zorder=3, linewidths=0)
    elif valid(pose[:11]):                    # no face file: the pose's own face points as a stand-in
        ax.scatter(pose[:11, 0] + dx, pose[:11, 1], s=ms, color=MUTED, zorder=3, linewidths=0)


def finish(ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def phrases_from_labels(lab):
    """[(start, end)] of phrases: B starts one; it runs over the following I frames."""
    out, cur = [], None
    for t, y in enumerate(lab):
        if y == 2:
            if cur is not None:
                out.append((cur, t - 1))
            cur = t
        elif y == 0 and cur is not None:
            out.append((cur, t - 1))
            cur = None
        elif y == 1 and cur is None:
            cur = t
    if cur is not None:
        out.append((cur, len(lab) - 1))
    return out


def pick_stretch(lab, L):
    """First stretch of L frames with 3 phrase starts, at least one O frame, and starts away from the edges."""
    starts = np.where(lab == 2)[0]
    for s in starts:
        a = max(0, s - L // 8)
        b = a + L
        inside = starts[(starts >= a + L // 10) & (starts < b - L // 10)]
        if b <= len(lab) and len(inside) >= 3 and np.any(lab[a:b] == 0):
            return a
    return max(0, (starts[0] if len(starts) else 0) - L // 8)


# ----------------------------------------------------------------------------- figures
def fig_frame(body, face, t, out):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_skeleton(ax, body[t], face[t] if face is not None else None, lw=2.6, ms=18)
    finish(ax)
    handles = [Patch(color=BODY_C, label="Body (MediaPipe)"), Patch(color=LEFT_C, label="Left hand"),
               Patch(color=RIGHT_C, label="Right hand")]
    if face is not None:
        handles.append(Patch(color=FACE_C, label=f"Face ({face.shape[1]} points)"))
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=2, fontsize=11)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_strip(body, face, lab, a, L, n, out):
    ts = np.linspace(a + L * 0.05, a + L * 0.95, n).astype(int)
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_axes([0.02, 0.32, 0.96, 0.66])
    span = 2.6                                  # horizontal spacing in shoulder widths
    for k, t in enumerate(ts):
        draw_skeleton(ax, body[t], face[t] if face is not None else None, lw=1.8, ms=5, dx=k * span)
    finish(ax)
    ax.set_xlim(-span / 2, (n - 0.5) * span)
    tl = fig.add_axes([0.02, 0.1, 0.96, 0.16])
    seg = lab[a:a + L]
    tl.add_patch(Rectangle((0, 0), L, 1, color="#D5DAE0"))
    for k, (s, e) in enumerate(phrases_from_labels(seg)):
        c = PHRASE_COLORS[k % len(PHRASE_COLORS)]
        tl.add_patch(Rectangle((s, 0), e - s + 1, 1, color=c))
        if e - s > L * 0.06:
            tl.text((s + e) / 2, 0.5, f"Phrase {k + 1}", ha="center", va="center", color="white", fontsize=11,
                    fontweight="bold")
    for t in ts:
        tl.plot([t - a, t - a], [1.0, 1.35], color=MUTED, lw=1, clip_on=False)
    tl.set_xlim(0, L)
    tl.set_ylim(0, 1)
    tl.set_yticks([])
    for sp in ("top", "right", "left"):
        tl.spines[sp].set_visible(False)
    secs = np.arange(0, L / FPS + 1e-9, 1.0)
    tl.set_xticks(secs * FPS)
    tl.set_xticklabels([f"{s:g} s" for s in secs], fontsize=9, color=MUTED)
    fig.text(0.5, 0.0, "Grey = no phrase (outside); coloured = annotated phrases", ha="center", fontsize=10,
             color=MUTED)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_bio(lab, a, L, out):
    seg = lab[a:a + L]
    col = {0: "#D5DAE0", 1: "#4C7FBF", 2: "#C0392B"}
    fig, ax = plt.subplots(figsize=(10, 1.6))
    for t, y in enumerate(seg):
        if y != 2:
            ax.add_patch(Rectangle((t, 0), 1, 1, color=col[int(y)], linewidth=0))
    for t in np.where(seg == 2)[0]:
        ax.add_patch(Rectangle((t - L * 0.002, -0.15), L * 0.004 + 1, 1.3, color=col[2], linewidth=0))
    ax.set_xlim(0, L)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    secs = np.arange(0, L / FPS + 1e-9, 1.0)
    ax.set_xticks(secs * FPS)
    ax.set_xticklabels([f"{s:g} s" for s in secs], fontsize=9, color=MUTED)
    ax.legend(handles=[Patch(color=col[2], label="B: beginning of a phrase (1 frame)"),
                       Patch(color=col[1], label="I: inside a phrase"), Patch(color=col[0], label="O: outside")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=10)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(video, kin_dir, face_dir, labels_dir, out_dir, start=None, seconds=8.0, frames=7, frame=None):
    os.makedirs(out_dir, exist_ok=True)
    body = load_body(kin_dir, video)
    face = find_face(face_dir, video)
    lab = np.load(os.path.join(labels_dir, f"{video}.npy")).astype(int)
    T = min(len(body), len(lab), len(face) if face is not None else 10 ** 9)
    body, lab = body[:T], lab[:T]
    face = face[:T] if face is not None else None
    L = int(seconds * FPS)
    a = pick_stretch(lab, L) if start is None else start
    a = min(a, max(0, T - L))
    if frame is None:                           # a frame inside a phrase with both hands and the face visible
        cand = [t for t in range(a, a + L) if lab[t] == 1 and valid(body[t, 23:44]) and valid(body[t, 44:65])
                and (face is None or valid(face[t]))]
        frame = cand[len(cand) // 2] if cand else a + L // 2
    fig_frame(body, face, frame, os.path.join(out_dir, "keypoints_frame.png"))
    fig_strip(body, face, lab, a, L, frames, os.path.join(out_dir, "keypoints_strip.png"))
    fig_bio(lab, a, min(L, int(4 * FPS)), os.path.join(out_dir, "bio_strip.png"))
    print(f"  {video}: stretch frames {a}-{a + L} ({a / FPS:.1f}-{(a + L) / FPS:.1f} s), single frame {frame}")
    print(f"wrote {out_dir}/keypoints_frame.png, keypoints_strip.png, bio_strip.png")


# ----------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        T = 2000
        base = np.zeros((T, 65, 3))
        pose = np.array([[0, -1.6], [-.1, -1.7], [-.15, -1.7], [-.2, -1.7], [.1, -1.7], [.15, -1.7], [.2, -1.7],
                         [-.3, -1.6], [.3, -1.6], [-.05, -1.4], [.05, -1.4], [-.5, 0], [.5, 0], [-.7, .8], [.7, .8],
                         [-.5, 1.2], [.5, 1.2], [-.55, 1.35], [.55, 1.35], [-.45, 1.38], [.45, 1.38], [-.5, 1.3],
                         [.5, 1.3]])
        hand = np.array([[0, 0]] + [[np.cos(a) * r, -np.sin(a) * r] for a in np.linspace(0.3, 2.8, 5)
                                    for r in (0.08, 0.14, 0.19, 0.23)])
        for t in range(T):
            ph = np.sin(t / 40)
            base[t, :23, :2] = pose
            base[t, 23:44, :2] = hand + pose[15] + [0, -0.3 * ph]
            base[t, 44:65, :2] = hand * [-1, 1] + pose[16] + [0, 0.3 * ph]
            base[t, 15, 1] += -0.3 * ph
            base[t, 16, 1] += 0.3 * ph
        os.makedirs(f"{d}/kin")
        np.save(f"{d}/kin/v_A.npy", np.array({"mediapipe": {"base": base}}, dtype=object))
        os.makedirs(f"{d}/face_norm")
        ang = np.linspace(0, 2 * np.pi, 31, endpoint=False)
        face = np.stack([np.stack([0.18 * np.cos(ang), -1.55 + 0.25 * np.sin(ang)], 1)] * T)
        np.save(f"{d}/face_norm/v_A.npy", np.concatenate([face, np.zeros((T, 31, 1))], -1))
        lab = np.zeros(T, int)
        t = 50
        while t < T - 200:
            L = int(rng.integers(80, 200))
            lab[t], lab[t + 1:t + L] = 2, 1
            t += L + int(rng.integers(0, 40))
        os.makedirs(f"{d}/lab")
        np.save(f"{d}/lab/v_A.npy", lab)
        out = os.environ.get("VIZ_SELFTEST_OUT", f"{d}/out")
        run("v_A", f"{d}/kin", f"{d}/face_norm", f"{d}/lab", out)
        ok = all(os.path.exists(os.path.join(out, f)) for f in ("keypoints_frame.png", "keypoints_strip.png",
                                                                "bio_strip.png"))
        print(f"  [{'ok' if ok else 'FAIL'}] three figures written")
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--kin-dir", default=os.path.join("processed_data", "kinematic_features"))
    ap.add_argument("--face-dir", default=None)
    ap.add_argument("--labels-dir", default=os.path.join("processed_data", "BIO_tags_phrase_v2"))
    ap.add_argument("--start", type=int, default=None, help="first frame of the stretch (default: automatic)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--frames", type=int, default=7, help="skeletons in the strip")
    ap.add_argument("--frame", type=int, default=None, help="frame for keypoints_frame.png (default: automatic)")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "figures"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.video:
        ap.error("--video is required (e.g. a test video such as 1209077_B)")
    run(a.video, a.kin_dir, a.face_dir, a.labels_dir, a.out_dir, a.start, a.seconds, a.frames, a.frame)


if __name__ == "__main__":
    main()