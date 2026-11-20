"""
viz_keypoints3d.py -- the body, hand and face keypoints in 3D, seen from an angle so the depth is visible.

  keypoints3d.png   the same frame from three viewpoints: from the camera, from an angle, and from the side
  keypoints3d.gif   (with --gif) the skeleton turning around, for a slide that plays animations

Depth comes from MediaPipe's own z estimate, which is coarser than x / y. MediaPipe gives each hand's z relative to
its own wrist and the face's z relative to the face, so the script moves each hand to the depth of the body's
wrist and the face to the depth of the body's nose; the body keeps its own z.

Data (defaults follow the repo layout; override with the flags):
  body + hands  processed_data/kinematic_features/<video>.pt   ["mediapipe"]["base"]: (T, 65, 3), shoulder-normalized
                (0-22 MediaPipe Pose upper body, 23-43 left hand, 44-64 right hand)
  face          the first processed_data/*face*normalized*/<video>.npy found, or --face-dir (same coordinates)
  labels        processed_data/BIO_tags_phrase_v2/<video>.npy   (used to pick a frame inside a phrase)

  python viz_keypoints3d.py                              # random test video, random frame inside a phrase
  python viz_keypoints3d.py --video 1209077_B --frame 12512 --gif
  python viz_keypoints3d.py --selftest
Output: evaluation_phrase/figures/
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BODY_C, LEFT_C, RIGHT_C, FACE_C, MUTED = "#6B7785", "#2F6DB5", "#E07B39", "#2E8B57", "#5F6B78"
ARM_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
VIEWS = [("From the camera", 0, -90), ("From an angle", 18, -55), ("From the side", 0, 0)]


# ----------------------------------------------------------------------------- data
def load_any(path):
    if path.endswith(".npy"):
        x = np.load(path, allow_pickle=True)
        return x.item() if x.dtype == object else x
    import torch
    return torch.load(path, weights_only=False, map_location="cpu")


def load_body(kin_dir, video):
    p = os.path.join(kin_dir, f"{video}.pt")
    d = load_any(p if os.path.exists(p) else os.path.join(kin_dir, f"{video}.npy"))
    if isinstance(d, dict):
        d = d.get("mediapipe", d)["base"]
    return np.asarray(d, dtype=np.float64)[:, :, :3]


def find_face(face_dir, video, prefer_points=31):
    """Shoulder-normalized face points. Without --face-dir, every processed_data/*face*normalized* folder is
    checked and the one with the model's compact 31-point face is preferred."""
    cands = [face_dir] if face_dir else [c for c in sorted(glob.glob(os.path.join("processed_data",
                                                                                  "*face*normalized*")))
                                         if os.path.isdir(c)]
    found = []
    for c in cands:
        p = os.path.join(c, f"{video}.npy")
        if os.path.exists(p):
            found.append((p, np.load(p, mmap_mode="r").shape))
    if not found:
        print("  no shoulder-normalized face file found -- drawing body and hands only")
        return None
    best = next((f for f in found if f[1][1] == prefer_points), found[0])
    p, shape = best
    print(f"  face points from {p}  ({shape[1]} points, {shape[2]} coordinates)"
          + ("" if len(found) == 1 else f"; other face folders: {', '.join(os.path.dirname(q) for q, _ in found if q != p)}"))
    return np.asarray(np.load(p), dtype=np.float64)


def valid(p):
    return p is not None and np.all(np.isfinite(p)) and not np.allclose(p, 0)


def random_video(rng, labels_dir, kin_dir, split_file, split):
    vids = sorted(f[:-4] for f in os.listdir(labels_dir) if f.endswith(".npy"))
    vids = [v for v in vids if os.path.exists(os.path.join(kin_dir, f"{v}.pt"))
            or os.path.exists(os.path.join(kin_dir, f"{v}.npy"))]
    if split_file and os.path.exists(split_file):
        keep = {v.replace(".npy", "").replace(".pt", "") for v in json.load(open(split_file)).get(split, [])}
        vids = [v for v in vids if v in keep] or vids
    if not vids:
        raise SystemExit(f"no video has both labels in {labels_dir} and keypoints in {kin_dir}")
    return str(rng.choice(vids))


# ----------------------------------------------------------------------------- one frame in 3D
def frame_points(body, face):
    """Body (65, 3) and face (F, 3) -> dict of 3D point sets with hand and face depth aligned to the body."""
    b = body.copy()
    for off, wrist in ((23, 15), (44, 16)):
        h = b[off:off + 21]
        if valid(h) and valid(b[wrist]):
            h[:, 2] += b[wrist, 2] - h[0, 2]          # hand wrist at the body's wrist depth
    f = None
    if face is not None and valid(face[:, :2]):
        f = face.copy()
        if f.shape[1] < 3:                            # x, y only: place the face flat at the nose's depth
            f = np.concatenate([f, np.full((len(f), 1), b[0, 2])], 1)
        else:
            f[:, 2] += b[0, 2] - f[:, 2].mean()       # face centred on the nose depth
    return b, f


def to_plot(p):
    """(x right, y down, z toward the camera is negative) -> plot axes (x, depth, up)."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def draw(ax, b, f, elev, azim, lim):
    B = to_plot(b)
    for i, j in ARM_EDGES:
        ax.plot(*zip(B[i], B[j]), color=BODY_C, lw=3.2, solid_capstyle="round")
    for off, c, wrist in ((23, LEFT_C, 15), (44, RIGHT_C, 16)):
        if valid(b[off:off + 21]):
            ax.plot(*zip(B[wrist], B[off]), color=BODY_C, lw=2)          # arm to hand
            for i, j in HAND_EDGES:
                ax.plot(*zip(B[off + i], B[off + j]), color=c, lw=1.8)
    if f is not None:
        F = to_plot(f)
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], color=FACE_C, s=6, depthshade=False)
    else:
        ax.scatter(B[:11, 0], B[:11, 1], B[:11, 2], color=MUTED, s=6, depthshade=False)
    c, r = lim
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x", color=MUTED, labelpad=-12)
    ax.set_ylabel("depth", color=MUTED, labelpad=-12)
    ax.set_zlabel("y", color=MUTED, labelpad=-12)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.98, 1.0))
        axis.pane.set_edgecolor((0.85, 0.87, 0.9, 1.0))


def limits(b, f):
    pts = [to_plot(b[11:17])] + [to_plot(b[o:o + 21]) for o in (23, 44) if valid(b[o:o + 21])]
    if f is not None:
        pts.append(to_plot(f))
    P = np.concatenate(pts)
    c = (P.max(0) + P.min(0)) / 2
    return c, np.abs(P - c).max() * 1.05


# ----------------------------------------------------------------------------- figures
def run(video, kin_dir, face_dir, labels_dir, out_dir, frame=None, gif=False, seed=None,
        split_file="dataset_splits.json", split="test"):
    os.makedirs(out_dir, exist_ok=True)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 100000)
    rng = np.random.default_rng(seed)
    if video is None:
        video = random_video(rng, labels_dir, kin_dir, split_file, split)
    body = load_body(kin_dir, video)
    face = find_face(face_dir, video)
    lab = np.load(os.path.join(labels_dir, f"{video}.npy")).astype(int)
    T = min(len(body), len(lab), len(face) if face is not None else 10 ** 9)
    if frame is None:                                   # a frame inside a phrase with both hands (and face) seen
        cand = [t for t in range(T) if lab[t] == 1 and valid(body[t, 23:44]) and valid(body[t, 44:65])
                and (face is None or valid(face[t, :, :2]))]
        if not cand:
            raise SystemExit(f"{video}: no frame with both hands visible inside a phrase; pass --frame")
        frame = int(rng.choice(cand))
    b, f = frame_points(body[frame], face[frame] if face is not None else None)
    lim = limits(b, f)

    fig = plt.figure(figsize=(12, 4.4))
    for k, (title, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        draw(ax, b, f, elev, azim, lim)
        ax.set_title(title, fontsize=12, color=MUTED, y=0.98)
    handles = [plt.Line2D([], [], color=BODY_C, lw=3, label="Body"), plt.Line2D([], [], color=LEFT_C, lw=3,
               label="Left hand"), plt.Line2D([], [], color=RIGHT_C, lw=3, label="Right hand")]
    if f is not None:
        handles.append(plt.Line2D([], [], color=FACE_C, marker="o", lw=0, label=f"Face ({len(f)} points)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    flat = face is not None and face.shape[-1] < 3
    fig.text(0.5, -0.07, "Depth is MediaPipe's own estimate (coarser than x and y); hands aligned to the body's "
             "wrists, face to the nose" + (" (face file has no depth: drawn flat)" if flat else ""),
             ha="center", fontsize=9, color=MUTED)
    p = os.path.join(out_dir, "keypoints3d.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {video}: frame {frame} ({frame / 50:.1f} s)")
    print(f"  to redraw exactly this: --video {video} --frame {frame}   (or --seed {seed})")
    print(f"wrote {p}")

    if gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig = plt.figure(figsize=(4.6, 4.6))
        ax = fig.add_subplot(111, projection="3d")

        def update(i):
            ax.cla()
            draw(ax, b, f, 12, -90 + i * 4, lim)
            return []
        anim = FuncAnimation(fig, update, frames=90, blit=False)
        p = os.path.join(out_dir, "keypoints3d.gif")
        anim.save(p, writer=PillowWriter(fps=20), dpi=110, savefig_kwargs={"facecolor": "white"})
        plt.close(fig)
        print(f"wrote {p}")


# ----------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        T = 300
        base = np.zeros((T, 65, 3))
        pose = np.zeros((23, 3))
        pose[0] = [0, -1.6, -0.3]
        pose[11], pose[12] = [-0.5, 0, 0], [0.5, 0, 0]
        pose[13], pose[14] = [-0.65, 0.8, -0.2], [0.6, 0.6, -0.5]
        pose[15], pose[16] = [-0.4, 1.1, -0.4], [0.3, 0.1, -0.9]
        hand = np.array([[0, 0, 0]] + [[np.cos(a) * r, -np.sin(a) * r, -0.05 * r] for a in np.linspace(0.3, 2.8, 5)
                                       for r in (0.08, 0.14, 0.19, 0.23)])
        for t in range(T):
            base[t, :23] = pose
            base[t, 23:44] = hand + [5, 5, 3]            # hand z relative to its wrist, offset on purpose
            base[t, 44:65] = hand * [-1, 1, 1] + [5, 5, 3]
            base[t, 23:44, :2] += pose[15, :2] - 5
            base[t, 44:65, :2] += pose[16, :2] - 5
        os.makedirs(f"{d}/kin")
        np.save(f"{d}/kin/v_A.npy", np.array({"mediapipe": {"base": base}}, dtype=object))
        ang = np.linspace(0, 2 * np.pi, 31, endpoint=False)
        face = np.stack([np.stack([0.18 * np.cos(ang), -1.55 + 0.25 * np.sin(ang), 0.05 * np.sin(2 * ang)], 1)] * T)
        os.makedirs(f"{d}/face")
        np.save(f"{d}/face/v_A.npy", face)
        lab = np.ones(T, int)
        lab[0] = 2
        os.makedirs(f"{d}/lab")
        np.save(f"{d}/lab/v_A.npy", lab)
        b, f = frame_points(base[10], face[10])
        ok1 = abs(b[23, 2] - pose[15, 2]) < 1e-9 and abs(b[44, 2] - pose[16, 2]) < 1e-9 and \
            abs(f[:, 2].mean() - pose[0, 2]) < 1e-9
        out = os.environ.get("VIZ_SELFTEST_OUT", f"{d}/out")
        run(None, f"{d}/kin", f"{d}/face", f"{d}/lab", out, gif=True, seed=1)
        ok2 = os.path.exists(os.path.join(out, "keypoints3d.png")) and os.path.exists(os.path.join(out, "keypoints3d.gif"))
    print(f"  [{'ok' if ok1 else 'FAIL'}] hand and face depth aligned to the body")
    print(f"  [{'ok' if ok2 else 'FAIL'}] figure and GIF written")
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="default: a random video of --split")
    ap.add_argument("--frame", type=int, default=None, help="default: a random frame inside a phrase")
    ap.add_argument("--gif", action="store_true", help="also write a turning animation")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--kin-dir", default=os.path.join("processed_data", "kinematic_features"))
    ap.add_argument("--face-dir", default=None)
    ap.add_argument("--labels-dir", default=os.path.join("processed_data", "BIO_tags_phrase_v2"))
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "figures"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run(a.video, a.kin_dir, a.face_dir, a.labels_dir, a.out_dir, a.frame, a.gif, a.seed, a.split_file, a.split)


if __name__ == "__main__":
    main()