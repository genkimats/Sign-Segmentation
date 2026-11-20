"""
viz_hamer_skeleton3d.py -- the hand-shape (HaMeR) input as 3D hand skeletons, WITHOUT the MANO model file.

The saved HaMeR features are MANO parameters: for each hand and frame, the rotation of 15 finger joints and of the
whole hand. Applying these rotations along a fixed rest-pose hand skeleton (forward kinematics, as MANO does
internally) gives the 3D position of the wrist, the 15 finger joints and 5 fingertips. The rest-pose skeleton here
is a built-in approximation of MANO's average right hand (joint positions in metres, MANO's own axes), so no model
file, smplx or chumpy is needed. Proportions are average; the finger bending and hand orientation come from the data.

  hamer_skeleton3d.png   one 3D hand skeleton per sign (middle of each sign in a stretch), seen from the camera
                         and slightly from above, with the sign's time below

Data (defaults follow the repo layout):
  HaMeR         the first processed_data/*hamer*/<video>_hamer.pt found, or --hamer-dir
  sign labels   processed_data/BIO_tags/<video>.npy   (sign level; 0 = O, 1 = I, 2 = B)

  python viz_hamer_skeleton3d.py                        # random test video and stretch
  python viz_hamer_skeleton3d.py --video 1209077_B --start 12000 --hand right
  python viz_hamer_skeleton3d.py --elev 20 --azim -60   # another viewing angle
  python viz_hamer_skeleton3d.py --selftest
Output: evaluation_phrase/figures/hamer_skeleton3d.png
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

FPS = 50
MUTED = "#5F6B78"
MIRROR = np.diag([-1.0, 1.0, 1.0])

# Approximate rest pose of MANO's average right hand (metres, MANO axes; fingers point along -x).
# Order = MANO joints: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb.
REST = np.array([
    [0.0957, 0.0064, 0.0062],
    [0.0069, 0.0047, 0.0209], [-0.0242, 0.0031, 0.0229], [-0.0464, 0.0020, 0.0218],
    [0.0127, 0.0047, -0.0019], [-0.0213, 0.0040, -0.0026], [-0.0478, 0.0029, -0.0040],
    [0.0316, 0.0013, -0.0391], [0.0118, -0.0011, -0.0479], [-0.0050, -0.0026, -0.0552],
    [0.0230, 0.0035, -0.0210], [-0.0094, 0.0024, -0.0253], [-0.0346, 0.0015, -0.0291],
    [0.0753, -0.0027, 0.0331], [0.0532, -0.0053, 0.0503], [0.0285, -0.0089, 0.0597],
])
PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
TIP_OF = {3: 16, 6: 17, 9: 18, 12: 19, 15: 20}        # last joint of each finger -> fingertip index
TIP_SCALE = 0.85                                        # tip bone = 0.85 x the previous bone
FINGERS = [("Thumb", [0, 13, 14, 15, 20], "#C0392B"), ("Index", [0, 1, 2, 3, 16], "#E07B39"),
           ("Middle", [0, 4, 5, 6, 17], "#C9A227"), ("Ring", [0, 10, 11, 12, 19], "#2E8B57"),
           ("Pinky", [0, 7, 8, 9, 18], "#2F6DB5")]
PALM = [1, 4, 10, 7]                                    # knuckles, drawn as a line across the palm


# ----------------------------------------------------------------------------- data
def load_any(path):
    if path.endswith(".npy"):
        x = np.load(path, allow_pickle=True)
        return x.item() if x.dtype == object else x
    import torch
    return torch.load(path, weights_only=False, map_location="cpu")


def hamer_dirs(hamer_dir):
    return [hamer_dir] if hamer_dir else sorted(c for c in glob.glob(os.path.join("processed_data", "*hamer*"))
                                                if os.path.isdir(c))


def find_hamer(hamer_dir, video):
    for c in hamer_dirs(hamer_dir):
        for ext in (".pt", ".npy"):
            p = os.path.join(c, f"{video}_hamer{ext}")
            if os.path.exists(p):
                print(f"  HaMeR features from {p}")
                return load_any(p)
    raise SystemExit(f"no {video}_hamer.pt found -- pass --hamer-dir")


def random_video(rng, hamer_dir, sign_dir, split_file, split):
    have = set()
    for c in hamer_dirs(hamer_dir):
        have |= {f.rsplit("_hamer", 1)[0] for f in os.listdir(c) if "_hamer." in f}
    vids = sorted(v for v in have if os.path.exists(os.path.join(sign_dir, f"{v}.npy")))
    if split_file and os.path.exists(split_file):
        keep = {v.replace(".npy", "").replace(".pt", "") for v in json.load(open(split_file)).get(split, [])}
        vids = [v for v in vids if v in keep] or vids
    if not vids:
        raise SystemExit("no video has both HaMeR features and sign labels")
    return str(rng.choice(vids))


def segments(lab):
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


# ----------------------------------------------------------------------------- forward kinematics
def joints_right(hand_pose, global_orient):
    """MANO forward kinematics for a right hand: (15,3,3) + (3,3) -> 21 joint positions (wrist at the origin)."""
    R = [global_orient] + [hand_pose[k] for k in range(15)]
    G = [None] * 16
    pos = np.zeros((21, 3))
    for j in range(16):
        if PARENT[j] < 0:
            G[j] = R[0]
            pos[j] = 0.0
        else:
            p = PARENT[j]
            pos[j] = pos[p] + G[p] @ (REST[j] - REST[p])
            G[j] = G[p] @ R[j]
    for j, tip in TIP_OF.items():                       # fingertips continue the last bone
        pos[tip] = pos[j] + G[j] @ ((REST[j] - REST[PARENT[j]]) * TIP_SCALE)
    return pos


def hand_joints(hp, go, is_right):
    """Left hands were stored un-mirrored (extract_hamer_features.py); mirror back, solve as a right hand, mirror."""
    if is_right:
        return joints_right(hp, go)
    p = joints_right(MIRROR @ hp @ MIRROR, MIRROR @ go @ MIRROR)
    p[:, 0] *= -1
    return p


# ----------------------------------------------------------------------------- drawing
def draw(ax, pos, elev, azim):
    """pos in HaMeR's camera frame (x right, y down, z away). Plotted as (x, z, -y): up on the page is up."""
    P = np.stack([pos[:, 0], pos[:, 2], -pos[:, 1]], 1)
    for _, chain, c in FINGERS:
        ax.plot(P[chain, 0], P[chain, 1], P[chain, 2], color=c, lw=3, solid_capstyle="round")
        ax.scatter(P[chain[1:], 0], P[chain[1:], 1], P[chain[1:], 2], color=c, s=14, depthshade=False)
    ax.plot(P[PALM, 0], P[PALM, 1], P[PALM, 2], color=MUTED, lw=1.5)
    ax.scatter(*P[0], color="#1F2933", s=30, depthshade=False)
    c = P.mean(0)
    r = np.abs(P - c).max() * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.98, 1.0))
        axis.pane.set_edgecolor((0.85, 0.87, 0.9, 1.0))


# ----------------------------------------------------------------------------- figure
def run(video, hamer_dir, sign_dir, out, start=None, seconds=6.0, hand=None, max_hands=6, seed=None,
        split_file="dataset_splits.json", split="test", elev=15.0, azim=-90.0):
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 100000)
    rng = np.random.default_rng(seed)
    if video is None:
        video = random_video(rng, hamer_dir, sign_dir, split_file, split)
    sign = np.load(os.path.join(sign_dir, f"{video}.npy")).astype(int)
    T = len(sign)
    d = find_hamer(hamer_dir, video)
    hp = np.asarray(d["hand_pose"], dtype=np.float64)
    go = np.asarray(d["global_orient"], dtype=np.float64)
    ds = int(d.get("temporal_downsample_factor", 1) or 1)
    idx = np.clip(np.arange(T) // ds, 0, len(hp) - 1)
    hp, go = hp[idx], go[idx]
    detected = ~np.all(hp.reshape(T, 2, -1) == 0, axis=-1)          # (T, 2)

    L = int(seconds * FPS)
    h = int(np.argmax(detected.sum(0))) if hand is None else {"left": 0, "right": 1}[hand]
    if start is None:                                               # random stretch: >= 3 signs, hand mostly seen
        good = [a for a in range(0, max(1, T - L), L // 4)
                if np.sum(sign[a:a + L] == 2) >= 3 and detected[a:a + L, h].mean() >= 0.6]
        start = int(rng.choice(good)) if good else 0
    a, b = start, min(T, start + L)
    signs = [(s, e) for s, e in segments(sign) if s >= a and e < b]
    mids = [m for m in ((s + e) // 2 for s, e in signs) if detected[m, h]][:max_hands]
    if not mids:
        raise SystemExit(f"no sign with a detected {['left', 'right'][h]} hand in frames {a}-{b}; try --hand / --start")

    fig = plt.figure(figsize=(2.6 * len(mids), 3.2))
    for k, m in enumerate(mids):
        ax = fig.add_subplot(1, len(mids), k + 1, projection="3d")
        draw(ax, hand_joints(hp[m, h], go[m, h], h == 1), elev, azim)
        ax.set_title(f"{(m - a) / FPS:.1f} s", fontsize=11, color=MUTED, y=-0.05)
    handles = [plt.Line2D([], [], color=c, lw=3, label=n) for n, _, c in FINGERS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{['Left', 'Right'][h]} hand at the middle of each sign: 3D skeleton from the HaMeR features "
                 f"(average hand proportions)", fontsize=11, color=MUTED, y=1.0)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {video}: frames {a}-{b} ({a / FPS:.1f}-{b / FPS:.1f} s), {len(mids)} signs, {['left', 'right'][h]} hand")
    print(f"  to redraw exactly this: --video {video} --start {a} --hand {['left', 'right'][h]}   (or --seed {seed})")
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- selftest
def _rot(axis, th):
    c, s = np.cos(th), np.sin(th)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    eye = np.tile(np.eye(3), (15, 1, 1))
    flat = joints_right(eye, np.eye(3))
    bone = np.linalg.norm(flat[1:16] - flat[[PARENT[j] for j in range(1, 16)]], axis=1)
    rest_bone = np.linalg.norm(REST[1:] - REST[[PARENT[j] for j in range(1, 16)]], axis=1)
    ok1 = np.allclose(bone, rest_bone)                                  # identity pose = rest pose
    bent = eye.copy()
    bent[[0, 1, 2]] = _rot("z", 0.8)
    pb = joints_right(bent, np.eye(3))
    ok2 = np.allclose(np.linalg.norm(pb[16] - pb[3]), np.linalg.norm(flat[16] - flat[3]))   # rigid bones
    ok3 = np.allclose(hand_joints(eye, np.eye(3), False)[:, 0], -flat[:, 0])                  # left = mirror
    with tempfile.TemporaryDirectory() as d:
        T = 1200
        sign = np.zeros(T, int)
        hp = np.tile(np.eye(3), (T, 2, 15, 1, 1))
        go = np.tile(_rot("y", np.pi / 2) @ _rot("z", np.pi / 2), (T, 2, 1, 1))
        t = 30
        while t < T - 100:
            L = int(rng.integers(30, 60))
            sign[t], sign[t + 1:t + L] = 2, 1
            curl = rng.uniform(0, 1.3, 5)
            for f, js in enumerate([(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]):
                for j in js:
                    hp[t:t + L, :, j] = _rot("z", curl[f] * 0.7)
            t += L + int(rng.integers(5, 20))
        os.makedirs(f"{d}/ham")
        np.save(f"{d}/ham/v_A_hamer.npy", np.array({"hand_pose": hp, "global_orient": go,
                                                    "temporal_downsample_factor": 1}, dtype=object))
        os.makedirs(f"{d}/sign")
        np.save(f"{d}/sign/v_A.npy", sign)
        out = os.environ.get("VIZ_SELFTEST_OUT", d)
        run(None, f"{d}/ham", f"{d}/sign", os.path.join(out, "hamer_skeleton3d.png"), seed=1, hand="right")
        ok4 = os.path.exists(os.path.join(out, "hamer_skeleton3d.png"))
    for name, ok in (("rest pose reproduced", ok1), ("bones keep their length", ok2), ("left hand = mirror", ok3),
                     ("figure written", ok4)):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return ok1 and ok2 and ok3 and ok4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="default: a random video of --split")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--hamer-dir", default=None)
    ap.add_argument("--sign-labels-dir", default=os.path.join("processed_data", "BIO_tags"))
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--hand", choices=["left", "right"], default=None)
    ap.add_argument("--max-hands", type=int, default=6)
    ap.add_argument("--elev", type=float, default=15.0, help="viewing angle above the hand (degrees)")
    ap.add_argument("--azim", type=float, default=-90.0, help="-90 = from the camera")
    ap.add_argument("--out", default=os.path.join("evaluation_phrase", "figures", "hamer_skeleton3d.png"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run(a.video, a.hamer_dir, a.sign_labels_dir, a.out, a.start, a.seconds, a.hand, a.max_hands, a.seed,
        a.split_file, a.split, a.elev, a.azim)


if __name__ == "__main__":
    main()