"""
viz_hamer3d.py -- the hand-shape (HaMeR) input as real 3D hand meshes, no video frames.

The saved HaMeR features are MANO parameters (15 finger-joint rotations + the hand rotation, per hand and frame).
Feeding them through the MANO hand model gives the 3D hand surface (778 vertices) that HaMeR itself reconstructs,
here with the average hand shape (the shape parameters were not saved, so every hand has the same proportions).

  hamer_hands3d.png   one 3D hand per sign, at the middle of each sign in a stretch, with the sign's time below

Run it in the environment where HaMeR runs: it needs torch, smplx and the MANO model file MANO_RIGHT.pkl, which
HaMeR's own setup downloaded (usually hamer/_DATA/data/mano/MANO_RIGHT.pkl). The MANO model has its own licence
(mano.is.tue.mpg.de): rendered images are fine for a talk, the model file itself must not be shared.

Data (defaults follow the repo layout):
  HaMeR         the first processed_data/*hamer*/<video>_hamer.pt found, or --hamer-dir
  sign labels   processed_data/BIO_tags/<video>.npy   (sign level; 0 = O, 1 = I, 2 = B)

  python viz_hamer3d.py --mano-dir ~/hamer/_DATA/data/mano            # random test video and stretch
  python viz_hamer3d.py --mano-dir ~/hamer/_DATA/data/mano --video 1209077_B --start 12000 --hand right
  python viz_hamer3d.py --selftest                                     # rendering check without MANO
Output: evaluation_phrase/figures/hamer_hands3d.png
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
from matplotlib.collections import PolyCollection

FPS = 50
MUTED = "#5F6B78"
SKIN = np.array([0.93, 0.78, 0.66])
MIRROR = np.diag([-1.0, 1.0, 1.0])


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


# ----------------------------------------------------------------------------- MANO
class ManoHand:
    """Right-hand MANO layer taking rotation matrices, as HaMeR calls it (pose2rot=False, mean shape)."""

    def __init__(self, mano_dir):
        import torch
        self.torch = torch
        path = mano_dir
        if os.path.isdir(mano_dir):
            cands = glob.glob(os.path.join(mano_dir, "**", "MANO_RIGHT.pkl"), recursive=True)
            if not cands:
                raise SystemExit(f"MANO_RIGHT.pkl not found under {mano_dir}")
            path = os.path.dirname(cands[0])
        try:
            from hamer.models.mano_wrapper import MANO      # HaMeR's own wrapper, if importable
            self.layer = MANO(model_path=path, gender="neutral", num_hand_joints=15, create_body_pose=False,
                              use_pca=False)
        except Exception:
            import smplx
            self.layer = smplx.MANOLayer(model_path=path, is_rhand=True, use_pca=False)
        self.faces = np.asarray(self.layer.faces, dtype=np.int64)

    def vertices(self, hand_pose, global_orient):
        """hand_pose (15, 3, 3), global_orient (3, 3) of a right hand -> (778, 3) in HaMeR's camera frame."""
        t = self.torch
        with t.no_grad():
            out = self.layer(global_orient=t.tensor(global_orient, dtype=t.float32).reshape(1, 1, 3, 3),
                             hand_pose=t.tensor(hand_pose, dtype=t.float32).reshape(1, 15, 3, 3),
                             betas=t.zeros(1, 10), pose2rot=False)
        return out.vertices[0].numpy()


def hand_vertices(mano, hp, go, is_right):
    """Left hands were stored un-mirrored (extract_hamer_features.py); HaMeR models them as mirrored right hands,
    so mirror the rotations back, run the right-hand model, and mirror the surface (as HaMeR's renderer does)."""
    if is_right:
        return mano.vertices(hp, go)
    v = mano.vertices(MIRROR @ hp @ MIRROR, MIRROR @ go @ MIRROR)
    v[:, 0] *= -1
    return v


# ----------------------------------------------------------------------------- rendering
def render(ax, verts, faces, color=SKIN, flip_winding=False):
    """Orthographic view along the camera axis (x right, y down, z away), painter's algorithm, Lambert shading."""
    v = verts - verts.mean(0)
    tri = v[faces]                                           # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if flip_winding:
        n = -n
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.3, -0.4, -1.0])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(np.abs(n @ light), 0, 1)
    order = np.argsort(-tri[:, :, 2].mean(1))                # far faces first
    polys = tri[order][:, :, :2] * np.array([1, -1])          # y down -> up on the page
    cols = np.clip(shade[order, None] * color[None], 0, 1)
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.2))
    r = np.abs(v[:, :2]).max() * 1.08
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- figure
def run(mano, video, hamer_dir, sign_dir, out, start=None, seconds=6.0, hand=None, max_hands=6, seed=None,
        split_file="dataset_splits.json", split="test"):
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
    if hand is None:
        h = int(np.argmax(detected.sum(0)))                         # the hand detected more often in the video
    else:
        h = {"left": 0, "right": 1}[hand]
    if start is None:                                               # random stretch: >= 3 signs, hand mostly seen
        good = [a for a in range(0, max(1, T - L), L // 4)
                if np.sum(sign[a:a + L] == 2) >= 3 and detected[a:a + L, h].mean() >= 0.6]
        start = int(rng.choice(good)) if good else 0
    a, b = start, min(T, start + L)
    signs = [(s, e) for s, e in segments(sign) if s >= a and e < b]
    mids = [m for m in ((s + e) // 2 for s, e in signs) if detected[m, h]][:max_hands]
    if not mids:
        raise SystemExit(f"no sign with a detected {['left', 'right'][h]} hand in frames {a}-{b}; try --hand / --start")

    fig, axes = plt.subplots(1, len(mids), figsize=(2.3 * len(mids), 2.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, mids):
        render(ax, hand_vertices(mano, hp[m, h], go[m, h], h == 1), mano.faces)
        ax.set_title(f"{(m - a) / FPS:.1f} s", fontsize=11, color=MUTED, y=-0.12)
    fig.suptitle(f"{['Left', 'Right'][h]} hand at the middle of each sign (3D hand from the HaMeR features, "
                 f"average hand proportions)", fontsize=11, color=MUTED, y=1.02)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {video}: frames {a}-{b} ({a / FPS:.1f}-{b / FPS:.1f} s), {len(mids)} signs, {['left', 'right'][h]} hand")
    print(f"  to redraw exactly this: --video {video} --start {a} --hand {['left', 'right'][h]}   (or --seed {seed})")
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- selftest (no MANO needed)
class _FakeMano:
    """Stand-in for the MANO layer: an ellipsoid 'palm' rotated by global_orient, to check data flow and drawing."""

    def __init__(self):
        u, w = np.meshgrid(np.linspace(0, np.pi, 18), np.linspace(0, 2 * np.pi, 30), indexing="ij")
        self.base = np.stack([0.04 * np.sin(u) * np.cos(w), 0.08 * np.sin(u) * np.sin(w), 0.02 * np.cos(u)], -1)
        self.base = self.base.reshape(-1, 3)
        nu, nw = u.shape
        f = []
        for i in range(nu - 1):
            for j in range(nw - 1):
                p = i * nw + j
                f += [[p, p + 1, p + nw], [p + 1, p + nw + 1, p + nw]]
        self.faces = np.array(f)

    def vertices(self, hand_pose, global_orient):
        bend = np.arccos(np.clip((np.trace(hand_pose, axis1=-2, axis2=-1).mean() - 1) / 2, -1, 1))
        v = self.base * np.array([1, 1 - 0.4 * bend, 1])
        return v @ global_orient.T


def selftest():
    import tempfile
    rng = np.random.default_rng(0)

    def rot(axis, th):
        c, s = np.cos(th), np.sin(th)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]

    with tempfile.TemporaryDirectory() as d:
        T = 1200
        sign = np.zeros(T, int)
        hp = np.zeros((T, 2, 15, 3, 3))
        go = np.zeros((T, 2, 3, 3))
        t = 30
        while t < T - 100:
            L = int(rng.integers(30, 60))
            sign[t], sign[t + 1:t + L] = 2, 1
            th, tz = rng.uniform(0, 1.2), rng.uniform(-0.8, 0.8)
            hp[t:t + L, :] = rot("x", th)
            go[t:t + L, :] = rot("z", tz)
            t += L + int(rng.integers(5, 20))
        os.makedirs(f"{d}/ham")
        np.save(f"{d}/ham/v_A_hamer.npy", np.array({"hand_pose": hp, "global_orient": go,
                                                    "temporal_downsample_factor": 1}, dtype=object))
        os.makedirs(f"{d}/sign")
        np.save(f"{d}/sign/v_A.npy", sign)
        fm = _FakeMano()
        v_r = hand_vertices(fm, hp[100, 1], go[100, 1], True)
        v_l = hand_vertices(fm, hp[100, 0], go[100, 0], False)
        ok1 = v_r.shape == v_l.shape and np.isfinite(v_l).all()
        out = os.environ.get("VIZ_SELFTEST_OUT", d)
        run(fm, None, f"{d}/ham", f"{d}/sign", os.path.join(out, "hamer_hands3d.png"), seed=1)
        ok2 = os.path.exists(os.path.join(out, "hamer_hands3d.png"))
    print(f"  [{'ok' if ok1 else 'FAIL'}] left / right hand vertices")
    print(f"  [{'ok' if ok2 else 'FAIL'}] figure written (stand-in mesh: real hands need --mano-dir)")
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mano-dir", default=None, help="folder containing MANO_RIGHT.pkl (HaMeR's _DATA/data/mano)")
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
    ap.add_argument("--out", default=os.path.join("evaluation_phrase", "figures", "hamer_hands3d.png"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    mano_dir = a.mano_dir
    if mano_dir is None:
        for c in ("_DATA/data/mano", "hamer/_DATA/data/mano", os.path.expanduser("~/hamer/_DATA/data/mano")):
            if os.path.exists(c):
                mano_dir = c
                break
    if mano_dir is None:
        ap.error("pass --mano-dir (the folder with MANO_RIGHT.pkl from HaMeR's setup)")
    run(ManoHand(mano_dir), a.video, a.hamer_dir, a.sign_labels_dir, a.out, a.start, a.seconds, a.hand,
        a.max_hands, a.seed, a.split_file, a.split)


if __name__ == "__main__":
    main()