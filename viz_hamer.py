"""
viz_hamer.py -- what the hand-shape (HaMeR) input looks like, without video frames.

HaMeR gives, for each hand and frame, the rotation of the 15 finger joints and of the whole hand (MANO
parameters): 2 x (15 + 1) rotation matrices = 288 numbers per frame, the model's hand-shape input. Raw rotation
matrices mean little to an audience, so the figure shows them three ways:

  (top)     schematic hands: how bent each finger is, at the middle of each sign in the stretch
  (middle)  finger bend over time (degrees) for one hand, with the annotated signs shaded and phrase starts
  (bottom)  the raw 288 numbers per frame as a heat map -- what the model actually receives

"Bend" of a finger = the summed rotation angle of its 3 joints, angle(R) = arccos((trace R - 1) / 2). It mixes
flexion with small sideways and twisting rotations, so the hands at the top are schematic (finger bend only, no
3D mesh: the MANO model files are not needed).

Data (defaults follow the repo layout; override with the flags):
  HaMeR         the first processed_data/*hamer*/<video>_hamer.pt found, or --hamer-dir
                ["hand_pose"] (T', 2, 15, 3, 3), ["global_orient"] (T', 2, 3, 3), index 0 = left, 1 = right;
                ["temporal_downsample_factor"]; a hand that was not detected is all zeros
  sign labels   processed_data/BIO_tags/<video>.npy           (sign level; 0 = O, 1 = I, 2 = B)
  phrase labels processed_data/BIO_tags_phrase_v2/<video>.npy

  python viz_hamer.py                                   # random test video, random stretch with >= 3 signs
  python viz_hamer.py --video 1209077_B
  python viz_hamer.py --video 1209077_B --start 12000 --seconds 6 --hand right
  python viz_hamer.py --selftest
Output: evaluation_phrase/figures/hamer_features.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FPS = 50
INK, MUTED = "#1F2933", "#5F6B78"
# MANO joint order of hand_pose: index 0-2, middle 3-5, pinky 6-8, ring 9-11, thumb 12-14
FINGERS = [("Thumb", (12, 13, 14), "#C0392B"), ("Index", (0, 1, 2), "#E07B39"), ("Middle", (3, 4, 5), "#C9A227"),
           ("Ring", (9, 10, 11), "#2E8B57"), ("Pinky", (6, 7, 8), "#2F6DB5")]


# ----------------------------------------------------------------------------- loading
def load_any(path):
    if path.endswith(".npy"):
        x = np.load(path, allow_pickle=True)
        return x.item() if x.dtype == object else x
    import torch
    return torch.load(path, weights_only=False, map_location="cpu")


def find_hamer(hamer_dir, video):
    cands = [hamer_dir] if hamer_dir else sorted(c for c in glob.glob(os.path.join("processed_data", "*hamer*"))
                                                 if os.path.isdir(c))
    for c in cands:
        for ext in (".pt", ".npy"):
            p = os.path.join(c, f"{video}_hamer{ext}")
            if os.path.exists(p):
                print(f"  HaMeR features from {p}")
                return load_any(p)
    raise SystemExit(f"no {video}_hamer.pt found in {cands} -- pass --hamer-dir")


def per_frame(d, T):
    hp = np.asarray(d["hand_pose"], dtype=np.float64)          # (T', 2, 15, 3, 3)
    go = np.asarray(d["global_orient"], dtype=np.float64)      # (T', 2, 3, 3)
    ds = int(d.get("temporal_downsample_factor", 1) or 1)
    idx = np.clip(np.arange(T) // ds, 0, len(hp) - 1)
    return hp[idx], go[idx]


def joint_angles(hp):
    """(T, 2, 15, 3, 3) -> degrees (T, 2, 15); NaN where the hand was not detected (all zeros)."""
    tr = np.trace(hp, axis1=-2, axis2=-1)
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    missing = np.all(hp.reshape(hp.shape[0], 2, -1) == 0, axis=-1)          # (T, 2)
    ang[missing] = np.nan
    return ang


def finger_bends(ang, hand):
    return np.stack([np.nansum(ang[:, hand, list(j)], axis=1) * np.where(np.isnan(ang[:, hand, 0]), np.nan, 1)
                     for _, j, _ in FINGERS], 1)                              # (T, 5)


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


# ----------------------------------------------------------------------------- schematic hand
def draw_hand(ax, joint_deg, x0, sx=1.0, sy=1.0):
    """Front view of a hand; each finger is shortened by how far its joints bend (foreshortening).
    sx, sy: data units per hand unit, so the hand keeps its shape on axes with unequal scales."""
    base = {"Index": (-0.27, 1.0, 0.42), "Middle": (-0.09, 1.04, 0.47), "Ring": (0.09, 1.0, 0.43),
            "Pinky": (0.26, 0.92, 0.34)}
    palm = np.array([[-0.36, 0.0], [0.34, 0.0], [0.36, 0.95], [-0.36, 1.0]])
    ax.fill(x0 + palm[:, 0] * sx, palm[:, 1] * sy, color="#E8ECF0", ec=MUTED, lw=1)
    for name, j, c in FINGERS:
        th = np.radians(np.nan_to_num(joint_deg[list(j)], nan=0.0))
        cum = np.cumsum(th)
        if name == "Thumb":
            p, d = np.array([-0.36, 0.3]), np.radians(135)
            pts = [p]
            for k in range(3):
                step = 0.2 * (1.0, 0.8, 0.6)[k] * max(np.cos(min(cum[k], np.pi)), 0.15)
                p = p + step * np.array([np.cos(d), np.sin(d)])
                pts.append(p)
        else:
            bx, by, L = base[name]
            p, pts = np.array([bx, by]), [np.array([bx, by])]
            for k in range(3):
                p = p + np.array([0.0, L * (0.45, 0.33, 0.22)[k] * np.cos(min(cum[k], np.pi))])
                pts.append(p)
        pts = np.array(pts)
        ax.plot(x0 + pts[:, 0] * sx, pts[:, 1] * sy, color=c, lw=3.2, solid_capstyle="round")


# ----------------------------------------------------------------------------- figure
def random_video(rng, hamer_dir, sign_dir, split_file, split):
    import json
    dirs = [hamer_dir] if hamer_dir else sorted(c for c in glob.glob(os.path.join("processed_data", "*hamer*"))
                                                if os.path.isdir(c))
    have = set()
    for c in dirs:
        have |= {f.rsplit("_hamer", 1)[0] for f in os.listdir(c) if "_hamer." in f}
    vids = sorted(v for v in have if os.path.exists(os.path.join(sign_dir, f"{v}.npy")))
    if split_file and os.path.exists(split_file):
        keep = {v.replace(".npy", "").replace(".pt", "") for v in json.load(open(split_file)).get(split, [])}
        vids = [v for v in vids if v in keep] or vids
    if not vids:
        raise SystemExit("no video has both HaMeR features and sign labels -- check --hamer-dir / --sign-labels-dir")
    return str(rng.choice(vids))


def run(video, hamer_dir, sign_dir, phrase_dir, out, start=None, seconds=6.0, hand=None, max_hands=6, seed=None,
        split_file="dataset_splits.json", split="test"):
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 100000)
    rng = np.random.default_rng(seed)
    if video is None:
        video = random_video(rng, hamer_dir, sign_dir, split_file, split)
    sign = np.load(os.path.join(sign_dir, f"{video}.npy")).astype(int)
    phrase = np.load(os.path.join(phrase_dir, f"{video}.npy")).astype(int) if phrase_dir else None
    T = len(sign)
    hp, go = per_frame(find_hamer(hamer_dir, video), T)
    ang = joint_angles(hp)
    L = int(seconds * FPS)
    if start is None:                          # a random stretch with several signs and the hands mostly detected
        scored = []
        for a0 in range(0, max(1, T - L), L // 4):
            n_sign = int(np.sum(sign[a0:a0 + L] == 2))
            det = float(np.mean(np.isfinite(ang[a0:a0 + L, :, 0])))
            scored.append((det * min(n_sign, max_hands), a0, n_sign, det))
        good = [a0 for sc, a0, n, det in scored if n >= 3 and det >= 0.5]
        start = int(rng.choice(good)) if good else max(scored)[1]
    a, b = start, min(T, start + L)
    if hand is None:                           # the hand that moves its fingers more in this stretch
        var = [np.nanstd(finger_bends(ang[a:b], h)) if np.isfinite(ang[a:b, h, 0]).any() else -1 for h in (0, 1)]
        hand = int(np.argmax(var))
    else:
        hand = {"left": 0, "right": 1}[hand]
    fb = finger_bends(ang[a:b], hand)
    t = np.arange(a, b)

    signs = [(s, e) for s, e in segments(sign) if e >= a and s < b]
    fig = plt.figure(figsize=(12, 7.2))
    ax_h = fig.add_axes([0.08, 0.70, 0.84, 0.26])
    ax_b = fig.add_axes([0.08, 0.36, 0.84, 0.30], sharex=None)
    ax_m = fig.add_axes([0.08, 0.08, 0.84, 0.22])

    # top: schematic hands at the middle of each sign
    mids = [int((max(s, a) + min(e, b - 1)) / 2) for s, e in signs][:max_hands]
    dur = (b - a) / FPS
    w_in, h_in = 12 * 0.84, 7.2 * 0.26                  # size of ax_h in inches
    sy = 0.55                                           # hand units -> y data units (ylim 0..1.6)
    sx = sy * (dur / w_in) / (1.6 / h_in)               # same physical size in x
    for m in mids:
        x = (m - a) / FPS
        if x - 0.45 * sx >= 0 and x + 0.45 * sx <= dur:     # skip a hand that would be cut at the edge
            draw_hand(ax_h, ang[m, hand], x, sx=sx, sy=sy)
    ax_h.set_xlim(0, dur)
    ax_h.set_ylim(-0.25, 1.35)
    ax_h.axis("off")
    ax_h.set_title(f"{['Left', 'Right'][hand]} hand at the middle of each sign (schematic: finger bend only)",
                   fontsize=11, color=MUTED, loc="left")

    # middle: finger bend over time
    for k, (s, e) in enumerate(signs):
        ax_b.add_patch(Rectangle(((max(s, a) - a) / FPS, 0), (min(e, b - 1) - max(s, a) + 1) / FPS, 1,
                                 transform=ax_b.get_xaxis_transform(), color="#EEF1F4" if k % 2 == 0 else "#E1E6EB",
                                 lw=0))
    for (name, _, c), col in zip(FINGERS, fb.T):
        ax_b.plot((t - a) / FPS, col, color=c, lw=2, label=name)
    if phrase is not None:
        for s in np.where(phrase[a:b] == 2)[0]:
            ax_b.axvline(s / FPS, color=INK, lw=1.5, ls="--")
        ax_b.plot([], [], color=INK, lw=1.5, ls="--", label="phrase start")
    for m in mids:
        ax_b.axvline((m - a) / FPS, color=MUTED, lw=0.6, ls=":")
    ax_b.set_xlim(0, (b - a) / FPS)
    ax_b.set_ylabel("finger bend (°)")
    ax_b.legend(frameon=False, ncol=6, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)
    ax_b.set_xticklabels([])

    # bottom: the 288 numbers
    feat = np.concatenate([hp.reshape(T, -1)[a:b], go.reshape(T, -1)[a:b]], 1).T      # (288, L)
    ax_m.imshow(feat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                extent=(0, (b - a) / FPS, feat.shape[0], 0))
    ax_m.set_yticks([0, 135, 270, 288])
    ax_m.set_yticklabels(["left fingers", "right fingers", "orientations", ""], fontsize=8, color=MUTED, va="top")
    ax_m.set_xlabel("time (s)")
    ax_m.set_title("Model input: 288 numbers per frame (finger-joint and hand rotations, both hands)", fontsize=11,
                   color=MUTED, loc="left")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {video}: frames {a}-{b} ({a / FPS:.1f}-{b / FPS:.1f} s), {len(signs)} signs, "
          f"{['left', 'right'][hand]} hand")
    print(f"  to redraw exactly this: --video {video} --start {a} --hand {['left', 'right'][hand]}   "
          f"(or --seed {seed})")
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    rng = np.random.default_rng(0)

    def rot_x(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    with tempfile.TemporaryDirectory() as d:
        T = 1500
        sign = np.zeros(T, int)
        phrase = np.zeros(T, int)
        hp = np.zeros((T, 2, 15, 3, 3))
        go = np.tile(np.eye(3), (T, 2, 1, 1))
        t, shapes = 20, []
        while t < T - 100:
            L = int(rng.integers(25, 60))
            sign[t], sign[t + 1:t + L] = 2, 1
            curl = rng.uniform(0, 1.4, 5)
            for k in range(t, t + L):
                for f, (_, j, _) in enumerate(FINGERS):
                    for jj in j:
                        hp[k, 1, jj] = rot_x(curl[f] * (0.8 + 0.2 * jj % 3))
                        hp[k, 0, jj] = rot_x(0.3)
            t += L + int(rng.integers(3, 15))
        hp[200:230, 1] = 0                       # hand not detected
        phrase[[20, 400, 900]] = 2
        os.makedirs(f"{d}/ham")
        np.save(f"{d}/ham/v_A_hamer.npy", np.array({"hand_pose": hp, "global_orient": go,
                                                    "temporal_downsample_factor": 1}, dtype=object))
        for name, arr in (("sign", sign), ("phrase", phrase)):
            os.makedirs(f"{d}/{name}")
            np.save(f"{d}/{name}/v_A.npy", arr)
        ang = joint_angles(hp)
        ok1 = np.isnan(ang[210, 1]).all() and abs(np.nanmax(ang[:, 0]) - np.degrees(0.3)) < 1e-6
        out = os.environ.get("VIZ_SELFTEST_OUT", d)
        run(None, f"{d}/ham", f"{d}/sign", f"{d}/phrase", os.path.join(out, "hamer_features.png"), seed=1)
        ok2 = os.path.exists(os.path.join(out, "hamer_features.png"))
    print(f"  [{'ok' if ok1 else 'FAIL'}] joint angles and missing hands")
    print(f"  [{'ok' if ok2 else 'FAIL'}] figure written")
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="default: a random video of --split")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: new each run, printed)")
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--hamer-dir", default=None)
    ap.add_argument("--sign-labels-dir", default=os.path.join("processed_data", "BIO_tags"))
    ap.add_argument("--phrase-labels-dir", default=os.path.join("processed_data", "BIO_tags_phrase_v2"))
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--hand", choices=["left", "right"], default=None)
    ap.add_argument("--out", default=os.path.join("evaluation_phrase", "figures", "hamer_features.png"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run(a.video, a.hamer_dir, a.sign_labels_dir, a.phrase_labels_dir, a.out, a.start, a.seconds, a.hand,
        seed=a.seed, split_file=a.split_file, split=a.split)


if __name__ == "__main__":
    main()