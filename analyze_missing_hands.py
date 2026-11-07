
"""
analyze_missing_hands.py -- how often MediaPipe lost the hands, and whether those gaps line up with phrases.

extract_poses.py fills every missing hand by straight-line interpolation (any gap length) and then
normalises each frame by its own shoulder midpoint / width. Nothing records which frames were real.
This script recovers them from processed_data/keypoints and measures the effect. Nothing is changed.

Detection. Inside a filled gap every joint moves on a straight line, so the hand SHAPE (joints minus
the wrist) is A + tB in the raw coordinates. Per-frame normalisation only translates and scales a frame,
so the normalised shape is (A + tB) / s(t): it stays in the 2-D span of A and B. Three consecutive
frames of a filled stretch therefore have rank <= 2 (3rd singular value ~ 0), which real detections
(21 joints with tracking noise) never have. Constant fills at the video edges have rank 1, never-
detected hands are all-equal (shape 0). Every frame filled by interpolation is found; the real frames
at both ends of a gap are not counted. python analyze_missing_hands.py --selftest checks this on
synthetic data run through the same interpolation + normalisation.

    python analyze_missing_hands.py                         # all videos in dataset_splits.json
    python analyze_missing_hands.py --split test            # one split
    python analyze_missing_hands.py --selftest

Reports (bootstrap 95% CI over videos):
  1. how much is filled: share of frames per hand / either / both, gap-length distribution
  2. filled hands vs phrase labels: share of filled frames OUTSIDE phrases (O) vs INSIDE (I/B)
  3. around starts: share of the 0.5 s before a phrase start with a filled hand, vs before sign starts
     inside phrases (needs processed_data/BIO_tags)
  4. vs errors (if evaluation_phrase/error_analysis/gold_boundaries_test.csv exists): filled share before
     phrase starts that (almost) no model finds vs starts most models find
Output: evaluation_phrase/missing_hands/summary.txt, per_video.csv
"""
import argparse
import json
import os

import numpy as np

BODY = 23
HANDS = {"left": slice(BODY, BODY + 21), "right": slice(BODY + 21, BODY + 42)}
FPS = 50
TOL = 1e-6


# ----------------------------------------------------------------------------- detection
def filled_frames(kp, hand):
    """kp (T, 65, 3) normalised keypoints -> bool (T,) True where this hand was filled, not detected."""
    h = np.asarray(kp[:, HANDS[hand], :], dtype=np.float64)
    q = (h - h[:, :1]).reshape(len(h), -1)                     # shape: joints minus wrist (T, 63)
    T = len(q)
    out = np.zeros(T, dtype=bool)
    norm = np.linalg.norm(q, axis=1)
    out |= norm < 1e-9                                         # never detected (all joints equal)
    if T >= 3:
        w = np.stack([q[:-2], q[1:-1], q[2:]], axis=1)         # (T-2, 3, 63)
        sv = np.linalg.svd(w, compute_uv=False)                # (T-2, 3)
        flat = sv[:, 2] <= TOL * np.maximum(sv[:, 0], 1e-12)
        out[1:-1] |= flat                                      # middle frame of a rank-<=2 triple
        # constant edge fills (rank 1) reach the first / last frame
        if flat[0] and sv[0, 1] <= TOL * max(sv[0, 0], 1e-12):
            out[0] = True
        if flat[-1] and sv[-1, 1] <= TOL * max(sv[-1, 0], 1e-12):
            out[-1] = True
    return out


def runs(mask):
    """Lengths of the runs of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def starts_of(lab):
    lab = np.asarray(lab)
    prev = np.r_[0, lab[:-1]]
    return np.flatnonzero((lab == 2) | ((lab == 1) & (prev == 0)))


# ----------------------------------------------------------------------------- statistics
def boot(per_video, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(per_video)
    vals = [fn([per_video[k] for k in rng.choice(keys, len(keys))]) for _ in range(n)]
    return fn([per_video[k] for k in keys]), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def ratio(a, b):
    return lambda vs: sum(v[a] for v in vs) / max(1e-9, sum(v[b] for v in vs))


def fmt(r, pct=True):
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(r[0])} [{f(r[1])}, {f(r[2])}]"


def analyse(args):
    with open(args.split_file) as f:
        splits = json.load(f)
    names = [args.split] if args.split else ["train", "val", "test"]
    vids = sorted({v.replace(".npy", "") for s in names for v in splits.get(s, [])})
    win = int(0.5 * FPS)
    per, gaps = {}, {h: [] for h in HANDS}
    rows = []
    for vid in vids:
        kp_path = os.path.join(args.keypoints_dir, f"{vid}.npy")
        if not os.path.exists(kp_path):
            continue
        kp = np.load(kp_path)
        m = {h: filled_frames(kp, h) for h in HANDS}
        for h in HANDS:
            gaps[h].extend(runs(m[h]).tolist())
        either, both = m["left"] | m["right"], m["left"] & m["right"]
        T = len(kp)
        v = {"T": T, "left": m["left"].sum(), "right": m["right"].sum(), "either": either.sum(), "both": both.sum()}
        lp = os.path.join(args.phrase_dir, f"{vid}.npy")
        if os.path.exists(lp):
            lab = np.load(lp)[:T]
            e = either[:len(lab)]
            v.update(out_n=(lab == 0).sum(), out_f=e[lab == 0].sum(), in_n=(lab > 0).sum(), in_f=e[lab > 0].sum())
            ph = starts_of(lab)
            pre = lambda s: e[max(0, s - win):s]
            v.update(ph_n=sum(len(pre(s)) for s in ph), ph_f=sum(pre(s).sum() for s in ph))
            sp = os.path.join(args.sign_dir, f"{vid}.npy")
            if os.path.exists(sp):
                sg = starts_of(np.load(sp)[:len(lab)])
                inside = [s for s in sg if lab[s] == 1 and np.min(np.abs(ph - s)) > 5] if len(ph) else []
                v.update(sg_n=sum(len(pre(s)) for s in inside), sg_f=sum(pre(s).sum() for s in inside))
            v["_either"], v["_lab"] = e, lab
        per[vid] = v
        rows.append({k: (int(x) if not isinstance(x, np.ndarray) else None) for k, x in v.items() if not k.startswith("_")})

    lines = [f"{len(per)} videos ({'+'.join(names)}); a 'filled' frame = the hand was not detected and "
             f"extract_poses.py interpolated it. 95% CI: bootstrap over videos.", ""]
    lines.append("1. HOW MUCH IS FILLED (share of all frames)")
    for h in ("left", "right", "either", "both"):
        lines.append(f"   {h:7s} {fmt(boot(per, ratio(h, 'T')))}")
    for h in HANDS:
        g = np.array(gaps[h])
        if len(g):
            lines.append(f"   {h} gaps: {len(g)}; median {np.median(g) / FPS:.2f} s; >= 0.5 s: {(g >= 25).mean():.0%} of gaps, "
                         f"holding {g[g >= 25].sum() / g.sum():.0%} of filled frames; longest {g.max() / FPS:.1f} s")
    lab_v = {k: v for k, v in per.items() if "out_n" in v}
    if lab_v:
        lines += ["", "2. FILLED HANDS vs PHRASES (either hand filled)"]
        out_r, in_r = boot(lab_v, ratio("out_f", "out_n")), boot(lab_v, ratio("in_f", "in_n"))
        lines.append(f"   outside phrases (O): {fmt(out_r)}")
        lines.append(f"   inside phrases (I/B): {fmt(in_r)}")
        d = boot(lab_v, lambda vs: ratio("out_f", "out_n")(vs) - ratio("in_f", "in_n")(vs))
        lines.append(f"   difference O - inside: {fmt(d, pct=False)} (share of frames)")
        lines += ["", f"3. THE 0.5 s BEFORE A START (either hand filled)"]
        lines.append(f"   before phrase starts:              {fmt(boot(lab_v, ratio('ph_f', 'ph_n')))}")
        sg_v = {k: v for k, v in lab_v.items() if "sg_n" in v}
        if sg_v:
            lines.append(f"   before sign starts inside phrases: {fmt(boot(sg_v, ratio('sg_f', 'sg_n')))}")
            d = boot(sg_v, lambda vs: ratio("ph_f", "ph_n")(vs) - ratio("sg_f", "sg_n")(vs))
            lines.append(f"   difference:                        {fmt(d, pct=False)}")
    gb = args.gold_boundaries
    if os.path.exists(gb):
        import pandas as pd
        g = pd.read_csv(gb)
        g = g[g.video.isin(lab_v.keys())]
        if len(g):
            win_f = lambda r: lab_v[r.video]["_either"][max(0, int(r.frame) - win):int(r.frame)].mean()
            g["filled_before"] = g.apply(win_f, axis=1)
            hard, easy = g[g.detection_rate <= 0.1], g[g.detection_rate >= 0.9]
            lines += ["", "4. vs ERRORS (test; share of the 0.5 s before the phrase start with a filled hand)",
                      f"   starts found by <= 10% of runs (n={len(hard)}): {hard.filled_before.mean():.1%}",
                      f"   starts found by >= 90% of runs (n={len(easy)}): {easy.filled_before.mean():.1%}"]
            from scipy.stats import mannwhitneyu
            if len(hard) and len(easy):
                lines.append(f"   Mann-Whitney p = {mannwhitneyu(hard.filled_before, easy.filled_before).pvalue:.2g}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out_dir, "per_video.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video"] + keys)
        w.writeheader()
        for vid, r in zip(per, rows):
            w.writerow({"video": vid, **r})
    print("\n".join(lines))
    print(f"\n-> {args.out_dir}/summary.txt, per_video.csv")


# ----------------------------------------------------------------------------- selftest
def _normalize(kp):                                       # = extract_poses.normalize_skeleton
    ls, rs = kp[:, 11, :], kp[:, 12, :]
    scale = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=1, keepdims=True) + 1e-6
    root = (ls + rs) / 2.0
    out = kp.copy()
    out[:, :, :3] = (kp[:, :, :3] - root[:, None]) / scale[:, None]
    return out


def selftest():
    import pandas as pd
    rng = np.random.default_rng(0)
    T = 3000
    t = np.arange(T) / FPS
    kp = np.zeros((T, 65, 3))
    kp[:, :BODY] = 0.5 + 0.05 * rng.normal(size=(1, BODY, 3)) + 0.003 * rng.normal(size=(T, BODY, 3))
    kp[:, 11, 0] -= 0.1
    kp[:, 12, 0] += 0.1
    for k, h in enumerate(HANDS.values()):                  # moving hands with tracking noise
        base = 0.4 + 0.1 * k + 0.05 * np.sin(2 * np.pi * (0.7 + 0.2 * k) * t)[:, None, None]
        kp[:, h] = base + 0.03 * rng.normal(size=(1, 21, 3)) + 0.002 * rng.normal(size=(T, 21, 3))
    truth = {h: np.zeros(T, bool) for h in HANDS}
    for h, sl in HANDS.items():
        for _ in range(25):
            a, L = int(rng.integers(0, T - 200)), int(rng.choice([1, 2, 5, 20, 80, 150]))
            truth[h][a:a + L] = True
        truth[h][:7] = True                                  # missing at the start (edge fill)
    raw = kp.copy()
    for h, sl in HANDS.items():
        raw[truth[h], sl] = np.nan
    df = pd.DataFrame(raw.reshape(T, -1)).interpolate(method="linear", limit_direction="both")
    norm = _normalize(df.values.reshape(raw.shape))
    ok = True
    for h in HANDS:
        pred = filled_frames(norm, h)
        rec = (pred & truth[h]).sum() / truth[h].sum()
        prec = (pred & truth[h]).sum() / max(1, pred.sum())
        print(f"  {h}: recall {rec:.3f}, precision {prec:.3f} ({truth[h].sum()} filled frames)")
        ok &= rec > 0.99 and prec > 0.99
    print("selftest:", "OK" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints-dir", default="processed_data/keypoints")
    ap.add_argument("--phrase-dir", default="processed_data/BIO_tags_phrase_v2")
    ap.add_argument("--sign-dir", default="processed_data/BIO_tags")
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--gold-boundaries", default="evaluation_phrase/error_analysis/gold_boundaries_test.csv")
    ap.add_argument("--out-dir", default="evaluation_phrase/missing_hands")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    analyse(a)


if __name__ == "__main__":
    main()