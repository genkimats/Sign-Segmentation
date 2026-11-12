"""
analyze_nonmanual_starts.py -- does the face stream help most at phrase starts that are marked by the face?

Every gold phrase start of the test split is classified, model-free, from the interpretable cue files
(processed_data/cue_features, data_processing/extract_cue_features.py) in a window of +-W frames
(default 10 = +-0.2 s) around the start:
  eyes/brows   a blink, or a brow movement (|brow vertical velocity| >= Z robust-z units)  -> what the
               compact face stream sees
  head only    no eye / brow marker, but a head movement (|head vertical velocity| or head rotation
               speed >= Z)  -> head points are in the BODY stream (nose, eyes, ears of the pose), not
               in the face stream
  none         neither
Starts where the face was detected in less than half of the window are left out.

For two groups of runs -- WITH the face stream and WITHOUT it (same machine, same recipe; default
115-117 = B+H+Fc and 112-114 = B+H) -- every run is decoded with peak picking tuned on validation
(count_oracle_decoding.tune_peaks_on_val, the thesis decoder), and a gold start counts as found when a
predicted start lies within +-tol frames (one-to-one, src.metrics.match_boundaries). Found = mean over
the group's runs.

Reported (paired bootstrap over test videos, 95% intervals):
  detection rate per category, with and without face, and the difference (face effect) per category
  interaction: face effect at eyes/brows starts minus face effect at unmarked starts
  the same within short pauses (gold pause before the start <= 0.24 s), where F4 found most misses
  marker base rates: how often each marker occurs at the starts vs. at random moments inside phrases
  (>= 0.5 s from any start), to show the markers are boundary-related at all

  python analyze_nonmanual_starts.py                                   # 115-117 vs 112-114
  python analyze_nonmanual_starts.py --with-face 373-375 --without-face <B+H prefixes on that machine>
  python analyze_nonmanual_starts.py --selftest
Outputs (--out-dir, default evaluation_phrase/nonmanual_starts): summary.txt, starts.csv, nonmanual_starts.png
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

from count_oracle_decoding import peaks_with_heights, tune_peaks_on_val
from phrase_stats import PRED_ROOT, expand_prefixes, gold_starts_of, load_predictions
from src.cue_sets import FACE_DETECTED, cue_names_path
from src.metrics import B_TAG, O_TAG, match_boundaries

CATS = ["eyes/brows", "head only", "none"]
FPS = 50


# ----------------------------------------------------------------------------- markers
def cue_matrix(cue_dir, vid, names):
    path = os.path.join(cue_dir, f"{vid}.npy")
    if not os.path.exists(path):
        return None
    arr = np.load(path).astype(np.float32)
    return {n: arr[:, i] for i, n in enumerate(names)}


def markers_at(cues, t, w, z):
    """Markers in [t - w, t + w] -> dict, or None if the face was not detected in >= half of the window."""
    T = len(cues["blink"])
    lo, hi = max(0, t - w), min(T, t + w + 1)
    if hi <= lo or cues[FACE_DETECTED][lo:hi].mean() < 0.5:
        return None
    blink = bool((cues["blink"][lo:hi] > 0.5).any())
    brow = bool((np.abs(cues["brow vertical velocity"][lo:hi]) >= z).any())
    head = bool(max(np.abs(cues["head vertical velocity"][lo:hi]).max(),
                    cues["head rotation speed"][lo:hi].max()) >= z)
    cat = "eyes/brows" if (blink or brow) else ("head only" if head else "none")
    return {"blink": blink, "brow": brow, "head": head, "cat": cat}


def pause_before(gold, s):
    """Gold O frames immediately before start s."""
    n = 0
    while s - n - 1 >= 0 and gold[s - n - 1] == O_TAG:
        n += 1
    return n


def in_phrase_frames(gold, starts, min_gap, rng, n_max=200):
    """Random frames inside phrases, >= min_gap frames from any start (marker base rate)."""
    inside = np.flatnonzero(gold != O_TAG)
    if len(starts):
        d = np.min(np.abs(inside[:, None] - np.asarray(starts)[None, :]), axis=1)
        inside = inside[d >= min_gap]
    if len(inside) > n_max:
        inside = rng.choice(inside, n_max, replace=False)
    return inside


# ----------------------------------------------------------------------------- detection
def found_flags(run, split, pred_root, tol):
    """{vid: bool array over gold starts} for one run, peak picking tuned on val."""
    pk = tune_peaks_on_val(load_predictions(pred_root, run, "val"))
    probs, _, gold = load_predictions(pred_root, run, split)
    out = {}
    for v in probs:
        idx, h = peaks_with_heights(probs[v][B_TAG], pk["min_dist"], pk["sigma"])
        g_ok, _, _ = match_boundaries(gold_starts_of(gold[v]), np.sort(idx[h >= pk["thr"]]), tol)
        out[v] = g_ok
    return out, gold, pk


def group_found(runs, split, pred_root, tol):
    flags, gold, settings = [], None, []
    for r in runs:
        f, g, pk = found_flags(r, split, pred_root, tol)
        flags.append(f)
        gold = gold or g
        settings.append(f"{r}: min_dist={pk['min_dist']} sigma={pk['sigma']} thr={pk['thr']}")
    vids = sorted(set.intersection(*[set(f) for f in flags]))
    return {v: np.mean([f[v] for f in flags], axis=0) for v in vids}, gold, settings


# ----------------------------------------------------------------------------- bootstrap
def rates_bootstrap(rows, key, n_boot, seed=0):
    """
    rows: list of dicts with vid, cat, found_face, found_noface (+ the filter already applied).
    Returns {cat: (rate_face, rate_noface, diff, (lo, hi))} and the interaction eyes/brows - none,
    resampling videos with replacement.
    """
    vids = sorted({r["vid"] for r in rows})
    vi = {v: i for i, v in enumerate(vids)}
    V, C = len(vids), len(CATS)
    N, F, G = np.zeros((V, C)), np.zeros((V, C)), np.zeros((V, C))
    for r in rows:
        i, c = vi[r["vid"]], CATS.index(r[key])
        N[i, c] += 1
        F[i, c] += r["found_face"]
        G[i, c] += r["found_noface"]

    def stat(w):
        n = w @ N
        with np.errstate(invalid="ignore", divide="ignore"):
            rf, rg = (w @ F) / n, (w @ G) / n
        return rf, rg, rf - rg

    rf, rg, d = stat(np.ones(V))
    rng = np.random.default_rng(seed)
    draws = np.stack([stat(np.bincount(rng.integers(0, V, V), minlength=V).astype(float))[2]
                      for _ in range(n_boot)])                                   # (n_boot, C)
    inter = draws[:, 0] - draws[:, 2]
    res = {c: (rf[k], rg[k], d[k], (np.nanpercentile(draws[:, k], 2.5), np.nanpercentile(draws[:, k], 97.5)),
               int(N[:, k].sum())) for k, c in enumerate(CATS)}
    res["interaction"] = (d[0] - d[2], (np.nanpercentile(inter, 2.5), np.nanpercentile(inter, 97.5)))
    return res


def fmt_ci(lo, hi):
    return f"95% CI: {lo:+.3f} to {hi:+.3f}"


# ----------------------------------------------------------------------------- figure
def figure(res_all, res_short, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200, sharey=True)
    for ax, (res, title) in zip(axes, ((res_all, "All starts"), (res_short, "Starts after pauses <= 0.24 s"))):
        x = np.arange(len(CATS))
        ax.bar(x - 0.19, [res[c][1] * 100 for c in CATS], 0.36, color="#9AA5B1", label="without face")
        ax.bar(x + 0.19, [res[c][0] * 100 for c in CATS], 0.36, color="#1F5FA8", label="with face")
        for k, c in enumerate(CATS):
            ax.text(k + 0.19, res[c][0] * 100 + 1.5, f"{res[c][2] * 100:+.1f}", ha="center", fontsize=8)
        ax.set_xticks(x, [f"{c}\n(n = {res[c][4]})" for c in CATS], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 100)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Phrase starts found (%), peak picking, +-0.1 s")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.5, -0.02, "Numbers above bars: with face minus without face (percentage points). "
             "Marker classes from the cue files, +-0.2 s around the gold start.", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


# ----------------------------------------------------------------------------- main
def run_analysis(args):
    model = args.model
    with_face = expand_prefixes(model, args.with_face)
    without_face = expand_prefixes(model, args.without_face)
    print(f"with face   : {', '.join(with_face)}\nwithout face: {', '.join(without_face)}")
    ff, gold, set_f = group_found(with_face, args.split, args.pred_root, args.tol)
    fg, _, set_g = group_found(without_face, args.split, args.pred_root, args.tol)
    with open(cue_names_path(args.cue_dir)) as f:
        names = json.load(f)
    need = ["blink", "brow vertical velocity", "head vertical velocity", "head rotation speed", FACE_DETECTED]
    miss = [n for n in need if n not in names]
    if miss:
        raise SystemExit(f"{args.cue_dir} lacks cue channels {miss} -- re-run data_processing/extract_cue_features.py")

    rng = np.random.default_rng(0)
    rows, base, no_cue, no_face = [], {"blink": [], "brow": [], "head": [], "eyes/brows": []}, [], 0
    for v in sorted(set(ff) & set(fg)):
        cues = cue_matrix(args.cue_dir, v, names)
        if cues is None:
            no_cue.append(v)
            continue
        T = min(len(gold[v]), len(cues["blink"]))
        cues = {k: a[:T] for k, a in cues.items()}
        g = gold[v][:T]
        starts = gold_starts_of(gold[v])
        for j, s in enumerate(starts):
            if s >= T:
                continue
            m = markers_at(cues, int(s), args.window, args.z)
            if m is None:
                no_face += 1
                continue
            p = pause_before(g, int(s))
            rows.append({"vid": v, "start": int(s), "cat": m["cat"], "blink": int(m["blink"]),
                         "brow": int(m["brow"]), "head": int(m["head"]), "pause_frames": p,
                         "short_pause": int(p <= args.short_pause), "found_face": float(ff[v][j]),
                         "found_noface": float(fg[v][j])})
        for t in in_phrase_frames(g, starts, int(0.5 * FPS), rng):
            m = markers_at(cues, int(t), args.window, args.z)
            if m is not None:
                for k in ("blink", "brow", "head"):
                    base[k].append(m[k])
                base["eyes/brows"].append(m["blink"] or m["brow"])
    if not rows:
        raise SystemExit("No phrase start with cue data -- check --cue-dir.")

    res_all = rates_bootstrap(rows, "cat", args.n_boot)
    short = [r for r in rows if r["short_pause"]]
    res_short = rates_bootstrap(short, "cat", args.n_boot) if short else None

    os.makedirs(args.out_dir, exist_ok=True)
    lines = [f"Face stream and non-manually marked phrase starts ({args.split}; peak picking tuned on val; "
             f"found = predicted start within +-{args.tol} frames; mean over runs)",
             f"with face: {', '.join(with_face)} | without face: {', '.join(without_face)}",
             f"markers: +-{args.window} frames around the gold start; brow / head threshold |z| >= {args.z}; "
             f"{len(rows)} starts with face data ({no_face} left out: face not detected"
             + (f"; {len(no_cue)} videos without cue file" if no_cue else "") + ")", ""]
    s_rows = {"blink": np.mean([r["blink"] for r in rows]), "brow": np.mean([r["brow"] for r in rows]),
              "head": np.mean([r["head"] for r in rows]),
              "eyes/brows": np.mean([r["blink"] or r["brow"] for r in rows])}
    lines.append("Marker rates: at phrase starts vs. at random moments inside phrases (>= 0.5 s from a start)")
    for k in ("blink", "brow", "head", "eyes/brows"):
        lines.append(f"  {k:<11s} {s_rows[k] * 100:5.1f}%  vs  {np.mean(base[k]) * 100:5.1f}%")
    for title, res in (("All starts", res_all), (f"Starts after gold pauses <= {args.short_pause} frames "
                                                  f"({args.short_pause / FPS:.2f} s)", res_short)):
        if res is None:
            continue
        lines += ["", title, f"  {'category':<12s} {'n':>5s} {'without face':>13s} {'with face':>10s}   face effect"]
        for c in CATS:
            rf, rg, d, (lo, hi), n = res[c]
            lines.append(f"  {c:<12s} {n:>5d} {rg * 100:12.1f}% {rf * 100:9.1f}%   {d:+.3f} ({fmt_ci(lo, hi)})")
        d, (lo, hi) = res["interaction"]
        lines.append(f"  interaction (eyes/brows effect - unmarked effect): {d:+.3f} ({fmt_ci(lo, hi)})")
    lines += ["", "Peak-picking settings tuned on val:"] + [f"  {s}" for s in set_f + set_g]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out_dir, "starts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if res_short is not None:
        figure(res_all, res_short, os.path.join(args.out_dir, "nonmanual_starts.png"))
    print(f"\nwrote {args.out_dir}/summary.txt, starts.csv, nonmanual_starts.png")


def selftest():
    ok = []
    T = 200
    cues = {"blink": np.zeros(T), "brow vertical velocity": np.zeros(T), "head vertical velocity": np.zeros(T),
            "head rotation speed": np.zeros(T), FACE_DETECTED: np.ones(T)}
    cues["blink"][50] = 1
    cues["head vertical velocity"][120] = -3.0
    cues["brow vertical velocity"][160] = 2.5
    ok.append(markers_at(cues, 55, 10, 2.0)["cat"] == "eyes/brows")
    ok.append(markers_at(cues, 115, 10, 2.0)["cat"] == "head only")
    ok.append(markers_at(cues, 90, 10, 2.0)["cat"] == "none")
    ok.append(markers_at(cues, 165, 10, 2.0)["cat"] == "eyes/brows")
    c2 = dict(cues, **{FACE_DETECTED: np.zeros(T)})
    ok.append(markers_at(c2, 55, 10, 2.0) is None)
    g = np.full(30, 1)
    g[5:9] = O_TAG
    ok.append(pause_before(g, 9) == 4 and pause_before(g, 3) == 0)
    rows = [{"vid": f"v{i % 4}", "cat": CATS[i % 3], "found_face": 1.0 if i % 3 == 0 else 0.5,
             "found_noface": 0.5} for i in range(60)]
    r = rates_bootstrap(rows, "cat", 200)
    ok.append(abs(r["eyes/brows"][2] - 0.5) < 1e-9 and abs(r["none"][2]) < 1e-9
              and abs(r["interaction"][0] - 0.5) < 1e-9)
    print(f"selftest: {sum(ok)}/{len(ok)} passed")
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="multistream_bilstm")
    ap.add_argument("--with-face", nargs="+", default=["115-117"])
    ap.add_argument("--without-face", nargs="+", default=["112-114"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--cue-dir", default="processed_data/cue_features")
    ap.add_argument("--window", type=int, default=10, help="frames around the start (10 = +-0.2 s)")
    ap.add_argument("--z", type=float, default=2.0, help="robust-z threshold for brow / head movement")
    ap.add_argument("--tol", type=int, default=5, help="boundary tolerance in frames (5 = +-0.1 s)")
    ap.add_argument("--short-pause", type=int, default=12, help="frames (12 = 0.24 s, the F4 threshold)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "nonmanual_starts"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    run_analysis(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
