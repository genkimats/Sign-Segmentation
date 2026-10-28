"""
Figures for the research presentation (phrase-level segmentation, Public DGS Corpus).

Run from the repo root after the evaluation / analysis scripts:

  python make_presentation_figures.py                 # all figures -> figures/
  python make_presentation_figures.py --only 3 5 12   # some figures
  python make_presentation_figures.py --pdf           # also save vector PDFs

Inputs (default paths, change in PATHS below):
  evaluation_phrase/{multistream_bilstm,bilstm_2023}_test_metrics.csv   evaluate_phrase.py (test, --merge-starts)
  evaluation_phrase/error_analysis/summary_test.txt                     boundary_error_analysis.py (no merge)
  evaluation_phrase/error_analysis_merge10/*                            boundary_error_analysis.py --merge-starts 10
  evaluation_phrase/oracle/{oracle_summary,oracle_lexical_weights,transitions_test}.csv   phrase_oracle_analysis.py
  evaluation_phrase/metric_analysis/metric_analysis_{argmax,argmax_merge,tuned}.json    metric_analysis.py
  evaluation_phrase/predictions/<run>/test.npz + processed_data/BIO_tags_phrase_v2   (figure 6 only)

Run groups (which run belongs to which bar) are set in GROUPS below; edit them if you re-run.
Every figure is independent: a missing input skips that figure with a message.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------- paths
EVAL = "evaluation_phrase"
PATHS = {
    "metrics": [os.path.join(EVAL, "multistream_bilstm_test_metrics.csv"),
                os.path.join(EVAL, "bilstm_2023_test_metrics.csv")],
    "err_plain": os.path.join(EVAL, "error_analysis"),
    "err_merge": os.path.join(EVAL, "error_analysis_merge10"),
    "oracle": os.path.join(EVAL, "oracle"),
    "metric_json": os.path.join(EVAL, "metric_analysis", "metric_analysis_{}.json"),
    "predictions": os.path.join(EVAL, "predictions"),
    "labels": os.path.join("processed_data", "BIO_tags_phrase_v2"),
}
OUT = "figures"
FPS = 50

# ----------------------------------------------------------------------------- run groups
MS, BL = "multistream_bilstm", "bilstm_2023"


def runs(model, *ranges):
    """runs(MS, (130, 132), 148) -> ['multistream_bilstm-130', ..., 'multistream_bilstm-148']"""
    out = []
    for r in ranges:
        a, b = (r, r) if isinstance(r, int) else r
        out += [f"{model}-{i}" for i in range(a, b + 1)]
    return out


GROUPS = {
    # input-combination grid: (multistream runs, 2023-model runs)
    "grid": {
        "Body": (runs(MS, (130, 132)), runs(BL, (250, 252))),
        "HaMeR": (runs(MS, (133, 135)), runs(BL, (253, 255))),
        "Body+Face": (runs(MS, (139, 141)), runs(BL, (262, 264))),
        "Face+HaMeR": (runs(MS, (142, 144)), runs(BL, (265, 267))),
        "Body+HaMeR": (runs(MS, (112, 114), (118, 120), (127, 129)), runs(BL, (259, 261))),
        "Body+Face+HaMeR": (runs(MS, (115, 117), (121, 126), (148, 150)), runs(BL, (268, 270))),
    },
    # body-only mixing ablation: label -> (runs, family)   family: 2023 / none / temporal
    "mixing": {
        "2023 model\nLinear": (runs(BL, (250, 252)), "2023"),
        "MS\nper-joint\nLinear": (runs(MS, 280, 282), "none"),
        "MS\ntemporal conv\nno graph": (runs(MS, (277, 279)), "temporal"),
        "MS\nST-GCN\ngraph + conv": (runs(MS, (130, 132)), "temporal"),
    },
    "recipe": {
        "2023 recipe": runs(BL, (238, 240)),
        "Your recipe": runs(BL, (244, 246)),
    },
    "example_run": f"{MS}-104",     # figure 6 (best configuration, compact face)
}
EXAMPLE_MERGE_D = 30               # D used in the stutter example
EXAMPLE_WINDOW = 750               # frames shown (15 s)

# ----------------------------------------------------------------------------- style
C = {"ms": "#2a78d6", "bl": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "gray": "#8a8984", "light": "#c9c8c3", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e1"}
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 13,
    "axes.edgecolor": C["light"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white", "text.color": C["ink"],
})


def esc(s):
    """Escape '$' so matplotlib does not read glosses like $GEST-OFF as math."""
    return str(s).replace("$", r"\$")


def save(fig, name, pdf=False):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p)
    if pdf:
        fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("  saved", p)


def need(*paths):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("missing " + ", ".join(miss))


# ----------------------------------------------------------------------------- data helpers
def decoder_kind(s):
    if s == "argmax":
        return "argmax"
    if s.startswith("argmax +merge"):
        return "merge"
    if s.startswith("thr"):
        return "thr"
    if s.startswith("tuned") and "merge" in s:
        return "tuned_merge"
    if s.startswith("tuned"):
        return "tuned"
    return "other"


_METRICS = None


def metrics():
    global _METRICS
    if _METRICS is None:
        need(*PATHS["metrics"])
        d = pd.concat([pd.read_csv(p, dtype={"prefix": str}) for p in PATHS["metrics"]], ignore_index=True)
        d["kind"] = d["decoder"].map(decoder_kind)
        d["cfg"] = d["model"] if "model" in d else d["run"].str.rsplit("-", n=1).str[0]
        d["cfg"] = d["cfg"] + " | " + d["description"].fillna("").str.replace(r"\s*\[seed=\d+\]", "", regex=True)
        _METRICS = d
    return _METRICS


def vals(run_list, kind, col):
    d = metrics()
    v = d[(d["kind"] == kind) & d["run"].isin(run_list)][col].astype(float).values
    if len(v) == 0:
        raise ValueError(f"no rows for {kind}/{col} in runs {run_list[:2]}...")
    return v


def summary_fa(path):
    """Parse 'N false alarms over R runs (X per run)' and the weighted category shares."""
    txt = open(path, encoding="utf-8").read()
    per_run = float(re.search(r"\((\d+) per run\)", txt).group(1))
    line = re.search(r"all false alarms \(weighted by runs\):(.*)", txt).group(1)
    shares = {m.group(1).strip(): float(m.group(2)) / 100 for m in re.finditer(r"([^,]+?) (\d+)%", line)}
    recall = float(re.search(r"mean single-run recall@5: ([\d.]+)", txt).group(1))
    n_gold = int(re.search(r"(\d+) gold phrase starts", txt).group(1))
    return per_run, shares, recall * n_gold


def bar_labels(ax, bars, fmt="{:.3f}", inside=False, color=None, dy=0.004):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (-dy * 3 if inside else dy), fmt.format(h),
                ha="center", va="top" if inside else "bottom", fontsize=11, color=color or C["ink2"])


# ============================================================================= figures
def fig01(pdf):
    """Task schematic: continuous signing -> per-frame O/I/B labels -> phrase segments."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    T = 100
    phrases = [(8, 38), (45, 70), (78, 95)]
    signs = [(9, 15), (17, 24), (26, 31), (33, 38), (46, 52), (55, 62), (64, 70), (79, 86), (88, 95)]
    # row 1: signs
    for a, b in signs:
        ax.add_patch(plt.Rectangle((a, 2.2), b - a, 0.55, color=C["light"], lw=0))
    ax.text(-1, 2.47, "signs", ha="right", va="center", color=C["ink2"])
    # row 2: phrases
    for i, (a, b) in enumerate(phrases):
        ax.add_patch(plt.Rectangle((a, 1.2), b - a, 0.6, color=C["ms"], alpha=0.25, lw=0))
        ax.text((a + b) / 2, 1.5, f"phrase {i + 1}", ha="center", va="center", fontsize=11)
    ax.text(-1, 1.5, "phrases", ha="right", va="center", color=C["ink2"])
    # row 3: BIO
    lab = np.zeros(T, int)
    for a, b in phrases:
        lab[a:b] = 1
        lab[a] = 2
    colors = {0: "#f1f0ec", 1: "#cde2fb", 2: C["bl"]}
    for t in range(T):
        ax.add_patch(plt.Rectangle((t, 0.3), 1, 0.55, color=colors[lab[t]], lw=0))
    ax.text(-1, 0.57, "frame labels", ha="right", va="center", color=C["ink2"])
    for a, _ in phrases:
        ax.annotate("B", (a + 0.5, 0.3), (a + 0.5, -0.25), ha="center", color=C["bl"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C["bl"], lw=1))
    ax.text(50, -0.75, "O = outside   I = inside a phrase   B = first frame of a phrase (≈1% of frames)",
            ha="center", va="center", fontsize=11, color=C["ink2"])
    ax.set_xlim(-14, T + 1)
    ax.set_ylim(-1.0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "fig01_task_schematic", pdf)


def fig02(pdf):
    """2023 recipe vs your recipe (same 2023 model): Frame F1 and boundary F1 under three decoders."""
    rec = GROUPS["recipe"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 2.6]})
    names = list(rec)
    cols = [C["bl"], C["ms"]]
    f1 = [vals(rec[n], "argmax", "frame_f1").mean() for n in names]
    b = a1.bar(names, f1, color=cols, width=0.55)
    bar_labels(a1, b)
    a1.set_ylim(0, 0.8)
    a1.set_title("Frame F1")
    decs = [("argmax", "raw argmax"), ("merge", "argmax + merge"), ("tuned", "tuned thresholds")]
    x = np.arange(len(decs))
    w = 0.36
    for i, n in enumerate(names):
        v = [vals(rec[n], k, "bnd_f1_5").mean() for k, _ in decs]
        ratio = [vals(rec[n], k, "segment_pct").mean() for k, _ in decs]
        bb = a2.bar(x + (i - 0.5) * (w + 0.02), v, w, color=cols[i], label=n)
        bar_labels(a2, bb)
        for xb, r in zip(x + (i - 0.5) * (w + 0.02), ratio):
            a2.text(xb, 0.012, f"×{r:.1f}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    a2.set_xticks(x, [d for _, d in decs])
    a2.set_ylim(0, 0.55)
    a2.set_title("Boundary F1 ±5 frames")
    a2.legend(loc="upper left")
    a2.text(0.99, 0.97, "white numbers = predicted / gold segments", transform=a2.transAxes, ha="right",
            va="top", fontsize=10, color=C["ink2"])
    fig.suptitle("Same 2023 model: the training recipe costs .05 Frame F1; most of the boundary gap is stutter",
                 fontsize=14, x=0.01, ha="left", y=1.05)
    save(fig, "fig02_recipe_decoders", pdf)


def fig03(pdf):
    """Multistream vs 2023 model per input combination: Frame F1 vs boundary F1 (merge)."""
    grid = GROUPS["grid"]
    rows = list(grid)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, (kind, col, title, xl) in zip(axes, [("argmax", "frame_f1", "Frame F1 (decoder-free)", (0.655, 0.695)),
                                                 ("merge", "bnd_f1_5", "Boundary F1 ±5 (argmax + merge)", (0.37, 0.53))]):
        for y, r in enumerate(rows):
            m, b = vals(grid[r][0], kind, col).mean(), vals(grid[r][1], kind, col).mean()
            ax.plot([b, m], [y, y], color=C["light"], lw=3, zorder=1)
            ax.scatter([b], [y], s=90, color=C["bl"], zorder=2, label="2023 model (early fusion)" if y == 0 else None)
            ax.scatter([m], [y], s=90, color=C["ms"], zorder=3, label="Multistream" if y == 0 else None)
            ax.text(max(m, b) + (xl[1] - xl[0]) * 0.02, y, f"{m - b:+.3f}", va="center", fontsize=11,
                    color=C["ink"], fontweight="bold" if abs(m - b) >= 0.03 else "normal")
        ax.set_xlim(xl[0], xl[1] + (xl[1] - xl[0]) * 0.12)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), rows)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 1.04), ncol=2)
    fig.suptitle("The multistream gain is ~+.01 on Frame F1 but +.04 to +.07 on boundaries",
                 fontsize=14, x=0.01, ha="left", y=1.11)
    save(fig, "fig03_input_grid_gaps", pdf)


def fig04(pdf):
    """Mixing ablation (body only): temporal conv, not the graph, carries the gain."""
    mix = GROUPS["mixing"]
    fam_col = {"2023": C["bl"], "none": C["gray"], "temporal": C["ms"]}
    labels = list(mix)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (kind, col, title) in zip(axes, [("argmax", "frame_f1", "Frame F1"),
                                             ("merge", "bnd_f1_5", "Boundary F1 ±5 (argmax + merge)")]):
        for i, l in enumerate(labels):
            v = vals(mix[l][0], kind, col)
            c = fam_col[mix[l][1]]
            ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=40, color=c, alpha=0.45, lw=0)
            ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=c, lw=3)
            ax.text(i, v.max() + 0.0025 if col == "frame_f1" else v.max() + 0.006, f"{v.mean():.3f}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(labels)), labels, fontsize=11)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, len(labels) - 0.5)
    axes[0].set_ylim(0.66, 0.685)
    axes[1].set_ylim(0.39, 0.49)
    fig.suptitle("Body only: removing the temporal convolution drops the multistream model to the 2023 level; "
                 "removing the graph costs nothing", fontsize=13, x=0.01, ha="left", y=1.04)
    fig.text(0.01, -0.08, "dots = seeds, line = mean;  gray = no temporal mixing, blue = with temporal "
             "convolution, orange = 2023 model", fontsize=10, color=C["ink2"])
    save(fig, "fig04_mixing_ablation", pdf)


def fig05(pdf):
    """Boundary F1 vs tolerance for each decoder (competitive runs)."""
    d = metrics()
    a = d[d["kind"] == "argmax"]
    comp = set(a[a["frame_f1"] >= 0.665]["run"])
    tols = [1, 2, 5, 10, 25]
    decs = [("argmax", "raw argmax", C["gray"]), ("tuned", "tuned thresholds (2023 style)", C["bl"]),
            ("merge", "argmax + merge (duplicate removal)", C["ms"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, lab, c in decs:
        s = d[(d["kind"] == k) & d["run"].isin(comp)]
        y = [s[f"bnd_f1_{t}"].mean() for t in tols]
        ax.plot(range(len(tols)), y, marker="o", ms=8, lw=2, color=c, label=lab)
        if k != "tuned":     # value labels: merge above its line, raw argmax below
            off, va = (0.02, "bottom") if k == "merge" else (-0.02, "top")
            for i in (0, 2):
                ax.text(i, y[i] + off, f"{y[i]:.2f}", ha="center", va=va, fontsize=11, color=C["ink"])
    ax.set_xticks(range(len(tols)), [f"±{t}\n({t / FPS:.2f} s)" for t in tols])
    ax.set_xlabel("tolerance (frames)")
    ax.set_ylabel("boundary F1")
    ax.set_ylim(0, 0.7)
    ax.set_xlim(-0.3, len(tols) - 0.7)
    ax.legend(loc="upper left")
    ax.set_title(f"Removing duplicate starts nearly doubles boundary F1 ({len(comp)} competitive runs)",
                 loc="left")
    save(fig, "fig05_decoder_tolerance", pdf)


def fig06(pdf):
    """Real example: P(B), gold starts, raw argmax starts, merged starts."""
    sys.path.insert(0, os.getcwd())
    from src.metrics import extract_segments, merge_close_starts
    run = GROUPS["example_run"]
    npz = os.path.join(PATHS["predictions"], run, "test.npz")
    need(npz)
    z = np.load(npz)
    best = None
    for v in z.files:
        lp = os.path.join(PATHS["labels"], f"{v}.npy")
        if not os.path.exists(lp):
            continue
        p = z[v].astype(np.float32)
        p = p if p.shape[0] == 3 else p.T                  # (3, T)
        gold = np.load(lp)
        raw = extract_segments(p.argmax(0))
        mer = merge_close_starts(raw, p[2], EXAMPLE_MERGE_D)
        rs = np.array([s for s, _ in raw])
        ms_ = np.array([s for s, _ in mer])
        gs = np.array([s for s, _ in extract_segments(gold)])
        T = p.shape[1]
        for t0 in range(0, max(1, T - EXAMPLE_WINDOW), EXAMPLE_WINDOW // 5):
            t1 = t0 + EXAMPLE_WINDOW
            ng = ((gs >= t0) & (gs < t1)).sum()
            removed = ((rs >= t0) & (rs < t1)).sum() - ((ms_ >= t0) & (ms_ < t1)).sum()
            score = removed if ng >= 3 else -1
            if best is None or score > best[0]:
                best = (score, v, t0, p, gold, rs, ms_, gs)
    if best is None:
        raise FileNotFoundError(f"no labels for videos in {npz} under {PATHS['labels']}")
    _, v, t0, p, gold, rs, ms_, gs = best
    t1 = t0 + EXAMPLE_WINDOW
    t = np.arange(t0, min(t1, p.shape[1]))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    inside = gold[t] != 0
    ax.fill_between(t / FPS, 0, 1.0, where=inside, color=C["ms"], alpha=0.10, lw=0, step="mid",
                    label="gold phrase")
    ax.plot(t / FPS, p[2, t], color=C["ink2"], lw=1.4, label="P(B)")
    sel = lambda a: a[(a >= t0) & (a < t1)]
    for x in sel(gs):
        ax.axvline(x / FPS, color=C["ink"], lw=1.2, ls="--")
    ax.vlines(sel(rs) / FPS, 1.05, 1.17, color=C["gray"], lw=2)
    ax.vlines(sel(ms_) / FPS, 1.22, 1.34, color=C["ms"], lw=2.5)
    ax.text(t0 / FPS, 1.11, "raw argmax starts  ", ha="right", va="center", fontsize=11, color=C["ink2"])
    ax.text(t0 / FPS, 1.28, f"after merge (D={EXAMPLE_MERGE_D})  ", ha="right", va="center", fontsize=11,
            color=C["ms"])
    ax.plot([], [], color=C["ink"], ls="--", label="gold phrase start")
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(t0 / FPS, t1 / FPS)
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", bbox_to_anchor=(1, -0.13), ncol=3)
    ax.set_title(f"{v}, run {run}: {len(sel(rs))} raw starts → {len(sel(ms_))} after merging "
                 f"({len(sel(gs))} gold)", loc="left")
    save(fig, "fig06_stutter_example", pdf)


def fig07(pdf):
    """IoU blind spot: IoU vs boundary F1 per configuration."""
    d = metrics()
    d = d[d["seed"].notna()]                      # same run set as metric_analysis.py
    a = d[d["kind"] == "argmax"].groupby("cfg").agg(iou=("iou", "mean"), f1=("frame_f1", "mean"),
                                                    ratio=("segment_pct", "mean"), n=("run", "count"))
    m = d[d["kind"] == "merge"].groupby("cfg")["bnd_f1_5"].mean()
    a = a.join(m.rename("bnd")).dropna()
    a = a[a["f1"] >= 0.6]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    col = [C["bl"] if c.startswith(BL) else C["ms"] for c in a.index]
    ax.scatter(a["bnd"], a["iou"], s=60, c=col, alpha=0.8, lw=0)
    ax.scatter([], [], color=C["ms"], label="multistream configs")
    ax.scatter([], [], color=C["bl"], label="2023-model configs")
    top = a["iou"].idxmax()
    ax.annotate(f"IoU #1 ({a.loc[top, 'iou']:.2f})\nbut boundary F1 only {a.loc[top, 'bnd']:.2f}", (a.loc[top, "bnd"], a.loc[top, "iou"]),
                (a.loc[top, "bnd"] + 0.03, a.loc[top, "iou"] - 0.005), fontsize=11,
                arrowprops=dict(arrowstyle="-", color=C["ink2"]))
    over = a["ratio"].idxmax()
    ax.annotate(f"×{a.loc[over, 'ratio']:.0f} too many segments\nstill IoU {a.loc[over, 'iou']:.2f}",
                (a.loc[over, "bnd"], a.loc[over, "iou"]), (a.loc[over, "bnd"] + 0.025, a.loc[over, "iou"] - 0.045),
                fontsize=11, arrowprops=dict(arrowstyle="-", color=C["ink2"]))
    ax.set_xlabel("boundary F1 ±5 (argmax + merge)")
    ax.set_ylabel("IoU (argmax)")
    ax.legend(loc="lower right")
    ax.set_title("IoU does not tell you whether boundaries are right", loc="left")
    save(fig, "fig07_iou_blindspot", pdf)


METRIC_NAMES = [("frame_f1", "Frame F1"), ("f1_B", "B-class F1"), ("iou", "IoU"),
                ("segment_f1_05", "Segment F1@0.5"), ("bnd_f1_1", "Boundary F1 ±1"),
                ("bnd_f1_2", "Boundary F1 ±2"), ("bnd_f1_5", "Boundary F1 ±5"),
                ("bnd_f1_10", "Boundary F1 ±10"), ("bnd_f1_25", "Boundary F1 ±25")]


def fig08(pdf):
    """Metric reliability: val->test ranking agreement and ICC per decoder."""
    js = {}
    for k in ("argmax", "argmax_merge", "tuned"):
        p = PATHS["metric_json"].format(k)
        if os.path.exists(p):
            js[k] = json.load(open(p))
    need(PATHS["metric_json"].format("argmax"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    y = np.arange(len(METRIC_NAMES))[::-1]
    vt = js["argmax"].get("val_test_spearman", {})
    v = [vt.get(m, np.nan) for m, _ in METRIC_NAMES]
    a1.barh(y, v, height=0.6, color=C["gray"])          # argmax decoder only (val has no merge rows)
    for yi, vi, (m, _) in zip(y, v, METRIC_NAMES):
        a1.text(vi + 0.01, yi, f"{vi:.2f}", va="center", fontsize=11, fontweight="bold" if m == "iou" else "normal")
    a1.set_yticks(y, [n for _, n in METRIC_NAMES])
    a1.set_xlim(0, 1)
    a1.set_title("Val → test ranking agreement (ρ, raw argmax)")
    a1.grid(axis="y", visible=False)
    decs = [("argmax", "raw argmax", C["gray"]), ("argmax_merge", "argmax + merge", C["ms"]),
            ("tuned", "tuned thresholds", C["bl"])]
    h = 0.26
    for i, (k, lab, c) in enumerate(decs):
        if k not in js:
            continue
        rel = js[k]["test_reliability"]
        a2.barh(y + (1 - i) * h, [rel.get(m, {}).get("icc1", np.nan) for m, _ in METRIC_NAMES], height=h - 0.03,
                color=c, label=lab)
    a2.set_xlim(0, 1)
    a2.set_title("Reliability across seeds (ICC, test)")
    a2.legend(loc="lower left", bbox_to_anchor=(0, -0.22), ncol=3, fontsize=11)
    a2.grid(axis="y", visible=False)
    n = len(js["argmax"].get("test_reliability", {}))
    fig.suptitle("All metrics are reliable across seeds; val→test agreement is only moderate; "
                 "tuned thresholds add noise", fontsize=13, x=0.01, ha="left", y=1.03)
    save(fig, "fig08_metric_reliability", pdf)


def fig09(pdf):
    """Error consistency: share of runs detecting each gold start vs independent errors."""
    g = os.path.join(PATHS["err_merge"], "gold_boundaries_test.csv")
    r = os.path.join(PATHS["err_merge"], "runs_test.csv")
    need(g, r)
    gb = pd.read_csv(g)
    R = len(pd.read_csv(r))
    rate = gb["detection_rate"].values
    bins = np.linspace(0, 1, 11)
    obs = np.histogram(np.clip(rate, 0, 0.9999), bins)[0] / len(rate)
    # independent errors: every boundary detected with the same probability (= mean recall)
    from math import comb
    pbar = rate.mean()
    k = np.arange(R + 1)
    pmf = np.array([comb(R, int(i)) * pbar ** i * (1 - pbar) ** (R - i) for i in k])
    exp = np.histogram(np.clip(k / R, 0, 0.9999), bins, weights=pmf)[0]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(10)
    ax.bar(x, obs, width=0.8, color=C["ms"], label=f"observed ({len(rate)} gold starts, {R} runs)")
    ax.plot(x, exp, marker="o", ms=7, lw=2, color=C["bl"], label="if errors were independent")
    ax.set_xticks(x, [f"{int(b * 100)}–{int(b * 100) + 10}%" for b in bins[:-1]], fontsize=10)
    ax.set_xlabel("share of runs that find the boundary")
    ax.set_ylabel("share of gold phrase starts")
    none, every = (rate == 0).mean(), (rate >= 0.9999).mean()
    ax.text(0, obs[0] + 0.01, f"{obs[0]:.0%}\n({none:.0%} found\nby no run)", ha="center", fontsize=11)
    ax.text(9, obs[9] + 0.01, f"{obs[9]:.0%}\n({every:.1%} found\nby every run)", ha="center", fontsize=11)
    ax.set_ylim(0, max(obs.max(), exp.max()) * 1.25)
    ax.legend(loc="upper center")
    ax.set_title("All models miss the same boundaries", loc="left")
    save(fig, "fig09_error_consistency", pdf)


def fig10(pdf):
    """Detection rate by the pause before the phrase."""
    g = os.path.join(PATHS["err_merge"], "gold_boundaries_test.csv")
    need(g)
    gb = pd.read_csv(g)
    gb = gb[gb["phrase_gap_before"] >= 0]
    edges = [0, 7, 13, 26, 51, 101, 10 ** 9]
    labels = ["0–6", "7–12", "13–25", "26–50", "51–100", ">100"]
    gb["bin"] = pd.cut(gb["phrase_gap_before"], edges, right=False, labels=labels)
    s = gb.groupby("bin", observed=False)["detection_rate"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    b = ax.bar(range(len(s)), s["mean"], width=0.65, color=C["ms"])
    for i, (m, n) in enumerate(zip(s["mean"], s["count"])):
        if n:
            ax.text(i, m + 0.015, f"{m:.2f}", ha="center", fontsize=11)
            ax.text(i, 0.015, f"n={n}", ha="center", fontsize=9, color="white")
    ax.set_xticks(range(len(s)), [f"{l}\n" for l in labels])
    ax.set_xlabel(f"pause before the phrase (frames, {FPS} fps)")
    ax.set_ylabel("mean share of runs detecting it")
    ax.set_ylim(0, 1)
    ax.set_title("Boundaries after short pauses are the ones every model misses", loc="left")
    save(fig, "fig10_detection_vs_pause", pdf)


def fig11(pdf):
    """Oracle analysis: what gold-annotation cues could achieve vs the models."""
    p = os.path.join(PATHS["oracle"], "oracle_summary.csv")
    need(p)
    o = pd.read_csv(p).set_index("group")
    order = [("gap only", "Oracle: pause length only", "oracle"),
             ("timing", "Oracle: sign timing", "oracle"),
             ("lexical", "Oracle: glosses (lexical)", "oracle"),
             ("all", "Oracle: timing + glosses + mouth", "oracle"),
             (None, None, None)]
    for k in o.index:
        if "(3 runs" in k or k.startswith(("MS", "BL2023")):
            fam = "ms" if k.startswith("MS") else "bl"
            lab = k.replace(" (3 runs, max P(B) +-5)", " alone").replace("BL2023", "2023 model") \
                   .replace("MS", "Multistream")
            if "mouth" in k:
                continue
            order.append((k, lab, fam))
    fig, ax = plt.subplots(figsize=(11, 6))
    y, ticks, yl = 0, [], []
    for k, lab, fam in order[::-1]:
        if k is None:
            y += 0.6
            continue
        if k not in o.index:
            continue
        f1 = o.loc[k, "f1"]
        col = {"oracle": C["gray"], "ms": C["ms"], "bl": C["bl"]}[fam]
        ax.barh(y, f1, height=0.65, color=col, alpha=1 if "alone" in lab or fam == "oracle" else 0.6)
        ax.text(f1 + 0.005, y, f"{f1:.3f}", va="center", fontsize=11)
        ticks.append(y)
        yl.append(lab)
        y += 1
    if "all" in o.index:
        ax.axvline(o.loc["all", "f1"], color=C["ink2"], ls="--", lw=1)
    ax.set_yticks(ticks, yl)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel(f"F1 for 'is this sign transition a phrase start?' (test, base rate {0.175:.3f})")
    ax.grid(axis="y", visible=False)
    ax.set_title("Even perfect knowledge of the signs reaches only F1 ≈ .66", loc="left")
    save(fig, "fig11_oracle", pdf)


FA_CATS = [("duplicate start near a gold start", "next to a real start\n(duplicate / near miss)", C["gray"]),
           ("within-phrase sign transition", "inside a phrase,\nat a sign transition", C["ms"]),
           ("between phrases", "in a gap\nbetween phrases", C["aqua"]),
           ("inside a sign / no transition", "inside a sign", C["yellow"])]


def fig12(pdf):
    """False alarms per run by type, before vs after duplicate removal."""
    p0 = os.path.join(PATHS["err_plain"], "summary_test.txt")
    p1 = os.path.join(PATHS["err_merge"], "summary_test.txt")
    need(p0, p1)
    rows = []
    for lab, p in (("raw argmax", p0), ("after merge", p1)):
        per_run, shares, correct = summary_fa(p)
        rows.append((lab, per_run, shares, correct))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for i, (lab, per_run, shares, correct) in enumerate(rows[::-1]):
        left = 0
        for key, name, c in FA_CATS:
            w = per_run * shares.get(key, 0)
            ax.barh(i, w, left=left, height=0.55, color=c, edgecolor="white", lw=2,
                    label=name if i == 0 else None)
            if w > 120:
                ax.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center", color="white", fontsize=11,
                        fontweight="bold")
            left += w
        ax.text(left + 30, i, f"{per_run:.0f} false starts  vs  {correct:.0f} correct", va="center", fontsize=11)
    ax.set_yticks([0, 1], [r[0] for r in rows[::-1]])
    ax.set_xlabel("false phrase starts per run (test set)")
    ax.set_xlim(0, rows[0][1] * 1.45)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=10)
    ax.set_title("Duplicate removal cuts false starts by ~77%; within-phrase transitions remain", loc="left")
    save(fig, "fig12_false_alarm_breakdown", pdf)


CONTEXTS = [("after palm-up ($GEST-OFF)", "gp", ["$GEST-OFF"]),
            ("after mouth-only ($ORAL)", "gp", ["$ORAL"]),
            ("after other gesture ($GEST)", "gp", ["$GEST"]),
            ("before ICH1 ('I')", "gc", ["ICH1"]),
            ("next to pointing ($INDEX1)", "both", ["$INDEX1"])]


def fig13(pdf):
    """Contexts where the annotation itself is split: gold boundary rate and consistent false alarms."""
    tp = os.path.join(PATHS["oracle"], "transitions_test.csv")
    fp = os.path.join(PATHS["err_merge"], "false_alarms_test.csv")
    need(tp, fp)
    norm = lambda s: str(s).rstrip("^*").rstrip("^*")
    t = pd.read_csv(tp)
    t["gp"], t["gc"] = t["gloss_prev"].map(norm), t["gloss_cur"].map(norm)
    fa = pd.read_csv(fp)
    fa["gp"], fa["gc"] = fa["gloss_prev"].map(norm), fa["gloss_cur"].map(norm)
    cfa = fa[(fa["where"] == "within-phrase sign transition") & (fa["share_of_runs"] >= 0.5)]
    within = t[t["y"] == 0]

    def mask(df, side, gl):
        if side == "both":
            return df["gp"].isin(gl) | df["gc"].isin(gl)
        return df[side].isin(gl)

    rows = [("all transitions", t["y"].mean(), len(cfa) / len(within), len(within))]
    for name, side, gl in CONTEXTS:
        mt, mw, mf = mask(t, side, gl), mask(within, side, gl), mask(cfa, side, gl)
        rows.append((name, t[mt]["y"].mean(), mf.sum() / max(mw.sum(), 1), mw.sum()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, j, title, xmax in ((a1, 1, "Gold: share followed by a phrase boundary", 1.0),
                               (a2, 2, "Model: share of within-phrase cases that are\n"
                                       "consistent false alarms (≥50% of runs)", 0.45)):
        v = [r[j] for r in rows]
        ax.barh(y, v, height=0.6, color=[C["gray"]] + [C["ms"]] * (len(rows) - 1))
        for yi, vi in zip(y, v):
            ax.text(vi + xmax * 0.01, yi, f"{vi:.0%}", va="center", fontsize=11)
        ax.axvline(rows[0][j], color=C["ink2"], ls="--", lw=1)
        ax.set_xlim(0, xmax)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y, [esc(f"{r[0]}\n(n within = {r[3]})") for r in rows], fontsize=10)
    fig.suptitle("Where the labels go both ways, the model guesses wrong both ways", fontsize=14, x=0.01,
                 ha="left", y=1.05)
    save(fig, "fig13_ambiguous_contexts", pdf)


def fig14(pdf):
    """Backup: strongest gloss weights of the lexical oracle."""
    p = os.path.join(PATHS["oracle"], "oracle_lexical_weights.csv")
    need(p)
    w = pd.read_csv(p)
    w = w[~w["feature"].str.contains("_cat=")]
    w["name"] = w["feature"].str.replace("lexical:gloss_", "", regex=False) \
        .str.replace("cur=", "phrase starts with ", regex=False) \
        .str.replace("prev2=", "two signs back: ", regex=False) \
        .str.replace("prev=", "previous sign ", regex=False)
    top = pd.concat([w.nlargest(12, "weight"), w.nsmallest(12, "weight")]).sort_values("weight")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top["weight"], height=0.65,
            color=[C["ms"] if v > 0 else C["gray"] for v in top["weight"]])
    ax.set_yticks(range(len(top)), [esc(n) for n in top["name"]], fontsize=10)
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.set_xlabel("logistic-regression weight  (→ boundary   ← within phrase)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Connectives (ABER, WENN, DANN, UND) open phrases; right after one, a phrase rarely breaks",
                 loc="left", fontsize=13)
    save(fig, "fig14_lexical_weights", pdf)


FIGS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08, 9: fig09,
        10: fig10, 11: fig11, 12: fig12, 13: fig13, 14: fig14}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="*", help="figure numbers to make (default: all)")
    ap.add_argument("--pdf", action="store_true", help="also save PDF versions")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    globals()["OUT"] = args.out
    for n in args.only or sorted(FIGS):
        print(f"figure {n}: {FIGS[n].__doc__.strip().splitlines()[0]}")
        try:
            FIGS[n](args.pdf)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"  skipped: {e}")


if __name__ == "__main__":
    main()