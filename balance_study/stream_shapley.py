"""
balance_study/stream_shapley.py -- what does each stream contribute to a trained multi-stream model?

For every coalition of streams (the other streams replaced by their mean over the split, the same ablation as
check_stream_balance.py) the run predicts the whole split. From these predictions:
  1. coalition table: Frame F1, B-class F1, IoU, % and boundary F1@5 (argmax decoder) of every coalition,
     including "none" (all streams at their mean);
  2. Wu et al.'s conditional utilization rate u(m | rest) = (M(all) - M(all without m)) / M(all), for Frame F1
     and boundary F1@5 (single-stream ablation underrates partly redundant streams -- body and HaMeR both
     see the hands -- which is why the Shapley values below are reported too);
  3. per-frame Shapley attribution of the B log-odds, log p(B) - log(1 - p(B)), to each stream, with the
     all-streams-at-mean prediction as the baseline (AGM's mono-modal responses / Dr. SHAP-AV's global SHAP):
     mean |phi| share, mean phi on gold O / I / B frames, and phi event-locked around gold phrase starts.

Usage (repo root):
    python balance_study/stream_shapley.py --runs multistream_bilstm-373 multistream_bilstm-428 --split test
    python balance_study/stream_shapley.py --runs multistream_bilstm-373 --split val --window 50 --plot
Outputs: evaluation_phrase/balance/shapley/<run>_<split>_{coalitions.csv,eventlocked.csv,summary.json}
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_phrase import get_dataset, EXP_DIR, MODEL_DIR  # noqa: E402
from src.model_factory import build_model_kwargs  # noqa: E402
from src.models import MultiStreamSegmenter  # noqa: E402
from src.evaluation import predict_split  # noqa: E402
from src.metrics import evaluate_videos  # noqa: E402
from balance_study import core  # noqa: E402

OUT_DIR = os.path.join("evaluation_phrase", "balance", "shapley")
SHORT = {"body_hands": "Body", "face": "Face", "hamer": "HaMeR", "cues": "Cues"}


def coalition_name(streams, members):
    return "+".join(SHORT.get(streams[i], streams[i]) for i in members) if members else "none"


def load_run(run, split, device):
    with open(os.path.join(EXP_DIR, run, "hyperparameters.json")) as f:
        config = json.load(f)
    dataset = get_dataset(config, split)
    cls, kw = build_model_kwargs(config, detected_hamer_dim=dataset.detected_hamer_dim,
                                 detected_dinov2_dim=dataset.detected_dinov2_dim,
                                 detected_cue_dim=getattr(dataset, "detected_cue_dim", None))
    model = cls(**kw).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{run}.pth"), map_location=device), strict=True)
    model.eval()
    if not isinstance(model, MultiStreamSegmenter) or model.fusion != "concat":
        raise SystemExit(f"{run}: needs a multi-stream model with concat fusion.")
    return config, dataset, model


def stream_means(model, config, dataset, device, batch_size):
    """Mean branch output of every stream over the split (also returns the full-model probabilities)."""
    sums = {s: None for s in model.streams}
    count = [0]

    def capture(z, contrib):
        for s, v in z.items():
            v = v.detach().float().reshape(-1, v.shape[-1])
            sums[s] = v.sum(0) if sums[s] is None else sums[s] + v.sum(0)
        count[0] += next(iter(z.values())).shape[0] * next(iter(z.values())).shape[1]

    model.set_capture(capture)
    try:
        probs, gold = predict_split(model, config, dataset, device, batch_size)
    finally:
        model.set_capture(None)
    return {s: sums[s] / max(count[0], 1) for s in model.streams}, probs, gold


def b_logodds(p):
    pb = np.clip(p[2].astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(pb) - np.log1p(-pb)


def analyse_run(run, split, device, batch_size, window, plot):
    config, dataset, model = load_run(run, split, device)
    streams = list(model.streams)
    M = len(streams)
    print(f"\n=== {run} ({split}): streams {streams} -- {config.get('description', '')}")
    means, full_probs, gold = stream_means(model, config, dataset, device, batch_size)

    coal_list = [()] + core.coalitions(M)                # empty first, full last
    probs = {}
    for c in coal_list:
        if len(c) == M:
            probs[c] = full_probs
            continue
        model.set_ablations([(s, means[s]) for i, s in enumerate(streams) if i not in c])
        try:
            probs[c], _ = predict_split(model, config, dataset, device, batch_size)
        finally:
            model.set_ablations(None)
        print(f"  coalition {coalition_name(streams, c):18s} predicted")

    rows = []
    for c in coal_list:
        m = evaluate_videos(probs[c], gold, decoder="argmax")
        rows.append({"run": run, "split": split, "coalition": coalition_name(streams, c), "size": len(c),
                     "Frame_F1": m["Frame_F1"], "F1_B": m["F1_B"], "IoU": m["IoU"], "Pct": m["Pct"],
                     "BndF1_5": m["BndF1_5"], "Bnd_n_pred": m["Bnd_n_pred"]})
    by = {r["coalition"]: r for r in rows}
    full = by[coalition_name(streams, tuple(range(M)))]
    util = {}
    for i, s in enumerate(streams):
        rest = by[coalition_name(streams, tuple(j for j in range(M) if j != i))]
        util[s] = {k: (full[k] - rest[k]) / full[k] if full[k] else float("nan") for k in ("Frame_F1", "BndF1_5")}

    # per-frame Shapley attribution of the B log-odds (baseline = no stream)
    coals, Wm = core.shapley_matrix(M)
    lags = np.arange(-window, window + 1)
    el_sum = np.zeros((M + 1, len(lags)))
    el_n = np.zeros(len(lags))
    abs_sum = np.zeros(M)
    by_class = {k: np.zeros(M) for k in (0, 1, 2)}
    by_class_n = {k: 0 for k in (0, 1, 2)}
    for vid in full_probs:
        g = np.asarray(gold[vid])
        T = min(len(g), full_probs[vid].shape[1])
        g = g[:T]
        v0 = b_logodds(probs[()][vid][:, :T])
        V = np.stack([b_logodds(probs[c][vid][:, :T]) - v0 for c in coals])      # (C, T)
        phi = Wm @ V                                                               # (M, T)
        abs_sum += np.abs(phi).sum(1)
        for k in (0, 1, 2):
            mk = g == k
            by_class[k] += phi[:, mk].sum(1)
            by_class_n[k] += int(mk.sum())
        starts = np.flatnonzero(g == 2)
        total = V[-1]                                                              # full minus baseline
        for s0 in starts:
            idx = s0 + lags
            ok = (idx >= 0) & (idx < T)
            el_sum[:M, ok] += phi[:, idx[ok]]
            el_sum[M, ok] += total[idx[ok]]
            el_n[ok] += 1
    share = abs_sum / max(abs_sum.sum(), 1e-12)
    summary = {
        "run": run, "split": split, "streams": streams, "description": config.get("description", ""),
        "coalitions": rows,
        "conditional_utilization": util,
        "abs_shapley_share": {s: float(share[i]) for i, s in enumerate(streams)},
        "mean_shapley_by_gold_class": {name: {s: float(by_class[k][i] / max(by_class_n[k], 1))
                                              for i, s in enumerate(streams)}
                                       for k, name in ((0, "O"), (1, "I"), (2, "B"))},
        "phi_at_start_minus_mean_inside": {
            s: float(el_sum[i, window] / max(el_n[window], 1) - by_class[1][i] / max(by_class_n[1], 1))
            for i, s in enumerate(streams)},
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, f"{run}_{split}")
    with open(base + "_coalitions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(base + "_eventlocked.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag_frames"] + [f"phi_{s}" for s in streams] + ["full_minus_baseline", "n_starts"])
        for j, lag in enumerate(lags):
            n = max(el_n[j], 1)
            w.writerow([int(lag)] + [round(float(el_sum[i, j] / n), 5) for i in range(M + 1)] + [int(el_n[j])])
    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  {'coalition':18s} {'Frame F1':>9s} {'B F1':>7s} {'IoU':>7s} {'%':>6s} {'BndF1@5':>8s}")
    for r in rows:
        print(f"  {r['coalition']:18s} {r['Frame_F1']:9.4f} {r['F1_B']:7.4f} {r['IoU']:7.4f} "
              f"{r['Pct']:6.2f} {r['BndF1_5']:8.4f}")
    print("  conditional utilization u(m | rest) [Frame F1 / BndF1@5]: " + ", ".join(
        f"{SHORT.get(s, s)} {util[s]['Frame_F1']:+.3f} / {util[s]['BndF1_5']:+.3f}" for s in streams))
    print("  |Shapley| share of the B log-odds: " + ", ".join(
        f"{SHORT.get(s, s)} {summary['abs_shapley_share'][s]:.3f}" for s in streams))
    print("  mean Shapley at gold B frames (log-odds): " + ", ".join(
        f"{SHORT.get(s, s)} {summary['mean_shapley_by_gold_class']['B'][s]:+.3f}" for s in streams))
    print(f"  saved {base}_*.csv/json")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        n = np.maximum(el_n, 1)
        for i, s in enumerate(streams):
            ax.plot(lags / 50.0, el_sum[i] / n, label=SHORT.get(s, s), lw=1.8)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("time from gold phrase start (s)")
        ax.set_ylabel("Shapley contribution to B log-odds")
        ax.set_title(f"{run} ({split})", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(base + "_eventlocked.png", dpi=150)
        plt.close(fig)
        print(f"  plot {base}_eventlocked.png")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="e.g. multistream_bilstm-373 multistream_bilstm-428")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--window", type=int, default=50, help="event-locked half width in frames (50 = 1 s)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.runs:
        analyse_run(run, args.split, device, args.batch_size, args.window, args.plot)


if __name__ == "__main__":
    main()
