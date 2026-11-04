"""
fusion_analysis.py -- what did each fusion model learn? Event-locked analysis of the internal
signals the fusion models expose (model._trace), on full videos of a split.

    python fusion_study/fusion_analysis.py --runs multistream_fusion-01 multistream_fusion-02 --split test
    python fusion_study/fusion_analysis.py --runs multistream_bilstm-NN    # F1a (gated_sum): prints its gates

Per run -> evaluation_phrase/fusion_analysis/<run>/
  traces_<split>.npz           per-video stitched signals (float16), for your own plots
  locked_<split>.csv           event-locked mean / SEM of every scalar signal, offsets -L..L frames, for
                               phrase starts (all, continuous, short_pause, long_pause), the control
                               (sign starts inside phrases, >= --exclusion frames from phrase boundaries)
                               and random inside-phrase frames
  summary_<split>.txt          per signal: value at the event (+-2 frames) per group, and
                               phrase start minus sign-start control (the boundary-specific part)
  locked_<split>.png           the profiles (if matplotlib is available)
Mode-specific
  frame_gate  signals gate_<stream>: does the model shift weight to the face AT phrase boundaries?
  lag_face    lag_<split>.csv: attention over offsets -K..K at all frames / phrase starts / control;
              signal lag_mean (expected offset, + = face information AFTER the frame is used)
  film_face   signals film_gamma / film_beta: how strongly the face modulates, over time
  late        signals pB_<stream>, late_weight_<stream>; per-stream predictions saved to
              evaluation_phrase/predictions/<run>_<stream>/ (for phrase_stats.py etc.) and
              complementarity_<split>.csv: per-stream recall@k, union recall, boundaries only one
              stream finds, pairwise Jaccard -- overall and per boundary type
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from evaluate_phrase import EXP_DIR, MODEL_DIR, get_dataset, labels_dir_for  # noqa: E402
from fusion_study import trace_core as tc  # noqa: E402
from src.evaluation import _prepare_window, video_windows  # noqa: E402
from src.metrics import extract_segments  # noqa: E402
from src.model_factory import build_model_kwargs, call_model  # noqa: E402

OUT_ROOT = os.path.join("evaluation_phrase", "fusion_analysis")
PRED_ROOT = os.path.join("evaluation_phrase", "predictions")


def _time_first(key, x):
    """Every trace -> (B, T, ...)."""
    if key == "stream_probs":            # (B, S, 3, T) -> (B, T, S, 3)
        return x.permute(0, 3, 1, 2)
    return x


@torch.no_grad()
def predict_with_traces(model, config, cached, device, batch_size):
    """src.evaluation.predict_video_probs + stitching of model._trace (same windows, averaged overlaps)."""
    window_size, overlap = config["window_size"], config.get("overlap", 0)
    k = max(1, int(config.get("temporal_downsample_factor", 1)))
    feats_all = cached["features"]
    T = feats_all.shape[0]
    target_T = window_size // k
    windows = video_windows(T, window_size, overlap)
    prob_sum = np.zeros((3, T))
    tr_sum, counts = {}, np.zeros(T)

    def side(name, ws):
        x = torch.stack([_prepare_window(cached[name], s, e, k, target_T, False) for s, e in ws])
        return torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)

    for b0 in range(0, len(windows), batch_size):
        ws = windows[b0:b0 + batch_size]
        feats = torch.stack([_prepare_window(feats_all, s, e, k, target_T, True) for s, e in ws])
        feats = torch.nan_to_num(feats.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        hamer = side("hamer_features", ws) if config.get("use_hamer_features") else None
        dinov2 = side("dinov2_features", ws) if config.get("use_dinov2_features") else None
        cues = side("cue_features", ws) if config.get("use_cue_features") else None
        logits = call_model(model, config, feats, hamer, dinov2, cues)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        traces = {key: _time_first(key, v.float()).cpu().numpy() for key, v in getattr(model, "_trace", {}).items()}
        for i, (s, e) in enumerate(ws):
            n = e - s
            p = np.repeat(probs[i], k, axis=1)[:, :n] if k > 1 else probs[i][:, :n]
            if p.shape[1] < n:
                p = np.concatenate([p, np.repeat(p[:, -1:], n - p.shape[1], axis=1)], axis=1)
            prob_sum[:, s:e] += p
            for key, v in traces.items():
                x = np.repeat(v[i], k, axis=0) if k > 1 else v[i]
                if x.shape[0] < n:
                    x = np.concatenate([x, np.repeat(x[-1:], n - x.shape[0], axis=0)], axis=0)
                x = x[:n]
                if key not in tr_sum:
                    tr_sum[key] = np.zeros((T,) + x.shape[1:])
                tr_sum[key][s:e] += x
            counts[s:e] += 1
    counts[counts == 0] = 1
    probs = (prob_sum / counts).astype(np.float32)
    traces = {key: (v / counts.reshape((-1,) + (1,) * (v.ndim - 1))).astype(np.float32) for key, v in tr_sum.items()}
    return probs, traces


def scalar_signals(model, probs, traces):
    """name -> (T,) for every scalar signal of this run."""
    out = {"P(B) model": probs[2]}
    streams = list(model.streams)
    if "gate" in traces:
        for i, s in enumerate(streams):
            out[f"gate_{s}"] = traces["gate"][:, i]
    if "lag_mean" in traces:
        out["lag_mean"] = traces["lag_mean"]
    for key in ("film_gamma", "film_beta"):
        if key in traces:
            out[key] = traces[key]
    if "stream_probs" in traces:
        for i, s in enumerate(streams):
            out[f"pB_{s}"] = traces["stream_probs"][:, i, 2]
            out[f"late_weight_{s}"] = traces["late_weight"][:, i]
    return out


def argmax_starts(p):
    return np.array([s for s, _ in extract_segments(p.argmax(0))], dtype=np.int64)


def analyse_run(run, split, args, device):
    with open(os.path.join(EXP_DIR, run, "hyperparameters.json")) as f:
        config = json.load(f)
    dataset = get_dataset(config, split)
    model_class, kwargs = build_model_kwargs(config, detected_hamer_dim=dataset.detected_hamer_dim,
                                             detected_dinov2_dim=dataset.detected_dinov2_dim,
                                             detected_cue_dim=getattr(dataset, "detected_cue_dim", None))
    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{run}.pth"), map_location=device), strict=True)
    model.eval()
    mode = getattr(model, "fusion_mode", config.get("fusion", "concat"))
    print(f"\n{run}: {config.get('description', '')}\n  fusion: {mode}")
    if mode == "gated_sum":
        print("  global stream gates: " + ", ".join(f"{k} {v:+.3f}" for k, v in model.stream_gates().items()))
    out_dir = os.path.join(OUT_ROOT, run)
    os.makedirs(out_dir, exist_ok=True)

    bs = args.batch_size or config.get("batch_size", 16)
    rng = np.random.default_rng(0)
    vids, sig_by_name, ev = [], {}, {g: [] for g in ("phrase_start",) + tc.PHRASE_TYPES + ("sign_start", "inside")}
    traces_store, lag_attn, per_stream_probs = {}, [], {}
    gold_starts_all, gold_types_all, stream_starts = [], [], {}
    missing_sign = 0
    for vid, cached in dataset.video_cache.items():
        probs, traces = predict_with_traces(model, config, cached, device, bs)
        gold = np.asarray(cached["hard_labels"]).astype(np.int64)
        sp = os.path.join(args.sign_labels, f"{vid}.npy")
        sign = np.load(sp).astype(np.int64)[:len(gold)] if os.path.exists(sp) else None
        missing_sign += sign is None
        vids.append(vid)
        for name, x in scalar_signals(model, probs, traces).items():
            sig_by_name.setdefault(name, []).append(x)
        events = tc.phrase_start_events(gold, args.short_gap)
        ev["phrase_start"].append([e for e, _ in events])
        for t in tc.PHRASE_TYPES:
            ev[t].append([e for e, ty in events if ty == t])
        ev["sign_start"].append(tc.sign_start_controls(gold, sign, args.exclusion) if sign is not None else [])
        ev["inside"].append(tc.inside_frames(gold, max(1, len(events)), rng, args.exclusion))
        for key, v in traces.items():
            traces_store[f"{vid}__{key}"] = v.astype(np.float16)
        traces_store[f"{vid}__probs"] = probs.astype(np.float16)
        if "lag_attn" in traces:
            lag_attn.append(traces["lag_attn"])
        if "stream_probs" in traces:
            for i, s in enumerate(model.streams):
                p = traces["stream_probs"][:, i, :].T                   # (3, T)
                per_stream_probs.setdefault(s, {})[vid] = p
                stream_starts.setdefault(s, []).append(argmax_starts(p))
            stream_starts.setdefault("fused", []).append(argmax_starts(probs))
            gold_starts_all.append(np.array([e for e, _ in events], dtype=np.int64))
            gold_types_all.append([ty for _, ty in events])
    if not vids:
        raise SystemExit(f"{run}: no {split} videos")
    if missing_sign:
        print(f"  note: no sign labels for {missing_sign} videos in {args.sign_labels} -- the sign-start control is smaller")
    np.savez_compressed(os.path.join(out_dir, f"traces_{split}.npz"), **traces_store)

    # ---- event-locked profiles
    L = args.half_window
    rows, summary = [], []
    for name, sigs in sig_by_name.items():
        at_event = {}
        for g, evs in ev.items():
            mean, sem, n = tc.locked_average(sigs, evs, L)
            for j, off in enumerate(range(-L, L + 1)):
                rows.append({"signal": name, "event": g, "offset": off, "mean": round(float(mean[j, 0]), 6),
                             "sem": round(float(sem[j, 0]), 6), "n": int(n[j])})
            c = slice(L - 2, L + 3)
            at_event[g] = float(np.nanmean(mean[c, 0])) if n[c].sum() else float("nan")
        diff = at_event["phrase_start"] - at_event["sign_start"]
        summary.append(f"  {name:22s} " + "  ".join(f"{g} {v:.4f}" for g, v in at_event.items())
                       + f"   | phrase start - sign start: {diff:+.4f}")
    with open(os.path.join(out_dir, f"locked_{split}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {g: sum(len(e) for e in evs) for g, evs in ev.items()}
    head = f"{run} [{split}] events: " + ", ".join(f"{g} {n}" for g, n in counts.items())
    text = [head, "value at the event (mean over offsets -2..+2), signals:"] + summary

    # ---- lag distributions
    if lag_attn:
        K = (lag_attn[0].shape[1] - 1) // 2
        lag_rows = []
        for g, evs in [("all frames", None)] + [(g, ev[g]) for g in ("phrase_start",) + tc.PHRASE_TYPES + ("sign_start", "inside")]:
            d, n = tc.lag_distribution(lag_attn, evs)
            if d is None:
                continue
            lag_rows.append({"frames": g, "n": n, "mean_offset": round(float((d * np.arange(-K, K + 1)).sum()), 3),
                             "argmax_offset": int(np.argmax(d) - K),
                             **{f"d{o:+d}": round(float(v), 5) for o, v in zip(range(-K, K + 1), d)}})
        with open(os.path.join(out_dir, f"lag_{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lag_rows[0]))
            w.writeheader()
            w.writerows(lag_rows)
        text.append("lag attention over offsets (+ = face information after the frame):")
        text += [f"  {r['frames']:14s} n {r['n']:7d}  mean offset {r['mean_offset']:+.2f}  argmax {r['argmax_offset']:+d}"
                 for r in lag_rows]

    # ---- late fusion: per-stream predictions + complementarity
    if per_stream_probs:
        labels_dir = labels_dir_for(config)
        for s, pv in per_stream_probs.items():
            d = os.path.join(PRED_ROOT, f"{run}_{s}")
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{split}.npz"), **{v: p.astype(np.float16) for v, p in pv.items()})
            with open(os.path.join(d, f"{split}_meta.json"), "w") as f:
                json.dump({"run": f"{run}_{s}", "split": split, "labels_dir": labels_dir, "seed": config.get("seed", ""),
                           "description": f"{config.get('description', run)} | stream head {s} [seed={config.get('seed', '')}]",
                           "merge_starts": {}}, f, indent=2)
        names = list(model.streams) + ["fused"]
        D = np.concatenate([tc.detection_matrix(g, [stream_starts[n][i] for n in names], args.k)
                            for i, g in enumerate(gold_starts_all)], axis=0)
        types = [t for ts in gold_types_all for t in ts]
        comp = tc.complementarity(D[:, :-1], list(model.streams), types)
        fused_rec = float(D[:, -1].mean()) if len(D) else float("nan")
        with open(os.path.join(out_dir, f"complementarity_{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "what", "value", "n"])
            w.writeheader()
            w.writerows(comp + [{"group": "all", "what": "recall fused", "value": fused_rec, "n": len(D)}])
        text.append(f"late fusion, gold phrase starts detected within +-{args.k} frames (argmax starts):")
        text += [f"  [{r['group']}] {r['what']:28s} {r['value']:.3f}  (n={r['n']})" for r in comp if r["group"] == "all"]
        text.append(f"  [all] recall fused                  {fused_rec:.3f}")
        text.append(f"  per-stream predictions saved as {', '.join(f'{run}_{s}' for s in model.streams)}")

    with open(os.path.join(out_dir, f"summary_{split}.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text))
    plot_profiles(rows, os.path.join(out_dir, f"locked_{split}.png"), run, L)


def plot_profiles(rows, path, run, L):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    signals = list(dict.fromkeys(r["signal"] for r in rows))
    groups = ["phrase_start", "continuous", "short_pause", "long_pause", "sign_start", "inside"]
    fig, axes = plt.subplots(len(signals), 1, figsize=(7, 2.2 * len(signals)), squeeze=False)
    for ax, s in zip(axes[:, 0], signals):
        for g in groups:
            rs = [r for r in rows if r["signal"] == s and r["event"] == g and r["n"] > 0]
            if not rs:
                continue
            x = np.array([r["offset"] for r in rs])
            m = np.array([r["mean"] for r in rs])
            e = np.array([r["sem"] for r in rs])
            ls = "--" if g in ("sign_start", "inside") else "-"
            ax.plot(x, m, ls, label=g, lw=1.2)
            ax.fill_between(x, m - e, m + e, alpha=0.15)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(s, fontsize=9)
        ax.set_xlim(-L, L)
    axes[0, 0].legend(fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("frames from the event (50 fps)")
    fig.suptitle(run, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Event-locked analysis of the fusion models' internal signals.")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sign-labels", default=os.path.join("processed_data", "BIO_tags"),
                    help="gloss (sign) BIO labels for the sign-start control")
    ap.add_argument("--half-window", type=int, default=50, help="offsets -L..L frames around each event")
    ap.add_argument("--exclusion", type=int, default=10, help="control events this far from phrase boundaries")
    ap.add_argument("--short-gap", type=int, default=12)
    ap.add_argument("--k", type=int, default=5, help="tolerance for the complementarity analysis")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.runs:
        analyse_run(run, args.split, args, device)


if __name__ == "__main__":
    main()
