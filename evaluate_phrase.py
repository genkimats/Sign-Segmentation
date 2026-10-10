"""
evaluate_phrase.py -- evaluates every saved run of ONE model on a split (default: val)
with the 2023-style full-video metrics: Frame F1, IoU and % of segments
(plus Segment F1@0.5).

Usage (from the repo root, same place you run train_phrase.py):
    python evaluate_phrase.py                       # asks which model, evaluates all its prefixes
    python evaluate_phrase.py --model stgcn_bilstm  # all stgcn_bilstm-XX runs
    python evaluate_phrase.py --model stgcn_bilstm --prefixes 10 12
    python evaluate_phrase.py --model stgcn_bilstm --sweep   # also tune decoding thresholds on this split

What it guarantees:
  - Each run is rebuilt from ITS OWN experiments_phrase/<run>/hyperparameters.json via the
    same factory train_phrase.py uses (window_size, overlap, d_model, n_layers, nhead,
    dim_feedforward, mamba/latent settings, hamer/dinov2 dims, features, downsampling ...).
    The checkpoint is loaded with strict=True, so any mismatch fails loudly instead of
    silently evaluating a different model.
  - Inference uses the same windows as training, stitched per full video.
  - Ground truth = raw hard labels at the raw frame rate. tolerance_window is ignored for
    evaluation (the dataset is built with tolerance_window=1 and hard labels are used anyway).

Only phrase runs are looked up: experiments_phrase/ and saved_models_phrase/ (an older,
same-named run elsewhere in the repo is never picked up).
"""
import argparse
import csv
import json
import os
import re
import time

import numpy as np
import torch

from src.dataset import SignSegmentationDataset
from src.model_factory import build_model_kwargs
from src.evaluation import predict_split
from src.metrics import evaluate_videos, pct_score, extract_segments, decode_threshold_2023, segment_iou
from src.metrics import boundary_type_analysis, BOUNDARY_TYPES, BOUNDARY_TYPE_LABELS

EXP_DIR = "experiments_phrase"
MODEL_DIR = "saved_models_phrase"
OUT_DIR = "evaluation_phrase"
KEYPOINTS_DIR = "processed_data/keypoints"
DEFAULT_LABELS_DIR = "processed_data/BIO_tags_phrase"  # v1; used when a run's config has no phrase_labels_dir
LABELS_OVERRIDE = None  # set by --labels-dir: evaluate every run against these labels instead
SPLIT_FILE = "dataset_splits.json"

RUN_RE = re.compile(r"^(?P<model>.+)-(?P<prefix>\d+)$")


# ==============================================================================
# Run discovery
# ==============================================================================
def discover_runs(exp_dir, model_dir):
    """Returns {model_name: [(prefix_str, run_name), ...]} for runs that have BOTH a config and a checkpoint."""
    runs = {}
    if not os.path.isdir(exp_dir):
        return runs
    for name in sorted(os.listdir(exp_dir)):
        m = RUN_RE.match(name)
        if not m:
            continue
        has_cfg = os.path.exists(os.path.join(exp_dir, name, "hyperparameters.json"))
        has_ckpt = os.path.exists(os.path.join(model_dir, f"{name}.pth"))
        if has_cfg and has_ckpt:
            runs.setdefault(m.group("model"), []).append((m.group("prefix"), name))
    for model in runs:
        runs[model].sort(key=lambda x: int(x[0]))
    return runs


def choose_model_interactively(runs):
    names = sorted(runs)
    print("\nModels with saved phrase runs:")
    for i, n in enumerate(names):
        prefixes = ", ".join(p for p, _ in runs[n])
        print(f"  [{i}] {n}  (prefixes: {prefixes})")
    while True:
        choice = input("Choose a model (number or name): ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(names):
            return names[int(choice)]
        if choice in runs:
            return choice
        print("Not a valid choice, try again.")


# ==============================================================================
# Dataset cache (one dataset per distinct FEATURE configuration)
# ==============================================================================
_DATASET_CACHE = {}


def labels_dir_for(config):
    """The run's own phrase labels, unless --labels-dir overrides them for all runs."""
    return LABELS_OVERRIDE or config.get("phrase_labels_dir", DEFAULT_LABELS_DIR)


def dataset_key(config):
    # Window size, overlap, tolerance and downsampling don't change what is cached
    # (windowing/downsampling happen at inference time), so they are not in the key.
    return json.dumps({
        "labels_dir": labels_dir_for(config),
        "base_features": config.get("base_features"),
        "kinematic_features": config.get("kinematic_features", []),
        "use_face_keypoints": config.get("use_face_keypoints", False),
        "face_dir": config.get("face_dir"),
        "face_subset": config.get("face_subset", "full"),
        "face_only": config.get("face_only", False),
        "face_dims": int(config.get("face_dims", 3)),
        "use_hamer_features": config.get("use_hamer_features", False),
        "hamer_dir": config.get("hamer_dir"),
        "use_dinov2_features": config.get("use_dinov2_features", False),
        "dinov2_dir": config.get("dinov2_dir"),
    }, sort_keys=True)


def report_skipped(dataset, split):
    """Prints how many videos of the split failed to load, per reason (names are saved in the per-video JSON)."""
    skipped = getattr(dataset, "skipped_videos", {}) or {}
    total = sum(len(v) for v in skipped.values())
    if total == 0:
        print(f"  All {len(dataset.video_cache)} {split} videos loaded.")
        return
    reasons = ", ".join(f"{reason}: {len(items)}" for reason, items in skipped.items())
    print(f"  ⚠️  {total} {split} video(s) failed to load ({len(dataset.video_cache)} loaded; "
          f"not in the metrics) -- {reasons}")


def get_dataset(config, split):
    key = (split, dataset_key(config))
    if key not in _DATASET_CACHE:
        t0 = time.time()
        _DATASET_CACHE[key] = SignSegmentationDataset(
            keypoints_dir=KEYPOINTS_DIR,
            labels_dir=labels_dir_for(config),
            split_file=SPLIT_FILE,
            split=split,
            window_size=config["window_size"],
            overlap=config.get("overlap", 0),
            tolerance_window=1,  # evaluation never uses smoothed labels
            use_full_length=False,
            base_features=config["base_features"],
            kinematic_features=config.get("kinematic_features", []),
            temporal_downsample_factor=config.get("temporal_downsample_factor", 1),
            use_face_keypoints=config.get("use_face_keypoints", False),
            face_dir=config.get("face_dir", "processed_data/face_keypoints_normalized"),
            face_subset=config.get("face_subset", "full"),
            face_only=config.get("face_only", False),
            face_dims=int(config.get("face_dims", 3)),  # old configs = legacy 3D face
            use_hamer_features=config.get("use_hamer_features", False),
            hamer_dir=config.get("hamer_dir", "processed_data/hamer_features"),
            use_dinov2_features=config.get("use_dinov2_features", False),
            dinov2_dir=config.get("dinov2_dir", "processed_data/dinov2_features"),
        )
        print(f"  labels: {labels_dir_for(config)}")
        print(f"  [time] loading {split} data: {time.time() - t0:.1f}s "
              f"(reused for later runs with the same input features)")
        report_skipped(_DATASET_CACHE[key], split)
    return _DATASET_CACHE[key]


# ==============================================================================
# Evaluation of one run
# ==============================================================================
def sweep_thresholds(video_probs, video_gold, grid):
    """
    Tunes (b, o) thresholds on THIS split by IoU + %-score. Results on the same split
    are optimistic. Each grid point only computes IoU and % (light mode, gold segments
    computed once); full metrics are computed once for the best pair.
    """
    gold_cache = {}
    best = None
    for b in grid:
        for o in grid:
            m = evaluate_videos(video_probs, video_gold, decoder="threshold", b_threshold=b,
                                o_threshold=o, light=True, gold_segments_cache=gold_cache)
            score = m["IoU"] + pct_score(m["Pct"])
            if best is None or score > best[0]:
                best = (score, b, o)
    _, b_best, o_best = best
    m_best = evaluate_videos(video_probs, video_gold, decoder="threshold",
                             b_threshold=b_best, o_threshold=o_best)
    return best[0], b_best, o_best, m_best


def diagnose(video_probs, video_gold, b_threshold, o_threshold):
    """
    Sanity checks that explain odd metric values:
      - gold vs predicted class distribution (is the model predicting ~one class?
        does the label file use the expected 0=O, 1=I, 2=B encoding?)
      - segment counts and lengths (flicker -> thousands of 1-2 frame segments)
      - a trivial "everything is Inside" baseline, to see what IoU / F1 you get for free
    """
    from sklearn.metrics import f1_score
    gold_all = np.concatenate([np.asarray(video_gold[v]) for v in video_probs])
    pred_all = np.concatenate([video_probs[v].argmax(axis=0) for v in video_probs])
    names = {0: "O", 1: "I", 2: "B"}

    def dist(a):
        c = np.bincount(a.astype(np.int64), minlength=3)
        return "  ".join(f"{names[i]}={c[i]} ({c[i] / max(1, len(a)):.1%})" for i in range(3))

    print("  [diag] gold classes : " + dist(gold_all))
    print("  [diag] argmax pred  : " + dist(pred_all))
    unexpected = sorted(set(np.unique(gold_all).tolist()) - {0, 1, 2})
    if unexpected:
        print(f"  [diag] ⚠️ gold contains unexpected label values {unexpected}")

    n_gold = n_arg = n_thr = 0
    len_gold, len_arg, len_thr = [], [], []
    trivial_ious = []
    max_pb = []
    for v, probs in video_probs.items():
        gold = np.asarray(video_gold[v])[:probs.shape[1]]
        gs = extract_segments(gold)
        a = extract_segments(probs.argmax(axis=0))
        t = decode_threshold_2023(probs, b_threshold, o_threshold)
        n_gold += len(gs); n_arg += len(a); n_thr += len(t)
        len_gold += [e - s + 1 for s, e in gs]
        len_arg += [e - s + 1 for s, e in a]
        len_thr += [e - s + 1 for s, e in t]
        trivial_ious.append(segment_iou([(0, len(gold) - 1)], gs, len(gold)))
        max_pb.append(float(probs[2].max()))

    def med(x):
        return f"{np.median(x):.0f}" if x else "-"

    print(f"  [diag] segments    : gold={n_gold} (median len {med(len_gold)} fr) | "
          f"argmax={n_arg} (median len {med(len_arg)} fr) | "
          f"thr={n_thr} (median len {med(len_thr)} fr)")
    print(f"  [diag] P(B)        : max over each video = "
          + ", ".join(f"{x:.2f}" for x in max_pb))
    trivial_f1 = f1_score(gold_all, np.ones_like(gold_all), labels=[0, 1, 2], average="macro", zero_division=0)
    print(f"  [diag] trivial 'all Inside' baseline: Frame F1 {trivial_f1:.4f} | IoU {np.mean(trivial_ious):.4f}")


def evaluate_run(run_name, split, device, b_threshold, o_threshold, do_sweep, batch_size_override,
                 sweep_step=0.1, do_diagnose=False, swap_ib=False):
    cfg_path = os.path.join(EXP_DIR, run_name, "hyperparameters.json")
    ckpt_path = os.path.join(MODEL_DIR, f"{run_name}.pth")
    with open(cfg_path) as f:
        config = json.load(f)

    dataset = get_dataset(config, split)
    model_class, model_kwargs = build_model_kwargs(
        config,
        detected_hamer_dim=dataset.detected_hamer_dim,
        detected_dinov2_dim=dataset.detected_dinov2_dim,
    )
    model = model_class(**model_kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    batch_size = batch_size_override or config.get("batch_size", 16)
    t0 = time.time()
    video_probs, video_gold = predict_split(model, config, dataset, device, batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  [time] model inference: {time.time() - t0:.1f}s "
          f"({len(video_probs)} videos, batch size {batch_size})")
    if swap_ib:
        # DIAGNOSTIC ONLY: reinterpret the model's class 1 as B and class 2 as I, i.e. test
        # whether the checkpoint was trained on label files with I and B swapped.
        video_probs = {v: p[[0, 2, 1]] for v, p in video_probs.items()}
        print("  ⚠️  --swap-ib: model outputs reinterpreted (1<->2). Diagnostic only, don't report these.")

    t0 = time.time()
    results = {
        "argmax": evaluate_videos(video_probs, video_gold, decoder="argmax"),
        "threshold": evaluate_videos(video_probs, video_gold, decoder="threshold",
                                     b_threshold=b_threshold, o_threshold=o_threshold),
    }
    print(f"  [time] metrics: {time.time() - t0:.1f}s")
    if do_diagnose:
        diagnose(video_probs, video_gold, b_threshold, o_threshold)

    best_thresholds = None
    if do_sweep:
        t0 = time.time()
        grid = [round(x, 4) for x in np.arange(0.3, 0.9 + 1e-9, sweep_step)]
        _, b_best, o_best, m_best = sweep_thresholds(video_probs, video_gold, grid)
        results["swept"] = m_best
        best_thresholds = (b_best, o_best)
        print(f"  [time] threshold sweep ({len(grid)}x{len(grid)} = {len(grid) ** 2} pairs): "
              f"{time.time() - t0:.1f}s")

    return config, results, best_thresholds, dataset, video_probs, video_gold


def print_boundary_types(bt):
    """Table: phrase-start detection per boundary type (recall@k, offsets, peak P(B))."""
    tols = bt["tolerances"]
    header = (f"      {'boundary type':<22} {'n':>5} {'share':>6} {'gap':>5}  "
              + "  ".join(f"{'R@' + str(k):>6}" for k in tols)
              + f"  {'offset':>7} {'|off|':>6} {'peakPB':>6}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    for t in BOUNDARY_TYPES + ("all",):
        r = bt["types"][t]
        if r["n"] == 0 and t != "all":
            continue
        name = "ALL" if t == "all" else BOUNDARY_TYPE_LABELS[t]
        gap = f"{r['median_gap']:>5.0f}" if not np.isnan(r["median_gap"]) else f"{'-':>5}"
        off = f"{r['median_offset']:>+7.1f}" if not np.isnan(r["median_offset"]) else f"{'-':>7}"
        aoff = f"{r['mean_abs_offset']:>6.1f}" if not np.isnan(r["mean_abs_offset"]) else f"{'-':>6}"
        print(f"      {name:<22} {r['n']:>5} {r['share']:>6.1%} {gap}  "
              + "  ".join(f"{r['recall'][k]:>6.3f}" for k in tols)
              + f"  {off} {aoff} {r['peak_pb']:>6.3f}")
    k_mid = tols[len(tols) // 2]
    print(f"      predicted starts: {bt['n_pred']} | "
          + " | ".join(f"P@{k} {bt['precision'][k]:.3f}" for k in tols)
          + f" | unmatched @±{k_mid}: {bt['fp_in_phrase'][k_mid]} inside a gold phrase, "
            f"{bt['fp_in_pause'][k_mid]} in a gold pause")


def print_per_video(per_video):
    """Per-video table: frame F1 (argmax), IoU, %, segment F1@0.5 and segment counts."""
    name_w = max(10, max(len(v) for v in per_video))
    header = (f"      {'video':<{name_w}}  {'frames':>7}  {'gold':>5}  {'pred':>6}  "
              f"{'F1':>6}  {'IoU':>6}  {'%':>8}  {'SegF1':>6}")
    print(header)
    print("      " + "-" * (len(header) - 6))
    for vid in sorted(per_video):
        d = per_video[vid]
        pct = d["Pct"]
        pct_str = f"{pct:8.3f}" if not np.isnan(pct) else f"{'n/a':>8}"
        print(f"      {vid:<{name_w}}  {d['num_frames']:>7}  {d['num_gold_segments']:>5}  "
              f"{d['num_pred_segments']:>6}  {d['Frame_F1']:>6.3f}  {d['IoU']:>6.3f}  "
              f"{pct_str}  {d['Segment_F1_05']:>6.3f}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all saved phrase runs of one model (Frame F1, IoU, % of segments, "
                    "Segment F1@0.5) on full videos against the hard gold labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python evaluate_phrase.py                                  # choose a model interactively, val split\n"
            "  python evaluate_phrase.py --model stgcn_bilstm             # all stgcn_bilstm-XX runs on val\n"
            "  python evaluate_phrase.py --model stgcn_bilstm --prefixes 10 12\n"
            "  python evaluate_phrase.py --model stgcn_bilstm --sweep     # tune thresholds on val\n"
            "  python evaluate_phrase.py --model stgcn_bilstm --split test --b-threshold 0.6 --o-threshold 0.5\n"
            "  python evaluate_phrase.py --model stgcn_bilstm --per-video --diagnose\n"
            "  python evaluate_phrase.py --model multistream_bilstm --boundary-types --sweep\n"
        ),
    )
    parser.add_argument("--model", help="Model basename, e.g. stgcn_bilstm (asks if omitted).")
    parser.add_argument("--prefixes", nargs="*", help="Only these prefixes (default: all).")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Dataset split to evaluate on: 'val' (default) or 'test'.")
    parser.add_argument("--per-video", action="store_true",
                        help="Also print the metrics for every video (each decoder).")
    parser.add_argument("--b-threshold", type=float, default=0.5, help="B threshold for 2023-style decoding.")
    parser.add_argument("--o-threshold", type=float, default=0.5, help="O threshold for 2023-style decoding.")
    parser.add_argument("--sweep", action="store_true", help="Also tune thresholds on this split (0.3..0.9).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Inference batch size (default: the run's training batch_size). "
                             "Larger = faster, more GPU memory; results are identical.")
    parser.add_argument("--swap-ib", action="store_true",
                        help="DIAGNOSTIC: swap the model's I and B outputs, to test whether a checkpoint "
                             "was trained on labels with I/B swapped.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print class distributions, segment counts/lengths and a trivial baseline.")
    parser.add_argument("--boundary-types", action="store_true",
                        help="Also evaluate phrase-start detection split by boundary type: after a "
                             "long pause, after a short pause, or within continuous signing (no gap).")
    parser.add_argument("--tolerances", type=int, nargs="+", default=[2, 5, 10],
                        help="Boundary tolerances in frames for --boundary-types (default: 2 5 10).")
    parser.add_argument("--short-gap", type=int, default=12,
                        help="Max Outside gap (frames) still counted as a SHORT pause (default 12).")
    parser.add_argument("--peak-window", type=int, default=5,
                        help="+-frames around a gold start for the peak P(B) column (default 5).")
    parser.add_argument("--labels-dir", default=None,
                        help="Evaluate every run against these phrase labels (e.g. "
                             "processed_data/BIO_tags_phrase_v2). Default: each run's own labels "
                             "(its phrase_labels_dir, or v1 for older runs).")
    parser.add_argument("--sweep-step", type=float, default=0.1,
                        help="Threshold grid step for --sweep over 0.3..0.9 (default 0.1 = 7x7 pairs). "
                             "Smaller = finer tuning but slower (0.05 = 13x13).")
    args = parser.parse_args()
    global LABELS_OVERRIDE
    LABELS_OVERRIDE = args.labels_dir

    runs = discover_runs(EXP_DIR, MODEL_DIR)
    if not runs:
        print(f"No runs found (need {EXP_DIR}/<run>/hyperparameters.json AND {MODEL_DIR}/<run>.pth).")
        return

    model_name = args.model or choose_model_interactively(runs)
    if model_name not in runs:
        print(f"No saved runs for '{model_name}'. Available: {sorted(runs)}")
        return

    selected = runs[model_name]
    if args.prefixes:
        wanted = {int(p) for p in args.prefixes}
        selected = [(p, r) for p, r in selected if int(p) in wanted]
        if not selected:
            print(f"None of the prefixes {sorted(wanted)} exist for '{model_name}'.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nEvaluating {len(selected)} run(s) of '{model_name}' on '{args.split}' ({device}).")

    rows = []
    boundary_rows = []
    for prefix, run_name in selected:
        print(f"\n--- {run_name} ---")
        try:
            t_run = time.time()
            config, results, best_thr, dataset, video_probs, video_gold = evaluate_run(
                run_name, args.split, device, args.b_threshold, args.o_threshold,
                args.sweep, args.batch_size, args.sweep_step, args.diagnose, args.swap_ib)
        except Exception as e:
            print(f"⚠️  Skipped {run_name}: {type(e).__name__}: {e}")
            continue

        print(f"  {config.get('description', '')}")
        print(f"  window={config['window_size']} overlap={config.get('overlap', 0)} "
              f"d_model={config['d_model']} n_layers={config['n_layers']} "
              f"downsample={config.get('temporal_downsample_factor', 1)} "
              f"(trained with tolerance_window={config.get('tolerance_window')}; ignored for evaluation)")

        decoders = [("argmax", "argmax", None),
                    ("threshold", f"thr b={args.b_threshold} o={args.o_threshold}", None)]
        if best_thr:
            decoders.append(("swept", f"swept b={best_thr[0]} o={best_thr[1]}", best_thr))

        for key, label, _ in decoders:
            m = results[key]
            print(f"  [{label:<22}] Frame F1 {m['Frame_F1']:.4f} | IoU {m['IoU']:.4f} | "
                  f"% (ratio) {m['Pct']:.4f} | SegF1@0.5 {m['Segment_F1_05']:.4f}")
            if key == "argmax":  # per-class F1 is decoder-independent (always argmax)
                print(f"  {'':<24} Frame F1 per class: O {m['F1_O']:.4f} | I {m['F1_I']:.4f} | "
                      f"B {m['F1_B']:.4f}  (Frame F1 = their average)")
            if args.per_video:
                print_per_video(m["per_video"])
            if args.boundary_types:
                thr = best_thr if key == "swept" else (args.b_threshold, args.o_threshold)
                bt = boundary_type_analysis(
                    video_probs, video_gold, decoder="argmax" if key == "argmax" else "threshold",
                    b_threshold=thr[0], o_threshold=thr[1], tolerances=args.tolerances,
                    short_gap=args.short_gap, peak_window=args.peak_window)
                print_boundary_types(bt)
                for t in BOUNDARY_TYPES + ("all",):
                    r = bt["types"][t]
                    row = {"run": run_name, "prefix": prefix, "decoder": label, "boundary_type": t,
                           "n": r["n"], "share": round(r["share"], 4),
                           "median_gap": r["median_gap"]}
                    for k in bt["tolerances"]:
                        row[f"recall@{k}"] = round(r["recall"][k], 4) if r["n"] else ""
                    row.update({"median_offset": r["median_offset"],
                                "mean_abs_offset": round(r["mean_abs_offset"], 3) if r["n"] else "",
                                "peak_pb": round(r["peak_pb"], 4) if r["n"] else "",
                                "n_pred": bt["n_pred"]})
                    for k in bt["tolerances"]:
                        row[f"precision@{k}"] = round(bt["precision"][k], 4)
                    row.update({"short_gap": args.short_gap, "seed": config.get("seed", ""),
                                "description": config.get("description", "")})
                    boundary_rows.append(row)
            rows.append({
                "run": run_name,
                "prefix": prefix,
                "decoder": label,
                "frame_f1": round(m["Frame_F1"], 4),
                "f1_O": round(m["F1_O"], 4),
                "f1_I": round(m["F1_I"], 4),
                "f1_B": round(m["F1_B"], 4),
                "iou": round(m["IoU"], 4),
                "segment_pct": round(m["Pct"], 4),
                "segment_f1_05": round(m["Segment_F1_05"], 4),
                "videos_evaluated": len(dataset.video_cache),
                "videos_skipped": sum(len(v) for v in getattr(dataset, "skipped_videos", {}).values()),
                "window_size": config["window_size"],
                "seed": config.get("seed", ""),
                "description": config.get("description", ""),
            })

        per_video_path = os.path.join(OUT_DIR, f"{run_name}_{args.split}_per_video.json")
        with open(per_video_path, "w") as f:
            json.dump({
                "decoders": {k: v["per_video"] for k, v in results.items()},
                "skipped_videos": {reason: [{"video": v, "detail": d} for v, d in items]
                                   for reason, items in getattr(dataset, "skipped_videos", {}).items()},
            }, f, indent=2)
        print(f"  [time] total for {run_name}: {time.time() - t_run:.1f}s")

    if not rows:
        print("\nNothing was evaluated.")
        return

    csv_path = os.path.join(OUT_DIR, f"{model_name}_{args.split}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Summary saved to {csv_path} (per-video details in {OUT_DIR}/).")
    if boundary_rows:
        bt_path = os.path.join(OUT_DIR, f"{model_name}_{args.split}_boundary_types.csv")
        with open(bt_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(boundary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(boundary_rows)
        print(f"✅ Boundary-type results saved to {bt_path}")


if __name__ == "__main__":
    main()