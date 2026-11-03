"""
evaluate_fact.py -- evaluates FACT-pilot runs on val / test with the project's metrics and saves their
predictions where the existing analysis tools look for them.

    python fact_phrase/evaluate_fact.py --runs fact-01 frozen_bilstm-01       # val + test
    python fact_phrase/evaluate_fact.py --runs fact-01 --splits test

For every (run, decoder) -- FACT: "token" and "head", frozen BiLSTM: "head" -- the decoder's own best
checkpoint (saved_models/<run>__<decoder>.pth) is loaded and the split is predicted on full videos.

Outputs
  fact_phrase/results/<split>_metrics.csv
        same columns as evaluate_phrase.py's argmax rows (Frame F1, IoU, %, Segment F1@0.5,
        boundary F1 / P / R ...), one row per (run, decoder); re-running replaces the rows.
  evaluation_phrase/predictions/<name>/<split>.npz + <split>_meta.json
        <name> = fact_token-NN / fact_head-NN / frozen_bilstm-NN, in evaluate_phrase.py's format, so
        phrase_stats.py (paired bootstrap), count_oracle_decoding.py, boundary_error_analysis.py and
        phrase_oracle_analysis.py --predictions work on them unchanged. For "token" the saved
        "probabilities" are the one-hot decoded labels (FACT's token decoding has no per-frame B score).
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

from fact_phrase.data import FeatureStore, decoded_probs, predict_store  # noqa: E402
from fact_phrase.fact_model import build_model  # noqa: E402
from src.metrics import BND_TOLERANCES, evaluate_videos  # noqa: E402

HERE = "fact_phrase"
EXP_DIR = os.path.join(HERE, "experiments")
MODEL_DIR = os.path.join(HERE, "saved_models")
RESULTS_DIR = os.path.join(HERE, "results")
PRED_ROOT = os.path.join("evaluation_phrase", "predictions")
DECODERS = {"fact": ["token", "head"], "frozen_bilstm": ["head"]}


def prediction_name(cfg, decoder):
    prefix = cfg["run"].rsplit("-", 1)[1]
    base = {"token": "fact_token", "head": "fact_head"}[decoder] if cfg["model"] == "fact" else cfg["model"]
    return f"{base}-{prefix}"


def load_run(run, device):
    with open(os.path.join(EXP_DIR, run, "hyperparameters.json")) as f:
        cfg = json.load(f)
    return cfg


def build_for(cfg, dim, decoder, device):
    model = build_model(cfg, dim).to(device)
    if cfg["model"] == "fact":
        model.attach_criterion(cfg["nullw_resolved"], device)
    path = os.path.join(MODEL_DIR, f"{cfg['run']}__{decoder}.pth")
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    model.eval()
    return model


def metrics_row(cfg, decoder, m, n_videos):
    row = {"run": prediction_name(cfg, decoder), "train_run": cfg["run"], "decoder": decoder,
           "frame_f1": round(m["Frame_F1"], 4), "f1_O": round(m.get("F1_O", float("nan")), 4),
           "f1_I": round(m.get("F1_I", float("nan")), 4), "f1_B": round(m.get("F1_B", float("nan")), 4),
           "iou": round(m["IoU"], 4), "segment_pct": round(m["Pct"], 4),
           "segment_f1_05": round(m["Segment_F1_05"], 4)}
    row.update({f"bnd_f1_{k}": round(m[f"BndF1_{k}"], 4) for k in BND_TOLERANCES})
    row.update({"bnd_p_5": round(m["BndP_5"], 4), "bnd_r_5": round(m["BndR_5"], 4),
                "bnd_n_pred": m["Bnd_n_pred"], "bnd_n_gold": m["Bnd_n_gold"], "videos_evaluated": n_videos,
                "source_run": cfg["source_run"], "window_frames": cfg["window_frames"], "seed": cfg.get("seed", ""),
                "n_params": cfg.get("n_params", ""), "description": cfg.get("description", "")})
    return row


def write_rows(split, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{split}_metrics.csv")
    old = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    keys = {(r["run"], r["decoder"]) for r in rows}
    merged = [r for r in old if (r["run"], r["decoder"]) not in keys] + rows
    fields = list(rows[0].keys())
    for r in merged:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)
    return path


def save_predictions(cfg, decoder, split, probs, labels_dir):
    name = prediction_name(cfg, decoder)
    d = os.path.join(PRED_ROOT, name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{split}.npz"), **{v: p.astype(np.float16) for v, p in probs.items()})
    meta = {"run": name, "model": name.rsplit("-", 1)[0], "prefix": name.rsplit("-", 1)[1], "split": split,
            "labels_dir": labels_dir, "train_run": cfg["run"], "decoder": decoder, "source_run": cfg["source_run"],
            "description": f"{cfg.get('description', cfg['model'])} | decoder={decoder} [seed={cfg.get('seed', '')}]",
            "seed": cfg.get("seed", ""), "merge_starts": {}}
    with open(os.path.join(d, f"{split}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d


def evaluate_run(run, splits, device):
    cfg = load_run(run, device)
    rows_by_split = {s: [] for s in splits}
    for split in splits:
        store = FeatureStore(cfg["source_run"], split, int(cfg["feature_stride"]))
        gold = {v: store.videos[v]["gold"] for v in store.videos}
        for decoder in DECODERS[cfg["model"]]:
            model = build_for(cfg, store.dim, decoder, device)
            preds = predict_store(model, cfg, store, device)
            probs = decoded_probs(preds, decoder)
            m = evaluate_videos(probs, gold)
            rows_by_split[split].append(metrics_row(cfg, decoder, m, len(gold)))
            save_predictions(cfg, decoder, split, probs, store.labels_dir)
            print(f"  [{split}] {prediction_name(cfg, decoder):18s} Frame F1 {m['Frame_F1']:.4f}  IoU {m['IoU']:.4f}  "
                  f"% {m['Pct']:.3f}  BndF1@5 {m['BndF1_5']:.4f}  SegF1@0.5 {m['Segment_F1_05']:.4f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for split, rows in rows_by_split.items():
        if rows:
            print(f"  -> {write_rows(split, rows)}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate FACT-pilot runs (val/test).")
    ap.add_argument("--runs", nargs="+", required=True, help="e.g. fact-01 frozen_bilstm-01")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.runs:
        print(f"\n{run}")
        evaluate_run(run, args.splits, device)


if __name__ == "__main__":
    main()
