"""
aux_head_eval.py -- did the auxiliary heads of Idea B learn their task at all?

For runs trained with aux_heads (queue_train_phrase.py AUX_STUDY), rebuilds the val / test data
WITH the auxiliary labels (force_aux=True; training-time datasets load them for train only),
runs the model with the same windows as training, stitches the head probabilities per video and
reports per task:

  accuracy, macro-F1 (classes present in the gold labels), per-class recall / precision,
  the confusion matrix, and the majority-class baseline (accuracy of always predicting the most
  frequent class) -- "learned mouthing" means clearly above that baseline, especially in macro-F1.

Together with check_stream_balance.py (face / HaMeR variance share) this gives the
information-vs-optimization reading of plan section 4.2:
  head good, boundaries unchanged, face share up -> limit is INFORMATION
  head good, boundaries up                        -> limit was OPTIMIZATION
  head poor                                       -> mouthing not recoverable from the keypoints

Usage (repo root):
  python aux_head_eval.py --model multistream_bilstm --prefixes 401 402 403 --splits val test
Outputs: evaluation_phrase/aux_heads/<run>_<split>.json and summary_<split>.csv (one row per run x task)
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.aux_labels import CLASS_NAMES, IGNORE
from src.dataset import SignSegmentationDataset, extra_dataset_kwargs
from src.model_factory import build_model_kwargs

EXP_DIR = "experiments_phrase"
MODEL_DIR = "saved_models_phrase"
OUT_DIR = os.path.join("evaluation_phrase", "aux_heads")


def build_dataset(config, split):
    """Same arguments as evaluate_phrase.get_dataset, plus force_aux=True."""
    return SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir=config.get("phrase_labels_dir", "processed_data/BIO_tags_phrase"),
        split_file="dataset_splits.json", split=split,
        window_size=config["window_size"], overlap=config.get("overlap", 0), tolerance_window=1,
        use_full_length=False, base_features=config["base_features"],
        kinematic_features=config.get("kinematic_features", []),
        temporal_downsample_factor=config.get("temporal_downsample_factor", 1),
        use_face_keypoints=config.get("use_face_keypoints", False),
        face_dir=config.get("face_dir", "processed_data/face_keypoints_normalized"),
        face_subset=config.get("face_subset", "full"), face_only=config.get("face_only", False),
        face_dims=int(config.get("face_dims", 3)),
        use_hamer_features=config.get("use_hamer_features", False),
        hamer_dir=config.get("hamer_dir", "processed_data/hamer_features"),
        use_dinov2_features=config.get("use_dinov2_features", False),
        dinov2_dir=config.get("dinov2_dir", "processed_data/dinov2_features"),
        force_aux=True, **extra_dataset_kwargs(config))


def task_metrics(gold, pred, n_cls):
    """gold / pred: 1-D int arrays (IGNORE already removed)."""
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (gold, pred), 1)
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = np.diag(cm)
    recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
    precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), np.nan)
    f1 = np.where((support > 0), 2 * tp / np.maximum(support + predicted, 1), np.nan)
    present = support > 0
    acc = float(tp.sum() / max(cm.sum(), 1))
    majority = float(support.max() / max(support.sum(), 1)) if support.sum() else float("nan")
    return {"frames": int(cm.sum()), "accuracy": round(acc, 4), "majority_baseline": round(majority, 4),
            "macro_f1": round(float(np.nanmean(f1[present])), 4) if present.any() else float("nan"),
            "recall": [None if np.isnan(r) else round(float(r), 4) for r in recall],
            "precision": [None if np.isnan(p) else round(float(p), 4) for p in precision],
            "support": support.tolist(), "confusion": cm.tolist()}


@torch.no_grad()
def eval_run(run, split, device, batch_size=None):
    with open(os.path.join(EXP_DIR, run, "hyperparameters.json")) as f:
        config = json.load(f)
    if not config.get("aux_heads"):
        print(f"{run}: no aux_heads in its config -- skipped")
        return None
    if int(config.get("temporal_downsample_factor", 1)) != 1:
        raise NotImplementedError("aux_head_eval.py supports temporal_downsample_factor = 1 only.")
    ds = build_dataset(config, split)
    tasks = list(ds.aux_tasks)
    heads = config["aux_heads"]
    model_class, kw = build_model_kwargs(config, detected_hamer_dim=ds.detected_hamer_dim,
                                         detected_dinov2_dim=ds.detected_dinov2_dim,
                                         detected_cue_dim=getattr(ds, "detected_cue_dim", None))
    model = model_class(**kw).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{run}.pth"), map_location=device), strict=True)
    model.eval()

    use_h, use_d = config.get("use_hamer_features", False), config.get("use_dinov2_features", False)
    use_c = config.get("use_cue_features", False)
    acc = {t: {v: np.zeros((int(heads[t]["classes"]), c["aux_labels"].shape[1]), dtype=np.float64)
               for v, c in ds.video_cache.items()} for t in tasks}
    cnt = {v: np.zeros(c["aux_labels"].shape[1]) for v, c in ds.video_cache.items()}
    loader = DataLoader(ds, batch_size=batch_size or config.get("batch_size", 16), shuffle=False, num_workers=2)
    for batch in loader:
        feats = torch.nan_to_num(batch[0].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        vids, starts, ends = batch[-3], batch[-2], batch[-1]
        extras = list(batch[2:-3])
        kwargs = {}
        if use_h:
            kwargs["hamer"] = torch.nan_to_num(extras.pop(0).to(device), nan=0.0, posinf=0.0, neginf=0.0)
        if use_d:
            kwargs["dinov2"] = torch.nan_to_num(extras.pop(0).to(device), nan=0.0, posinf=0.0, neginf=0.0)
        if use_c:
            kwargs["cues"] = torch.nan_to_num(extras.pop(0).to(device), nan=0.0, posinf=0.0, neginf=0.0)
        model(feats, **kwargs)
        probs = {t: torch.softmax(model._aux_logits[t].float(), dim=1).cpu().numpy() for t in tasks}
        for i, v in enumerate(vids):
            s, e = int(starts[i]), int(ends[i])
            for t in tasks:
                acc[t][v][:, s:e] += probs[t][i][:, :e - s]
            cnt[v][s:e] += 1

    out = {"run": run, "split": split, "description": config.get("description", ""), "tasks": {}}
    for j, t in enumerate(tasks):
        n_cls = int(heads[t]["classes"])
        golds, preds = [], []
        for v, c in ds.video_cache.items():
            g = c["aux_labels"][j].numpy().astype(np.int64)
            m = (g != IGNORE) & (cnt[v] > 0)
            if m.any():
                golds.append(g[m])
                preds.append(acc[t][v][:, m].argmax(0))
        if not golds:
            out["tasks"][t] = {"frames": 0}
            continue
        res = task_metrics(np.concatenate(golds), np.concatenate(preds), n_cls)
        names = CLASS_NAMES.get(t, [])
        res["classes"] = names[:n_cls] if len(names) >= n_cls else [str(k) for k in range(n_cls)]
        res["stream"] = heads[t]["stream"]
        out["tasks"][t] = res
        rec = ", ".join(f"{n} {r if r is not None else '-'}" for n, r in zip(res["classes"], res["recall"]))
        print(f"  {run} [{split}] {t} on '{heads[t]['stream']}': acc {res['accuracy']:.3f} "
              f"(majority {res['majority_baseline']:.3f}), macro-F1 {res['macro_f1']:.3f} | recall: {rec}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Accuracy of the Idea-B auxiliary heads on val / test.")
    ap.add_argument("--model", default="multistream_bilstm")
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in args.splits:
        rows = []
        for p in args.prefixes:
            run = f"{args.model}-{p}"
            res = eval_run(run, split, device, args.batch_size)
            if res is None:
                continue
            with open(os.path.join(args.out_dir, f"{run}_{split}.json"), "w") as f:
                json.dump(res, f, indent=2)
            for t, r in res["tasks"].items():
                if not r.get("frames"):
                    continue
                rows.append({"run": run, "description": res["description"], "task": t, "stream": r["stream"],
                             "frames": r["frames"], "accuracy": r["accuracy"],
                             "majority_baseline": r["majority_baseline"], "macro_f1": r["macro_f1"],
                             **{f"recall_{n}": rc for n, rc in zip(r["classes"], r["recall"])}})
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            path = os.path.join(args.out_dir, f"summary_{split}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
