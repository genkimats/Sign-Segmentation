"""
train_fact.py -- trains the FACT pilot and its frozen-BiLSTM reference on exported features.

    python fact_phrase/train_fact.py                     # works through fact_phrase/train_queue_fact.json
    python fact_phrase/train_fact.py --job '{"model": "fact", "source_run": "multistream_bilstm-310", "seed": 42}'
    python fact_phrase/train_fact.py --max-jobs 1        # just the next queued job

The queue (built by queue_fact.py) is a JSON list; the first entry is popped and moved to
train_queue_fact_done.json with its run name, so you can edit the queue while a job runs.

Per run (fact_phrase/experiments/<run>/):
  hyperparameters.json   the fully resolved config (incl. ntoken, null weight, parameter count)
  history.csv            per epoch: train loss, val BndF1@5 / IoU / % for every decoder
  saved_models/<run>__<decoder>.pth   best epoch per decoder by val BndF1@5 (as R0: "epoch by bnd5")
After training the run is evaluated on val and test (evaluate_fact.evaluate_run).

Decoders: FACT has two -- "token" (FACT's own token decoding) and "head" (argmax of its frame-branch
B/I/O probabilities); each keeps its own best checkpoint. The frozen BiLSTM has "head" only.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fact_phrase import fact_core as core  # noqa: E402
from fact_phrase.data import (FeatureStore, TrainWindows, list_collate, pad_batch, predict_store,  # noqa: E402
                              decoded_probs, samples_per_epoch)
from fact_phrase.fact_model import BILSTM_DEFAULTS, FACT_DEFAULTS, build_model, count_parameters  # noqa: E402
from src.metrics import evaluate_videos  # noqa: E402

HERE = "fact_phrase"
EXP_DIR = os.path.join(HERE, "experiments")
MODEL_DIR = os.path.join(HERE, "saved_models")
QUEUE = os.path.join(HERE, "train_queue_fact.json")
DONE = os.path.join(HERE, "train_queue_fact_done.json")

# Training recipes: FACT keeps FACT's own recipe (Adam 1e-4, constant LR, gradient clipping 10),
# the frozen BiLSTM keeps R0's (AdamW 1e-4 with cosine annealing, batch 16, windows of 512 frames).
TRAIN_DEFAULTS = {
    "fact": {"window_frames": 1500, "feature_stride": 1, "batch_size": 4, "optimizer": "adam", "lr": 1e-4,
             "weight_decay": 0.0, "scheduler": None, "grad_clip": 10.0, "epochs": 60, "patience": 10,
             "epoch_coverage": 0.5, "eval_step_frac": 0.5, "num_workers": 4},
    "frozen_bilstm": {"window_frames": 512, "feature_stride": 1, "batch_size": 16, "optimizer": "adamw",
                      "lr": 1e-4, "weight_decay": 0.01, "scheduler": "cosine", "grad_clip": None, "epochs": 100,
                      "patience": 10, "epoch_coverage": 1.0, "eval_step_frac": 1.0, "num_workers": 4},
}
MODEL_DEFAULTS = {"fact": FACT_DEFAULTS, "frozen_bilstm": BILSTM_DEFAULTS}
DECODERS = {"fact": ["token", "head"], "frozen_bilstm": ["head"]}


def resolve_config(job):
    model = job.get("model", "fact")
    if model not in MODEL_DEFAULTS:
        raise ValueError(f"model must be one of {sorted(MODEL_DEFAULTS)}")
    cfg = {**MODEL_DEFAULTS[model], **TRAIN_DEFAULTS[model]}
    unknown = [k for k in job if k not in cfg and k not in ("source_run", "seed", "description", "model")]
    if unknown:   # a typo would otherwise silently leave the intended setting at its default
        raise ValueError(f"unknown config key(s) {unknown}; valid: {sorted(cfg)}")
    cfg.update(job)
    if "source_run" not in cfg:
        raise ValueError("every job needs source_run (the exported features, e.g. multistream_bilstm-310)")
    cfg.setdefault("seed", 42)
    cfg.setdefault("description", f"{model} on {cfg['source_run']}")
    return cfg


def next_run_name(basename):
    os.makedirs(EXP_DIR, exist_ok=True)
    nums = [int(m.group(1)) for d in os.listdir(EXP_DIR) for m in [re.match(rf"^{basename}-(\d+)$", d)] if m]
    return f"{basename}-{(max(nums) + 1 if nums else 1):02d}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val_scores(model, cfg, store, device):
    preds = predict_store(model, cfg, store, device)
    gold = {v: store.videos[v]["gold"] for v in preds}
    out = {}
    for dec in DECODERS[cfg["model"]]:
        m = evaluate_videos(decoded_probs(preds, dec), gold, light=True)
        out[dec] = {"bnd5": m["BndF1_5"], "iou": m["IoU"], "pct": m["Pct"]}
    return out


def train_job(job, device, evaluate_after=True):
    cfg = resolve_config(job)
    run = next_run_name(cfg["model"])
    run_dir = os.path.join(EXP_DIR, run)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    set_seed(int(cfg["seed"]))
    print(f"\n{'=' * 90}\n{run}: {cfg['description']}  [seed={cfg['seed']}]\n{'=' * 90}")

    stride = int(cfg["feature_stride"])
    train_store = FeatureStore(cfg["source_run"], "train", stride)
    val_store = FeatureStore(cfg["source_run"], "val", stride)
    W = int(cfg["window_frames"]) // stride
    print(f"  features {cfg['source_run']} (dim {train_store.dim}), {len(train_store.videos)} train / "
          f"{len(val_store.videos)} val videos, window {W} model frames ({cfg['window_frames'] / 50:.1f} s)")

    if cfg["model"] == "fact":
        mx, mean, nwin = core.window_segment_stats(train_store.model_rate_labels(), W, max(1, W // 4))
        if cfg["ntoken"] == "auto":
            cfg["ntoken"] = core.auto_ntoken(mx)
        if cfg["ntoken"] < mx:
            print(f"  ⚠️  ntoken {cfg['ntoken']} < {mx} segments in the busiest training window: "
                  f"some segments will stay unmatched there")
        cfg["nullw_resolved"] = core.null_weight(cfg["ntoken"], mean) if cfg["nullw"] == "auto" else float(cfg["nullw"])
        cfg["window_segments"] = {"max": mx, "mean": round(mean, 2), "windows": nwin}
        print(f"  segments per {W}-frame window (O runs + phrases): max {mx}, mean {mean:.1f} "
              f"-> ntoken {cfg['ntoken']}, null weight {cfg['nullw_resolved']:.3f}")

    model = build_model(cfg, train_store.dim).to(device)
    if cfg["model"] == "fact":
        model.attach_criterion(cfg["nullw_resolved"], device)
    cfg["n_params"] = count_parameters(model)
    cfg["run"] = run
    print(f"  parameters: {cfg['n_params'] / 1e6:.2f} M")
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    if cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"]) if cfg["scheduler"] == "cosine" else None

    n_samples = samples_per_epoch(train_store, W, cfg["epoch_coverage"])
    loader = DataLoader(TrainWindows(train_store, W, n_samples), batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], collate_fn=list_collate,
                        persistent_workers=cfg["num_workers"] > 0, pin_memory=device.type == "cuda")
    print(f"  {n_samples} windows per epoch, batch {cfg['batch_size']}")

    decoders = DECODERS[cfg["model"]]
    best = {d: -1.0 for d in decoders}
    best_epoch = {d: 0 for d in decoders}
    since_improved = 0
    hist_path = os.path.join(run_dir, "history.csv")
    hist_fields = ["epoch", "train_loss", "skipped_steps", "lr", "seconds"] + \
        [f"val_{d}_{m}" for d in decoders for m in ("bnd5", "iou", "pct")]
    with open(hist_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=hist_fields).writeheader()

    try:
        from tqdm import tqdm
    except ImportError:                                                     # pragma: no cover
        def tqdm(x, **_):
            return x

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for batch in tqdm(loader, desc=f"{run} epoch {epoch}", leave=False):
            opt.zero_grad(set_to_none=True)
            if cfg["model"] == "fact":
                loss = sum(model.loss(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                           for x, y in batch) / len(batch)
            else:
                X, Y = pad_batch(batch, device)
                loss = model.loss_batch(X, Y)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += float(loss.item())
            n += 1
        if sched is not None:
            sched.step()
        if skipped > 0.2 * max(1, n + skipped):
            raise RuntimeError(f"{skipped} non-finite losses in epoch {epoch} -- stopping {run}")

        scores = val_scores(model, cfg, val_store, device)
        improved = []
        for d in decoders:
            if scores[d]["bnd5"] > best[d]:
                best[d], best_epoch[d] = scores[d]["bnd5"], epoch
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{run}__{d}.pth"))
                improved.append(d)
        since_improved = 0 if improved else since_improved + 1
        row = {"epoch": epoch, "train_loss": round(tot / max(n, 1), 5), "skipped_steps": skipped,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        for d in decoders:
            row.update({f"val_{d}_{m}": round(scores[d][m], 4) for m in ("bnd5", "iou", "pct")})
        with open(hist_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=hist_fields).writerow(row)
        print(f"  epoch {epoch:3d}  loss {row['train_loss']:.4f}  " +
              "  ".join(f"{d}: bnd5 {scores[d]['bnd5']:.4f} %{scores[d]['pct']:.2f}" for d in decoders) +
              (f"  * best {','.join(improved)}" if improved else "") + f"  ({row['seconds']:.0f}s)")
        if since_improved >= cfg["patience"]:
            print(f"  early stop: no decoder improved for {cfg['patience']} epochs")
            break

    cfg["best_val_bnd5"] = best
    cfg["best_epoch"] = best_epoch
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("  best val BndF1@5: " + ", ".join(f"{d} {best[d]:.4f} (epoch {best_epoch[d]})" for d in decoders))

    if evaluate_after:
        from fact_phrase.evaluate_fact import evaluate_run
        evaluate_run(run, ["val", "test"], device)
    return run


def pop_job():
    if not os.path.exists(QUEUE):
        return None
    with open(QUEUE) as f:
        jobs = json.load(f)
    if not jobs:
        return None
    job = jobs.pop(0)
    with open(QUEUE, "w") as f:
        json.dump(jobs, f, indent=1)
    return job


def mark_done(job, run, error=None):
    done = json.load(open(DONE)) if os.path.exists(DONE) else []
    done.append({**job, "run": run, **({"error": error} if error else {})})
    with open(DONE, "w") as f:
        json.dump(done, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description="Train the FACT pilot / frozen-BiLSTM reference.")
    ap.add_argument("--job", help="one job as JSON (bypasses the queue)")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="skip the val/test evaluation after training")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️  no GPU found -- training will be slow")

    if args.job:
        train_job(json.loads(args.job), device, not args.no_eval)
        return
    done = 0
    while args.max_jobs is None or done < args.max_jobs:
        job = pop_job()
        if job is None:
            print("Queue empty.")
            break
        try:
            run = train_job(job, device, not args.no_eval)
            mark_done(job, run)
        except Exception as e:                                             # keep the queue moving
            import traceback
            traceback.print_exc()
            mark_done(job, None, error=repr(e))
        done += 1


if __name__ == "__main__":
    main()
