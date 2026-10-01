"""
train_novelty.py -- trains STGCN_Novelty (models_novelty.py) from its own queue
file (train_queue_novelty.json, built by queue_train_novelty.py).

Differences from the project's train.py, and why
  * Checkpoint selection uses the DECODED prediction (CRF Viterbi, or argmax for
    non-CRF arms), scored with the corrected one-to-one segment metric plus
    frame macro-F1:  score = 0.5 * (frame_macro_F1 + segF1@0.5).
    Frame macro-F1 alone barely moves between a good and a bad decoder (it was
    0.588 vs 0.606 while segment F1 went 0.30 -> 0.66 in the decoder study), so
    selecting on it would be blind to what this model is built to improve.
    train.py's legacy many-to-one segment F1 can exceed 1.0 and is NOT used.
    Metrics are POOLED over all validation windows (not averaged per window),
    which avoids the per-window zero-division artifact.
  * Raw targets by default (tolerance_window=1): the CRF's global normalisation
    supplies the boundary tolerance that label dilation was faking.
  * Loss = CRF NLL per frame + ce_weight * weighted per-frame cross-entropy.
    With use_crf=False the loss is the weighted cross-entropy alone (ablation arm).

Paths are resolved from this file's location; every output goes under
novelty_crf/. Run from anywhere.
"""
import os
import sys
import json
import time
import random
import copy
import socket
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "decoder_study")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models_novelty import STGCN_Novelty
from src.dataset import SignSegmentationDataset
from src.loss import WeightedCrossEntropyLoss

try:
    import study_metrics as M   # decoder_study/ -- the corrected one-to-one metrics
except Exception as _e:         # pragma: no cover
    M = None
    print(f"[WARN] could not import decoder_study/study_metrics ({_e}); checkpoints will be "
          f"selected on frame macro-F1 only.")

DEFAULT_KEYPOINTS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "keypoints")
DEFAULT_LABELS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "BIO_tags")
DEFAULT_HAMER_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "hamer_features")
DEFAULT_DINOV2_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "dinov2_features")
DEFAULT_SPLIT_FILE = os.path.join(_PROJECT_ROOT, "dataset_splits.json")

QUEUE_FILE = os.path.join(_SCRIPT_DIR, "train_queue_novelty.json")
MODEL_DIR = os.path.join(_SCRIPT_DIR, "saved_models")
EXPERIMENTS_DIR = os.path.join(_SCRIPT_DIR, "experiments")
MODULE_DIR_NAME = os.path.basename(_SCRIPT_DIR)   # recorded so the exporter can find models_novelty.py

MODEL_CONFIG_KEYS = ["d_model", "n_layers", "backbone", "xlstm_pattern", "xlstm_heads", "dropout", "n_local_blocks",
                     "adapter_layers", "use_similarity", "similarity_K", "novelty_scales", "d_sim", "use_crf",
                     "crf_forbid_penalty", "hamer_proj_dim", "hamer_smooth", "dinov2_proj_dim"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_next_job():
    if not os.path.exists(QUEUE_FILE):
        return None
    with open(QUEUE_FILE, "r") as f:
        try:
            queue = json.load(f)
        except json.JSONDecodeError:
            return None
    if not queue or len(queue) <= 1:
        return None
    job = queue.pop(1)
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
    return job


def run_name_for(cfg):
    parts = [cfg.get("backbone", "xlstm")]
    parts.append("sim" if cfg.get("use_similarity", True) else "nosim")
    parts.append("crf" if cfg.get("use_crf", True) else "ce")
    return f"nov_{'_'.join(parts)}-{cfg.get('prefix', '01')}"


def pooled_selection_metrics(preds, golds):
    """preds/golds: dict key -> (T,) int8 BIO. Returns (score, summary dict)."""
    if M is None:
        conf = np.zeros((3, 3))
        for k in golds:
            conf += np.bincount(golds[k].astype(int) * 3 + preds[k].astype(int), minlength=9).reshape(3, 3)
        tp = np.diag(conf); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9), 0.0)
        return float(f1.mean()), {"frame_macro_f1": float(f1.mean()), "segF1@0.5": None}
    summ, _, _ = M.evaluate(preds, golds)
    return 0.5 * (summ["frame_macro_f1"] + summ["segF1@0.5"]), summ


def train_model(cfg):
    print(f"\n{'=' * 60}\nSTARTING QUEUED NOVELTY JOB\n{'=' * 60}")
    print(json.dumps(cfg, indent=4))

    SEED = cfg.get("seed", 42)
    set_seed(SEED)
    WINDOW = cfg.get("window_size", 64)
    BATCH = cfg.get("batch_size", 16)
    EPOCHS = cfg.get("epochs", 100)
    PATIENCE = cfg.get("patience", 10)
    EARLY_STOP = cfg.get("early_stopping", True)
    LR = cfg.get("learning_rate", 3e-4)
    WD = cfg.get("weight_decay", 0.01)
    TOL = cfg.get("tolerance_window", 1)
    CLASS_W = cfg.get("class_weights", [0.6, 0.8, 1.0])
    CE_WEIGHT = cfg.get("ce_weight", 0.5)
    USE_CRF = cfg.get("use_crf", True)
    USE_H, USE_D = cfg.get("use_hamer_features", True), cfg.get("use_dinov2_features", False)
    HAMER_DIR = cfg.get("hamer_dir", DEFAULT_HAMER_DIR)
    DINO_DIR = cfg.get("dinov2_dir", DEFAULT_DINOV2_DIR)
    if not os.path.isabs(HAMER_DIR): HAMER_DIR = os.path.join(_PROJECT_ROOT, HAMER_DIR)
    if not os.path.isabs(DINO_DIR): DINO_DIR = os.path.join(_PROJECT_ROOT, DINO_DIR)

    # SignSegmentationDataset hard-codes its kinematic-features path as a bare
    # relative string, so it resolves against the CURRENT directory. Same fix as
    # export_logits.py / train_stage2_finetune.py.
    os.chdir(_PROJECT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_kw = dict(keypoints_dir=DEFAULT_KEYPOINTS_DIR, labels_dir=DEFAULT_LABELS_DIR, split_file=DEFAULT_SPLIT_FILE,
                 window_size=WINDOW, overlap=cfg.get("overlap", 0), tolerance_window=TOL,
                 base_features=cfg.get("base_features", ["x-cord", "y-cord", "z-cord"]),
                 kinematic_features=cfg.get("kinematic_features", []),
                 use_hamer_features=USE_H, hamer_dir=HAMER_DIR, use_dinov2_features=USE_D, dinov2_dir=DINO_DIR)
    train_ds = SignSegmentationDataset(split="train", **ds_kw)
    val_ds = SignSegmentationDataset(split="val", **ds_kw)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # in_channels is read off the real data rather than trusted from the config:
    # enabling kinematic features silently changes it, and a mismatch would only
    # surface as an obscure shape error three layers into the first batch.
    n_ch = int(train_ds[0][0].shape[0])
    if n_ch != cfg.get("in_channels", 3):
        print(f"[SETUP] in_channels in config ({cfg.get('in_channels', 3)}) != channels in the data ({n_ch}); using {n_ch}.")
    cfg["in_channels"] = n_ch

    model_kwargs = {k: cfg[k] for k in MODEL_CONFIG_KEYS if k in cfg}
    model_kwargs.update(num_vertices=cfg.get("num_vertices", 65), in_channels=n_ch)
    if USE_H:
        if train_ds.detected_hamer_dim is None: raise RuntimeError("use_hamer_features=True but no hamer_dim detected.")
        model_kwargs["hamer_dim"] = train_ds.detected_hamer_dim
    if USE_D:
        if train_ds.detected_dinov2_dim is None: raise RuntimeError("use_dinov2_features=True but no dinov2_dim detected.")
        model_kwargs["dinov2_dim"] = train_ds.detected_dinov2_dim
    model = STGCN_Novelty(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SETUP] STGCN_Novelty: {n_params:,} parameters | backbone={cfg.get('backbone', 'xlstm')} "
          f"similarity={cfg.get('use_similarity', True)} crf={USE_CRF} hamer={USE_H} dinov2={USE_D} "
          f"tolerance_window={TOL} metric-selection={'decoded one-to-one + frame F1' if M is not None else 'frame F1 only'}")

    criterion = WeightedCrossEntropyLoss(weights=CLASS_W)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    def unpack(batch):
        i = 0
        feats = batch[i].to(device); i += 1
        labels = batch[i].to(device); i += 1
        hamer = batch[i].to(device) if USE_H else None
        if USE_H: i += 1
        dino = batch[i].to(device) if USE_D else None
        return feats, labels, hamer, dino

    def forward(feats, hamer, dino):
        kw = {}
        if USE_H: kw["hamer"] = hamer
        if USE_D: kw["dinov2"] = dino
        return model(feats, **kw)

    best = {"score": -1.0, "epoch": 0, "state": None, "summary": None}
    history, bad_epochs, t0 = [], 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses, n_nan, n_batches = [], 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False):
            feats, labels, hamer, dino = unpack(batch)
            n_batches += 1
            optimizer.zero_grad()
            logits, _ = forward(feats, hamer, dino)                       # (B,3,T)
            hard = labels.argmax(dim=1)                                   # (B,T)
            ce = criterion(logits, hard)
            loss = (model.crf.nll(logits.permute(0, 2, 1), hard) + CE_WEIGHT * ce) if USE_CRF else ce
            if not torch.isfinite(loss):
                n_nan += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if n_batches and n_nan / n_batches > 0.2:
            raise RuntimeError(f"{n_nan}/{n_batches} training batches produced a non-finite loss in epoch {epoch}. "
                               f"Training is unstable -- lower learning_rate, or try backbone='bilstm' to isolate xLSTM.")

        model.eval()
        preds, golds, k = {}, {}, 0
        with torch.no_grad():
            for batch in val_loader:
                feats, labels, hamer, dino = unpack(batch)
                logits, _ = forward(feats, hamer, dino)
                p = model.decode(logits).cpu().numpy().astype(np.int8)
                g = labels.argmax(dim=1).cpu().numpy().astype(np.int8)
                for i in range(p.shape[0]):
                    preds[f"{k:08d}"], golds[f"{k:08d}"] = p[i], g[i]; k += 1
        score, summ = pooled_selection_metrics(preds, golds)
        seg = summ.get("segF1@0.5")
        print(f"Epoch {epoch}: loss={np.mean(losses) if losses else float('nan'):.4f}  frameF1={summ['frame_macro_f1']:.4f}  "
              f"segF1@0.5={'n/a' if seg is None else f'{seg:.4f}'}  ratio={summ.get('segment_ratio', float('nan')):.3f}  B-F1={summ.get('frame_f1_B', float('nan')):.4f}  score={score:.4f}"
              + (f"  [{n_nan} non-finite batches skipped]" if n_nan else ""))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "score": score,
                        **{kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else None) for kk, vv in summ.items()
                           if kk in ("frame_macro_f1", "frame_f1_B", "segF1@0.5", "segment_ratio")}})
        if score > best["score"]:
            best.update(score=score, epoch=epoch, state=copy.deepcopy(model.state_dict()), summary=summ)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if EARLY_STOP and bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best epoch {best['epoch']}, score {best['score']:.4f})")
            break

    total = time.time() - t0
    run = run_name_for(cfg)
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt = os.path.join(MODEL_DIR, f"{run}.pth")
    torch.save(best["state"], ckpt)
    print(f"Best model (epoch {best['epoch']}, score {best['score']:.4f}) saved to {ckpt}")
    if USE_CRF:
        W = model.crf.effective_transitions().detach().cpu().numpy()
        print("[CRF] learned transition scores [from -> to], order O, I, B (O->I is the fixed forbidden entry):")
        print(np.array2string(W, precision=2, suppress_small=True))

    exp_dir = os.path.join(EXPERIMENTS_DIR, run)
    os.makedirs(exp_dir, exist_ok=True)
    hp = {**cfg, "basename": "novelty", "model_module": "models_novelty", "model_module_dir": MODULE_DIR_NAME,
          "model_class": "STGCN_Novelty", "model_kwargs": model_kwargs, "temporal_downsample_factor": 1,
          "use_face_keypoints": False, "tolerance_window": TOL, "use_hamer_features": USE_H,
          "use_dinov2_features": USE_D, "hamer_dir": cfg.get("hamer_dir", "processed_data/hamer_features"),
          "dinov2_dir": cfg.get("dinov2_dir", "processed_data/dinov2_features"),
          "base_features": ds_kw["base_features"], "kinematic_features": ds_kw["kinematic_features"],
          "window_size": WINDOW}
    with open(os.path.join(exp_dir, "hyperparameters.json"), "w") as f:
        json.dump(hp, f, indent=4, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(exp_dir, "hardware_summary.json"), "w") as f:
        json.dump({"seed": SEED, "machine_hostname": socket.gethostname(), "best_epoch": best["epoch"],
                   "best_score": best["score"], "n_parameters": n_params,
                   "total_training_seconds": round(total, 2),
                   "total_training_time": f"{int(total // 3600)}h {int(total % 3600 // 60)}m"}, f, indent=4)

    print(f"\nDone: {run}. Evaluate it with the decoder study (run from anywhere):\n"
          f"  python decoder_study/export_logits.py --run {run} --root {MODULE_DIR_NAME} --splits val test\n"
          f"  python decoder_study/run_decoder_eval.py --run {run}\n"
          f"  python decoder_study/rescore_protocol.py --run {run}")


if __name__ == "__main__":
    print("Starting novelty_crf queue manager...")
    while True:
        job = get_next_job()
        if job is None:
            print("No more jobs in queue. Exiting.")
            break
        train_model(job)