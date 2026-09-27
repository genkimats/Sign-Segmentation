"""
train_multitask.py -- trains the STGCN_*_Multitask models (segmentation +
auxiliary gloss classification) via a combined loss. Reuses the existing,
proven window-based training paradigm (same window_size/overlap semantics,
same WeightedCrossEntropyLoss for the BIO half, same seeding/NaN-restart
approach as train.py) -- the only new pieces are the gloss-id data stream
and the second loss term.

Combined loss: bio_loss + gloss_loss_weight * gloss_loss, where gloss_loss
uses ignore_index=-1 to exclude Outside/padding frames (nothing meaningful
to predict a gloss for there) -- computed only over frames where a real
gloss is actually active.

Fully separate from your existing infrastructure: own queue file
(train_queue_multitask.json), own model directory (saved_models_multitask/),
own experiment directory (experiments_multitask/).

Run extract_gloss_ids.py first. Queue jobs via queue_train_multitask.py.
"""
import os
import sys
import json
import time
import random
import copy
import socket
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_multitask import SignSegmentationDatasetMultitask
from models_multitask import STGCN_BiLSTM_Multitask, STGCN_Transformer_Multitask, STGCN_BiMamba_Multitask
from src.loss import WeightedCrossEntropyLoss

MODEL_REGISTRY = {
    "stgcn_bilstm_multitask": STGCN_BiLSTM_Multitask,
    "stgcn_transformer_multitask": STGCN_Transformer_Multitask,
    "stgcn_bimamba_multitask": STGCN_BiMamba_Multitask,
}
MAMBA_BASED_MODELS = ["stgcn_bimamba_multitask"]

DEFAULT_KEYPOINTS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "keypoints")
DEFAULT_LABELS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "BIO_tags")
DEFAULT_HAMER_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "hamer_features")
DEFAULT_DINOV2_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "dinov2_features")
DEFAULT_GLOSS_IDS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "gloss_ids")
DEFAULT_SPLIT_FILE = os.path.join(_PROJECT_ROOT, "dataset_splits.json")

QUEUE_FILE = os.path.join(_SCRIPT_DIR, "train_queue_multitask.json")
MODEL_DIR = os.path.join(_SCRIPT_DIR, "saved_models_multitask")
EXPERIMENTS_DIR = os.path.join(_SCRIPT_DIR, "experiments_multitask")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    next_job = queue.pop(1)
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
    return next_job


def compute_frame_f1(logits, hard_labels, num_classes=3):
    preds = torch.argmax(logits, dim=1)  # (B, T)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (hard_labels == c)).sum().item()
        fp = ((preds == c) & (hard_labels != c)).sum().item()
        fn = ((preds != c) & (hard_labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def train_model(config):
    print(f"\n{'='*60}\n🚀 STARTING QUEUED MULTITASK JOB\n{'='*60}")
    print(json.dumps(config, indent=4))

    SEED = config.get("seed", 42)
    set_seed(SEED)

    MODEL_NAME = config["basename"]
    if MODEL_NAME not in MODEL_REGISTRY:
        raise ValueError(f"Unknown basename '{MODEL_NAME}'. Available: {list(MODEL_REGISTRY.keys())}")

    WINDOW_SIZE = config.get("window_size", 64)
    OVERLAP = config.get("overlap", 0)
    BATCH_SIZE = config.get("batch_size", 16)
    EPOCHS = config.get("epochs", 100)
    EARLY_STOPPING = config.get("early_stopping", True)
    PATIENCE = config.get("patience", 10)
    LEARNING_RATE = config.get("learning_rate", 0.0001)
    TOLERANCE_WINDOW = config.get("tolerance_window", 5)
    CLASS_WEIGHTS = config.get("class_weights", [0.6, 0.8, 1.0])
    GLOSS_LOSS_WEIGHT = config.get("gloss_loss_weight", 0.1)
    # Normalizing by log(vocab_size) -- the loss a uniform-random classifier
    # would get -- puts the gloss loss on a comparable scale to the 3-class
    # BIO loss regardless of vocabulary size, so gloss_loss_weight means
    # roughly the same thing across different vocab sizes instead of being
    # silently dominated by however large the vocabulary happens to be.
    # Confirmed necessary empirically: at weight=0.3 unnormalized, the
    # weighted gloss term was over 3x LARGER than the bio loss itself
    # (0.3 x ~5.5 vs ~0.53), and F1 dropped to ~0.55 from a ~0.81 baseline --
    # the auxiliary task was dominating, not assisting.
    NORMALIZE_GLOSS_LOSS = config.get("normalize_gloss_loss", True)
    BASE_FEATURES = config.get("base_features", ["x-cord", "y-cord", "z-cord"])
    KINEMATIC_FEATURES = config.get("kinematic_features", [])
    IN_CHANNELS = config.get("in_channels", 3)
    USE_HAMER_FEATURES = config.get("use_hamer_features", False)
    HAMER_DIR = config.get("hamer_dir", DEFAULT_HAMER_DIR)
    USE_DINOV2_FEATURES = config.get("use_dinov2_features", False)
    DINOV2_DIR = config.get("dinov2_dir", DEFAULT_DINOV2_DIR)
    D_MODEL = config.get("d_model", 256)
    N_LAYERS = config.get("n_layers", 4)
    MAMBA_D_STATE = config.get("mamba_d_state", 16)
    MAMBA_D_CONV = config.get("mamba_d_conv", 4)
    MAMBA_EXPAND = config.get("mamba_expand", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SignSegmentationDatasetMultitask(
        keypoints_dir=DEFAULT_KEYPOINTS_DIR, labels_dir=DEFAULT_LABELS_DIR, split_file=DEFAULT_SPLIT_FILE,
        split="train", window_size=WINDOW_SIZE, overlap=OVERLAP, tolerance_window=TOLERANCE_WINDOW,
        base_features=BASE_FEATURES, kinematic_features=KINEMATIC_FEATURES,
        use_hamer_features=USE_HAMER_FEATURES, hamer_dir=HAMER_DIR,
        use_dinov2_features=USE_DINOV2_FEATURES, dinov2_dir=DINOV2_DIR,
        gloss_ids_dir=DEFAULT_GLOSS_IDS_DIR,
    )
    val_dataset = SignSegmentationDatasetMultitask(
        keypoints_dir=DEFAULT_KEYPOINTS_DIR, labels_dir=DEFAULT_LABELS_DIR, split_file=DEFAULT_SPLIT_FILE,
        split="val", window_size=WINDOW_SIZE, overlap=OVERLAP, tolerance_window=TOLERANCE_WINDOW,
        base_features=BASE_FEATURES, kinematic_features=KINEMATIC_FEATURES,
        use_hamer_features=USE_HAMER_FEATURES, hamer_dir=HAMER_DIR,
        use_dinov2_features=USE_DINOV2_FEATURES, dinov2_dir=DINOV2_DIR,
        gloss_ids_dir=DEFAULT_GLOSS_IDS_DIR,
    )

    gloss_vocab_size = train_dataset.gloss_vocab_size
    print(f"[SETUP] Gloss vocabulary size: {gloss_vocab_size} (including UNK id {train_dataset.gloss_unk_id})")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_kwargs = dict(
        num_vertices=config.get("num_vertices", 65), in_channels=IN_CHANNELS, d_model=D_MODEL,
        n_layers=N_LAYERS, gloss_vocab_size=gloss_vocab_size,
    )
    if MODEL_NAME in MAMBA_BASED_MODELS:
        model_kwargs.update(mamba_d_state=MAMBA_D_STATE, mamba_d_conv=MAMBA_D_CONV, mamba_expand=MAMBA_EXPAND)
    if USE_HAMER_FEATURES:
        if train_dataset.detected_hamer_dim is None:
            raise RuntimeError("use_hamer_features=True but no hamer_dim was detected during validation.")
        model_kwargs["hamer_dim"] = train_dataset.detected_hamer_dim
    if USE_DINOV2_FEATURES:
        if train_dataset.detected_dinov2_dim is None:
            raise RuntimeError("use_dinov2_features=True but no dinov2_dim was detected during validation.")
        model_kwargs["dinov2_dim"] = train_dataset.detected_dinov2_dim

    model = MODEL_REGISTRY[MODEL_NAME](**model_kwargs).to(device)
    bio_criterion = WeightedCrossEntropyLoss(weights=CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    def unpack_batch(batch):
        idx = 0
        features = batch[idx].to(device); idx += 1
        labels = batch[idx].to(device); idx += 1
        gloss_ids = batch[idx].to(device); idx += 1
        hamer = batch[idx].to(device) if USE_HAMER_FEATURES else None
        if USE_HAMER_FEATURES: idx += 1
        dinov2 = batch[idx].to(device) if USE_DINOV2_FEATURES else None
        if USE_DINOV2_FEATURES: idx += 1
        return features, labels, gloss_ids, hamer, dinov2

    def call_model(model, features, hamer, dinov2):
        kwargs = {}
        if USE_HAMER_FEATURES: kwargs["hamer"] = hamer
        if USE_DINOV2_FEATURES: kwargs["dinov2"] = dinov2
        return model(features, **kwargs)

    best_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_bio_losses, epoch_gloss_losses = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for batch in loop:
            features, labels, gloss_ids, hamer, dinov2 = unpack_batch(batch)

            optimizer.zero_grad()
            bio_logits, gloss_logits, _ = call_model(model, features, hamer, dinov2)

            hard_bio_labels = torch.argmax(labels, dim=1)
            bio_loss = bio_criterion(bio_logits, hard_bio_labels)

            # ignore_index=-1 excludes Outside/padding frames from the gloss
            # loss entirely -- only frames where a real gloss is active
            # contribute. If a batch happens to contain no real-gloss frames
            # at all (e.g. an all-Outside window), this would produce a NaN
            # from F.cross_entropy's 0/0 mean -- guarded below.
            has_real_gloss = (gloss_ids != -1).any()
            if has_real_gloss:
                gloss_loss = F.cross_entropy(gloss_logits, gloss_ids, ignore_index=-1)
            else:
                gloss_loss = torch.tensor(0.0, device=device)

            # Normalize by log(vocab_size) -- the loss a uniform-random
            # classifier would get -- before applying the weight. Without
            # this, gloss_loss_weight's effective strength depends heavily on
            # vocabulary size, and a "modest" 0.3 can still dominate the bio
            # loss by 3x+ purely because a 2813-class problem starts with a
            # much larger raw loss than a 3-class one, not because the weight
            # itself was set high.
            if NORMALIZE_GLOSS_LOSS and gloss_vocab_size > 1:
                gloss_loss_scaled = gloss_loss / math.log(gloss_vocab_size)
            else:
                gloss_loss_scaled = gloss_loss

            total_loss = bio_loss + GLOSS_LOSS_WEIGHT * gloss_loss_scaled

            if torch.isnan(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_bio_losses.append(bio_loss.item())
            epoch_gloss_losses.append(gloss_loss.item())
            loop.set_postfix(bio=f"{bio_loss.item():.3f}", gloss=f"{gloss_loss.item():.3f}")

        scheduler.step()

        model.eval()
        val_f1s = []
        with torch.no_grad():
            for batch in val_loader:
                features, labels, gloss_ids, hamer, dinov2 = unpack_batch(batch)
                bio_logits, gloss_logits, _ = call_model(model, features, hamer, dinov2)
                hard_bio_labels = torch.argmax(labels, dim=1)
                val_f1s.append(compute_frame_f1(bio_logits, hard_bio_labels))

        mean_val_f1 = float(np.mean(val_f1s)) if val_f1s else 0.0
        mean_bio_loss = float(np.mean(epoch_bio_losses)) if epoch_bio_losses else float("nan")
        mean_gloss_loss = float(np.mean(epoch_gloss_losses)) if epoch_gloss_losses else float("nan")
        print(f"Epoch {epoch}: bio_loss={mean_bio_loss:.4f}, gloss_loss={mean_gloss_loss:.4f}, "
              f"val_f1={mean_val_f1:.4f}")

        if mean_val_f1 > best_f1:
            best_f1 = mean_val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch}, F1={best_f1:.4f})")
            break

    total_time = time.time() - start_time

    prefix = config.get("prefix", "01")
    run_name = f"{MODEL_NAME}-{prefix}"

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, f"{run_name}.pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"✅ Best model (epoch {best_epoch}, F1={best_f1:.4f}) saved to {model_save_path}")

    exp_dir = os.path.join(EXPERIMENTS_DIR, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    hardware_summary = {
        "seed": SEED,
        "machine_hostname": socket.gethostname(),
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "gloss_loss_weight": GLOSS_LOSS_WEIGHT,
        "normalize_gloss_loss": NORMALIZE_GLOSS_LOSS,
        "gloss_vocab_size": gloss_vocab_size,
        "total_training_seconds": round(total_time, 2),
        "total_training_time": f"{int(total_time // 60)}m {int(total_time % 60)}s",
    }
    with open(os.path.join(exp_dir, "hardware_summary.json"), "w") as f:
        json.dump(hardware_summary, f, indent=4)

    print(f"\nDone. Best F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    print("🚦 Starting Multitask Train Queue Manager...")
    while True:
        job_config = get_next_job()
        if job_config is None:
            print("No more jobs in queue. Exiting.")
            break
        train_model(job_config)