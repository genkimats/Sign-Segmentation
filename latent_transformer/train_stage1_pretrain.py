"""
train_stage1_pretrain.py -- Stage 1 of the two-stage latent Transformer:
trains TransformerReconstructionAutoencoder via masked MSE reconstruction
loss. No BIO labels involved at all -- this is purely self-supervised,
matching Sign-Mamba's own Stage I training (their Eq. 7 MSE loss).

Padded frames (only present for videos shorter than window_size) are masked
OUT of the loss -- see dataset_reconstruction.py's docstring for why zero-
padding a reconstruction TARGET, unlike a BIO label, has no meaningful value
to predict.

Output: a checkpoint containing the FULL autoencoder state dict (encoder +
decoder). Stage 2 (train_stage2_finetune.py) loads this and keeps only the
encoder portion (stgcn_blocks, encoder_proj, transformer_encoder) -- the
decoder is discarded after Stage 1, exactly as in the paper (their S-Decoder
is kept for generation; here the segmentation head replaces that role, so
the pretraining decoder itself is scaffolding, not part of the deployed
model).

Own queue file, own model/experiment directories, fully separate from every
other training script in this project -- run from anywhere.
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
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_reconstruction import ReconstructionDataset, DEFAULT_SPLIT_FILE
from models_latent_transformer import TransformerReconstructionAutoencoder

QUEUE_FILE = os.path.join(_SCRIPT_DIR, "train_queue_stage1.json")
MODEL_DIR = os.path.join(_SCRIPT_DIR, "saved_models_stage1")
EXPERIMENTS_DIR = os.path.join(_SCRIPT_DIR, "experiments_stage1")


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


def masked_mse_loss(pred, target, valid_lengths, device):
    B, T, D = pred.shape
    idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    mask = (idx < valid_lengths.to(device).unsqueeze(1)).float()  # (B, T)
    sq_err = (pred - target) ** 2
    masked = sq_err * mask.unsqueeze(-1)
    denom = mask.sum() * D
    return masked.sum() / denom.clamp(min=1.0)


def train_model(config):
    print(f"\n{'='*60}\n🚀 STARTING QUEUED STAGE 1 (PRETRAINING) JOB\n{'='*60}")
    print(json.dumps(config, indent=4))

    SEED = config.get("seed", 42)
    set_seed(SEED)

    WINDOW_SIZE = config.get("window_size", 64)
    OVERLAP = config.get("overlap", 0)
    BATCH_SIZE = config.get("batch_size", 16)
    EPOCHS = config.get("epochs", 100)
    EARLY_STOPPING = config.get("early_stopping", True)
    PATIENCE = config.get("patience", 10)
    LEARNING_RATE = config.get("learning_rate", 0.0001)
    BASE_FEATURES = config.get("base_features", ["x-cord", "y-cord", "z-cord"])
    KINEMATIC_FEATURES = config.get("kinematic_features", [])
    IN_CHANNELS = config.get("in_channels", 3)
    D_MODEL = config.get("d_model", 256)
    N_LAYERS = config.get("n_layers", 4)
    NHEAD = config.get("nhead", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ReconstructionDataset(
        split_file=DEFAULT_SPLIT_FILE, split="train", window_size=WINDOW_SIZE, overlap=OVERLAP,
        base_features=BASE_FEATURES, kinematic_features=KINEMATIC_FEATURES,
    )
    val_dataset = ReconstructionDataset(
        split_file=DEFAULT_SPLIT_FILE, split="val", window_size=WINDOW_SIZE, overlap=OVERLAP,
        base_features=BASE_FEATURES, kinematic_features=KINEMATIC_FEATURES,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerReconstructionAutoencoder(
        num_vertices=config.get("num_vertices", 65), in_channels=IN_CHANNELS, d_model=D_MODEL,
        n_layers=N_LAYERS, nhead=NHEAD,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for features, valid_lengths, _ in loop:
            features = features.to(device)
            optimizer.zero_grad()
            reconstruction, raw_target, _ = model(features)
            loss = masked_mse_loss(reconstruction, raw_target, valid_lengths, device)

            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            loop.set_postfix(mse=f"{loss.item():.5f}")

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, valid_lengths, _ in val_loader:
                features = features.to(device)
                reconstruction, raw_target, _ = model(features)
                loss = masked_mse_loss(reconstruction, raw_target, valid_lengths, device)
                val_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {epoch}: train_mse={mean_train_loss:.5f}, val_mse={mean_val_loss:.5f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch}, val_mse={best_val_loss:.5f})")
            break

    total_time = time.time() - start_time

    prefix = config.get("prefix", "01")
    run_name = f"stgcn_transformer_autoencoder-{prefix}"

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, f"{run_name}.pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"✅ Best model (epoch {best_epoch}, val_mse={best_val_loss:.5f}) saved to {model_save_path}")
        print(f"   -> pass this path to train_stage2_finetune.py's pretrained_checkpoint config field")

    exp_dir = os.path.join(EXPERIMENTS_DIR, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    hardware_summary = {
        "seed": SEED, "machine_hostname": socket.gethostname(),
        "best_epoch": best_epoch, "best_val_mse": best_val_loss,
        "d_model": D_MODEL, "n_layers": N_LAYERS, "nhead": NHEAD,
        "total_training_seconds": round(total_time, 2),
        "total_training_time": f"{int(total_time // 60)}m {int(total_time % 60)}s",
    }
    with open(os.path.join(exp_dir, "hyperparameters.json"), "w") as f:
        json.dump({**config, "resolved_model_path": model_save_path}, f, indent=4)
    with open(os.path.join(exp_dir, "hardware_summary.json"), "w") as f:
        json.dump(hardware_summary, f, indent=4)

    print(f"\nDone. Best val MSE: {best_val_loss:.5f} at epoch {best_epoch}")


if __name__ == "__main__":
    print("🚦 Starting Stage 1 (Pretraining) Queue Manager...")
    while True:
        job_config = get_next_job()
        if job_config is None:
            print("No more jobs in queue. Exiting.")
            break
        train_model(job_config)