import time
import json
import csv
import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import copy
from torch.amp import GradScaler, autocast

# Import our custom modules
from src.dataset import SignSegmentationDataset
from src.models import PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_BiMamba, Decoupled_STGCN_Mamba, Decoupled_STGCN_BiMamba, BiLSTM_Baseline
from src.metrics import evaluate_batch
from src.loss import CombinedBoundaryLoss, FocalLoss, StandardCrossEntropyLoss, WeightedCrossEntropyLoss
from src.decoder import decode_predictions

QUEUE_FILE = "train_queue.json"

# ==============================================================================
# Model Registry Mapping
# ==============================================================================
MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_bimamba": STGCN_BiMamba,
    "decoupled_stgcn_mamba": Decoupled_STGCN_Mamba,
    "decoupled_stgcn_bimamba": Decoupled_STGCN_BiMamba,
    "bilstm_baseline": BiLSTM_Baseline
}

def get_next_job():
    """Reads the queue file, pops the first job, updates the file, and returns the job."""
    if not os.path.exists(QUEUE_FILE):
        return None
        
    with open(QUEUE_FILE, "r") as f:
        try:
            queue = json.load(f)
        except json.JSONDecodeError:
            return None
            
    if not queue:
        return None
        
    # Pop the top job
    next_job = queue.pop(0)
    
    # Save the remaining queue
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    return next_job

def get_next_experiment_prefix(model_name):
    """Finds the next available prefix like 01, 02, etc."""
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        return "01"
        
    existing_dirs = os.listdir(experiments_dir)
    model_dirs = [d for d in existing_dirs if d.startswith(f"{model_name}-")]
    
    if not model_dirs:
        return "01"
        
    prefixes = []
    for d in model_dirs:
        try:
            prefix = int(d.split('-')[-1])
            prefixes.append(prefix)
        except ValueError:
            continue
            
    if not prefixes:
        return "01"
        
    next_prefix = max(prefixes) + 1
    return f"{next_prefix:02d}"

def train_model(config):
    """Executes a single training run based on the provided configuration dictionary."""
    print(f"\\n{'='*60}\\n🚀 STARTING QUEUED JOB\\n{'='*60}")
    print(json.dumps(config, indent=4))
    
    # Unpack config
    BATCH_SIZE = config["batch_size"]
    EPOCHS = config["epochs"]
    LEARNING_RATE = config["learning_rate"]
    WINDOW_SIZE = config["window_size"]
    OVERLAP = config["overlap"]
    NUM_VERTICES = config["num_vertices"]
    TOLERANCE_WINDOW = config["tolerance_window"]
    LOSS_FUNCTION = config["loss_function"]
    CLASS_WEIGHTS = config["class_weights"]
    USE_FULL_LENGTH = config["use_full_length"]
    BASE_FEATURES = config["base_features"]
    KINEMATIC_FEATURES = config["kinematic_features"]
    IN_CHANNELS = config["in_channels"]
    DECODER_STRATEGY = config["decoder_strategy"]
    DECODER_THRESHOLD = config["decoder_threshold"]
    D_MODEL = config["d_model"]
    N_LAYERS = config["n_layers"]
    FOCAL_LOSS_GAMMA = config["focal_loss_gamma"]
    OPTIMIZER_NAME = config["optimizer"]
    SCHEDULER_NAME = config["scheduler"]
    MODEL_NAME = config["basename"]
    
    prefix = get_next_experiment_prefix(MODEL_NAME)
    run_name = f"{MODEL_NAME}-{prefix}"
    print(f"📁 Assigned Run Name: {run_name}")
    
    # Setup directories
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    exp_dir = os.path.join("experiments", run_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save hyperparams
    with open(os.path.join(exp_dir, "hyperparameters.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    # Dataset
    full_dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/labels",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        tolerance_window=TOLERANCE_WINDOW,
        use_full_length=USE_FULL_LENGTH,
        base_features=BASE_FEATURES,
        kinematic_features=KINEMATIC_FEATURES 
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    loader_batch_size = 1 if USE_FULL_LENGTH else BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = MODEL_REGISTRY.get(MODEL_NAME)
    
    if not model_class:
        print(f"❌ Error: Model '{MODEL_NAME}' not found in registry. Skipping.")
        return
        
    if MODEL_NAME == "bilstm_baseline":
        model = model_class(
            in_channels=IN_CHANNELS,
            num_vertices=NUM_VERTICES,
            num_classes=3,
            d_model=D_MODEL,
            n_layers=N_LAYERS
        ).to(device)
    else:
        model = model_class(
            in_channels=IN_CHANNELS,
            num_vertices=NUM_VERTICES,
            num_classes=3,
            d_model=D_MODEL,
            n_layers=N_LAYERS
        ).to(device)

    # Loss Selection
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
    
    if LOSS_FUNCTION == "bcl":
        criterion = CombinedBoundaryLoss(
            gamma=FOCAL_LOSS_GAMMA, 
            weights=weights, 
            contrastive_weight=config.get("contrastive_weight", 0.15)
        )
    elif LOSS_FUNCTION == "standard_ce":
        criterion = StandardCrossEntropyLoss()
    elif LOSS_FUNCTION == "weighted_ce":
        criterion = WeightedCrossEntropyLoss(weights=weights)
    elif LOSS_FUNCTION == "focal":
        criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weights=weights)
    else:
        raise ValueError(f"Unknown loss function '{LOSS_FUNCTION}'")

    # Optimizer & Scheduler
    if OPTIMIZER_NAME == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
    if SCHEDULER_NAME == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        scheduler = None
        
    scaler = GradScaler('cuda')
    metrics_log_path = os.path.join(exp_dir, f"{run_name}_training_metrics.csv")
    
    with open(metrics_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'frame_f1', 'mean_iou', 'segment_f1', 'epoch_time'])
    
    # Trackers for Checkpointing
    best_iou = -1.0
    best_epoch = 0
    best_model_state = None
    
    # 4. Training Loop
    start_train_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for features, masks, labels in loop:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                if LOSS_FUNCTION == "bcl":
                    logits, embeddings = model(features)
                    loss = criterion(logits, labels, embeddings)
                else:
                    logits, _ = model(features)
                    loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        if scheduler:
            scheduler.step()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_frame_f1, val_iou, val_seg_f1 = [], [], []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
            for features, masks, labels in val_loop:
                features = features.to(device)
                labels = labels.to(device)
                
                with autocast('cuda'):
                    if LOSS_FUNCTION == "bcl":
                        logits, embeddings = model(features)
                        loss = criterion(logits, labels, embeddings)
                    else:
                        logits, _ = model(features)
                        loss = criterion(logits, labels)
                        
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                
                for i in range(features.size(0)):
                    valid_len = int(masks[i].sum().item())
                    if valid_len == 0: continue
                    
                    true_seq = labels[i, :valid_len].cpu().numpy()
                    pred_logits = logits[i, :valid_len].cpu().numpy() 
                    
                    pred_seq = decode_predictions(
                        pred_logits, 
                        strategy=DECODER_STRATEGY, 
                        threshold=DECODER_THRESHOLD
                    )
                    
                    f_f1, iou, s_f1 = evaluate_batch([true_seq], [pred_seq])
                    val_frame_f1.append(f_f1)
                    val_iou.append(iou)
                    val_seg_f1.append(s_f1)
                    
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_f1 = float(np.mean(val_frame_f1)) if val_frame_f1 else 0.0
        epoch_iou = float(np.mean(val_iou)) if val_iou else 0.0
        epoch_seg = float(np.mean(val_seg_f1)) if val_seg_f1 else 0.0
        
        epoch_end_time = time.time()
        epoch_duration = round(epoch_end_time - epoch_start_time, 2)
        
        # --- Checkpoint Best Model ---
        if epoch_iou > best_iou:
            best_iou = epoch_iou
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Frame F1: {epoch_f1:.4f} | "
              f"Mean IoU: {epoch_iou:.4f} | "
              f"Seg F1: {epoch_seg:.4f} | "
              f"Time: {epoch_duration}s")
              
        with open(metrics_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_train_loss, avg_val_loss, epoch_f1, epoch_iou, epoch_seg, epoch_duration])

    total_time = time.time() - start_train_time
    total_minutes, total_seconds = divmod(int(total_time), 60)
    avg_epoch_time = int(total_time / EPOCHS)
    avg_minutes, avg_seconds = divmod(avg_epoch_time, 60)
    
    with open(metrics_log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Average Time Per Epoch:', f"{avg_minutes}m {avg_seconds}s"])
        writer.writerow(['Total Training Time:', f"{total_minutes}m {total_seconds}s"])
        
    model_save_path = os.path.join(model_dir, f"{run_name}.pth")
    if best_model_state:
        torch.save(best_model_state, model_save_path)
        print(f"✅ Best Model (Epoch {best_epoch} | IoU: {best_iou:.4f}) saved to {model_save_path}")
    else:
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Final Model saved to {model_save_path}")
        
    print(f"🏁 Finished {run_name}\\n")

if __name__ == "__main__":
    print("🚦 Starting Train Queue Manager...")
    
    jobs_processed = 0
    while True:
        job_config = get_next_job()
        
        if job_config is None:
            if jobs_processed == 0:
                print("📭 Queue is empty. No jobs to run.")
            else:
                print(f"🎉 All {jobs_processed} queued jobs completed successfully. Shutting down.")
            break
            
        train_model(job_config)
        jobs_processed += 1
