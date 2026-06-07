import time
import json
import csv
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import our custom modules
from src.dataset import SignSegmentationDataset
from src.models import PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_BiMamba
from src.loss import FocalLoss
from src.metrics import evaluate_batch

# ==============================================================================
# 🎛️ EXPERIMENT CONFIGURATION PANEL
# ==============================================================================
RUN_ALL_MODELS = False           # <-- THE NEW TOGGLE (True = Train all 4 sequentially)

# Options: "pure_mamba", "bi_mamba", "stgcn_mamba", "stgcn_bimamba"
CHOSEN_MODEL = "stgcn_bimamba"    # Only used if RUN_ALL_MODELS is False

BATCH_SIZE = 16      
EPOCHS = 30
LEARNING_RATE = 1e-4
WINDOW_SIZE = 1000
OVERLAP = 200
NUM_VERTICES = 65
TOLERANCE_WINDOW = 5            # 1 (Strict), 3, or 5 frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_bimamba": STGCN_BiMamba
}

# Base hyperparameters (basename is injected dynamically during training)
HYPERPARAMETERS = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "window_size": WINDOW_SIZE,
    "overlap": OVERLAP,
    "num_vertices": NUM_VERTICES,
    "tolerance_window": TOLERANCE_WINDOW,
    "in_channels": 3,
    "d_model": 256,
    "n_layers": 4,
    "focal_loss_gamma": 2.0,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
}
# ==============================================================================

def setup_experiment_dirs(basename):
    logs_parent = "experiments"
    models_parent = "saved_models"
    
    os.makedirs(logs_parent, exist_ok=True)
    os.makedirs(models_parent, exist_ok=True)
    
    existing_dirs = os.listdir(logs_parent)
    max_index = 0
    
    for d in existing_dirs:
        if d.startswith(f"{basename}-"):
            try:
                idx = int(d.split('-')[-1])
                if idx > max_index:
                    max_index = idx
            except ValueError:
                continue
                
    next_index = max_index + 1
    run_name = f"{basename}-{next_index:02d}"
    
    log_dir = os.path.join(logs_parent, run_name)
    model_dir = models_parent 
    
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir, model_dir, run_name

def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}m {int(secs)}s"

def save_plots(history, log_dir):
    epochs = history['epoch']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red', marker='o', markersize=4)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Focal Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(epochs, history['frame_f1'], label='Frame F1', color='green', marker='s', markersize=4)
    ax2.plot(epochs, history['mean_iou'], label='Mean IoU', color='purple', marker='^', markersize=4)
    ax2.plot(epochs, history['segment_f1'], label='Segment % (F1@0.5)', color='orange', marker='D', markersize=4)
    ax2.set_title('Validation Metrics over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.0) 
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved training curves to '{plot_path}'")

def train_model(model_name, model_class, train_loader, val_loader):
    """Encapsulated training function so it can be looped."""
    print("\n" + "="*50)
    print(f"🚀 STARTING TRAINING: {model_name.upper()}")
    print("="*50)
    
    log_dir, model_dir, run_name = setup_experiment_dirs(model_name)
    print(f"📁 Run Path: {log_dir}/")
    
    # Inject dynamic basename and save hyperparams
    local_hp = HYPERPARAMETERS.copy()
    local_hp["basename"] = model_name
    with open(os.path.join(log_dir, "hyperparameters.json"), "w") as f:
        json.dump(local_hp, f, indent=4)
    
    # Initialize Model, Loss, and Optimizer
    model = model_class(num_vertices=NUM_VERTICES, in_channels=3, d_model=256, n_layers=4).to(DEVICE)
    
    criterion = FocalLoss(gamma=HYPERPARAMETERS['focal_loss_gamma']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    training_start_time = time.time()
    epoch_durations = []
    
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'frame_f1': [], 'mean_iou': [], 'segment_f1': [], 'epoch_time': []
    }
    
    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        model.train()
        total_train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for features, targets in loop:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(features)  
            
            loss = criterion(logits, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 4. Validation Loop
        model.eval()
        total_val_loss = 0.0
        val_frame_f1, val_iou, val_seg_f1 = [], [], []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                logits = model(features) 
                
                loss = criterion(logits, targets)
                total_val_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1) 
                preds_np = predictions.cpu().numpy()
                
                if targets.dim() == 3:
                    hard_targets = torch.argmax(targets, dim=1)
                    targets_np = hard_targets.cpu().numpy()
                else:
                    targets_np = targets.cpu().numpy()
                
                batch_metrics = evaluate_batch(preds_np, targets_np)
                val_frame_f1.append(batch_metrics['Frame_F1'])
                val_iou.append(batch_metrics['Mean_IoU'])
                val_seg_f1.append(batch_metrics['Segment_F1_05'])
                
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_f1 = float(np.mean(val_frame_f1))
        epoch_iou = float(np.mean(val_iou))
        epoch_seg = float(np.mean(val_seg_f1))
        
        epoch_time_taken = time.time() - epoch_start_time
        epoch_durations.append(epoch_time_taken)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['frame_f1'].append(epoch_f1)
        history['mean_iou'].append(epoch_iou)
        history['segment_f1'].append(epoch_seg)
        history['epoch_time'].append(round(epoch_time_taken, 2))
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ⏱️ {format_time(epoch_time_taken)}")
        print(f"         └─> Frame F1: {epoch_f1:.4f} | Mean IoU: {epoch_iou:.4f} | Segments %: {epoch_seg:.4f}")

    # --- Exports ---
    total_training_time = time.time() - training_start_time
    average_epoch_time = sum(epoch_durations) / len(epoch_durations)
    
    csv_path = os.path.join(log_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        for row_data in zip(*history.values()):
            writer.writerow(row_data)
        writer.writerow([]) 
        writer.writerow(["Average Time Per Epoch:", format_time(average_epoch_time)])
        writer.writerow(["Total Training Time:", format_time(total_training_time)])

    save_plots(history, log_dir)

    model_save_path = os.path.join(model_dir, f"{run_name}.pth")
    torch.save(model.state_dict(), model_save_path)
    
    print("\n" + "-"*40)
    print(f"🏁 {run_name.upper()} COMPLETE 🏁")
    print(f"Total Time: {format_time(total_training_time)}")
    print("-"*40)

    # --- CRITICAL MEMORY FLUSH ---
    # Delete the model and optimizer to free up GPU VRAM for the next model in the loop
    del model
    del optimizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("📦 Loading Dataset into RAM (Only happens once)...")
    full_dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        tolerance_window=TOLERANCE_WINDOW
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create the dataloaders ONCE
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    if RUN_ALL_MODELS:
        print("🔄 RUN_ALL_MODELS is TRUE. Training all architectures sequentially...")
        for name, cls in MODEL_REGISTRY.items():
            train_model(name, cls, train_loader, val_loader)
        print("\n🎉 ALL ARCHITECTURES HAVE FINISHED TRAINING 🎉")
    else:
        print(f"▶️ RUN_ALL_MODELS is FALSE. Training only {CHOSEN_MODEL}...")
        train_model(CHOSEN_MODEL, MODEL_REGISTRY[CHOSEN_MODEL], train_loader, val_loader)