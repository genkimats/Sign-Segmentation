import time
import json
import csv
import os
import socket
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

# --- Imports for Confusion Matrix ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import our custom modules
from src.dataset import SignSegmentationDataset
from src.models import (PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_MLP_Mamba, 
                        STGCN_BiMamba, Decoupled_STGCN_Mamba, BiLSTM_Baseline, STGCN_BiLSTM, 
                        TransformerBaseline, STGCN_Transformer, Latent_STGCN_Mamba,
                        CTRGCN_Mamba, InfoGCN_Mamba, ShiftGCN_Mamba, SpatialTransformer_Mamba,
                        HDGCN_Mamba, HyperSign_Mamba)
from src.metrics import evaluate_batch
from src.loss import CombinedBoundaryLoss, FocalLoss, StandardCrossEntropyLoss, WeightedCrossEntropyLoss, UnifiedCTCLoss, WeightedCE_TMSE_Loss, WeightedNLLLoss
# (Removed decoder import since we no longer use it in training/validation)

QUEUE_FILE = "train_queue.json"

# ==============================================================================
# Model Registry Mapping
# ==============================================================================
MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_mlp_mamba": STGCN_MLP_Mamba,
    "stgcn_bimamba": STGCN_BiMamba,
    "decoupled_stgcn_mamba": Decoupled_STGCN_Mamba,
    "bilstm_baseline": BiLSTM_Baseline,
    "stgcn_bilstm": STGCN_BiLSTM,
    "transformer_baseline": TransformerBaseline,
    "stgcn_transformer": STGCN_Transformer,
    "latent_stgcn_mamba": Latent_STGCN_Mamba,
    "ctrgcn_mamba": CTRGCN_Mamba,
    "infogcn_mamba": InfoGCN_Mamba,
    "shiftgcn_mamba": ShiftGCN_Mamba,
    "spatial_transformer_mamba": SpatialTransformer_Mamba,
    "hdgcn_mamba": HDGCN_Mamba,
    "hypersign_mamba": HyperSign_Mamba
}

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

def train_model(config):
    print(f"\n{'='*60}\n🚀 STARTING QUEUED JOB\n{'='*60}")
    print(json.dumps(config, indent=4))
    
    BATCH_SIZE = config["batch_size"]
    EPOCHS = config["epochs"]
    MIN_EPOCHS = config.get("min_epochs", 15) 
    EARLY_STOPPING = config.get("early_stopping", True)
    PATIENCE = config.get("patience", 10)
    LEARNING_RATE = config["learning_rate"]
    WINDOW_SIZE = config["window_size"]
    OVERLAP = config["overlap"]
    NUM_VERTICES = config["num_vertices"]
    TOLERANCE_WINDOW = config["tolerance_window"]
    DOWNSAMPLE_FACTOR = config.get("temporal_downsample_factor", 1) 
    LOSS_FUNCTION = config["loss_function"]
    CLASS_WEIGHTS = config["class_weights"]
    USE_FULL_LENGTH = config.get("use_full_length", False)
    BASE_FEATURES = config["base_features"]
    KINEMATIC_FEATURES = config["kinematic_features"]
    IN_CHANNELS = config["in_channels"]
    USE_FACE_KEYPOINTS = config.get("use_face_keypoints", False)
    FACE_DIR = config.get("face_dir", "processed_data/face_keypoints_normalized")
    D_MODEL = config["d_model"]
    N_LAYERS = config["n_layers"]
    FOCAL_LOSS_GAMMA = config.get("focal_loss_gamma", 2.0)
    OPTIMIZER_NAME = config["optimizer"]
    SCHEDULER_NAME = config["scheduler"]
    MODEL_NAME = config["basename"]
    
    prefix = config.get("prefix", "01")
    run_name = f"{MODEL_NAME}-{prefix}"
    print(f"📁 Assigned Run Name: {run_name}")
    
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    exp_dir = os.path.join("experiments", run_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, "hyperparameters.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    train_dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        split_file="dataset_splits.json",
        split="train",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        tolerance_window=TOLERANCE_WINDOW,
        use_full_length=USE_FULL_LENGTH,
        base_features=BASE_FEATURES,
        kinematic_features=KINEMATIC_FEATURES,
        temporal_downsample_factor=DOWNSAMPLE_FACTOR,
        use_face_keypoints=USE_FACE_KEYPOINTS,
        face_dir=FACE_DIR
    )
    
    val_dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        split_file="dataset_splits.json",
        split="val",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        tolerance_window=TOLERANCE_WINDOW,
        use_full_length=USE_FULL_LENGTH,
        base_features=BASE_FEATURES,
        kinematic_features=KINEMATIC_FEATURES,
        temporal_downsample_factor=DOWNSAMPLE_FACTOR,
        use_face_keypoints=USE_FACE_KEYPOINTS,
        face_dir=FACE_DIR
    )
    
    loader_batch_size = 1 if USE_FULL_LENGTH else BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = MODEL_REGISTRY.get(MODEL_NAME)
    
    if not model_class:
        print(f"❌ Error: Model '{MODEL_NAME}' not found in registry. Skipping.")
        return
        
    model_kwargs = {
        "in_channels": IN_CHANNELS,
        "num_vertices": NUM_VERTICES,
        "num_classes": 3,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS
    }
    
    if MODEL_NAME in ["transformer_baseline", "stgcn_transformer"]:
        model_kwargs["nhead"] = config.get("nhead", 8)
        model_kwargs["dim_feedforward"] = config.get("dim_feedforward", D_MODEL * 4)
        
    if MODEL_NAME == "stgcn_mlp_mamba":
        model_kwargs["mlp_expansion_factor"] = config.get("mlp_expansion_factor", 4)
        
    if MODEL_NAME in ["latent_stgcn_mamba", "ctrgcn_mamba", "infogcn_mamba", "shiftgcn_mamba", "spatial_transformer_mamba", "hdgcn_mamba", "hypersign_mamba"]:
        model_kwargs["latent_dim"] = config.get("latent_dim", 128)

    MAX_REDOS = 5
    redo_count = 0
    total_nan_this_run = 0
    
    while redo_count <= MAX_REDOS:
        if redo_count > 0:
            print(f"\n🔄 RESTARTING TRAINING (Attempt {redo_count + 1}/{MAX_REDOS + 1}) DUE TO NAN EXPLOSION...")
            
        model = model_class(**model_kwargs).to(device)
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
        
        if LOSS_FUNCTION == "bcl":
            criterion = CombinedBoundaryLoss(focal_gamma=FOCAL_LOSS_GAMMA, contrastive_weight=config.get("contrastive_weight", 0.15))
        elif LOSS_FUNCTION == "unified_ctc":
            criterion = UnifiedCTCLoss(blank_idx=0, ctc_weight=config.get("ctc_weight", 0.5))
        elif LOSS_FUNCTION == "standard_ce":
            criterion = StandardCrossEntropyLoss()
        elif LOSS_FUNCTION == "weighted_ce":
            criterion = WeightedCrossEntropyLoss(weights=weights)
        elif LOSS_FUNCTION == "focal":
            criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA)
        elif LOSS_FUNCTION == "wce_tmse":
            criterion = WeightedCE_TMSE_Loss(
                weights=weights,
                tmse_weight=config.get("tmse_weight", 0.15),
                threshold=config.get("tmse_threshold", 0.1)
            )
        elif LOSS_FUNCTION == "weighted_nll":
            criterion = WeightedNLLLoss(weights=weights)
        else:
            raise ValueError(f"Unknown loss function '{LOSS_FUNCTION}'")

        if OPTIMIZER_NAME == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
        if SCHEDULER_NAME == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        else:
            scheduler = None
            
        metrics_log_path = os.path.join(exp_dir, "training_metrics.csv")
        with open(metrics_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'frame_f1', 'mean_iou', 'segment_f1', 'epoch_time', 'epoch_nan_count'])
        
        best_combined_score = -1.0
        best_epoch = 0
        best_model_state = None
        epochs_without_improvement = 0
        actual_epochs_ran = 0
        total_nan_this_run = 0
        gpu_utilization_samples = []
        # Reset here (not once before the while loop) so total_training_time only ever
        # reflects the LATEST attempt -- a NaN-triggered restart no longer inflates it
        # with time spent on earlier, discarded attempts.
        start_train_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            
        needs_restart = False

        for epoch in range(1, EPOCHS + 1):
            actual_epochs_ran = epoch
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            epoch_nan_count = 0
            valid_batches = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
            # Add the 3 underscores to absorb the vid, start, and end metadata!
            for features, labels, _, _, _ in loop:
                features = features.to(device)
                labels = labels.to(device)
                
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                optimizer.zero_grad()
                
                if LOSS_FUNCTION == "bcl":
                    logits, embeddings = model(features)
                    loss, _, _ = criterion(logits, embeddings, labels)
                elif LOSS_FUNCTION == "unified_ctc":
                    logits, _ = model(features)
                    hard_labels = torch.argmax(labels, dim=1)
                    loss, _, _ = criterion(logits, hard_labels)
                else:
                    # Depending on the model, it might return (logits, embeddings) or just logits
                    output = model(features)
                    logits = output[0] if isinstance(output, tuple) else output
                    hard_labels = torch.argmax(labels, dim=1)
                    loss = criterion(logits, hard_labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    epoch_nan_count += 1
                    total_nan_this_run += 1
                    loop.set_postfix(loss="NaN", nans=epoch_nan_count)
                    
                    if epoch_nan_count > 50 and redo_count < MAX_REDOS:
                        print(f"\n⚠️ CRITICAL: Epoch {epoch} encountered >50 NaN losses ({epoch_nan_count}). Aborting run.")
                        needs_restart = True
                        break
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                valid_batches += 1
                loop.set_postfix(loss=loss.item(), nans=epoch_nan_count)

                try:
                    if torch.cuda.is_available():
                        gpu_utilization_samples.append(torch.cuda.utilization(device))
                except Exception:
                    pass
            
            if needs_restart:
                break 
                
            avg_train_loss = train_loss / valid_batches if valid_batches > 0 else float('inf')
            
            if scheduler:
                scheduler.step()
                
            model.eval()
            val_loss = 0.0
            val_frame_f1, val_iou, val_seg_f1 = [], [], []
            epoch_val_true = []
            epoch_val_pred = []
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
                # It should look something like this:
                for features, labels, vids, start_indices, end_indices in val_loop:
                    features = features.to(device)
                    labels = labels.to(device)
                    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if LOSS_FUNCTION == "bcl":
                        logits, embeddings = model(features)
                        loss, _, _ = criterion(logits, embeddings, labels)
                    elif LOSS_FUNCTION == "unified_ctc":
                        logits, _ = model(features)
                        hard_labels = torch.argmax(labels, dim=1)
                        loss, _, _ = criterion(logits, hard_labels)
                    else:
                        output = model(features)
                        logits = output[0] if isinstance(output, tuple) else output
                        hard_labels = torch.argmax(labels, dim=1)
                        loss = criterion(logits, hard_labels)
                        
                    val_loss += loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0
                    
                    try:
                        if torch.cuda.is_available():
                            gpu_utilization_samples.append(torch.cuda.utilization(device))
                    except Exception:
                        pass
                    
                    for i in range(features.size(0)):
                        valid_len = labels.size(-1) 
                        if valid_len == 0: continue
                        
                        true_seq = torch.argmax(labels[i, :, :valid_len], dim=0).cpu().numpy().astype(float)
                        pred_logits_tensor = logits[i:i+1, :, :valid_len]
                        
                        # --- REMOVED DECODER - USING PURE RAW ARGMAX FOR VALIDATION EVAL ---
                        pred_seq_tensor = torch.argmax(pred_logits_tensor, dim=1)
                        
                        pred_seq = pred_seq_tensor[0].cpu().numpy().astype(float)
                        epoch_val_true.extend(true_seq.tolist())
                        epoch_val_pred.extend(pred_seq.tolist())
                        
                        try:
                            metrics_out = evaluate_batch(np.array([pred_seq.tolist()]), np.array([true_seq.tolist()]))
                            if isinstance(metrics_out, dict):
                                vals = list(metrics_out.values())
                                val_frame_f1.append(float(vals[0]))
                                val_iou.append(float(vals[1]))
                                val_seg_f1.append(float(vals[2]))
                            else:
                                f_f1, iou, s_f1 = metrics_out
                                val_frame_f1.append(float(f_f1))
                                val_iou.append(float(iou))
                                val_seg_f1.append(float(s_f1))
                        except Exception as e:
                            pass
                        
            avg_val_loss = val_loss / len(val_loader)
            
            epoch_f1 = float(np.mean(val_frame_f1)) if val_frame_f1 else 0.0
            epoch_iou = float(np.mean(val_iou)) if val_iou else 0.0
            epoch_seg = float(np.mean(val_seg_f1)) if val_seg_f1 else 0.0
            
            epoch_end_time = time.time()
            epoch_duration = round(epoch_end_time - epoch_start_time, 2)
            
            combined_score = epoch_f1 + epoch_iou + epoch_seg
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                is_best = True
                
                try:
                    cm = confusion_matrix(epoch_val_true, epoch_val_pred, labels=[0, 1, 2])
                    cm_normalized = confusion_matrix(epoch_val_true, epoch_val_pred, labels=[0, 1, 2], normalize='true')
                    
                    annot_labels = np.empty_like(cm, dtype=object)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            annot_labels[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
                            
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues', 
                                xticklabels=['Outside (0)', 'Inside (1)', 'Begin (2)'], 
                                yticklabels=['Outside (0)', 'Inside (1)', 'Begin (2)'],
                                vmin=0.0, vmax=1.0)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual (Row %)')
                    plt.title(f'Validation Confusion Matrix (Best Epoch {epoch})\n{run_name}')
                    
                    cm_save_path = os.path.join(exp_dir, "best_confusion_matrix.png")
                    plt.savefig(cm_save_path, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"⚠️ Failed to generate confusion matrix: {e}")
                    
            else:
                epochs_without_improvement += 1
                is_best = False
                
            nan_string = f" | NaNs: {epoch_nan_count}" if epoch_nan_count > 0 else ""
            
            print(f"Epoch [{epoch:02d}/{EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Frame F1: {epoch_f1:.4f} | "
                  f"Mean IoU: {epoch_iou:.4f} | "
                  f"Seg F1: {epoch_seg:.4f} | "
                  f"Time: {epoch_duration}s{nan_string}" + (" 🌟 (New Best!)" if is_best else f" (No improvement x{epochs_without_improvement})"))
                  
            with open(metrics_log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, avg_train_loss, avg_val_loss, epoch_f1, epoch_iou, epoch_seg, epoch_duration, epoch_nan_count])

            if EARLY_STOPPING and epochs_without_improvement >= PATIENCE and epoch >= MIN_EPOCHS:
                print(f"\n🛑 Early stopping triggered! No improvement in combined score for {PATIENCE} epochs (Minimum {MIN_EPOCHS} epochs met).")
                break
                
        if not needs_restart:
            break
        
        redo_count += 1
        
    if needs_restart and redo_count > MAX_REDOS:
        print(f"\n❌ FAILED TO STABILIZE TRAINING. Maximum redos ({MAX_REDOS}) reached. Saving partial/broken run.")

    total_time = time.time() - start_train_time
    total_minutes, total_seconds = divmod(int(total_time), 60)
    
    avg_epoch_time = int(total_time / actual_epochs_ran) if actual_epochs_ran > 0 else 0
    avg_minutes, avg_seconds = divmod(avg_epoch_time, 60)
    
    avg_gpu_util = sum(gpu_utilization_samples) / len(gpu_utilization_samples) if gpu_utilization_samples else 0.0
    max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    
    hostname = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    
    hardware_summary = {
        "machine_hostname": hostname,
        "gpu_name": gpu_name,
        "total_training_time": f"{int(total_minutes)}m {int(total_seconds)}s",
        "total_training_seconds": round(total_time, 2),
        "total_epochs_ran": actual_epochs_ran,
        "early_stopping_triggered": epochs_without_improvement >= PATIENCE,
        "average_time_per_epoch": f"{int(avg_minutes)}m {int(avg_seconds)}s",
        "max_gpu_memory_used_gb": round(max_mem_gb, 4),
        "average_gpu_utilization_percent": round(avg_gpu_util, 2),
        "total_nan_loss_count_latest_run": total_nan_this_run, 
        "total_training_restarts": min(redo_count, MAX_REDOS)  
    }

    summary_save_path = os.path.join(exp_dir, "hardware_summary.json")
    with open(summary_save_path, "w") as f:
        json.dump(hardware_summary, f, indent=4)
        
    print(f"📊 Hardware summary saved to {summary_save_path}")
        
    model_save_path = os.path.join(model_dir, f"{run_name}.pth")
    if best_model_state:
        torch.save(best_model_state, model_save_path)
        print(f"✅ Best Model (Epoch {best_epoch} | Combined Score: {best_combined_score:.4f}) saved to {model_save_path}")
    else:
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Final Model saved to {model_save_path}")
        
    print(f"🏁 Finished {run_name}\n")

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