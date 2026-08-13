import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import Custom Modules
from src.dataset import SignSegmentationDataset
from src.decoder import decode_predictions
from src.metrics import evaluate_batch
from inference_class_plot import InteractiveViewer

# Import Model Registry
from src.models import (
    PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_MLP_Mamba, 
    STGCN_BiMamba, Decoupled_STGCN_Mamba, BiLSTM_Baseline, STGCN_BiLSTM, 
    TransformerBaseline, STGCN_Transformer, Latent_STGCN_Mamba,
    CTRGCN_Mamba, InfoGCN_Mamba, ShiftGCN_Mamba, SpatialTransformer_Mamba
)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
CHOSEN_MODEL = "stgcn_mamba"
PREFIX = "257"

# Decoder Settings
DECODING_STRATEGY = "linguistic" # Options: "argmax", "threshold", "linguistic"
DECODING_THRESHOLD = 0.60

# Dataset Settings
TARGET_SPLIT = "val" # "val" or "test"
BATCH_SIZE = 16      

# Window Display Logic (Strictly for the Viewer, NOT the model)
USE_DEFAULT_WINDOW_SIZE = False
CUSTOM_WINDOW_SIZE = 256

WEIGHTS_PATH = f"saved_models/{CHOSEN_MODEL}-{PREFIX}.pth"
HYPERPARAMETER_PATH = f"experiments/{CHOSEN_MODEL}-{PREFIX}/hyperparameters.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_mlp_mamba": STGCN_MLP_Mamba,
    "stgcn_bimamba": STGCN_BiMamba,
    "decoupled_stgcn_mamba": Decoupled_STGCN_Mamba,
    "bilstm": BiLSTM_Baseline,
    "stgcn_bilstm": STGCN_BiLSTM,
    "transformer": TransformerBaseline,
    "stgcn_transformer": STGCN_Transformer,
    "latent_stgcn_mamba": Latent_STGCN_Mamba,
    "ctrgcn_mamba": CTRGCN_Mamba,
    "infogcn_mamba": InfoGCN_Mamba,
    "shiftgcn_mamba": ShiftGCN_Mamba,
    "spatial_transformer_mamba": SpatialTransformer_Mamba
}

if __name__ == "__main__":
    print(f"🚀 Initializing Evaluation for {CHOSEN_MODEL}-{PREFIX} using '{DECODING_STRATEGY}' decoder...")
    
    # 1. Load Hyperparameters
    with open(HYPERPARAMETER_PATH, 'r') as f:
        hp = json.load(f)
        
    # --- PROPER WINDOW ISOLATION ---
    # The model MUST run on its trained window size to maintain valid hidden states
    train_window_size = hp.get("window_size")
    display_window_size = train_window_size if USE_DEFAULT_WINDOW_SIZE else CUSTOM_WINDOW_SIZE
        
    # 2. Initialize Dataset (Locked to the exact configuration used during training)
    dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        window_size=train_window_size,  # Model strict limit
        overlap=0, 
        tolerance_window=hp.get("tolerance_window"),
        use_full_length=False, 
        base_features=hp.get("base_features"),
        kinematic_features=hp.get("kinematic_features"),
        split=TARGET_SPLIT  
    )
    
    if len(dataset) == 0:
        raise ValueError(f"❌ Error: Dataset initialized with 0 slices for split '{TARGET_SPLIT}'.")
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Initialize Model
    model_class = MODEL_REGISTRY[CHOSEN_MODEL]
    model_kwargs = {
        "num_vertices": hp.get("num_vertices"),
        "in_channels": hp.get("in_channels"),
        "d_model": hp.get("d_model"),
        "n_layers": hp.get("n_layers")
    }
    
    if CHOSEN_MODEL in ["latent_stgcn_mamba", "latent_mamba", "ctrgcn_mamba", "infogcn_mamba", "shiftgcn_mamba", "spatial_transformer_mamba"]:
        model_kwargs["latent_dim"] = hp.get("latent_dim", 128)
        
    model = model_class(**model_kwargs).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # 4. Dictionary to hold chunks before stitching
    video_predictions = {}
    video_ground_truth = {}

    print(f"\n🧪 Running Inference on '{TARGET_SPLIT}' split (Stitching 16-frame chunks)...")
    with torch.no_grad():
        for features, labels, vids, starts, ends in tqdm(dataloader, desc="Evaluating"):
            features = features.to(DEVICE)
            
            outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            preds = decode_predictions(logits, strategy=DECODING_STRATEGY, threshold=DECODING_THRESHOLD)
            hard_labels = torch.argmax(labels, dim=1)
            
            preds_cpu = preds.cpu().numpy()
            labels_cpu = hard_labels.cpu().numpy()
            
            # Save the chunks to their respective videos
            for i in range(len(vids)):
                vid = vids[i]
                s = starts[i].item()
                
                if vid not in video_predictions:
                    video_predictions[vid] = []
                    video_ground_truth[vid] = []
                    
                video_predictions[vid].append((s, preds_cpu[i]))
                video_ground_truth[vid].append((s, labels_cpu[i]))

    # 5. Evaluate the Stitched Videos
    total_frame_f1 = 0
    total_mean_iou = 0
    total_segment_f1 = 0
    num_videos = len(video_predictions)

    for vid in video_predictions.keys():
        # Sort chunks by start index to ensure chronological order
        video_predictions[vid].sort(key=lambda x: x[0])
        video_ground_truth[vid].sort(key=lambda x: x[0])
        
        # Concatenate chunks into continuous arrays
        full_pred = np.concatenate([chunk[1] for chunk in video_predictions[vid]])
        full_true = np.concatenate([chunk[1] for chunk in video_ground_truth[vid]])
        
        # Calculate F1 on the complete timeline!
        metrics = evaluate_batch(torch.tensor([full_pred]), torch.tensor([full_true]))
        vals = list(metrics.values())
        
        total_frame_f1 += float(vals[0])
        total_mean_iou += float(vals[1])
        total_segment_f1 += float(vals[2])

    # 6. Print Final Benchmark Results
    print("\n" + "="*50)
    print(f"📊 METRICS RESULTS ({DECODING_STRATEGY.upper()})")
    print("="*50)
    print(f"Frame F1:   {total_frame_f1 / num_videos:.4f}")
    print(f"Mean IoU:   {total_mean_iou / num_videos:.4f}")
    print(f"Segment F1: {total_segment_f1 / num_videos:.4f}")
    print("="*50 + "\n")

    # 7. Seamless Interactive Viewer Transfer
    user_choice = input("Do you want to visually inspect the predictions in the Interactive Viewer? (y/N): ").strip().lower()
    
    if user_choice == 'y':
        print(f"Launching viewer... (Using display window size: {display_window_size})")
        # Pass the separated display window size to the viewer
        viewer = InteractiveViewer(model, dataset, hp, display_window_size)
        plt.show() 
    else:
        print("Exiting evaluation.")