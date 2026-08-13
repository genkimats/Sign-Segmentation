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
BATCH_SIZE = 16      # Batch size for rapid evaluation

# Window Display Logic (from inference_class_plot.py)
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
    "latent_mamba": Latent_Mamba,
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
        
    # 2. Extract Window Size Logic
    if USE_DEFAULT_WINDOW_SIZE:
        display_window_size = hp.get("window_size")
        print(f"Using trained window size for display: {display_window_size}")
    else:
        display_window_size = CUSTOM_WINDOW_SIZE
        print(f"Overriding dataset window size for display. Using custom window size: {display_window_size}")
        
    # 3. Initialize Dataset (Streaming directly from RAM/Disk based on your dataset.py)
    dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        window_size=display_window_size,  
        overlap=0, # 0 overlap is standard for clean metric evaluation
        tolerance_window=hp.get("tolerance_window"),
        use_full_length=False, 
        base_features=hp.get("base_features"),
        kinematic_features=hp.get("kinematic_features"),
        split=TARGET_SPLIT  
    )
    
    if len(dataset) == 0:
        raise ValueError(f"❌ Error: Dataset initialized with 0 slices for split '{TARGET_SPLIT}'.")
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Initialize Model
    model_class = MODEL_REGISTRY[CHOSEN_MODEL]
    model_kwargs = {
        "num_vertices": hp.get("num_vertices"),
        "in_channels": hp.get("in_channels"),
        "d_model": hp.get("d_model"),
        "n_layers": hp.get("n_layers")
    }
    
    # Inject latent_dim safely if evaluating modern architectures
    if CHOSEN_MODEL in ["latent_stgcn_mamba", "latent_mamba", "ctrgcn_mamba", "infogcn_mamba", "shiftgcn_mamba", "spatial_transformer_mamba"]:
        model_kwargs["latent_dim"] = hp.get("latent_dim", 128)
        
    model = model_class(**model_kwargs).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # 5. Rapid Evaluation Loop
    total_frame_f1 = 0
    total_mean_iou = 0
    total_segment_f1 = 0
    batches = 0

    print(f"\n🧪 Running Inference on '{TARGET_SPLIT}' split...")
    with torch.no_grad():
        # Using 5 variables to catch the metadata safely from dataset.py
        for features, labels, _, _, _ in tqdm(dataloader, desc="Evaluating"):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass (handles tuples if Mamba returns hidden states)
            outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Decode the sequence using the linguistic rules
            preds = decode_predictions(logits, strategy=DECODING_STRATEGY, threshold=DECODING_THRESHOLD)
            
            # Calculate F1/IoU metrics for this batch
            metrics = evaluate_batch(preds, labels)
            total_frame_f1 += metrics['frame_f1']
            total_mean_iou += metrics['mean_iou']
            total_segment_f1 += metrics['segment_f1']
            batches += 1

    # 6. Print Final Benchmark Results
    print("\n" + "="*50)
    print(f"📊 METRICS RESULTS ({DECODING_STRATEGY.upper()})")
    print("="*50)
    print(f"Frame F1:   {total_frame_f1 / batches:.4f}")
    print(f"Mean IoU:   {total_mean_iou / batches:.4f}")
    print(f"Segment F1: {total_segment_f1 / batches:.4f}")
    print("="*50 + "\n")

    # 7. Seamless Interactive Viewer Transfer
    user_choice = input("Do you want to visually inspect the predictions in the Interactive Viewer? (y/N): ").strip().lower()
    
    if user_choice == 'y':
        print("Launching viewer... (Press Right Arrow to load optimal sequences)")
        
        # Passes the loaded model & dataset natively into your background thread plot engine
        viewer = InteractiveViewer(model, dataset, hp, display_window_size)
        
        # Holds the script open while you interact with Matplotlib
        plt.show() 
    else:
        print("Exiting evaluation.")