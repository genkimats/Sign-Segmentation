import os
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules from your project
from src.dataset import SignSegmentationDataset
from src.decoder import decode_predictions
from src.models import (
    PureMambaBaseline, 
    BiMambaBaseline, 
    STGCN_Mamba, 
    STGCN_BiMamba, 
    Decoupled_STGCN_Mamba, 
    Decoupled_STGCN_BiMamba
)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
CHOSEN_MODEL = "stgcn_mamba"
PREFIX = 46

WEIGHTS_PATH = f"saved_models/{CHOSEN_MODEL}-{PREFIX}.pth"
HYPERPARAMETER_PATH = f"experiments/{CHOSEN_MODEL}-{PREFIX}/hyperparameters.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_bimamba": STGCN_BiMamba,
    "decoupled_stgcn_mamba": Decoupled_STGCN_Mamba,
    "decoupled_stgcn_bimamba": Decoupled_STGCN_BiMamba
}

# ==============================================================================
# 🔍 SEQUENCE SELECTION LOGIC
# ==============================================================================
def find_optimal_sequence(dataset):
    """
    Scans the dataset to find a sequence with:
    - At least 3 glosses (3 'Begin' frames)
    - At least 7 frames of 'Outside' at the start
    - At least 7 frames of 'Outside' at the end
    """
    print("🔍 Scanning dataset for a sequence matching linguistic criteria...")
    valid_indices = []
    
    for i in range(len(dataset.slice_index)):
        slice_info = dataset.slice_index[i]
        label_path = os.path.join(dataset.labels_dir, f"{slice_info['base_name']}.npy")
        
        # Load just the labels for fast scanning
        labels = np.load(label_path, mmap_mode='r')[slice_info['start']:slice_info['end']]
        
        # Convert to hard BIO states (T) safely whether it's 1D or 2D
        if labels.ndim > 1:
            hard_labels = np.argmax(labels, axis=1)
        else:
            hard_labels = labels
        
        # Criteria 1: At least 3 glosses (Class 2 is 'Begin')
        if np.sum(hard_labels == 2) < 3:
            continue
            
        # Criteria 2 & 3: Starts and ends with at least 7 frames of 'Outside' (Class 0)
        if not np.all(hard_labels[:7] == 0) or not np.all(hard_labels[-7:] == 0):
            continue
            
        valid_indices.append(i)
        
    if valid_indices:
        chosen = random.choice(valid_indices)
        print(f"✅ Found {len(valid_indices)} valid sequences. Randomly selected index {chosen}.")
        return chosen
    else:
        print("⚠️ No sequence strictly matched all criteria. Falling back to a random sequence.")
        return random.randint(0, len(dataset.slice_index) - 1)

# ==============================================================================
# 🚀 MAIN INFERENCE PIPELINE
# ==============================================================================
def main():
    if not os.path.exists(HYPERPARAMETER_PATH):
        raise FileNotFoundError(f"Could not find {HYPERPARAMETER_PATH}")
        
    # 1. Load Hyperparameters dynamically
    with open(HYPERPARAMETER_PATH, 'r') as f:
        hp = json.load(f)
        
    print(f"Loaded Hyperparameters for {CHOSEN_MODEL}-{PREFIX}")
    
    # 2. Initialize Dataset using exact training params
    # We force use_full_length=False so we visualize exactly one window_size chunk
    dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        window_size=hp.get("window_size", 512),
        overlap=hp.get("overlap", 200),
        tolerance_window=hp.get("tolerance_window", 5),
        use_full_length=False, 
        base_features=hp.get("base_features", ["x-cord", "y-cord"]),
        kinematic_features=hp.get("kinematic_features", [])
    )
    
    # 3. Find the perfect sequence
    seq_idx = find_optimal_sequence(dataset)
    features, targets = dataset[seq_idx]
    
    # 4. Initialize Model
    model_class = MODEL_REGISTRY[CHOSEN_MODEL]
    model = model_class(
        num_vertices=hp.get("num_vertices", 65),
        in_channels=hp.get("in_channels", 3),
        d_model=hp.get("d_model", 256),
        n_layers=hp.get("n_layers", 4)
    ).to(DEVICE)
    
    # Load Weights
    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    # 5. Run Inference
    features_tensor = features.unsqueeze(0).to(DEVICE) # Add batch dim
    
    with torch.no_grad():
        outputs = model(features_tensor)
        
        # Safely handle tuple outputs (Logits, Embeddings) for BCL models
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        # Decode using the exact strategy from training
        predictions = decode_predictions(
            logits, 
            strategy=hp.get("decoder_strategy", "argmax"), 
            threshold=hp.get("decoder_threshold", 0.6)
        )
        
    # Prepare arrays for plotting
    # Convert targets from (3, T) to (T) BIO classes
    gt_np = np.argmax(targets.numpy(), axis=0)
    pred_np = predictions[0].cpu().numpy()
    time_axis = np.arange(len(gt_np))
    
    # 6. Plotting
    plt.figure(figsize=(16, 6))
    
    # We offset them slightly on the Y-axis so they don't perfectly overlap visually
    plt.step(time_axis, gt_np + 0.05, label="Ground Truth", linewidth=2.5, color="#2ca02c", where='mid')
    plt.step(time_axis, pred_np - 0.05, label="Predicted", linewidth=2.5, color="#d62728", linestyle="--", where='mid')
    
    # Aesthetics
    plt.yticks([0, 1, 2], ['Outside (O)', 'Inside (I)', 'Begin (B)'], fontsize=12)
    plt.xlabel(f"Frames (Window Size: {hp.get('window_size', 512)})", fontsize=12, fontweight='bold')
    plt.ylabel("BIO State", fontsize=12, fontweight='bold')
    plt.title(f"Sign Language Segmentation \n Model: {CHOSEN_MODEL} | Decoder: {hp.get('decoder_strategy', 'argmax').upper()}", fontsize=14)
    
    # Highlight the regions where predictions differ from ground truth
    errors = gt_np != pred_np
    plt.fill_between(time_axis, -0.5, 2.5, where=errors, color='gray', alpha=0.15, label='Mismatched Frames')
    
    plt.ylim(-0.5, 2.5)
    plt.legend(loc="upper right", fontsize=11, framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()