import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from src.dataset import SignSegmentationDataset
from src.models import PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_BiMamba, STGCN_MLP_Mamba, Decoupled_STGCN_Mamba

# ==============================================================================
# 🎛️ DEFAULT CONFIGURATION
# ==============================================================================
# Change this variable to set the default model when only a prefix is provided
CHOSEN_MODEL = "stgcn_mamba"  

# Safely expand registry based on your train.py imports
MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_bimamba": STGCN_BiMamba,
    "stgcn_mlp_mamba": STGCN_MLP_Mamba,
    "decoupled_stgcn_mamba": Decoupled_STGCN_Mamba
}

# Mapping integers to labels for the graph
LABEL_MAP = {0: "Outside (O)", 1: "Inside (I)", 2: "Begin (B)"}
# ==============================================================================

def main():
    # 1. Setup Argument Parsing
    parser = argparse.ArgumentParser(description="Generate Confusion Matrix for trained models.")
    parser.add_argument("args_list", nargs="*", help="Provide [model] [prefix] OR just [prefix]. Examples: 'stgcn_mamba 100' or '100'")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for validation loader.")
    args = parser.parse_args()
    
    chosen_model = CHOSEN_MODEL
    
    # Dynamically handle flexible positional arguments
    if len(args.args_list) == 0:
        raise ValueError("❌ Error: You must specify a prefix! Usage: 'python evaluate_matrix.py 100' or 'python evaluate_matrix.py stgcn_mamba 100'")
    elif len(args.args_list) == 1:
        try:
            prefix = int(args.args_list[0])
        except ValueError:
            raise ValueError(f"❌ Error: If providing 1 argument, it must be the numeric prefix (e.g., 100). Got: '{args.args_list[0]}'")
    else:
        chosen_model = args.args_list[0]
        try:
            prefix = int(args.args_list[1])
        except ValueError:
            raise ValueError(f"❌ Error: Second argument must be the numeric prefix. Got: '{args.args_list[1]}'")
            
    prefix_formatted = f"{prefix:02d}"

    if chosen_model not in MODEL_REGISTRY:
        raise ValueError(f"❌ Error: Unknown model '{chosen_model}'. Available models: {list(MODEL_REGISTRY.keys())}")
    
    # 2. Locate Hyperparameters and Weights
    exp_dir = f"experiments/{chosen_model}-{prefix_formatted}"
    hyperparameter_path = os.path.join(exp_dir, "hyperparameters.json")
    weights_path = f"saved_models/{chosen_model}-{prefix_formatted}.pth"
    
    if not os.path.exists(hyperparameter_path):
        print(f"❌ Error: Could not find {hyperparameter_path}")
        return
    if not os.path.exists(weights_path):
        print(f"❌ Error: Could not find weights at {weights_path}")
        return

    # 3. Load Hyperparameters
    with open(hyperparameter_path, 'r') as f:
        hp = json.load(f)

    WINDOW_SIZE = hp.get("window_size", 1000)
    OVERLAP = hp.get("overlap", 200)
    NUM_VERTICES = hp.get("num_vertices", 65)
    IN_CHANNELS = hp.get("in_channels", 3)
    D_MODEL = hp.get("d_model", 256)
    N_LAYERS = hp.get("n_layers", 4)
    TOLERANCE_WINDOW = hp.get("tolerance_window", 5)
    
    # Pulling dataset feature formatting from your new JSON
    BASE_FEATURES = hp.get("base_features", ["x-cord", "y-cord", "z-cord"])
    KINEMATIC_FEATURES = hp.get("kinematic_features", [])
    USE_FULL_LENGTH = hp.get("use_full_length", False)

    print("="*60)
    print(f"📊 CONFUSION MATRIX EVALUATION: {chosen_model.upper()} (Run {prefix_formatted})")
    print("="*60)
    
    # 4. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device.upper()}")
    
    model_class = MODEL_REGISTRY[chosen_model]
    model = model_class(
        num_vertices=NUM_VERTICES, 
        in_channels=IN_CHANNELS, 
        d_model=D_MODEL, 
        n_layers=N_LAYERS
    ).to(device)
    
    print(f"📥 Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 5. Smart Path Resolution for Dataset
    # Checks if a dedicated val folder exists, otherwise falls back to processed_data
    if os.path.exists("processed_data/val/keypoints"):
        kp_dir = "processed_data/val/keypoints"
        lbl_dir = "processed_data/val/BIO_tags"
    else:
        kp_dir = "processed_data/keypoints"
        lbl_dir = "processed_data/BIO_tags"

    print(f"📂 Loading Dataset from: {kp_dir}")
    val_dataset = SignSegmentationDataset(
        keypoints_dir=kp_dir, 
        labels_dir=lbl_dir, 
        window_size=WINDOW_SIZE, 
        overlap=OVERLAP, 
        tolerance_window=TOLERANCE_WINDOW,
        use_full_length=USE_FULL_LENGTH,
        base_features=BASE_FEATURES,
        kinematic_features=KINEMATIC_FEATURES
    )
    
    # Explicit check to stop empty matrices
    if len(val_dataset) == 0:
        print(f"❌ Error: The dataset found 0 valid slices. Check your directories: {kp_dir} and {lbl_dir}")
        return

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. Run Inference
    all_preds = []
    all_truths = []

    print("🚀 Running Inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Processing Batches"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            
            if targets.dim() == 3:
                if targets.shape[1] == 3:
                    truths = torch.argmax(targets, dim=1) 
                else:
                    truths = torch.argmax(targets, dim=2) 
            else:
                truths = targets

            all_preds.extend(preds.view(-1).cpu().numpy())
            all_truths.extend(truths.view(-1).cpu().numpy())

    # 7. Generate Confusion Matrix
    print("🧮 Calculating Matrix...")
    cm = confusion_matrix(all_truths, all_preds, labels=[0, 1, 2])
    
    cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentages = np.nan_to_num(cm_percentages)

    # 8. Plot the Matrix
    plt.figure(figsize=(10, 8))
    
    annot_data = [[f"{count}\n({percent:.1%})" for count, percent in zip(row_count, row_percent)] 
                  for row_count, row_percent in zip(cm, cm_percentages)]
    
    sns.heatmap(cm, annot=annot_data, fmt='', cmap='Blues', 
                xticklabels=[LABEL_MAP[i] for i in [0, 1, 2]], 
                yticklabels=[LABEL_MAP[i] for i in [0, 1, 2]],
                cbar_kws={'label': 'Number of Frames'})
    
    plt.title(f'Confusion Matrix: {chosen_model} (Run {prefix_formatted})\nDataset: {kp_dir}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Grammatical Label', fontsize=14, fontweight='bold')
    plt.xlabel('Model Prediction', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(exp_dir, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Matrix saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()