import os
import json
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
    STGCN_BiMamba
)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
CHOSEN_MODEL = "stgcn_mamba"
PREFIX = "59"
WEIGHTS_PATH = f"saved_models/{CHOSEN_MODEL}-{PREFIX}.pth"
HYPERPARAMETER_PATH = f"experiments/{CHOSEN_MODEL}-{PREFIX}/hyperparameters.json"

# --- NEW CONFIGURATION ---
# Options: "train", "val", "test", or "all"
TARGET_SPLIT = "val" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_bimamba": STGCN_BiMamba
}

# ==============================================================================
# 🖼️ SEQUENTIAL VIEWER
# ==============================================================================
class InteractiveViewer:
    def __init__(self, model, dataset, hp):
        self.model = model
        self.dataset = dataset
        self.hp = hp
        
        # Start exactly at the beginning of the split list
        self.current_idx = 0 
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(15, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.manager.set_window_title(f"Viewer - {CHOSEN_MODEL.upper()} [{TARGET_SPLIT.upper()}]")
        
        # Draw the first plot
        self.draw_graph()

    def run_inference(self):
        inputs, targets = self.dataset[self.current_idx]
        
        # Add batch dimension and move to device
        inputs = inputs.unsqueeze(0).to(DEVICE)
        targets = targets.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # Handle models using Boundary Contrastive Loss which return (logits, embeddings)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
        # Decode probabilities into hard predictions
        preds = decode_predictions(
            logits, 
            strategy=self.hp.get("decoder_strategy", "argmax"), 
            threshold=self.hp.get("decoder_threshold", 0.5)
        )
        
        # Extract ground truth
        if targets.dim() == 3:
            if targets.shape[1] == 3:
                truths = torch.argmax(targets, dim=1) 
            else:
                truths = torch.argmax(targets, dim=2) 
        else:
            truths = targets
            
        return truths[0].cpu().numpy(), preds[0].cpu().numpy()

    def draw_graph(self):
        truths, preds = self.run_inference()
        
        self.ax.clear()
        
        # BIO Mapping: 0=Outside, 1=Inside, 2=Begin
        x_axis = np.arange(len(truths))
        
        # Plot Ground Truth (Thick transparent line in background)
        self.ax.step(x_axis, truths, label="Ground Truth", color="blue", alpha=0.3, linewidth=8, where='post')
        
        # Plot Prediction (Thin sharp line in foreground)
        self.ax.step(x_axis, preds, label="Prediction", color="red", linewidth=2, where='post')
        
        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(["Outside (0)", "Inside (1)", "Begin (2)"])
        self.ax.set_xlabel("Frames")
        
        # --- NEW: Fix dataset attribute names based on your updated dataset.py ---
        if hasattr(self.dataset, 'use_full_length') and self.dataset.use_full_length:
            slice_info = self.dataset.samples[self.current_idx]
            file_id = slice_info.get('video_id', 'Unknown')
            start_f = 0
            end_f = "Full"
        else:
            slice_info = self.dataset.windows[self.current_idx]
            file_id = slice_info.get('video_id', 'Unknown')
            start_f = slice_info.get('start_idx', 0)
            end_f = slice_info.get('end_idx', 0)
        
        title = (f"File: {file_id} | Split: {TARGET_SPLIT.upper()} | "
                 f"Frames: {start_f}-{end_f}\n"
                 f"Dataset Index: {self.current_idx + 1} / {len(self.dataset)} (Use Left/Right Arrows to navigate)")
                 
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.legend(loc="upper right")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        self.fig.canvas.draw()

    def on_press(self, event):
        if event.key == 'right':
            if self.current_idx < len(self.dataset) - 1:
                self.current_idx += 1
                self.draw_graph()
            else:
                print("Already at the end of the dataset.")
        elif event.key == 'left':
            if self.current_idx > 0:
                self.current_idx -= 1
                self.draw_graph()
            else:
                print("Already at the beginning of the dataset.")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not os.path.exists(HYPERPARAMETER_PATH):
        raise FileNotFoundError(f"Could not find {HYPERPARAMETER_PATH}")
        
    with open(HYPERPARAMETER_PATH, 'r') as f:
        hp = json.load(f)
        
    print(f"Loaded Hyperparameters for {CHOSEN_MODEL}-{PREFIX}")
    print(f"Targeting '{TARGET_SPLIT}' split via Dataset parameter.")
    
    # Load the custom dataset and pass the split parameter directly
    dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        window_size=hp.get("window_size"),
        overlap=hp.get("overlap"),
        tolerance_window=hp.get("tolerance_window"),
        use_full_length=False, 
        base_features=hp.get("base_features"),
        kinematic_features=hp.get("kinematic_features"),
        split=TARGET_SPLIT  # <-- Let the Dataset handle the filtering
    )
    
    if len(dataset) == 0:
        raise ValueError(f"❌ Error: Dataset initialized with 0 slices for split '{TARGET_SPLIT}'.")

    # Load Model
    model_class = MODEL_REGISTRY[CHOSEN_MODEL]
    model = model_class(
        num_vertices=hp.get("num_vertices"),
        in_channels=hp.get("in_channels"),
        d_model=hp.get("d_model"),
        n_layers=hp.get("n_layers")
    ).to(DEVICE)
    
    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    # Start the Sequential Viewer
    viewer = InteractiveViewer(model, dataset, hp)
    plt.show() 

if __name__ == "__main__":
    main()