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
PREFIX = "257"
WEIGHTS_PATH = f"saved_models/{CHOSEN_MODEL}-{PREFIX}.pth"
HYPERPARAMETER_PATH = f"experiments/{CHOSEN_MODEL}-{PREFIX}/hyperparameters.json"

# --- NEW CONFIGURATION ---
# Options: "train", "val", "test", or "all"
TARGET_SPLIT = "val" 

USE_DEFAULT_WINDOW_SIZE = False
CUSTOM_WINDOW_SIZE = 256

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
    def __init__(self, model, dataset, hp, target_window_size):
        self.model = model
        self.dataset = dataset
        self.hp = hp
        self.target_window_size = target_window_size
        self.trained_window_size = hp.get("window_size", 512)
        self.temporal_downsample = hp.get("temporal_downsample_factor", 1)
        
        # Calculate the chunk size expected by the model during inference.
        self.chunk_size = self.trained_window_size // self.temporal_downsample
        
        # Start exactly at the beginning of the split list
        self.current_idx = 0 
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(15, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.manager.set_window_title(f"Viewer - {CHOSEN_MODEL.upper()} [{TARGET_SPLIT.upper()}]")
        
        # Draw the first plot
        self.draw_graph()

    def run_inference(self):
        # --- FIX: Unpack all 5 variables returned by the new dataset.py ---
        inputs, targets, vid, start_idx, end_idx = self.dataset[self.current_idx]
        
        seq_len = inputs.shape[1] # Time dimension
        all_logits = []
        
        with torch.no_grad():
            # Process in chunks
            for i in range(0, seq_len, self.chunk_size):
                end_i = min(i + self.chunk_size, seq_len)
                chunk_inputs = inputs[:, i:end_i, :].unsqueeze(0).to(DEVICE)
                
                outputs = self.model(chunk_inputs)
                
                if isinstance(outputs, tuple):
                    chunk_logits = outputs[0]
                else:
                    chunk_logits = outputs
                    
                all_logits.append(chunk_logits)
                
        # Concatenate all logits along the time dimension
        concatenated_logits = torch.cat(all_logits, dim=2)

        # Decode probabilities into hard predictions
        preds = decode_predictions(
            concatenated_logits, 
            strategy=self.hp.get("decoder_strategy", "argmax"), 
            threshold=self.hp.get("decoder_threshold", 0.5)
        )
        
        targets = targets.unsqueeze(0).to(DEVICE)
        
        # Extract ground truth 
        if targets.dim() == 3:
            if targets.shape[1] == 3:
                truths = torch.argmax(targets, dim=1) 
            else:
                truths = torch.argmax(targets, dim=2) 
        else:
            truths = targets
            
        # Return the extracted video metadata as well
        return truths[0].cpu().numpy(), preds[0].cpu().numpy(), vid, start_idx, end_idx

    def draw_graph(self):
        # Pull the metadata natively directly from the inference loop
        truths, preds, file_id, start_f, end_f = self.run_inference()
        
        self.ax.clear()
        
        # BIO Mapping: 0=Outside, 1=Inside, 2=Begin
        x_axis = np.arange(len(truths))
        
        # Plot Ground Truth: Dashed yellow line, slightly transparent so it doesn't overpower
        self.ax.step(x_axis, truths, label="Ground Truth", color="gold", linestyle="--", alpha=0.8, linewidth=4, where='post')
        
        # Plot Prediction: Solid blue line, slightly offset vertically (+0.05) to avoid overlap
        self.ax.step(x_axis, preds + 0.05, label="Prediction", color="dodgerblue", linestyle="-", linewidth=2.5, where='post')
        
        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(["Outside (0)", "Inside (1)", "Begin (2)"])
        self.ax.set_ylim(-0.2, 2.2)
        self.ax.set_xlabel("Frames")
        
        model_chunk_str = f"Trained Window Size: {self.trained_window_size}"
        if self.temporal_downsample > 1:
             model_chunk_str += f" (Downsampled to {self.chunk_size})"
        
        # --- FIX: Removed the clunky string slicing code and used the exact variables! ---
        title = (f"File: {file_id} | Split: {TARGET_SPLIT.upper()} | "
                 f"Displayed Frames: {start_f}-{end_f} | {model_chunk_str}\n"
                 f"Dataset Index: {self.current_idx + 1} / {len(self.dataset)} (Use Left/Right Arrows to navigate)")
                 
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.legend(loc="upper right")
        self.ax.set_axisbelow(True)
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
    
    if USE_DEFAULT_WINDOW_SIZE:
        display_window_size = hp.get("window_size")
        print(f"Using trained window size for display: {display_window_size}")
    else:
        display_window_size = CUSTOM_WINDOW_SIZE
        print(f"Overriding dataset window size for display. Using custom window size: {display_window_size}")
    
    dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        window_size=display_window_size,  
        overlap=hp.get("overlap"),
        tolerance_window=hp.get("tolerance_window"),
        use_full_length=False, 
        base_features=hp.get("base_features"),
        kinematic_features=hp.get("kinematic_features"),
        split=TARGET_SPLIT
    )
    
    if len(dataset) == 0:
        raise ValueError(f"❌ Error: Dataset initialized with 0 slices for split '{TARGET_SPLIT}'.")

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
    
    viewer = InteractiveViewer(model, dataset, hp, target_window_size=display_window_size)
    plt.show() 

if __name__ == "__main__":
    main()