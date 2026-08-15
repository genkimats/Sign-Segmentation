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
PREFIX = "135"
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
# 🖼️ CONFIDENCE PROBABILITY VIEWER
# ==============================================================================
class InteractiveViewer:
    def __init__(self, model, dataset, hp, target_window_size):
        self.model = model
        self.dataset = dataset
        self.hp = hp
        self.target_window_size = target_window_size
        self.trained_window_size = hp.get("window_size", 512)
        self.temporal_downsample = hp.get("temporal_downsample_factor", 1)
        
        self.chunk_size = self.trained_window_size // self.temporal_downsample
        self.current_idx = 0 
        
        # --- FIX: Changed to 2 subplots so probabilities and predictions don't overlap ---
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, 
            figsize=(15, 9), 
            sharex=True, 
            gridspec_kw={'height_ratios': [2.5, 1]}
        )
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.manager.set_window_title(f"Confidence Viewer - {CHOSEN_MODEL.upper()} [{TARGET_SPLIT.upper()}]")
        
        self.draw_graph()

    def run_inference(self):
        inputs, targets, vid, start_idx, end_idx = self.dataset[self.current_idx]
        
        seq_len = inputs.shape[1] 
        all_logits = []
        
        with torch.no_grad():
            for i in range(0, seq_len, self.chunk_size):
                end_i = min(i + self.chunk_size, seq_len)
                chunk_inputs = inputs[:, i:end_i, :].unsqueeze(0).to(DEVICE)
                
                outputs = self.model(chunk_inputs)
                
                if isinstance(outputs, tuple):
                    chunk_logits = outputs[0]
                else:
                    chunk_logits = outputs
                    
                all_logits.append(chunk_logits)
                
        concatenated_logits = torch.cat(all_logits, dim=2)
        
        # 1. Get continuous probabilities
        probs = torch.softmax(concatenated_logits, dim=1) 
        
        # 2. Get final decoded hard predictions
        preds = decode_predictions(
            concatenated_logits, 
            strategy=self.hp.get("decoder_strategy", "argmax"), 
            threshold=self.hp.get("decoder_threshold", 0.5)
        )
        
        targets = targets.unsqueeze(0).to(DEVICE)
        
        if targets.dim() == 3:
            if targets.shape[1] == 3:
                truths = torch.argmax(targets, dim=1) 
            else:
                truths = torch.argmax(targets, dim=2) 
        else:
            truths = targets
            
        return truths[0].cpu().numpy(), probs[0].cpu().numpy(), preds[0].cpu().numpy(), vid, start_idx, end_idx

    def draw_graph(self):
        truths, probs, preds, file_id, start_f, end_f = self.run_inference()
        
        self.ax1.clear()
        self.ax2.clear()
        x_axis = np.arange(len(truths))
        
        # =====================================================================
        # TOP PLOT: Probabilities & Shaded Ground Truth
        # =====================================================================
        self.ax1.fill_between(x_axis, 0, 1.05, where=(truths == 0), color='gray', alpha=0.15, label="GT: Outside", step="post")
        self.ax1.fill_between(x_axis, 0, 1.05, where=(truths == 1), color='blue', alpha=0.10, label="GT: Inside", step="post")
        self.ax1.fill_between(x_axis, 0, 1.05, where=(truths == 2), color='red', alpha=0.25, label="GT: Begin", step="post")
        
        self.ax1.plot(x_axis, probs[0], color='black', linewidth=2, label="P(Outside)")
        self.ax1.plot(x_axis, probs[1], color='dodgerblue', linewidth=2, label="P(Inside)")
        self.ax1.plot(x_axis, probs[2], color='red', linewidth=2.5, label="P(Begin)")
        
        self.ax1.set_ylim(-0.05, 1.05)
        self.ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        self.ax1.set_ylabel("Confidence", fontsize=12, fontweight='bold')
        self.ax1.grid(True, linestyle='--', alpha=0.5)
        self.ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)
        
        model_chunk_str = f"Trained Window Size: {self.trained_window_size}"
        if self.temporal_downsample > 1:
             model_chunk_str += f" (Downsampled to {self.chunk_size})"
        
        title = (f"Confidence & Prediction Plot | File: {file_id} | Split: {TARGET_SPLIT.upper()} | "
                 f"Displayed Frames: {start_f}-{end_f} | {model_chunk_str}\n"
                 f"Dataset Index: {self.current_idx + 1} / {len(self.dataset)} (Use Left/Right Arrows to navigate)")
        self.ax1.set_title(title, fontsize=12, fontweight='bold')

        # =====================================================================
        # BOTTOM PLOT: Discrete Hard Predictions vs Ground Truth
        # =====================================================================
        self.ax2.step(x_axis, truths, label="Ground Truth", color="gold", linestyle="--", alpha=0.8, linewidth=4, where='post')
        self.ax2.step(x_axis, preds + 0.05, label="Final Prediction", color="dodgerblue", linestyle="-", linewidth=2.5, where='post')
        
        self.ax2.set_yticks([0, 1, 2])
        self.ax2.set_yticklabels(["Outside (0)", "Inside (1)", "Begin (2)"])
        self.ax2.set_ylim(-0.2, 2.2)
        self.ax2.set_ylabel("Predicted Class", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("Frames", fontsize=12, fontweight='bold')
        self.ax2.grid(True, linestyle='--', alpha=0.5)
        self.ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)
        
        plt.tight_layout()
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