import os
import json
import threading
import queue
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
PREFIX = "50"
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
# 🔍 BACKGROUND SEARCHER
# ==============================================================================
def background_scanner(dataset, search_queue, stop_event):
    """
    Runs in a background thread. Continuously scans the dataset.
    When it finds a sequence matching the criteria, it pushes the index to the queue.
    """
    for i in range(len(dataset.slice_index)):
        if stop_event.is_set():
            break
            
        slice_info = dataset.slice_index[i]
        label_path = os.path.join(dataset.labels_dir, f"{slice_info['base_name']}.npy")
        
        # Fast load
        labels = np.load(label_path, mmap_mode='r')[slice_info['start']:slice_info['end']]
        
        if labels.ndim > 1:
            hard_labels = np.argmax(labels, axis=1)
        else:
            hard_labels = labels
            
        # Criteria
        if np.sum(hard_labels == 2) < 3: continue
        if not np.all(hard_labels[:7] == 0) or not np.all(hard_labels[-7:] == 0): continue
            
        # Match found! Send to UI.
        search_queue.put(i)

# ==============================================================================
# 🚀 INTERACTIVE VIEWER CLASS
# ==============================================================================
class InteractiveViewer:
    def __init__(self, dataset, model, hp):
        self.dataset = dataset
        self.model = model
        self.hp = hp
        
        # State Tracking
        self.history = []
        self.current_pos = -1
        
        # Concurrency
        self.search_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.search_thread = threading.Thread(
            target=background_scanner, 
            args=(self.dataset, self.search_queue, self.stop_event),
            daemon=True
        )
        
        # Plotting Setup
        self.fig, self.ax = plt.subplots(figsize=(16, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Start search and trigger first plot
        print("🔍 Starting background scanner...")
        self.search_thread.start()
        self.load_next_sequence()

    def run_inference_and_plot(self, idx):
        """Runs the model on the requested index and updates the graph."""
        slice_info = self.dataset.slice_index[idx]
        file_desc = f"{slice_info['base_name']} | Frames: {slice_info['start']} to {slice_info['end']}"
        
        features, targets = self.dataset[idx]
        features_tensor = features.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
            predictions = decode_predictions(
                logits, 
                # strategy=self.hp.get("decoder_strategy", "argmax"),
                strategy="argmax", 
                threshold=self.hp.get("decoder_threshold", 0.6)
            )
            
        gt_np = np.argmax(targets.numpy(), axis=0)
        pred_np = predictions[0].cpu().numpy()
        time_axis = np.arange(len(gt_np))
        
        # Clear previous plot
        self.ax.clear()
        
        # Re-draw
        self.ax.step(time_axis, gt_np + 0.05, label="Ground Truth", linewidth=2.5, color="#2ca02c", where='mid')
        self.ax.step(time_axis, pred_np - 0.05, label="Predicted", linewidth=2.5, color="#d62728", linestyle="--", where='mid')
        
        errors = gt_np != pred_np
        self.ax.fill_between(time_axis, -0.5, 2.5, where=errors, color='gray', alpha=0.15, label='Mismatched Frames')
        
        # Formatting
        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(['Outside (O)', 'Inside (I)', 'Begin (B)'], fontsize=12)
        self.ax.set_xlabel(f"Frames (Window Size: {self.hp.get('window_size', 512)})", fontsize=12, fontweight='bold')
        self.ax.set_ylabel("BIO State", fontsize=12, fontweight='bold')
        
        self.ax.set_ylim(-0.5, 2.5)
        self.ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        self.ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Set Window Title and Graph Title
        self.fig.canvas.manager.set_window_title(file_desc)
        self.ax.set_title(f"Model: {CHOSEN_MODEL} | Sequence: {file_desc}\n[Right Arrow: Next]  [Left Arrow: Previous]", fontsize=14)
        
        self.fig.canvas.draw()

    def load_next_sequence(self):
        """Moves forward in history, or waits for background scanner if at the end."""
        if self.current_pos < len(self.history) - 1:
            self.current_pos += 1
            self.run_inference_and_plot(self.history[self.current_pos])
        else:
            print("⏳ Waiting for background scanner to find the next match...")
            try:
                # Block until the queue yields a new index
                new_idx = self.search_queue.get(timeout=10.0) 
                self.history.append(new_idx)
                self.current_pos += 1
                self.run_inference_and_plot(new_idx)
            except queue.Empty:
                print("⚠️ Scanner timed out. No more matching sequences found.")

    def load_prev_sequence(self):
        """Moves backward in history."""
        if self.current_pos > 0:
            self.current_pos -= 1
            self.run_inference_and_plot(self.history[self.current_pos])
        else:
            print("🛑 Already at the first sequence.")

    def on_key_press(self, event):
        """Matplotlib event listener."""
        if event.key == 'right':
            self.load_next_sequence()
        elif event.key == 'left':
            self.load_prev_sequence()

    def show(self):
        plt.tight_layout()
        plt.show()
        self.stop_event.set() # Kill background thread when window is closed

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not os.path.exists(HYPERPARAMETER_PATH):
        raise FileNotFoundError(f"Could not find {HYPERPARAMETER_PATH}")
        
    with open(HYPERPARAMETER_PATH, 'r') as f:
        hp = json.load(f)
        
    print(f"Loaded Hyperparameters for {CHOSEN_MODEL}-{PREFIX}")
    
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
    
    model_class = MODEL_REGISTRY[CHOSEN_MODEL]
    model = model_class(
        num_vertices=hp.get("num_vertices", 65),
        in_channels=hp.get("in_channels", 3),
        d_model=hp.get("d_model", 256),
        n_layers=hp.get("n_layers", 4)
    ).to(DEVICE)
    
    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    # Launch the interactive GUI
    viewer = InteractiveViewer(dataset, model, hp)
    viewer.show()

if __name__ == "__main__":
    main()