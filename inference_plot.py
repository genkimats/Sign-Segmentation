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
PREFIX = "59"
WEIGHTS_PATH = f"saved_models/{CHOSEN_MODEL}-{PREFIX}.pth"
HYPERPARAMETER_PATH = f"experiments/{CHOSEN_MODEL}-{PREFIX}/hyperparameters.json"

# --- NEW CONFIGURATION ---
# Options: "train", "val", "test", or "all" (to ignore the split file)
TARGET_SPLIT = "val" 
SPLIT_FILE_PATH = "dataset_splits.json"

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
def background_scanner(dataset, search_queue, stop_event, hp):
    """
    Runs on a separate thread. Scans the dataset sequentially.
    When it finds an index that meets the criteria, it pushes it to the queue.
    """
    WINDOW_SIZE = hp.get("window_size", 512)
    
    # We want at least 3 distinct glosses (3 Begin tags followed by Insides)
    target_glosses = 3
    # We want padding on both sides to ensure the boundaries aren't cut off
    padding_frames = 7
    
    for i in range(len(dataset)):
        if stop_event.is_set():
            break
            
        # We only need to load the label array to check conditions
        slice_info = dataset.slice_index[i]
        label_path = os.path.join(dataset.labels_dir, f"{slice_info['base_name']}.npy")
        
        try:
            # We use mmap to check labels instantly without loading the whole file into RAM
            label_array = np.load(label_path, mmap_mode='r')[slice_info['start']:slice_info['end']]
            
            # If the window isn't full size, skip
            if len(label_array) != WINDOW_SIZE:
                continue
                
            # Convert soft labels to hard labels if necessary
            if label_array.ndim > 1:
                hard_labels = np.argmax(label_array, axis=1)
            else:
                hard_labels = label_array
                
            # Rule 1: Check padding at the start and end (Class 0 = Outside)
            if not (np.all(hard_labels[:padding_frames] == 0) and np.all(hard_labels[-padding_frames:] == 0)):
                continue
                
            # Rule 2: Count glosses. A new gloss happens when we see a '2' (Begin)
            num_begins = np.sum(hard_labels == 2)
            if num_begins >= target_glosses:
                # We found a perfect sequence! Push the index to the main thread.
                search_queue.put(i)
                
        except Exception as e:
            continue

# ==============================================================================
# 🖼️ INTERACTIVE VIEWER
# ==============================================================================
class InteractiveViewer:
    def __init__(self, model, dataset, search_queue, hp):
        self.model = model
        self.dataset = dataset
        self.search_queue = search_queue
        self.hp = hp
        self.history = []
        self.active_index = -1
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(15, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.manager.set_window_title(f"Interactive Viewer - {CHOSEN_MODEL.upper()} [{TARGET_SPLIT.upper()}]")
        
        # Load the very first plot
        self.next_plot()

    def run_inference(self, data_idx):
        inputs, targets = self.dataset[data_idx]
        
        # Add batch dimension and move to device
        inputs = inputs.unsqueeze(0).to(DEVICE)
        targets = targets.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            # Handle decoupled models that return (logits, features)
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

    def draw_graph(self, data_idx):
        truths, preds = self.run_inference(data_idx)
        
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
        
        slice_info = self.dataset.slice_index[data_idx]
        title = (f"File: {slice_info['base_name']} | Split: {TARGET_SPLIT.upper()} | "
                 f"Frames: {slice_info['start']}-{slice_info['end']}\n"
                 f"Dataset Index: {data_idx} (Press Right Arrow for Next)")
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.legend(loc="upper right")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        self.fig.canvas.draw()

    def next_plot(self):
        print("Waiting for background thread to find a sequence...")
        try:
            # Wait up to 10 seconds for the scanner to find the next valid sequence
            next_idx = self.search_queue.get(timeout=10.0)
            self.history.append(next_idx)
            self.active_index = len(self.history) - 1
            self.draw_graph(next_idx)
            print(f"Rendered index {next_idx}")
        except queue.Empty:
            print("Scanner timeout. No valid sequences found recently.")

    def prev_plot(self):
        if self.active_index > 0:
            self.active_index -= 1
            prev_idx = self.history[self.active_index]
            self.draw_graph(prev_idx)
            print(f"Re-rendered previous index {prev_idx}")
        else:
            print("Already at the oldest viewed plot.")

    def on_press(self, event):
        if event.key == 'right':
            # If we are looking at an old history item, just move forward in history
            if self.active_index < len(self.history) - 1:
                self.active_index += 1
                self.draw_graph(self.history[self.active_index])
            else:
                # If we are at the edge, pull a new one from the queue
                self.next_plot()
        elif event.key == 'left':
            self.prev_plot()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not os.path.exists(HYPERPARAMETER_PATH):
        raise FileNotFoundError(f"Could not find {HYPERPARAMETER_PATH}")
        
    with open(HYPERPARAMETER_PATH, 'r') as f:
        hp = json.load(f)
        
    print(f"Loaded Hyperparameters for {CHOSEN_MODEL}-{PREFIX}")
    
    # ---------------------------------------------------------
    # NEW: Filter by Split JSON before passing to dataset loader
    # ---------------------------------------------------------
    allowed_files = None
    if TARGET_SPLIT in ["train", "val", "test"]:
        if not os.path.exists(SPLIT_FILE_PATH):
            raise FileNotFoundError(f"Missing {SPLIT_FILE_PATH}! Required to filter by '{TARGET_SPLIT}'.")
        
        with open(SPLIT_FILE_PATH, 'r') as f:
            splits = json.load(f)
            
        if TARGET_SPLIT not in splits:
            raise KeyError(f"Key '{TARGET_SPLIT}' not found in {SPLIT_FILE_PATH}")
            
        # Strip .npy extension to match how dataset.py checks base_names
        allowed_files = set([f.replace('.npy', '') for f in splits[TARGET_SPLIT]])
        print(f"Targeting '{TARGET_SPLIT}' split: Found {len(allowed_files)} allowed sequences.")
    
    # Load the custom dataset
    dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        window_size=hp.get("window_size"),
        overlap=hp.get("overlap"),
        tolerance_window=hp.get("tolerance_window"),
        use_full_length=False, 
        base_features=hp.get("base_features"),
        kinematic_features=hp.get("kinematic_features")
    )
    
    # Apply the split filter explicitly
    if allowed_files is not None:
        original_len = len(dataset.slice_index)
        dataset.slice_index = [s for s in dataset.slice_index if s['base_name'] in allowed_files]
        print(f"Dataset filtered from {original_len} total slices down to {len(dataset.slice_index)} '{TARGET_SPLIT}' slices.")
    
    if len(dataset.slice_index) == 0:
        raise ValueError(f"No slices remained after filtering for '{TARGET_SPLIT}'. Cannot run viewer.")

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
    
    search_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    
    scanner_thread = threading.Thread(
        target=background_scanner, 
        args=(dataset, search_queue, stop_event, hp),
        daemon=True
    )
    scanner_thread.start()
    
    viewer = InteractiveViewer(model, dataset, search_queue, hp)
    plt.show() 
    stop_event.set()

if __name__ == "__main__":
    main()