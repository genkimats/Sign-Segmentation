import json
import os

QUEUE_FILE = "train_queue.json"

# ==============================================================================
# 🚦 MODEL TYPE SELECTOR
# ==============================================================================
# Change this variable to determine which defaults and experiments are queued:
# Options: 'mamba'  - Queues ST-GCN Mamba models with their optimized hyper-redundant settings.
#          'lstm'   - Queues Bidirectional LSTM baselines with sequence-adapted settings.
CHOSEN_TYPE = 'mamba'

# ==============================================================================
# 🐍 MAMBA-BASED DEFAULT HYPERPARAMETERS
# ==============================================================================
MAMBA_DEFAULTS = {
    "basename": "stgcn_mamba",  # "stgcn_mamba" or "stgcn_bimamba"
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "loss_function": "standard_ce",
    "class_weights": [0.1, 0.3, 1.0],
    "use_full_length": False,
    "base_features": ["x-cord", "y-cord", "z-cord"],
    "kinematic_features": ["velocity", "acceleration", "jerk", "velocity-mag", "angular-vel"],
    "in_channels": 14,
    "decoder_strategy": "threshold",
    "decoder_threshold": 0.5,
    "d_model": 256,
    "n_layers": 4,
    "focal_loss_gamma": 2.0,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
}

# ==============================================================================
# 📈 LSTM-BASED DEFAULT HYPERPARAMETERS
# ==============================================================================
LSTM_DEFAULTS = {
    "basename": "bilstm_baseline",
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "loss_function": "weighted_ce",
    "class_weights": [0.1, 0.3, 1.0],
    "use_full_length": False,
    "base_features": ["x-cord", "y-cord", "z-cord"],
    "kinematic_features": ["velocity", "acceleration", "jerk", "velocity-mag", "angular-vel"],
    "in_channels": 14,
    "decoder_strategy": "threshold",
    "decoder_threshold": 0.5,
    "d_model": 256,  # 256 per direction = 512 total hidden units
    "n_layers": 4,   # 4 layers matching SOTA paper baseline
    "focal_loss_gamma": 2.0,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
}

def add_to_queue(job_config):
    """Appends a new hyperparameter configuration to the queue file."""
    queue = []
    
    # Load existing queue if it exists
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            try:
                queue = json.load(f)
            except json.JSONDecodeError:
                queue = []
                
    queue.append(job_config)
    
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    print(f"✅ Enqueued Job: {job_config.get('description')}")

if __name__ == "__main__":
    print(f"🛠️ Preparing experiment list for CHOSEN_TYPE: '{CHOSEN_TYPE.upper()}'")
    
    experiments_to_run = []
    defaults = {}

    if CHOSEN_TYPE == 'mamba':
        defaults = MAMBA_DEFAULTS
        # Define Mamba ablation experiments you wish to queue
        experiments_to_run = [
            {
                "window_size": 128,
                "overlap": 112,
                "loss_function": "standard_ce",
                "description": "Mamba SOTA: 128 Window | 112 Overlap (87.5% Overlap)"
            },
            {
                "window_size": 64,
                "overlap": 56,
                "loss_function": "standard_ce",
                "description": "Mamba SOTA: 64 Window | 56 Overlap (87.5% Overlap)"
            },
            {
                "window_size": 512,
                "overlap": 448,
                "loss_function": "weighted_ce",
                "description": "Mamba SOTA: 512 Window | 448 Overlap (87.5% Overlap, Weighted CE)"
            }
        ]
        
    elif CHOSEN_TYPE == 'lstm':
        defaults = LSTM_DEFAULTS
        # Define LSTM ablation experiments you wish to queue
        experiments_to_run = [
            {
                "window_size": 128,
                "overlap": 64,
                "description": "LSTM-baseline overlap ratio (64/128)"
            },
            {
                "window_size": 256,
                "overlap": 128,
                "description": "LSTM-baseline overlap ratio (128/256)"
            },
            {
                "window_size": 512,
                "overlap": 256,
                "description": "LSTM-baseline overlap ratio (256/512)"
            }
        ]
        
    else:
        raise ValueError(f"Unknown CHOSEN_TYPE: '{CHOSEN_TYPE}'. Choose either 'mamba' or 'lstm'.")

    # Merge each experiment variant with the selected default parameter set
    count = 0
    for exp in experiments_to_run:
        full_config = defaults.copy()
        full_config.update(exp)
        add_to_queue(full_config)
        count += 1
        
    print(f"\n🎉 Successfully added {count} '{CHOSEN_TYPE}' runs to '{QUEUE_FILE}'!")