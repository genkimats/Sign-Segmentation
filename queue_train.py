import json
import os

QUEUE_FILE = "train_queue.json"

# ==============================================================================
# 🚦 MODEL TYPE SELECTOR
# ==============================================================================
# Change this variable to determine which defaults and experiments are queued:
# Options: 'mamba'        - ST-GCN + Mamba architecture
#          'pure_mamba'   - Pure Mamba baseline architecture (No GCN)
#          'lstm'         - Baseline BiLSTM architecture (No GCN)
#          'stgcn_lstm'   - Hybrid ST-GCN + BiLSTM architecture
#          'transformer'  - Pure Multi-Head Attention architecture (No GCN)
CHOSEN_TYPE = 'transformer'

# ==============================================================================
# 🐍 MAMBA-BASED DEFAULT HYPERPARAMETERS (ST-GCN + Mamba)
# ==============================================================================
MAMBA_DEFAULTS = {
    "basename": "stgcn_mamba",  
    "batch_size": 16,
    "epochs": 50,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0003,
    "num_vertices": 65,
    "tolerance_window": 5,
    "temporal_downsample_factor": 1, 
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
# 🐍 PURE MAMBA DEFAULT HYPERPARAMETERS (No GCN)
# ==============================================================================
PURE_MAMBA_DEFAULTS = {
    "basename": "pure_mamba",  
    "batch_size": 16,
    "epochs": 50,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0003,
    "num_vertices": 65,
    "tolerance_window": 5,
    "temporal_downsample_factor": 1, 
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
# 🔄 LSTM-BASED DEFAULT HYPERPARAMETERS (Baseline)
# ==============================================================================
LSTM_DEFAULTS = {
    "basename": "bilstm_baseline",
    "batch_size": 16,
    "epochs": 50,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "temporal_downsample_factor": 1, 
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
# 🔗 HYBRID STGCN + LSTM DEFAULTS
# ==============================================================================
STGCN_LSTM_DEFAULTS = {
    "basename": "stgcn_bilstm",
    "batch_size": 16,
    "epochs": 50,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "temporal_downsample_factor": 1,
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
# 🤖 TRANSFORMER DEFAULT HYPERPARAMETERS
# ==============================================================================
TRANSFORMER_DEFAULTS = {
    "basename": "transformer_baseline",
    "batch_size": 16, # Beware of OOM errors on large window sizes!
    "epochs": 50,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "temporal_downsample_factor": 1,
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
    "nhead": 8,              # <-- TRANSFORMER SPECIFIC
    "dim_feedforward": 1024, # <-- TRANSFORMER SPECIFIC (Usually 4x d_model)
    "focal_loss_gamma": 2.0,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
}

def calculate_in_channels(base_features, kinematic_features):
    """Dynamically calculates the exact number of input channels based on requested features."""
    base_count = len(base_features)
    deriv_count = base_count if base_count > 0 else 2
    total_channels = base_count
    
    if "velocity" in kinematic_features: total_channels += deriv_count
    if "acceleration" in kinematic_features: total_channels += deriv_count
    if "jerk" in kinematic_features: total_channels += deriv_count
    if "velocity-mag" in kinematic_features: total_channels += 1
    if "angular-vel" in kinematic_features: total_channels += 1
        
    return total_channels


if __name__ == "__main__":
    if CHOSEN_TYPE == 'mamba':
        defaults = MAMBA_DEFAULTS
        experiments_to_run = [
            {
                "window_size": 256,
                "overlap": 128,
                "temporal_downsample_factor": 2, 
                "description": "Mamba SOTA: 2x Temp Downsample (Network sees 128 frames spanning 10 seconds)"
            }
        ]
        
    elif CHOSEN_TYPE == 'pure_mamba':
        defaults = PURE_MAMBA_DEFAULTS
        experiments_to_run = [
            {
                "window_size": 128,
                "overlap": 64,
                "description": "Pure Mamba Baseline (No ST-GCN feature extraction)"
            }
        ]
        
    elif CHOSEN_TYPE == 'lstm':
        defaults = LSTM_DEFAULTS
        experiments_to_run = [
            {
                "window_size": 128,
                "overlap": 64,
                "description": "LSTM-baseline overlap ratio (64/128)"
            }
        ]
        
    elif CHOSEN_TYPE == 'stgcn_lstm':
        defaults = STGCN_LSTM_DEFAULTS
        experiments_to_run = [
            {
                "window_size": 128,
                "overlap": 64,
                "description": "Hybrid ST-GCN + BiLSTM (Evaluating GCN impact over standard MLPs)"
            }
        ]
        
    elif CHOSEN_TYPE == 'transformer':
        defaults = TRANSFORMER_DEFAULTS
        experiments_to_run = [
            {
                "window_size": 128,
                "overlap": 64,
                "description": "Pure Transformer Baseline (Evaluating Multi-Head Attention vs. Mamba/LSTM)"
            }
        ]
        
    else:
        raise ValueError(f"Unknown CHOSEN_TYPE: '{CHOSEN_TYPE}'. Choose either 'mamba', 'pure_mamba', 'lstm', 'stgcn_lstm', or 'transformer'.")

    # Load existing queue to maintain the "prefixes" tracker
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            try:
                queue = json.load(f)
                if len(queue) == 0 or "prefixes" not in queue[0]:
                    queue.insert(0, {"prefixes": []})
            except json.JSONDecodeError:
                queue = [{"prefixes": []}]
    else:
        queue = [{"prefixes": []}]

    # --- Manual Prefix Selection ---
    prefixes = queue[0]["prefixes"]
    print(f"Current tracked prefixes in {QUEUE_FILE}: {prefixes}")
    
    suggested_prefix = max(prefixes) + 1 if prefixes else 1
    
    while True:
        user_input = input(f"\nEnter starting prefix number [Press Enter to use {suggested_prefix}]: ").strip()
        
        if not user_input:
            next_prefix = suggested_prefix
            break
        try:
            next_prefix = int(user_input)
            if next_prefix <= 0:
                print("⚠️ Prefix must be a positive integer.")
            elif next_prefix in prefixes:
                print(f"⚠️ Warning: Prefix {next_prefix} is already in the queue array!")
                override = input("Do you want to override and use it anyway? (y/N): ").strip().lower()
                if override == 'y':
                    break
            else:
                break
        except ValueError:
            print("⚠️ Please enter a valid number.")

    count = 0
    for exp in experiments_to_run:
        full_config = defaults.copy()
        full_config.update(exp)
        
        calculated_channels = calculate_in_channels(
            base_features=full_config.get("base_features", []),
            kinematic_features=full_config.get("kinematic_features", [])
        )
        full_config["in_channels"] = calculated_channels
        
        prefix_str = f"{next_prefix:02d}"
        full_config["prefix"] = prefix_str
        
        queue.append(full_config)
        if next_prefix not in queue[0]["prefixes"]:
            queue[0]["prefixes"].append(next_prefix)
        
        print(f"Added to queue (Prefix {prefix_str} | Channels: {calculated_channels}): {full_config['description']}")
        next_prefix += 1
        count += 1
        
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    print(f"\n✅ Successfully added {count} {CHOSEN_TYPE.upper()} experiments to {QUEUE_FILE}")