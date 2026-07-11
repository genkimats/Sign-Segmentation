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
# 🔄 LSTM-BASED DEFAULT HYPERPARAMETERS
# ==============================================================================
LSTM_DEFAULTS = {
    "basename": "bilstm_baseline",
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
    "d_model": 256,  # Note: Serves as hidden_dim for LSTM
    "n_layers": 4,   # Note: Number of LSTM layers
    "focal_loss_gamma": 2.0,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
}


if __name__ == "__main__":
    if CHOSEN_TYPE == 'mamba':
        defaults = MAMBA_DEFAULTS
        # Define Mamba ablation experiments you wish to queue
        experiments_to_run = [
            {
                "window_size": 128,
                "overlap": 112,
                "loss_function": "standard_ce",
                "learning_rate": 0.0005,
                "description": "ST-GCN Mamba: learning rate 0.0005"
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

    # Load existing queue to maintain the "prefixes" tracker
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            try:
                queue = json.load(f)
                # Ensure the queue has the correct structural format at index 0
                if len(queue) == 0 or "prefixes" not in queue[0]:
                    queue.insert(0, {"prefixes": []})
            except json.JSONDecodeError:
                queue = [{"prefixes": []}]
    else:
        queue = [{"prefixes": []}]

    # --- Manual Prefix Selection ---
    prefixes = queue[0]["prefixes"]
    print(f"Current tracked prefixes in {QUEUE_FILE}: {prefixes}")
    
    # Suggest the next prefix, but allow the user to override
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
        
        # Inject the designated prefix string
        prefix_str = f"{next_prefix:02d}"
        full_config["prefix"] = prefix_str
        
        queue.append(full_config)
        # Avoid duplicating the prefix tracker if user forces an override of an existing one
        if next_prefix not in queue[0]["prefixes"]:
            queue[0]["prefixes"].append(next_prefix)
        
        print(f"Added to queue (Prefix {prefix_str}): {full_config['description']}")
        next_prefix += 1
        count += 1
        
    # Write the entire payload back at once
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    print(f"\n✅ Successfully added {count} {CHOSEN_TYPE.upper()} experiments to {QUEUE_FILE}")