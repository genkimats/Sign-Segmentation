import json
import os

QUEUE_FILE = "train_queue.json"

# ==============================================================================
# 🚦 MODEL TYPE SELECTOR
# ==============================================================================
CHOSEN_TYPE = 'mamba'

# ==============================================================================
# 🐍 MAMBA-BASED DEFAULT HYPERPARAMETERS
# ==============================================================================
MAMBA_DEFAULTS = {
    "basename": "stgcn_mamba",  
    "batch_size": 16,
    "epochs": 50,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "temporal_downsample_factor": 1, 
    "loss_function": "standard_ce",  
    "ctc_weight": 0.5,             
    "class_weights": [0.6, 0.8, 1.0], 
    "tmse_weight": 0.15,            
    "tmse_threshold": 0.1,          
    "base_features": ["x-cord", "y-cord", "z-cord"], 
    "kinematic_features": ["spatial_angles"],        
    "in_channels": 5, 
    "d_model": 256,
    "n_layers": 4,     
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
}

EXPERIMENTS_TO_RUN = [
    {
        "basename": "latent_stgcn_mamba",             
        "description": "Model 1/5: Baseline ST-GCN + Mamba",
        "latent_dim": 128,
        "loss_function": "weighted_ce",
        "window_size": 16,
        "overlap": 0
    },
    {
        "basename": "ctrgcn_mamba",             
        "description": "Model 2/5: CTR-GCN (Dynamic Channel Graph) + Mamba",
        "latent_dim": 128,
        "loss_function": "weighted_ce",
        "window_size": 16,
        "overlap": 0
    },
    {
        "basename": "infogcn_mamba",             
        "description": "Model 3/5: InfoGCN (Multi-Scale Attention) + Mamba",
        "latent_dim": 128,
        "loss_function": "weighted_ce",
        "window_size": 16,
        "overlap": 0
    },
    {
        "basename": "shiftgcn_mamba",             
        "description": "Model 4/5: ShiftGCN (No Graph / Spatial Shifts) + Mamba",
        "latent_dim": 128,
        "loss_function": "weighted_ce",
        "window_size": 16,
        "overlap": 0
    },
    {
        "basename": "spatial_transformer_mamba",             
        "description": "Model 5/5: Spatial Self-Attention + Mamba",
        "latent_dim": 128,
        "loss_function": "weighted_ce",
        "window_size": 16,
        "overlap": 0
    }
]

if CHOSEN_TYPE == 'mamba':
    defaults = MAMBA_DEFAULTS
    experiments_to_run = EXPERIMENTS_TO_RUN
else:
    raise ValueError(f"Unknown CHOSEN_TYPE: {CHOSEN_TYPE}")


if __name__ == "__main__":
    # ==============================================================================
    # 📝 BUILD QUEUE
    # ==============================================================================
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
    else:
        queue = [{"prefixes": {}}]
        
    # --- MIGRATION: Convert old flat list to dict format ---
    prefixes_data = queue[0].get("prefixes", {})
    if isinstance(prefixes_data, list):
        prefixes_data = {"legacy_models": prefixes_data}
        queue[0]["prefixes"] = prefixes_data

    print("Current tracked prefixes by model in train_queue.json:")
    if not prefixes_data:
        print("  (None)")
    else:
        for m_name, p_list in prefixes_data.items():
            print(f"  - {m_name}: {p_list}")
    print()

    # --- PREFIX INPUT PHASE ---
    while True:
        try:
            user_input = input("Enter a starting prefix for this batch [Press Enter to auto-assign per model]: ").strip()
            
            if not user_input:
                base_prefix = None
                break
                
            base_prefix = int(user_input)
            if base_prefix <= 0:
                print("⚠️ Prefix must be a positive integer.")
                continue
                
            # Check for collisions with the specific models we are about to queue
            collision = False
            for exp in experiments_to_run:
                m_name = exp.get("basename", defaults.get("basename", "unknown"))
                if base_prefix in prefixes_data.get(m_name, []):
                    collision = True
                    break
                    
            if collision:
                print(f"⚠️ Warning: Prefix {base_prefix} is already tracked for one of the models in this batch!")
                override = input("Do you want to override and use it anyway? (y/N): ").strip().lower()
                if override == 'y':
                    break
            else:
                break
                
        except ValueError:
            print("⚠️ Please enter a valid number.")

    current_model_prefix = {}
    count = 0
    
    for exp in experiments_to_run:
        full_config = defaults.copy()
        full_config.update(exp)
        
        m_name = full_config.get("basename", "unknown")
        
        # 1. Determine the exact prefix for this specific architecture
        if m_name not in current_model_prefix:
            if base_prefix is not None:
                current_model_prefix[m_name] = base_prefix
            else:
                existing = prefixes_data.get(m_name, [])
                current_model_prefix[m_name] = max(existing) + 1 if existing else 1
                
        assigned_prefix = current_model_prefix[m_name]
        
        # Increment just in case you queue two of the EXACT SAME model in one batch
        current_model_prefix[m_name] += 1 
        
        # 2. Calculate Input Channels dynamically
        calculated_channels = calculate_in_channels(full_config)
        full_config["in_channels"] = calculated_channels
        
        prefix_str = f"{assigned_prefix:02d}"
        full_config["prefix"] = prefix_str
        
        # 3. Add to Queue Array
        queue.append(full_config)
        
        # 4. Save to Tracker Dictionary
        if m_name not in queue[0]["prefixes"]:
            queue[0]["prefixes"][m_name] = []
            
        if assigned_prefix not in queue[0]["prefixes"][m_name]:
            queue[0]["prefixes"][m_name].append(assigned_prefix)
        
        print(f"Added to queue ({m_name}-{prefix_str} | Channels: {calculated_channels}): {full_config['description']}")
        count += 1
        
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    print(f"\n✅ Successfully added {count} {CHOSEN_TYPE.upper()} experiments to the queue.")
    print(f"▶️  Run 'python train.py' to start processing.")