#region
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
    "window_size": 16,
    "overlap": 0,
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
    "use_face_keypoints": False,
    "face_dir": "processed_data/face_keypoints_normalized",
    "use_hamer_features": False,
    "hamer_dir": "processed_data/hamer_features",
    "d_model": 256,
    "n_layers": 4,     
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
}

# ==============================================================================
# 🚦 DYNAMIC IN_CHANNELS CALCULATOR
# ==============================================================================
def calculate_in_channels(config):
    base_features = config.get("base_features", [])
    kinematic_features = config.get("kinematic_features", [])
    
    # --- PURE HAMER CALCULATION ---
    if "pure_hamer" in base_features:
        total = config.get("hamer_dim", 288) 
        
        for feat in kinematic_features:
            if feat in ["velocity", "acceleration", "jerk"]:
                total += (65 * 3) # 195
            elif feat in ["velocity-mag"]:
                total += (65 * 1) # 65
            elif feat in ["angular-vel"]:
                total += (65 * 1) # 65
            elif feat in ["spatial_angles", "temporal_angles"]:
                total += (65 * 2) # 130
        return total
    
    # --- STANDARD / HYBRID CALCULATION ---
    valid_base_cords = [f for f in base_features if f in ["x-cord", "y-cord", "z-cord"]]
    total_channels = len(valid_base_cords)
    deriv_channels = len(valid_base_cords) if valid_base_cords else 3
    
    for feat in kinematic_features:
        if feat in ["velocity", "acceleration", "jerk"]:
            total_channels += deriv_channels
        elif feat in ["velocity-mag", "angular-vel"]:
            total_channels += 1
        elif feat in ["spatial_angles", "temporal_angles"]:
            total_channels += 2 
            
    return total_channels

# ==============================================================================
# 🚦 DYNAMIC NUM_VERTICES CALCULATOR
# ==============================================================================
# Must match len(SELECTED_INDICES) in extract_face_keypoints.py -- update both
# together if that index list ever changes.
NUM_FACE_VERTICES = 83
BASE_BODY_HAND_VERTICES = 65

def calculate_num_vertices(config):
    base_features = config.get("base_features", [])

    if "pure_hamer" in base_features:
        # pure_hamer mode is a flat per-frame feature vector (not a per-vertex point
        # cloud), so num_vertices isn't meaningful the same way here -- leave whatever
        # the config already specifies untouched rather than guessing.
        return config.get("num_vertices", BASE_BODY_HAND_VERTICES)

    total = BASE_BODY_HAND_VERTICES
    if config.get("use_face_keypoints", False):
        total += NUM_FACE_VERTICES
    return total
#endregion

EXPERIMENTS_TO_RUN = [
    {
        "basename": "transformer_baseline",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": False,
        "description": "overlap ratio (0/64), loss=weighted_ce, no kinetic"
    },
    {
        "basename": "transformer_baseline",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": True,
        "description": "overlap ratio (0/64), loss=weighted_ce, hamer"
    },
    {
        "basename": "stgcn_transformer",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": False,
        "description": "overlap ratio (0/64), loss=weighted_ce, no kinetic"
    },
    {
        "basename": "stgcn_transformer",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": True,
        "description": "overlap ratio (0/64), loss=weighted_ce, hamer"
    },
    {
        "basename": "bilstm_baseline",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": False,
        "description": "overlap ratio (0/64), loss=weighted_ce, no kinetic"
    },
    {
        "basename": "bilstm_baseline",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": True,
        "description": "overlap ratio (0/64), loss=weighted_ce, hamer"
    },
    {
        "basename": "stgcn_bilstm",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": False,
        "description": "overlap ratio (0/64), loss=weighted_ce, no kinetic"
    },
    {
        "basename": "stgcn_bilstm",
        "window_size": 64,
        "overlap": 0,
        "use_hamer_features": True,
        "description": "overlap ratio (0/64), loss=weighted_ce, hamer"
    },
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

    # --- PREFIX INPUT PHASE (per model variation) ---
    # Determine every distinct model basename present in this batch, in the order
    # they first appear, and ask for a starting prefix separately for EACH one --
    # useful now that a single queuing session often mixes several architectures
    # (e.g. stgcn_mamba, stgcn_bilstm, stgcn_transformer) that you may want to
    # start at different prefix numbers, rather than one shared prompt forcing
    # the same starting point on all of them.
    distinct_model_names = []
    for exp in experiments_to_run:
        m_name = exp.get("basename", defaults.get("basename", "unknown"))
        if m_name not in distinct_model_names:
            distinct_model_names.append(m_name)

    base_prefix_by_model = {}

    for m_name in distinct_model_names:
        existing = prefixes_data.get(m_name, [])
        existing_str = f"existing: {existing}" if existing else "no existing prefixes"
        while True:
            try:
                user_input = input(
                    f"Enter a starting prefix for '{m_name}' ({existing_str}) "
                    f"[Press Enter to auto-assign]: "
                ).strip()

                if not user_input:
                    base_prefix_by_model[m_name] = None
                    break

                candidate = int(user_input)
                if candidate <= 0:
                    print("⚠️ Prefix must be a positive integer.")
                    continue

                if candidate in existing:
                    print(f"⚠️ Warning: Prefix {candidate} is already tracked for '{m_name}'!")
                    override = input("Do you want to override and use it anyway? (y/N): ").strip().lower()
                    if override != 'y':
                        continue

                base_prefix_by_model[m_name] = candidate
                break

            except ValueError:
                print("⚠️ Please enter a valid number.")

    print()

    current_model_prefix = {}
    count = 0
    
    for exp in experiments_to_run:
        full_config = defaults.copy()
        full_config.update(exp)
        
        m_name = full_config.get("basename", "unknown")
        
        # 1. Determine the exact prefix for this specific architecture
        if m_name not in current_model_prefix:
            chosen_base = base_prefix_by_model.get(m_name)
            if chosen_base is not None:
                current_model_prefix[m_name] = chosen_base
            else:
                existing = prefixes_data.get(m_name, [])
                current_model_prefix[m_name] = max(existing) + 1 if existing else 1
                
        assigned_prefix = current_model_prefix[m_name]
        
        # Increment just in case you queue two of the EXACT SAME model in one batch
        current_model_prefix[m_name] += 1 
        
        # 2. Calculate Input Channels & Num Vertices dynamically
        calculated_channels = calculate_in_channels(full_config)
        full_config["in_channels"] = calculated_channels

        calculated_vertices = calculate_num_vertices(full_config)
        full_config["num_vertices"] = calculated_vertices
        
        prefix_str = f"{assigned_prefix:02d}"
        full_config["prefix"] = prefix_str
        
        # 3. Add to Queue Array
        queue.append(full_config)
        
        # 4. Save to Tracker Dictionary
        if m_name not in queue[0]["prefixes"]:
            queue[0]["prefixes"][m_name] = []
            
        if assigned_prefix not in queue[0]["prefixes"][m_name]:
            queue[0]["prefixes"][m_name].append(assigned_prefix)
        
        print(f"Added to queue ({m_name}-{prefix_str} | Channels: {calculated_channels} | "
              f"Vertices: {calculated_vertices}): {full_config['description']}")
        count += 1
        
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    print(f"\n✅ Successfully added {count} {CHOSEN_TYPE.upper()} experiments to the queue.")
    print(f"▶️  Run 'python train.py' to start processing.")