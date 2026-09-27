"""
queue_train_multitask.py -- builds train_queue_multitask.json for
train_multitask.py. Separate from every other queue script in this project:
own queue file, tracked independently.
"""
import os
import json

MULTITASK_DEFAULTS = {
    "basename": "stgcn_bilstm_multitask",
    "window_size": 64,
    "overlap": 0,
    "batch_size": 16,
    "epochs": 100,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "base_features": ["x-cord", "y-cord", "z-cord"],
    "kinematic_features": [],
    "in_channels": 3,
    "class_weights": [0.6, 0.8, 1.0],
    "gloss_loss_weight": 0.1,
    "normalize_gloss_loss": True,
    "use_hamer_features": False,
    "hamer_dir": "processed_data/hamer_features",
    "use_dinov2_features": False,
    "dinov2_dir": "processed_data/dinov2_features",
    "d_model": 256,
    "n_layers": 4,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
}

# ==============================================================================
# EDIT THIS: each entry overrides MULTITASK_DEFAULTS for one queued job.
# ==============================================================================
EXPERIMENTS_TO_RUN = [
    {"description": "first multitask run, hamer, gloss_loss_weight=0.3"},
]


def with_seeds(exp, seeds=(42, 123, 2024)):
    variants = []
    for seed in seeds:
        variant = dict(exp)
        variant["seed"] = seed
        variant["description"] = f"{exp.get('description', '')} [seed={seed}]".strip()
        variants.append(variant)
    return variants


def select_seed_experiments(experiments):
    print(f"\n{'='*75}")
    print("📋 EXPERIMENTS DEFINED IN EXPERIMENTS_TO_RUN")
    print(f"{'='*75}")
    for i, exp in enumerate(experiments):
        basename = exp.get("basename", MULTITASK_DEFAULTS["basename"])
        desc = exp.get("description", "No description provided")
        print(f"[ID: {i}] {basename}")
        print(f"        └─ 📝 {desc}\n")
    print(f"{'='*75}")

    user_input = input(
        "🎲 Enter IDs to run with 3 seeds (comma-separated, e.g. 0, 2), "
        "'all' for all of them, or press Enter for none (1 seed each): "
    ).strip()

    if not user_input:
        return set()
    if user_input.lower() == 'all':
        return set(range(len(experiments)))
    try:
        ids = [int(x.strip()) for x in user_input.split(',')]
    except ValueError:
        print("⚠️ Invalid input. Defaulting to 1 seed for every experiment.")
        return set()

    valid_ids = set(i for i in ids if 0 <= i < len(experiments))
    invalid_ids = set(ids) - valid_ids
    if invalid_ids:
        print(f"⚠️ Ignoring out-of-range ID(s): {sorted(invalid_ids)}")
    return valid_ids


def expand_experiments_with_seeds(experiments, seed_indices, seeds=(42, 123, 2024)):
    final = []
    for i, exp in enumerate(experiments):
        if i in seed_indices:
            final.extend(with_seeds(exp, seeds=seeds))
        else:
            final.append(exp)
    return final


if __name__ == "__main__":
    # Resolved relative to THIS FILE's own location, not the terminal's current
    # directory, so this always points at the same train_queue_multitask.json
    # that train_multitask.py reads from, regardless of where either script is
    # run from.
    QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_queue_multitask.json")

    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
    else:
        queue = [{"prefixes": {}}]

    prefixes_data = queue[0].get("prefixes", {})
    queue[0]["prefixes"] = prefixes_data

    print("Current tracked prefixes by model in train_queue_multitask.json:")
    if not prefixes_data:
        print("  (None)")
    else:
        for m_name, p_list in prefixes_data.items():
            print(f"  - {m_name}: {p_list}")
    print()

    experiments_to_run = EXPERIMENTS_TO_RUN
    seed_indices = select_seed_experiments(experiments_to_run)
    experiments_to_run = expand_experiments_with_seeds(experiments_to_run, seed_indices)
    print()

    distinct_model_names = []
    for exp in experiments_to_run:
        m_name = exp.get("basename", MULTITASK_DEFAULTS.get("basename", "unknown"))
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

    current_model_prefix = {}
    count = 0
    for exp in experiments_to_run:
        full_config = MULTITASK_DEFAULTS.copy()
        full_config.update(exp)

        m_name = full_config.get("basename", "unknown")
        if m_name not in current_model_prefix:
            chosen_base = base_prefix_by_model.get(m_name)
            if chosen_base is not None:
                current_model_prefix[m_name] = chosen_base
            else:
                existing = prefixes_data.get(m_name, [])
                current_model_prefix[m_name] = max(existing) + 1 if existing else 1

        assigned_prefix = current_model_prefix[m_name]
        current_model_prefix[m_name] += 1

        full_config["prefix"] = str(assigned_prefix)
        prefixes_data.setdefault(m_name, [])
        if assigned_prefix not in prefixes_data[m_name]:
            prefixes_data[m_name].append(assigned_prefix)

        queue.append(full_config)
        count += 1
        print(f"Added to queue ({m_name}-{assigned_prefix:02d} | gloss_loss_weight="
              f"{full_config['gloss_loss_weight']}): {full_config.get('description', '')}")

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)

    print(f"\n✅ Successfully added {count} multitask experiments to the queue.")
    print("▶️  Run 'python train_multitask.py' to start processing.")