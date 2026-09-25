"""
queue_train_detr.py -- builds train_queue_detr.json for train_detr.py.
Separate from queue_train.py/queue_train_phrase.py: this paradigm's config
schema is genuinely different (no window_size/overlap/loss_function/
class_weights -- those are BIO-tagging concepts that don't apply to set
prediction).
"""
import os
import json

DETR_DEFAULTS = {
    "basename": "stgcn_detr",
    "epochs": 100,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "base_features": ["x-cord", "y-cord", "z-cord"],
    "kinematic_features": [],
    "in_channels": 3,
    "use_hamer_features": False,
    "hamer_dir": "processed_data/hamer_features",
    "use_dinov2_features": False,
    "dinov2_dir": "processed_data/dinov2_features",
    "d_model": 256,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "num_queries": 50000,
    "class_weight": 1.0,
    "l1_weight": 5.0,
    "iou_weight": 2.0,
    "no_object_weight": 0.1,
    "confidence_threshold": 0.5,
    "iou_match_threshold": 0.5,
}

# ==============================================================================
# EDIT THIS: each entry overrides DETR_DEFAULTS for one queued job.
# ==============================================================================
EXPERIMENTS_TO_RUN = [
    {"description": "first DETR run, coords only"},
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
        basename = exp.get("basename", DETR_DEFAULTS["basename"])
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
    # Resolved relative to THIS FILE's own location (not the terminal's current
    # directory), so this always points at the same train_queue_detr.json that
    # train_detr.py reads from, regardless of where either script is run from.
    QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_queue_detr.json")

    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
    else:
        queue = [{"prefixes": {}}]

    prefixes_data = queue[0].get("prefixes", {})
    queue[0]["prefixes"] = prefixes_data

    print("Current tracked prefixes by model in train_queue_detr.json:")
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
        m_name = exp.get("basename", DETR_DEFAULTS.get("basename", "unknown"))
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
        full_config = DETR_DEFAULTS.copy()
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
        print(f"Added to queue ({m_name}-{assigned_prefix:02d} | num_queries={full_config['num_queries']}): "
              f"{full_config.get('description', '')}")

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)

    print(f"\n✅ Successfully added {count} DETR experiments to the queue.")
    print("▶️  Run 'python train_detr.py' to start processing.")