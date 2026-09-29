"""
queue_train_stage1.py -- builds train_queue_stage1.json for
train_stage1_pretrain.py.
"""
import os
import json

STAGE1_DEFAULTS = {
    "window_size": 64,
    "overlap": 0,
    "batch_size": 16,
    "epochs": 100,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "base_features": ["x-cord", "y-cord", "z-cord"],
    "kinematic_features": [],
    "in_channels": 3,
    "d_model": 256,
    "n_layers": 4,
    "nhead": 8,
}

# ==============================================================================
# EDIT THIS: each entry overrides STAGE1_DEFAULTS for one queued job.
# ==============================================================================
EXPERIMENTS_TO_RUN = [
    {"description": "first stage 1 pretraining run, coords only"},
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
        desc = exp.get("description", "No description provided")
        print(f"[ID: {i}] stage1_pretrain")
        print(f"        └─ 📝 {desc}\n")
    print(f"{'='*75}")
    user_input = input(
        "🎲 Enter IDs to run with 3 seeds (comma-separated), 'all', or Enter for none: "
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
    QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_queue_stage1.json")

    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
    else:
        queue = [{"prefixes": []}]

    prefixes_data = queue[0].get("prefixes", [])
    queue[0]["prefixes"] = prefixes_data

    print(f"Current tracked prefixes in train_queue_stage1.json: {prefixes_data or '(none)'}\n")

    experiments_to_run = EXPERIMENTS_TO_RUN
    seed_indices = select_seed_experiments(experiments_to_run)
    experiments_to_run = expand_experiments_with_seeds(experiments_to_run, seed_indices)
    print()

    next_prefix = (max(prefixes_data) + 1) if prefixes_data else 1
    count = 0
    for exp in experiments_to_run:
        full_config = STAGE1_DEFAULTS.copy()
        full_config.update(exp)
        full_config["prefix"] = str(next_prefix)
        prefixes_data.append(next_prefix)
        queue.append(full_config)
        print(f"Added to queue (stage1_pretrain-{next_prefix:02d}): {full_config.get('description', '')}")
        next_prefix += 1
        count += 1

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)

    print(f"\n✅ Successfully added {count} Stage 1 pretraining jobs to the queue.")
    print("▶️  Run 'python train_stage1_pretrain.py' to start processing.")