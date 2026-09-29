"""
queue_train_stage2.py -- builds train_queue_stage2.json for
train_stage2_finetune.py. Every entry MUST set "pretrained_checkpoint" to a
Stage 1 checkpoint path (printed at the end of train_stage1_pretrain.py, or
found in latent_transformer/saved_models_stage1/) -- this script refuses to
queue an entry that's missing one, rather than let it fail only after you've
already started training.
"""
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_MODEL_DIR = os.path.join(_SCRIPT_DIR, "saved_models_stage1")

STAGE2_DEFAULTS = {
    "pretrained_checkpoint": None,  # MUST be set per-experiment or here globally -- see below
    "freeze_encoder": True,
    "window_size": 64,
    "overlap": 0,
    "batch_size": 16,
    "epochs": 100,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 5,
    "class_weights": [0.6, 0.8, 1.0],
    "base_features": ["x-cord", "y-cord", "z-cord"],
    "kinematic_features": [],
    "in_channels": 3,
    "use_hamer_features": False,
    "hamer_dir": "processed_data/hamer_features",
    "use_dinov2_features": False,
    "dinov2_dir": "processed_data/dinov2_features",
    "d_model": 256,
    "n_layers": 4,
    "nhead": 8,
}

# ==============================================================================
# EDIT THIS: each entry overrides STAGE2_DEFAULTS for one queued job.
# Set "pretrained_checkpoint" here, or override it per-entry below.
# ==============================================================================
GLOBAL_PRETRAINED_CHECKPOINT = os.path.join(STAGE1_MODEL_DIR, "stgcn_transformer_autoencoder-1.pth")  # e.g. os.path.join(STAGE1_MODEL_DIR, "stgcn_transformer_autoencoder-01.pth")

EXPERIMENTS_TO_RUN = [
    {
        "description": "first stage 2 fine-tune, frozen encoder, hamer",
        "frozen_encoder": True,
        "pretrained_checkpoint": GLOBAL_PRETRAINED_CHECKPOINT,
    },
    {
        "description": "first stage 2 fine-tune, fine-tune encoder, hamer",
        "frozen_encoder": False,
        "pretrained_checkpoint": GLOBAL_PRETRAINED_CHECKPOINT,
    },

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
        print(f"[ID: {i}] stage2_finetune")
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
    QUEUE_FILE = os.path.join(_SCRIPT_DIR, "train_queue_stage2.json")

    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
    else:
        queue = [{"prefixes": []}]

    prefixes_data = queue[0].get("prefixes", [])
    queue[0]["prefixes"] = prefixes_data

    print(f"Current tracked prefixes in train_queue_stage2.json: {prefixes_data or '(none)'}\n")

    if os.path.exists(STAGE1_MODEL_DIR):
        available = [f for f in os.listdir(STAGE1_MODEL_DIR) if f.endswith(".pth")]
        print(f"Available Stage 1 checkpoints in {STAGE1_MODEL_DIR}: {available or '(none found)'}\n")

    experiments_to_run = EXPERIMENTS_TO_RUN
    seed_indices = select_seed_experiments(experiments_to_run)
    experiments_to_run = expand_experiments_with_seeds(experiments_to_run, seed_indices)
    print()

    # Refuse to queue anything without a resolvable pretrained_checkpoint --
    # this fails LOUD here, rather than silently after training has already started.
    unresolved = []
    for i, exp in enumerate(experiments_to_run):
        ckpt = exp.get("pretrained_checkpoint", GLOBAL_PRETRAINED_CHECKPOINT)
        if not ckpt:
            unresolved.append(i)
        elif not os.path.exists(ckpt):
            print(f"⚠️ Entry {i}: pretrained_checkpoint={ckpt!r} does not exist on disk.")
            unresolved.append(i)
    if unresolved:
        print(f"\n❌ {len(unresolved)} experiment(s) have no valid pretrained_checkpoint "
              f"(entries {unresolved}). Set GLOBAL_PRETRAINED_CHECKPOINT at the top of this "
              f"file, or add \"pretrained_checkpoint\": \"<path>\" to each entry in "
              f"EXPERIMENTS_TO_RUN. Nothing was queued.")
        sys.exit(1)

    next_prefix = (max(prefixes_data) + 1) if prefixes_data else 1
    count = 0
    for exp in experiments_to_run:
        full_config = STAGE2_DEFAULTS.copy()
        full_config.update(exp)
        if full_config["pretrained_checkpoint"] is None:
            full_config["pretrained_checkpoint"] = GLOBAL_PRETRAINED_CHECKPOINT
        full_config["prefix"] = str(next_prefix)
        prefixes_data.append(next_prefix)
        queue.append(full_config)
        print(f"Added to queue (stage2_finetune-{next_prefix:02d}, "
              f"freeze_encoder={full_config['freeze_encoder']}): {full_config.get('description', '')}")
        next_prefix += 1
        count += 1

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)

    print(f"\n✅ Successfully added {count} Stage 2 fine-tuning jobs to the queue.")
    print("▶️  Run 'python train_stage2_finetune.py' to start processing.")