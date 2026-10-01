"""
queue_train_novelty.py -- builds train_queue_novelty.json for train_novelty.py.

EXPERIMENTS_TO_RUN is the ablation ladder, in the order I recommend running it.
Every arm differs from the one before it by ONE change, so each comparison
answers one question (use the paired bootstrap in decoder_study/compare_runs.py
to say whether a difference is real):

  0  bilstm, CE only, no similarity   control: the new training script/recipe with
                                      nothing new switched on. Must land near your
                                      existing raw-target BiLSTM before you trust
                                      any later arm.
  1  + CRF loss                       does training for Viterbi help?
  2  + similarity (CE loss)           do similarity/novelty features help on their own?
  3  + similarity + CRF               do they combine?
  4  xlstm + similarity + CRF         the full model: does xLSTM beat BiLSTM here?
  5  arm 4 + HaMeR smoothing          optional: does de-jittering the hand features help?
  6  arm 4 + velocity/acceleration    optional: explicit kinematic channels

Every key you write in an experiment must be a key of NOVELTY_DEFAULTS (plus
"description" and "seed"): dict.update() would otherwise silently ADD a typo'd
key and leave the setting you meant at its default, so a typo aborts the queue.
"""
import os
import sys
import json

NOVELTY_DEFAULTS = {
    # data / training
    "window_size": 64, "overlap": 0, "batch_size": 16, "epochs": 100, "early_stopping": True, "patience": 10,
    "learning_rate": 3e-4, "weight_decay": 0.01,
    "tolerance_window": 1,                       # raw targets: the CRF supplies the tolerance
    "class_weights": [0.6, 0.8, 1.0], "ce_weight": 0.5,
    "num_vertices": 65, "in_channels": 3,        # in_channels is re-read from the data at train time
    "base_features": ["x-cord", "y-cord", "z-cord"], "kinematic_features": [],
    "use_hamer_features": True, "hamer_dir": "processed_data/hamer_features",
    "use_dinov2_features": False, "dinov2_dir": "processed_data/dinov2_features",
    # model
    "d_model": 256, "n_layers": 4, "backbone": "xlstm", "xlstm_pattern": "mmms", "xlstm_heads": 4,
    "dropout": 0.2, "n_local_blocks": 3, "adapter_layers": 2,
    "use_similarity": True, "similarity_K": 16, "novelty_scales": [2, 4, 8, 16], "d_sim": 64,
    "use_crf": True, "crf_forbid_penalty": -20.0,
    "hamer_proj_dim": 128, "hamer_smooth": 0, "dinov2_proj_dim": 128,
}

EXPERIMENTS_TO_RUN = [
    {"description": "0 control: bilstm, CE only, no similarity",
     "backbone": "bilstm", "use_similarity": False, "use_crf": False},
    {"description": "1 bilstm + CRF loss",
     "backbone": "bilstm", "use_similarity": False, "use_crf": True},
    {"description": "2 bilstm + similarity (CE loss)",
     "backbone": "bilstm", "use_similarity": True, "use_crf": False},
    {"description": "3 bilstm + similarity + CRF",
     "backbone": "bilstm", "use_similarity": True, "use_crf": True},
    {"description": "4 xlstm + similarity + CRF (full model)",
     "backbone": "xlstm", "use_similarity": True, "use_crf": True},
    # --- optional follow-ups -------------------------------------------------
    # {"description": "5 full model + HaMeR smoothing", "hamer_smooth": 5},
    # {"description": "6 full model + velocity/acceleration channels",
    #  "kinematic_features": ["velocity", "acceleration"]},      # in_channels is auto-detected
]


def with_seeds(exp, seeds=(42, 123, 2024)):
    out = []
    for seed in seeds:
        v = dict(exp)
        v["seed"] = seed
        v["description"] = f"{exp.get('description', '')} [seed={seed}]".strip()
        out.append(v)
    return out


def select_seed_experiments(experiments):
    print(f"\n{'=' * 75}\nEXPERIMENTS DEFINED IN EXPERIMENTS_TO_RUN\n{'=' * 75}")
    for i, exp in enumerate(experiments):
        print(f"[ID: {i}] {exp.get('description', 'No description provided')}")
    print("=" * 75)
    raw = input("Enter IDs to run with 3 seeds (comma-separated), 'all', or Enter for none: ").strip()
    if not raw:
        return set()
    if raw.lower() == "all":
        return set(range(len(experiments)))
    try:
        ids = [int(x.strip()) for x in raw.split(",")]
    except ValueError:
        print("Invalid input. Using 1 seed for every experiment.")
        return set()
    valid = {i for i in ids if 0 <= i < len(experiments)}
    if set(ids) - valid:
        print(f"Ignoring out-of-range ID(s): {sorted(set(ids) - valid)}")
    return valid


def expand_experiments_with_seeds(experiments, seed_indices, seeds=(42, 123, 2024)):
    final = []
    for i, exp in enumerate(experiments):
        final.extend(with_seeds(exp, seeds) if i in seed_indices else [exp])
    return final


if __name__ == "__main__":
    QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_queue_novelty.json")
    queue = json.load(open(QUEUE_FILE)) if os.path.exists(QUEUE_FILE) else [{"prefixes": []}]
    prefixes = queue[0].get("prefixes", [])
    queue[0]["prefixes"] = prefixes
    print(f"Tracked prefixes in train_queue_novelty.json: {prefixes or '(none)'}")

    experiments = EXPERIMENTS_TO_RUN
    experiments = expand_experiments_with_seeds(experiments, select_seed_experiments(experiments))
    print()

    known = set(NOVELTY_DEFAULTS) | {"description", "seed"}
    bad = [(i, sorted(set(e) - known)) for i, e in enumerate(experiments) if set(e) - known]
    if bad:
        print("Unrecognized config key(s) -- likely a typo. dict.update() would silently ADD them as new, "
              "ignored keys instead of overriding the setting you meant:")
        for i, keys in bad:
            print(f"   entry {i}: {keys}")
        print("Nothing was queued. Fix the key name(s) in EXPERIMENTS_TO_RUN and rerun.")
        sys.exit(1)

    nxt = (max(prefixes) + 1) if prefixes else 1
    for exp in experiments:
        cfg = NOVELTY_DEFAULTS.copy()
        cfg.update(exp)
        cfg["prefix"] = str(nxt)
        prefixes.append(nxt)
        queue.append(cfg)
        print(f"Queued #{nxt}: backbone={cfg['backbone']} sim={cfg['use_similarity']} crf={cfg['use_crf']} "
              f"-- {cfg.get('description', '')}")
        nxt += 1

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
    print(f"\nAdded {len(experiments)} job(s). Run 'python train_novelty.py' to start.")