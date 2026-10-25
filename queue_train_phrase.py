#region
import json
import os

QUEUE_FILE = "train_queue_phrase.json"

# ==============================================================================
# 🚦 MODEL TYPE SELECTOR
# ==============================================================================
CHOSEN_TYPE = 'mamba'

# ==============================================================================
# 🐍 MAMBA-BASED DEFAULT HYPERPARAMETERS
# ==============================================================================
# Default [O, I, B] class weights for weighted_ce (your recipe; used for the multistream runs).
DEFAULT_CLASS_WEIGHTS = [1.0, 1.1, 8.5]

MAMBA_DEFAULTS = {
    "basename": "stgcn_mamba",  
    "window_size": 256,
    "overlap": 0,
    "batch_size": 16,
    "epochs": 100,
    "early_stopping": True,
    "patience": 10,
    "learning_rate": 0.0001,
    "num_vertices": 65,
    "tolerance_window": 1,
    "temporal_downsample_factor": 1, 
    "loss_function": "weighted_ce",  
    "ctc_weight": 0.5,             
    "class_weights": list(DEFAULT_CLASS_WEIGHTS),
    "tmse_weight": 0.15,            
    "tmse_threshold": 0.1,          
    "base_features": ["x-cord", "y-cord", "z-cord"], 
    "kinematic_features": [],        
    "in_channels": 5, 
    "use_face_keypoints": False,
    # Face keypoints: face-local, 2D (x, y) files from data_processing/normalize_face_keypoints_v2.py
    "face_dir": "processed_data/face_keypoints_normalized_v2",
    "face_dims": 2,
    # Which saved face points to load when use_face_keypoints=True (see src/face_subsets.py):
    #   "full" 83 | "compact" 31 (eyes, brows, mouth) | "eyes_brows" 22 | "minimal" 18
    "face_subset": "full",
    # "saved83" = the 83-point files above; "full478" = all landmarks from
    # data_processing/extract_face_keypoints_full.py (+ normalize ... --layout full), needed for
    # the 2023 subset ("contours2023"), "contours2023_no_oval", "face_oval", "compact_iris"
    "face_layout": "saved83",
    # True = face vertices ONLY (body + hands dropped); needs use_face_keypoints=True and
    # kinematic_features=[]. Works with stgcn_bilstm / stgcn_transformer / *_baseline.
    "face_only": False,
    "use_pose_input": True,
    "stream_encoder": "stgcn",       # multistream keypoint streams: stgcn | blockgcn | skateformer | linear          # bilstm_2023 only: False = ignore the keypoints (HaMeR only)
    "use_hamer_features": False,
    "hamer_dir": "processed_data/hamer_features",
    # Phrase labels to train on. v2 = corrected boundaries (data_processing/
    # create_phrase_bio_tags_v2.py); v1 = processed_data/BIO_tags_phrase (all earlier runs).
    "phrase_labels_dir": "processed_data/BIO_tags_phrase_v2",
    # Stream balancing (stgcn_bilstm / stgcn_transformer): project the flattened ST-GCN
    # output to this size before it is concatenated with HaMeR/DINOv2. None = off
    # (original behaviour). hamer_proj_dim / dinov2_proj_dim can also be set per
    # experiment (model defaults: 64 / 128).
    "stgcn_proj_dim": None,
    # --- Multi-stream models (multistream_bilstm / multistream_transformer) ---
    # Switch each stream on/off. The queue sets use_face_keypoints / use_hamer_features
    # automatically from these for multistream models.
    "use_stream_body_hands": True,   # ST-GCN on body + hands (65 vertices)
    "use_stream_face": False,        # ST-GCN on the face subset (see face_subset)
    "use_stream_hamer": False,       # MLP on HaMeR (288)
    # "concat" (option A): each stream -> its *_proj_dim, concatenated, shared projection -> d_model
    # "gated_sum" (option B): each stream -> d_model, learnable gate per stream, summed
    "fusion": "concat",
    # Transformer encoder of multistream_transformer (ignored by BiLSTM models):
    #   pos_encoding: "sinusoidal" (absolute) | "rope" (rotary, relative) | "alibi" (relative,
    #                 favours nearby frames) | "none"
    #   transformer_norm: "post" (original) | "pre" (LayerNorm before each sublayer; more stable)
    #   sinusoidal + post = the original PyTorch encoder (older runs rebuild unchanged)
    "pos_encoding": "sinusoidal",
    "transformer_norm": "post",
    "rope_base": 10000.0,
    "nhead": 8,
    "body_hands_proj_dim": 256,      # concat only (gated_sum always uses d_model)
    "face_proj_dim": 256,
    # (hamer_proj_dim also sets the HaMeR branch size in concat mode; model default 256 there)
    # d_model above = the size going into the encoder (BiLSTM hidden / Transformer width).
    "use_dinov2_features": False,
    "dinov2_dir": "processed_data/dinov2_features",
    # --- Interpretable cue stream (multistream models) ---
    # Per-frame cues from data_processing/extract_cue_features.py (eye openness, brows,
    # mouth, head nod/height, wrist height/speed, holds, finger flexion ...). Not a
    # skeleton -> no ST-GCN, only a projection branch. use_cue_features is set by the
    # queue from use_stream_cues.
    "use_stream_cues": False,
    "cue_dir": "processed_data/cue_features",
    # "selected" (36, cues with boundary signal) | "all" (48) | "face" (12, facial only) |
    # "nonmanual" (19, face + head) | "manual" (17, hands / handshape / HaMeR / body)
    # or an explicit list of cue names (src/cue_sets.py)
    "cue_set": "selected",
    "cue_proj_dim": 128,              # concat only (gated_sum uses d_model)
    # 1 = per-frame projection only; 9 = + depthwise temporal conv over each cue (same
    # temporal kernel as the ST-GCN), residual, initialised as identity
    "cue_temporal_kernel": 1,
    # --- Front-end options (src/frontend.py; multistream models). Defaults = original. ---
    # Handedness: mirror every video whose dominant hand is MediaPipe "left", so the
    # dominant hand is always on the same side (all splits, no labels involved) ...
    "hand_canonicalize": False,
    # ... or instead randomly mirror training windows with this probability.
    "mirror_augment_p": 0.0,
    # ST-GCN output -> branch: "flatten" (V x C -> Linear, original) | "part_mean" |
    # "part_attn" (body-part tokens: head, arms, hands; face: eyes, brows, mouth)
    "vertex_pooling": "flatten",
    # HaMeR branch: "matrix" (raw 288) | "rot6d" (192); + detected-hand mask; + per-joint
    # rotation speed (lag in frames; HaMeR was extracted at the full frame rate);
    # temporal kernel > 1 = residual temporal conv after the HaMeR projection
    "hamer_rep": "matrix",
    "hamer_mask": False,
    "hamer_motion": False,
    "hamer_motion_lag": 1,
    "hamer_temporal_kernel": 1,
    # Training only: zero each stream of each sample with this probability
    "stream_dropout": 0.0,
    # Change-point layer after fusion: future-minus-past feature differences at these
    # scales (frames, 50 fps) before the encoder. None = off.
    "change_point_scales": None,
    # Per-job override of the selection metric asked at train_phrase.py start:
    # None (use the prompt) | "combined" | "frame_f1" | "bnd5" (boundary F1 within +-5 frames)
    "force_selection_metric": None,
    "d_model": 256,
    "n_layers": 4,     
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
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
    
    # --- FACE ONLY (2D): only the face's x / y channels are kept ---
    if config.get("face_only", False) and int(config.get("face_dims", 3)) == 2:
        return len([f for f in base_features if f in ("x-cord", "y-cord")])

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
# Face vertex count now depends on the chosen face_subset (src/face_subsets.py is the
# single source of truth, shared with dataset.py and graph.py).
from src.face_subsets import face_subset_size
BASE_BODY_HAND_VERTICES = 65

def calculate_num_vertices(config):
    base_features = config.get("base_features", [])

    if "pure_hamer" in base_features:
        # pure_hamer mode is a flat per-frame feature vector (not a per-vertex point
        # cloud), so num_vertices isn't meaningful the same way here -- leave whatever
        # the config already specifies untouched rather than guessing.
        return config.get("num_vertices", BASE_BODY_HAND_VERTICES)

    if config.get("face_only", False):
        return face_subset_size(config.get("face_subset", "full"))

    total = BASE_BODY_HAND_VERTICES
    if config.get("use_face_keypoints", False):
        total += face_subset_size(config.get("face_subset", "full"))
    return total
#endregion

def with_seeds(exp, seeds=(42, 123, 2024)):
    """
    Expands one experiment dict into len(seeds) copies, identical except for a
    "seed" field and a description suffix noting which seed it is -- for
    running the same config multiple times with different (recorded,
    reproducible) random seeds, matching the 2023 paper's practice of
    repeating each final config 3x and reporting mean +/- std.

    You normally don't need to call this yourself -- define EXPERIMENTS_TO_RUN
    as plain experiment dicts, and select_seed_experiments()/
    expand_experiments_with_seeds() below apply this automatically based on
    what you choose interactively when the script runs.
    """
    variants = []
    for seed in seeds:
        variant = dict(exp)
        variant["seed"] = seed
        variant["description"] = f"{exp.get('description', '')} [seed={seed}]".strip()
        variants.append(variant)
    return variants


def select_seed_experiments(experiments):
    """
    Dequeue_train.py-style selection: prints EXPERIMENTS_TO_RUN as a numbered
    list, then asks which ones should run with 3 seeds instead of 1.

      - comma-separated indices (e.g. "0, 2")             -> only those get 3 seeds
      - "all"                                              -> every experiment gets 3 seeds
      - empty input                                        -> none do; everything runs with 1 seed

    Returns a set of indices (into `experiments`) selected for 3 seeds.
    """
    print(f"\n{'='*75}")
    print("📋 EXPERIMENTS DEFINED IN EXPERIMENTS_TO_RUN")
    print(f"{'='*75}")
    for i, exp in enumerate(experiments):
        basename = exp.get("basename", "unknown")
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
        print("⚠️ Invalid input. Please enter numbers separated by commas, 'all', or leave blank. "
              "Defaulting to 1 seed for every experiment.")
        return set()

    valid_ids = set(i for i in ids if 0 <= i < len(experiments))
    invalid_ids = set(ids) - valid_ids
    if invalid_ids:
        print(f"⚠️ Ignoring out-of-range ID(s): {sorted(invalid_ids)}")

    return valid_ids


def expand_experiments_with_seeds(experiments, seed_indices, seeds=(42, 123, 2024)):
    """
    Builds the final experiment list in original order: experiments at
    seed_indices get expanded into len(seeds) seeded copies (via with_seeds);
    everything else passes through unchanged (1 seed -- whatever "seed" it
    already has, or train.py's config.get("seed", 42) default).
    """
    final = []
    for i, exp in enumerate(experiments):
        if i in seed_indices:
            final.extend(with_seeds(exp, seeds=seeds))
        else:
            final.append(exp)
    return final


# ------------------------------------------------------------------------------
# PURE HAMER: HaMeR (288 = 2 hands x (15x3x3 + 3x3)) is the ONLY main input.
#   - base_features=["pure_hamer"] -> dataset.py puts HaMeR in the main tensor (288, T, 1)
#   - use_hamer_features=False     -> HaMeR is NOT also passed as a side branch
#   - num_vertices=1               -> calculate_num_vertices() keeps this as-is
#   - hamer_dim=288                -> calculate_in_channels() returns 288
#   - only graph-free models: bilstm_baseline / transformer_baseline
# Add "use_dinov2_features": True to fuse DINOv2 as a separate side branch.
# ------------------------------------------------------------------------------
_PURE_HAMER = {
    "base_features": ["pure_hamer"],
    "kinematic_features": [],
    "use_hamer_features": False,
    "use_face_keypoints": False,
    "use_dinov2_features": False,
    "num_vertices": 1,
    "hamer_dim": 288,
    "overlap": 0,
    "loss_function": "weighted_ce",
    "tolerance_window": 1,
}

# ------------------------------------------------------------------------------
# Position-encoding study for the multi-stream BiLSTM.
# RoPE / ALiBi only exist inside self-attention, so the BiLSTM gets ONE pre-norm
# self-attention layer on top (multistream_bilstm_attn) and the position encoding is
# varied there: none / sinusoidal / rope / alibi. The plain multistream_bilstm (no
# attention) is the reference that shows what the attention layer itself adds.
# Body + HaMeR, v2 phrase labels, tolerance 1 -- same setup as the Transformer study.
# ------------------------------------------------------------------------------
_MS_BODY_HAMER = {
    "use_stream_body_hands": True,
    "use_stream_face": False,
    "use_stream_hamer": True,
    "fusion": "concat",
    "d_model": 256,
    "n_layers": 4,
    "nhead": 8,
    "overlap": 0,
    "loss_function": "weighted_ce",
    "class_weights": list(DEFAULT_CLASS_WEIGHTS),
    "tolerance_window": 1,
    "kinematic_features": [],
    "phrase_labels_dir": "processed_data/BIO_tags_phrase_v2",
}

# ------------------------------------------------------------------------------
# CUE-STREAM STUDY: do the interpretable cues from boundary_cue_analysis.py help?
# Run data_processing/extract_cue_features.py once before training.
# Same setup as the stream study (multistream_bilstm, window 512, v2 labels, tolerance 1,
# class weights DEFAULT_CLASS_WEIGHTS); the references are Body+HaMeR and
# Body+HaMeR+Face(compact) -- re-queue them (REFERENCES below) only if your earlier runs
# used different settings.
# ------------------------------------------------------------------------------
_MS_CUE_STUDY = {**_MS_BODY_HAMER, "basename": "multistream_bilstm", "window_size": 512}
_CUES_ONLY = {**_MS_CUE_STUDY, "use_stream_body_hands": False, "use_stream_hamer": False,
              "use_stream_face": False, "use_stream_cues": True}

CUE_STUDY = [  # already run (multistream_bilstm 157-177)
    # Q1: how far do ~36 interpretable numbers per frame get on their own (vs Body .676 test)?
    {**_CUES_ONLY, "cue_set": "selected",
     "description": "Cues: cues only (selected 36), projection only"},
    {**_CUES_ONLY, "cue_set": "selected", "cue_temporal_kernel": 9,
     "description": "Cues: cues only (selected 36), + temporal conv k=9"},
    {**_CUES_ONLY, "cue_set": "all",
     "description": "Cues: cues only (all 48), projection only"},
    # Q2: can explicit facial cues replace the raw face ST-GCN stream?
    {**_MS_CUE_STUDY, "use_stream_cues": True, "cue_set": "face",
     "description": "Cues: Body + HaMeR + face cues (12) instead of the face stream"},
    {**_MS_CUE_STUDY, "use_stream_cues": True, "cue_set": "nonmanual",
     "description": "Cues: Body + HaMeR + non-manual cues (face + head, 19)"},
    # Q3: do the cues add anything on top of the best raw model?
    {**_MS_CUE_STUDY, "use_stream_cues": True, "cue_set": "selected",
     "description": "Cues: Body + HaMeR + selected cues (36)"},
    {**_MS_CUE_STUDY, "use_stream_face": True, "face_subset": "compact",
     "use_stream_cues": True, "cue_set": "selected",
     "description": "Cues: Body + HaMeR + Face(compact) + selected cues (4 streams)"},
]

# ------------------------------------------------------------------------------
# FRONT-END STUDY: changes between the raw inputs and the encoder (src/frontend.py).
# Everything is on top of the best model, Body + HaMeR + Face(compact), same setup as
# before (multistream_bilstm, window 512, v2 labels, tolerance 1, weights [1.0, 1.1, 8.5]).
# Each idea is one switch, so they can be compared one by one and then combined.
# ------------------------------------------------------------------------------
_MS_BEST = {**_MS_CUE_STUDY, "use_stream_face": True, "face_subset": "compact"}
_HAMER_FRONTEND = {"hamer_rep": "rot6d", "hamer_mask": True, "hamer_motion": True,
                   "hamer_motion_lag": 1, "hamer_temporal_kernel": 9}
_MS_HAMER_ONLY = {**_MS_CUE_STUDY, "use_stream_body_hands": False, "use_stream_face": False,
                  "use_stream_hamer": True}

FRONTEND_STUDY = [
    # 1. Handedness: does it cost anything that left-dominant signers use the other hand block?
    {**_MS_BEST, "hand_canonicalize": True,
     "description": "FE: B+H+Fc, dominant-hand canonicalization"},
    {**_MS_BEST, "mirror_augment_p": 0.5,
     "description": "FE: B+H+Fc, mirror augmentation p=0.5"},
    # 2. HaMeR branch: 6D rotations + detected mask + rotation speed + temporal conv k=9
    {**_MS_BEST, **_HAMER_FRONTEND,
     "description": "FE: B+H+Fc, HaMeR front-end (rot6d + mask + motion + temporal k9)"},
    {**_MS_HAMER_ONLY, **_HAMER_FRONTEND,
     "description": "FE: HaMeR only, HaMeR front-end (rot6d + mask + motion + temporal k9)"},
    # 3. Stream dropout
    {**_MS_BEST, "stream_dropout": 0.2,
     "description": "FE: B+H+Fc, stream dropout 0.2"},
    # 4. Body-part pooling instead of flattening all vertices
    {**_MS_BEST, "vertex_pooling": "part_mean",
     "description": "FE: B+H+Fc, part pooling (mean)"},
    {**_MS_BEST, "vertex_pooling": "part_attn",
     "description": "FE: B+H+Fc, part pooling (attention)"},
    # All four together
    {**_MS_BEST, **_HAMER_FRONTEND, "hand_canonicalize": True, "stream_dropout": 0.2,
     "vertex_pooling": "part_attn",
     "description": "FE: B+H+Fc, canonicalization + HaMeR front-end + stream dropout + part attn"},
]

# References (already trained: B+H+Fc = multistream_bilstm 124-126, HaMeR only = 133-135).
# Re-queue only if their settings differ; otherwise just re-evaluate them on the current val.
FRONTEND_REFERENCES = [
    {**_MS_BEST, "description": "FE ref: Body + HaMeR + Face(compact)"},
    {**_MS_HAMER_ONLY, "description": "FE ref: HaMeR only"},
]
# EXPERIMENTS_TO_RUN += FRONTEND_REFERENCES   # uncomment if the reference runs need redoing

# ------------------------------------------------------------------------------
# FULL-FACE STUDY: the 2023 paper's face subset (MediaPipe FACEMESH_CONTOURS, 128 points)
# and what the full-mesh files make possible. Needs, once:
#   python data_processing/extract_face_keypoints_full.py --blendshapes --workers 4
#   python data_processing/normalize_face_keypoints_v2.py --layout full
# Same setup as before: multistream_bilstm, window 512, v2 labels, tolerance 1,
# weights [1.0, 1.1, 8.5]. References: B+H+Fc 124-126 (.6802 test), B+H 127-129 (.6764),
# Face only 136-138 (.5234, B F1 0).
# ------------------------------------------------------------------------------
_FULL_FACE = {"face_layout": "full478", "face_dir": "processed_data/face_keypoints_full_normalized_v2"}
_MS_BH_FULLFACE = {**_MS_CUE_STUDY, **_FULL_FACE, "use_stream_face": True}
_BLENDSHAPES = {"use_stream_cues": True, "cue_dir": "processed_data/face_keypoints_full/_blendshapes",
                "cue_set": "file"}

FACE_FULL_STUDY = [
    # 0. Control: our compact subset cut from the NEW files. Should match 124-126; if not, the
    #    re-extraction itself (no forward-fill, different missing frames) changed something.
    {**_MS_BH_FULLFACE, "face_subset": "compact",
     "description": "FullFace: B+H+F compact (from full-mesh files) -- extraction control"},
    # 1. The 2023 paper's face subset
    {**_MS_BH_FULLFACE, "face_subset": "contours2023",
     "description": "FullFace: B+H+F contours2023 (2023 paper, 128)"},
    # 2./3. Which part of it matters: everything but the oval (= our 83 + outer upper lip) vs
    #       the oval alone (head shape / pose seen through the face outline)
    {**_MS_BH_FULLFACE, "face_subset": "contours2023_no_oval",
     "description": "FullFace: B+H+F contours2023 without face oval (92)"},
    {**_MS_BH_FULLFACE, "face_subset": "face_oval",
     "description": "FullFace: B+H+F face oval only (36 + 2 anchors)"},
    # 4. Does the 2023 subset carry boundary information on its own? (ours: B F1 = 0)
    {**_MS_BH_FULLFACE, "use_stream_body_hands": False, "use_stream_hamer": False,
     "face_subset": "contours2023",
     "description": "FullFace: Face only, contours2023"},
    # 5. Eye gaze: compact + both irises (gaze shifts are a known prosodic marker)
    {**_MS_BH_FULLFACE, "face_subset": "compact_iris",
     "description": "FullFace: B+H+F compact + irises (41)"},
    # 6./7. MediaPipe's 52 blendshape scores (eye blink, brow raise, jaw open ...): learned
    #       expression features, unlike the hand-made face cues that added nothing
    {**_MS_CUE_STUDY, **_BLENDSHAPES,
     "description": "FullFace: B+H + blendshapes (52) instead of the face stream"},
    {**_MS_BH_FULLFACE, **_BLENDSHAPES, "face_subset": "compact",
     "description": "FullFace: B+H+F compact + blendshapes (52)"},
]

# ------------------------------------------------------------------------------
# CHANGE-POINT STUDY (Tier 2): explicit "what comes next vs what came before" features
# at several time scales, between fusion and the BiLSTM (src/frontend.py ChangePointLayer).
# Target: the ~40% of phrase starts that are missed (mostly after SHORT pauses), so judge
# it by boundary F1 +-5 (bnd_f1_5) and short-pause recall at tuned thresholds, not Frame F1.
# Base = the best model, Body + HaMeR + Face(compact), 83-point face files.
# ------------------------------------------------------------------------------
CHANGE_POINT_STUDY = [
    {**_MS_BEST, "change_point_scales": [5, 10, 25],
     "description": "CP: B+H+Fc, change-point scales 5/10/25 (0.1-0.5 s)"},
    {**_MS_BEST, "change_point_scales": [10, 25, 50],
     "description": "CP: B+H+Fc, change-point scales 10/25/50 (0.2-1 s)"},
    # Selecting the saved epoch by boundary F1 instead of the combined score: separates the
    # effect of the selection criterion from the effect of the layer.
    {**_MS_BEST, "force_selection_metric": "bnd5",
     "description": "CP ref: B+H+Fc, no change-point, best epoch by boundary F1@5"},
    {**_MS_BEST, "change_point_scales": [5, 10, 25], "force_selection_metric": "bnd5",
     "description": "CP: B+H+Fc, change-point 5/10/25, best epoch by boundary F1@5"},
]

# ------------------------------------------------------------------------------
# 2023 BASELINE STUDY: Moryossef et al. (2023) E1s / E2s re-implemented on OUR split and
# v2 labels (model "bilstm_2023", src/models.py), each trained with
#   (a) THEIR recipe -- taken from their released code (sign-language-processing/
#       segmentation, jobs/job_batch.sh + src/args.py + src/train.py + src/model.py):
#       25 fps, whole videos in batches of 8 (zero-padded, padding masked out of the loss),
#       Adam lr 1e-3, no LR scheduler, no gradient clipping, up to 100 epochs, early
#       stopping patience 20 on validation Frame F1, class weights = inverse class ratio
#   (b) YOUR recipe -- the one used for all multistream runs: 50 fps, 512-frame windows,
#       batch 16, AdamW lr 1e-4 + cosine schedule, grad clip 1.0, patience 10 (min 15
#       epochs), weights DEFAULT_CLASS_WEIGHTS [1.0, 1.1, 8.5], best epoch by Frame F1
# 2 x 2: {E1s body+hands, E2s + 2023 face contours} x {their recipe, your recipe}.
# Compare (b) with your multistream runs (same recipe, different model) and (a) with (b)
# (same model, different recipe).
# Known differences to the paper, kept on purpose:
#   - phrase head only (they train a sign head jointly on the same encoder)
#   - our pose: 23 upper-body + 2 x 21 hand points (they drop the legs too), our labels v2
#   - E2s face: our face-local 2D contour points (theirs: 3D, in the body frame)
# Needs the full-mesh face files for E2s (see FULL-FACE STUDY).
# ------------------------------------------------------------------------------
_BL2023_BASE = {"basename": "bilstm_2023", "d_model": 256, "n_layers": 4,
                "base_features": ["x-cord", "y-cord", "z-cord"], "kinematic_features": [],
                "use_face_keypoints": False, "use_hamer_features": False,
                "loss_function": "weighted_ce", "tolerance_window": 1,
                "phrase_labels_dir": "processed_data/BIO_tags_phrase_v2",
                "force_selection_metric": "frame_f1"}
_RECIPE_2023 = {"use_full_length": True, "full_length_batch_size": 8, "batch_size": 8,
                "window_size": 100000, "overlap": 0, "temporal_downsample_factor": 2,
                "optimizer": "Adam", "learning_rate": 0.001, "scheduler": "none",
                "grad_clip_norm": None, "epochs": 100, "patience": 20, "min_epochs": 1,
                "class_weights": "inverse_ratio",
                # batch of 8 videos run one video at a time with gradient accumulation (same
                # update, ~1/8 of the memory, and no padding inside the BiLSTM)
                "micro_batch_size": 1}
_RECIPE_MINE = {"use_full_length": False, "batch_size": 16, "window_size": 512, "overlap": 0,
                "temporal_downsample_factor": 1, "optimizer": "AdamW", "learning_rate": 0.0001,
                "scheduler": "CosineAnnealingLR", "grad_clip_norm": 1.0, "epochs": 100,
                "patience": 10, "min_epochs": 15, "class_weights": list(DEFAULT_CLASS_WEIGHTS)}
_E2S_FACE = {"use_face_keypoints": True, "face_layout": "full478",
             "face_dir": "processed_data/face_keypoints_full_normalized_v2",
             "face_subset": "contours2023", "face_dims": 2}

BASELINE_2023_STUDY = [
    {**_BL2023_BASE, **_RECIPE_2023,
     "description": "BL2023: E1s (body+hands), 2023 recipe"},
    {**_BL2023_BASE, **_RECIPE_2023, **_E2S_FACE,
     "description": "BL2023: E2s (body+hands+face contours2023), 2023 recipe"},
    {**_BL2023_BASE, **_RECIPE_MINE,
     "description": "BL2023: E1s (body+hands), my recipe"},
    {**_BL2023_BASE, **_RECIPE_MINE, **_E2S_FACE,
     "description": "BL2023: E2s (body+hands+face contours2023), my recipe"},
]

# Optional 5th cell -- YOUR model with THEIR recipe (completes the model x recipe table).
# Off by default: whole-video batches through the ST-GCN streams need a lot of GPU memory
# (8 videos x ~11k frames x 96 vertices); try one run first if you want it.
OPTIONAL_MY_MODEL_2023_RECIPE = [
    {**_MS_BEST, **_RECIPE_2023, "force_selection_metric": "frame_f1",
     "description": "BL2023: my B+H+Fc multistream, 2023 recipe"},
]

# ------------------------------------------------------------------------------
# 2023-ARCHITECTURE STREAM COMBINATIONS: the same 7 input combinations as the multistream
# stream study, but through the 2023 model (early fusion: everything flattened and
# concatenated per frame -> ONE plain Linear -> BiLSTM 4 x 128/dir). Your recipe, weights
# DEFAULT_CLASS_WEIGHTS, v2 labels, best epoch by Frame F1 -- identical to the multistream runs,
# so each cell compares directly with the multistream cell of the same inputs.
#   B   = body + hands (65)  -> already trained: "BL2023: E1s, my recipe" (bilstm_2023 244-246)
#   F   = face, COMBO_FACE subset (face-local 2D)
#   H   = HaMeR (288), concatenated to the flattened keypoints
# Set COMBO_FACE to the subset your multistream combination study used.
# ------------------------------------------------------------------------------
COMBO_FACE = {"use_face_keypoints": True, "face_subset": "compact", "face_dims": 2,
              "face_layout": "saved83", "face_dir": "processed_data/face_keypoints_normalized_v2"}
_C = {**_BL2023_BASE, **_RECIPE_MINE}
_FS = COMBO_FACE["face_subset"]
BL2023_COMBO_STUDY = [
    {**_C, "use_pose_input": False, "use_hamer_features": True,
     "description": "BL2023 combo: H (HaMeR only)"},
    {**_C, **COMBO_FACE, "face_only": True,
     "description": f"BL2023 combo: F (face {_FS} only)"},
    {**_C, "use_hamer_features": True,
     "description": "BL2023 combo: B+H"},
    {**_C, **COMBO_FACE,
     "description": f"BL2023 combo: B+F ({_FS})"},
    {**_C, **COMBO_FACE, "face_only": True, "use_hamer_features": True,
     "description": f"BL2023 combo: F+H ({_FS})"},
    {**_C, **COMBO_FACE, "use_hamer_features": True,
     "description": f"BL2023 combo: B+F+H ({_FS})"},
    # B alone = "BL2023: E1s (body+hands), my recipe" -- uncomment to re-train it here too:
    # {**_C, "description": "BL2023 combo: B"},
]

# ------------------------------------------------------------------------------
# STREAM-ENCODER STUDY: replace the 2-block ST-GCN of the body_hands and face streams of
# the main model (B+H+Fc, multistream_bilstm, same recipe) by a modern skeleton encoder of
# the SAME width (64) and depth (2) -- src/skeleton_encoders.py.
#   blockgcn     BlockGCN (CVPR 2024): per-group learned adjacency + hop-distance encoding
#                + per-frame persistent-homology encoding, block-diagonal projection
#   skateformer  SkateFormer (ECCV 2024): G-Conv + T-Conv + partition attention over
#                {same body part, same position across parts} x {32 consecutive frames,
#                every 32nd frame}
# Reference = your existing B+H+Fc runs (ST-GCN), e.g. multistream_bilstm 124-126.
# ------------------------------------------------------------------------------
ENCODER_STUDY = [
    # {**_MS_BEST, "stream_encoder": "blockgcn",
    #  "description": "ENC: B+H+Fc, BlockGCN stream encoders"},
    {**_MS_BEST, "stream_encoder": "skateformer",
     "description": "ENC: B+H+Fc, SkateFormer stream encoders"},
    # Optional controls:
    # {**_MS_BEST, "stream_encoder": "blockgcn", "blockgcn_ph": False,
    #  "description": "ENC: B+H+Fc, BlockGCN without persistent homology"},
    # {**_MS_BEST, "stream_encoder": "linear",
    #  "description": "ENC: B+H+Fc, per-vertex Linear (no graph, no temporal conv)"},
]

# What runs now (earlier studies above are already trained):
EXPERIMENTS_TO_RUN = ENCODER_STUDY
# EXPERIMENTS_TO_RUN += OPTIONAL_MY_MODEL_2023_RECIPE

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

    print("Current tracked prefixes by model in train_queue_phrase.json:")
    if not prefixes_data:
        print("  (None)")
    else:
        for m_name, p_list in prefixes_data.items():
            print(f"  - {m_name}: {p_list}")
    print()

    # --- SEED SELECTION PHASE ---
    # Choose which experiments run with 3 seeds vs 1, THEN expand the list --
    # everything below this point (prefix assignment, channel/vertex calculation,
    # queueing) operates on the already-expanded list, so a 3-seed selection
    # correctly becomes 3 separate, sequentially-prefixed queue entries.
    seed_indices = select_seed_experiments(experiments_to_run)
    experiments_to_run = expand_experiments_with_seeds(experiments_to_run, seed_indices)
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
        
        # 1b. Multi-stream models: the data that gets loaded follows the stream switches
        if m_name in ("multistream_bilstm", "multistream_transformer", "multistream_bilstm_attn"):
            full_config["use_face_keypoints"] = bool(full_config.get("use_stream_face", False))
            full_config["use_hamer_features"] = bool(full_config.get("use_stream_hamer", False))
            full_config["use_cue_features"] = bool(full_config.get("use_stream_cues", False))
            if not any(full_config.get(k, d) for k, d in (("use_stream_body_hands", True),
                                                          ("use_stream_face", False),
                                                          ("use_stream_hamer", False),
                                                          ("use_stream_cues", False))):
                raise ValueError(f"{full_config.get('description', m_name)}: all streams are off.")

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