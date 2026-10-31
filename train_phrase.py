import time
import json
import csv
import os
import socket
import random
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

# --- Imports for Confusion Matrix ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import our custom modules
from src.dataset import SignSegmentationDataset, extra_dataset_kwargs
from src.model_factory import MODEL_REGISTRY, build_model_kwargs
# Full-video validation: same windows as training, probabilities stitched per video,
# metrics against HARD gold labels (never the tolerance-smoothed ones).
from src.evaluation import evaluate_model_on_split
from src.boundary_losses import boundary_aware_loss
from src.aux_labels import IGNORE as AUX_IGNORE, inv_sqrt_weights, mapped_class_counts
from src.loss import CombinedBoundaryLoss, FocalLoss, StandardCrossEntropyLoss, WeightedCrossEntropyLoss, UnifiedCTCLoss, WeightedCE_TMSE_Loss, WeightedNLLLoss
# (Removed decoder import since we no longer use it in training/validation)

"""
train_phrase.py -- trains against PHRASE-level labels (processed_data/
BIO_tags_phrase/, built by create_phrase_bio_tags.py) instead of the
existing SIGN/gloss-level labels (processed_data/BIO_tags/). Otherwise
IDENTICAL to train.py: same restart-on-NaN loop, same model registry, same
seeding, same loss functions -- only the label source, queue file, and model
save directory differ, kept fully separate so nothing here can collide with
or overwrite your existing sign-level experiments/checkpoints.

Run create_phrase_bio_tags.py first. Queue jobs via queue_train_phrase.py
(writes to train_queue_phrase.json, tracked separately from your existing
train_queue.json).
"""
QUEUE_FILE = "train_queue_phrase.json"

# MODEL_REGISTRY now lives in src/model_factory.py (shared with evaluate_phrase.py).

def get_next_job():
    if not os.path.exists(QUEUE_FILE):
        return None
        
    with open(QUEUE_FILE, "r") as f:
        try:
            queue = json.load(f)
        except json.JSONDecodeError:
            return None
            
    if not queue or len(queue) <= 1:
        return None
        
    next_job = queue.pop(1)
    
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    return next_job


def pad_collate(batch):
    """
    Batches FULL videos of different lengths (use_full_length + full_length_batch_size > 1,
    as in the 2023 recipe: batch of 8 zero-padded videos). Every tensor is padded along
    its time axis (dim 1); padded LABEL frames are all-zero, which the training loop turns
    into ignore_index (-100) so they never enter the loss (the 2023 code masks them too).
    """
    out = []
    for field in zip(*batch):
        first = field[0]
        if torch.is_tensor(first) and first.dim() >= 2:
            T = max(t.shape[1] for t in field)
            padded = []
            for t in field:
                if t.shape[1] < T:
                    pad_shape = list(t.shape)
                    pad_shape[1] = T - t.shape[1]
                    t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=1)
                padded.append(t)
            out.append(torch.stack(padded))
        elif isinstance(first, str):
            out.append(list(field))
        else:
            out.append(torch.as_tensor(field))
    return out


def mask_padding(hard_labels, soft_labels):
    """Frames whose soft label is all-zero are batch padding -> ignore_index -100."""
    return hard_labels.masked_fill(soft_labels.sum(dim=1) == 0, -100)


def micro_batches(labels, micro_size):
    """
    Splits a padded full-video batch into chunks of `micro_size` videos, each trimmed to the
    longest REAL length inside the chunk (padding = all-zero soft labels). Returns a list of
    (sample_indices, T_real). With micro_size = 1 every video runs at its own length, so the
    backward LSTM direction never sees padding.
    """
    real = labels.sum(dim=1) != 0                                   # (B, T)
    lengths = [int(r.nonzero().max().item()) + 1 if r.any() else 0 for r in real]
    order = list(range(labels.shape[0]))
    return [(order[i:i + micro_size], max(lengths[j] for j in order[i:i + micro_size]) or 1)
            for i in range(0, len(order), micro_size)]


def accumulated_ce(call, features, labels, hamer, dinov2, cues, class_weights, micro_size,
                   device, backward=True):
    """
    Weighted cross-entropy over a padded batch, computed in micro-batches to bound memory,
    with gradients IDENTICAL to one pass over the whole batch: every chunk's summed weighted
    loss is divided by the weight total of the WHOLE batch (= PyTorch's weighted-mean
    reduction), and gradients accumulate. The batch stays on the CPU; only one chunk (cut to
    its real length, cast to float32, NaN-cleaned) is on the GPU at a time.
    Returns the batch loss (float), or None if a chunk produced NaN/Inf (step is skipped).
    """
    hard = mask_padding(torch.argmax(labels, dim=1), labels)          # CPU
    valid = hard != -100
    w = (class_weights.detach().float().cpu() if class_weights is not None else torch.ones(3))
    w_total = float(w[hard[valid]].sum().clamp_min(1e-8))
    w_dev = w.to(device)
    total = 0.0
    for idx, T in micro_batches(labels, micro_size):
        def take(x):
            if x is None:
                return None
            x = x[idx].narrow(2, 0, min(T, x.shape[2]))
            x = x.to(device, non_blocking=True).float()
            return torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
        out = call(take(features), take(hamer), take(dinov2), take(cues))
        logits = out[0] if isinstance(out, tuple) else out          # (b, 3, T)
        tgt = hard[idx][:, :logits.shape[-1]].to(device)
        loss = torch.nn.functional.cross_entropy(logits.float(), tgt, weight=w_dev, ignore_index=-100,
                                                 reduction="sum") / w_total
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        if backward:
            loss.backward()
        total += loss.item()
        del out, logits, loss
    return total


def set_seed(seed):
    """
    Seeds every source of randomness that affects a run: model weight init,
    DataLoader shuffling order, dropout masks. Called ONCE at the very start of
    train_model(), before anything randomness-dependent happens (including the
    first model construction) -- so if a NaN-triggered restart re-initializes
    the model partway through, that re-init still consumes the SAME seeded RNG
    stream deterministically, rather than needing to be reseeded itself.

    Deliberately does NOT set torch.backends.cudnn.deterministic=True. That
    trades real training speed for bit-exact reproducibility, which isn't
    what's needed here (you need 3 DIFFERENT, recorded seeds for variance
    estimation, not "rerun must produce identical numbers") -- and it would
    make every future seeded run slower than your past unseeded ones, muddying
    the epoch-timing comparisons you've been tracking carefully. Mamba's custom
    CUDA kernel (selective_scan_cuda) isn't written with determinism guarantees
    in mind either way, so that guarantee would be partial at best.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SELECTION_METRICS = {
    "combined": "Combined score (Frame F1 + IoU + %-score)",
    "frame_f1": "Frame F1 only",
    "bnd5": "Boundary F1 within +-5 frames (phrase starts, argmax decoding)",
}


def ask_selection_metric():
    """
    Asked once when train_phrase.py starts; applies to every job in this session.
    Decides which validation score picks the saved best epoch ("New Best") and drives
    early stopping. Empty input keeps the current method (combined score).
    """
    print("\n📏 Which validation score should pick the best epoch and drive early stopping?")
    print("   [Enter] combined  -> Frame F1 + IoU + %-score (current method)")
    print("   f1               -> Frame F1 only")
    print("   bnd              -> Boundary F1 within +-5 frames (phrase starts)")
    while True:
        choice = input("Selection metric: ").strip().lower()
        if choice in ("", "combined", "c"):
            metric = "combined"
        elif choice in ("f1", "frame_f1", "frame f1"):
            metric = "frame_f1"
        elif choice in ("bnd", "bnd5", "b"):
            metric = "bnd5"
        else:
            print("⚠️  Please press Enter (combined) or type 'f1' or 'bnd'.")
            continue
        print(f"✅ Using: {SELECTION_METRICS[metric]}\n")
        return metric


def train_model(config, selection_metric="combined"):
    # A queue entry can force its own selection metric (e.g. "bnd5"), overriding the prompt.
    if config.get("force_selection_metric"):
        selection_metric = config["force_selection_metric"]
        if selection_metric not in SELECTION_METRICS:
            raise ValueError(f"force_selection_metric must be one of {sorted(SELECTION_METRICS)}.")
        print(f"📏 This job selects its best epoch by: {SELECTION_METRICS[selection_metric]} (forced by its config)")
    print(f"\n{'='*60}\n🚀 STARTING QUEUED JOB\n{'='*60}")
    print(json.dumps(config, indent=4))

    SEED = config.get("seed", 42)
    set_seed(SEED)
    
    BATCH_SIZE = config["batch_size"]
    EPOCHS = config["epochs"]
    MIN_EPOCHS = config.get("min_epochs", 15) 
    EARLY_STOPPING = config.get("early_stopping", True)
    PATIENCE = config.get("patience", 10)
    LEARNING_RATE = config["learning_rate"]
    WINDOW_SIZE = config["window_size"]
    OVERLAP = config["overlap"]
    NUM_VERTICES = config["num_vertices"]
    TOLERANCE_WINDOW = config["tolerance_window"]
    DOWNSAMPLE_FACTOR = config.get("temporal_downsample_factor", 1) 
    LOSS_FUNCTION = config["loss_function"]
    CLASS_WEIGHTS = config["class_weights"]
    USE_FULL_LENGTH = config.get("use_full_length", False)
    BASE_FEATURES = config["base_features"]
    KINEMATIC_FEATURES = config["kinematic_features"]
    IN_CHANNELS = config["in_channels"]
    USE_FACE_KEYPOINTS = config.get("use_face_keypoints", False)
    FACE_DIR = config.get("face_dir", "processed_data/face_keypoints_normalized")
    # Which saved face points to load: "full" (83), "compact" (31), "eyes_brows" (22), "minimal" (18).
    # Old configs without this key load all 83 points, exactly as before.
    FACE_SUBSET = config.get("face_subset", "full")
    # Which phrase labels to train on: v1 (processed_data/BIO_tags_phrase, the default for
    # older configs) or v2 (processed_data/BIO_tags_phrase_v2, corrected boundaries).
    PHRASE_LABELS_DIR = config.get("phrase_labels_dir", "processed_data/BIO_tags_phrase")
    print(f"🏷️  Phrase labels: {PHRASE_LABELS_DIR}")
    FACE_ONLY = config.get("face_only", False)
    # 2 = 2D face keypoints (x, y). Configs without the key are old 3D-face runs.
    FACE_DIMS = int(config.get("face_dims", 3))  # True: face vertices only (body + hands dropped)
    USE_HAMER_FEATURES = config.get("use_hamer_features", False)
    HAMER_DIR = config.get("hamer_dir", "processed_data/hamer_features")
    USE_DINOV2_FEATURES = config.get("use_dinov2_features", False)
    USE_CUE_FEATURES = config.get("use_cue_features", False)
    DINOV2_DIR = config.get("dinov2_dir", "processed_data/dinov2_features")
    # Boundary studies (src/boundary_losses.py), weighted_ce only, windowed training only:
    #   soft_label_dir     soft training targets from make_soft_labels.py (ambiguity study)
    #   count_loss_weight  > 0: + weight * "one start per boundary" count loss (count study)
    #   count_loss_radius  half-width of the count window in frames (default 5 = +-5 tolerance)
    SOFT_LABEL_DIR = config.get("soft_label_dir")
    COUNT_LOSS_WEIGHT = float(config.get("count_loss_weight", 0.0))
    COUNT_LOSS_RADIUS = int(config.get("count_loss_radius", 5))
    COUNT_LOSS_TAU = float(config.get("count_loss_tau", 0.1))       # sharpness of the soft "is B" indicator
    COUNT_LOSS_MODE = config.get("count_loss_mode", "argmax")      # "argmax" (default) | "mass"
    USE_BOUNDARY_LOSS = bool(SOFT_LABEL_DIR) or COUNT_LOSS_WEIGHT > 0
    if USE_BOUNDARY_LOSS:
        if LOSS_FUNCTION != "weighted_ce":
            raise ValueError("soft_label_dir / count_loss_weight need loss_function 'weighted_ce'.")
        if USE_FULL_LENGTH:
            raise ValueError("soft_label_dir / count_loss_weight are implemented for windowed training only.")
        print(f"🎯 Boundary loss: soft targets={'on (' + SOFT_LABEL_DIR + ')' if SOFT_LABEL_DIR else 'off'}, "
              f"count loss weight={COUNT_LOSS_WEIGHT}, radius={COUNT_LOSS_RADIUS}, "
              f"tau={COUNT_LOSS_TAU}, mode={COUNT_LOSS_MODE}")
    # Idea B -- stream-specific auxiliary supervision (src/aux_labels.py, make_aux_labels.py):
    #   aux_heads            {task: {"stream": "face" | "hamer" | ... | "fused", "classes": C}}
    #   aux_label_dir        processed_data/aux_labels (train split only; validation stays BIO-only)
    #   aux_tasks            order of the tasks in the label tensor (default: aux_heads order)
    #   aux_loss_weights     {task: lambda} (default 0.3 each)
    #   aux_class_weighting  "inv_sqrt" (w_c ~ 1/sqrt(train frames of c), mean 1) | "none"
    #   aux_label_map        {task: {src: dst}} class merges, e.g. {"signcat": {"6": 0}} (arm B6)
    #   aux_temporal_kernel  depthwise temporal conv of each head (default 9)
    AUX_HEADS = config.get("aux_heads") or {}
    USE_AUX = bool(AUX_HEADS)
    AUX_TASKS = list(config.get("aux_tasks") or AUX_HEADS.keys())
    AUX_LAMBDA, AUX_CLASS_W = {}, {}
    if USE_AUX:
        if set(AUX_TASKS) != set(AUX_HEADS):
            raise ValueError(f"aux_tasks {AUX_TASKS} must match the aux_heads keys {list(AUX_HEADS)}.")
        if not config.get("aux_label_dir"):
            raise ValueError("aux_heads needs aux_label_dir (run data_processing/make_aux_labels.py).")
        if USE_FULL_LENGTH:
            raise ValueError("aux_heads is implemented for windowed training only (not use_full_length / micro-batches).")
        if LOSS_FUNCTION in ("bcl", "unified_ctc"):
            raise ValueError(f"aux_heads is not implemented for loss_function '{LOSS_FUNCTION}'.")
        lam = config.get("aux_loss_weights") or {}
        AUX_LAMBDA = {t: float(lam.get(t, 0.3)) for t in AUX_TASKS}
        with open(os.path.join(config["aux_label_dir"], "_meta.json")) as f:
            aux_meta = json.load(f)
        label_map = config.get("aux_label_map") or {}
        for t in AUX_TASKS:
            n_cls = int(AUX_HEADS[t]["classes"])
            counts = mapped_class_counts(aux_meta["train_class_frames"][t], label_map.get(t), n_cls)
            if config.get("aux_class_weighting", "inv_sqrt") == "inv_sqrt":
                AUX_CLASS_W[t] = [round(w, 4) for w in inv_sqrt_weights(counts)]
            else:
                AUX_CLASS_W[t] = [1.0 if c > 0 else 0.0 for c in counts]
        config["aux_class_weights_resolved"] = AUX_CLASS_W
        config["aux_connectives"] = aux_meta.get("connectives")
        print(f"🧭 Auxiliary heads: " + ", ".join(
            f"{t} on '{AUX_HEADS[t]['stream']}' ({AUX_HEADS[t]['classes']} classes, lambda={AUX_LAMBDA[t]}, "
            f"weights={AUX_CLASS_W[t]})" for t in AUX_TASKS))
    D_MODEL = config["d_model"]
    N_LAYERS = config["n_layers"]
    FOCAL_LOSS_GAMMA = config.get("focal_loss_gamma", 2.0)
    OPTIMIZER_NAME = config["optimizer"]
    SCHEDULER_NAME = config["scheduler"]
    MODEL_NAME = config["basename"]
    
    prefix = config.get("prefix", "01")
    run_name = f"{MODEL_NAME}-{prefix}"
    print(f"📁 Assigned Run Name: {run_name}")
    
    model_dir = "saved_models_phrase"
    os.makedirs(model_dir, exist_ok=True)
    
    exp_dir = os.path.join("experiments_phrase", run_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Recorded with the run so you can always tell how its best epoch was chosen.
    config["selection_metric"] = selection_metric
    with open(os.path.join(exp_dir, "hyperparameters.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    train_dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir=PHRASE_LABELS_DIR,
        split_file="dataset_splits.json",
        split="train",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        tolerance_window=TOLERANCE_WINDOW,
        use_full_length=USE_FULL_LENGTH,
        base_features=BASE_FEATURES,
        kinematic_features=KINEMATIC_FEATURES,
        temporal_downsample_factor=DOWNSAMPLE_FACTOR,
        use_face_keypoints=USE_FACE_KEYPOINTS,
        face_dir=FACE_DIR,
        face_subset=FACE_SUBSET,
        face_only=FACE_ONLY,
        face_dims=FACE_DIMS,
        use_hamer_features=USE_HAMER_FEATURES,
        hamer_dir=HAMER_DIR,
        use_dinov2_features=USE_DINOV2_FEATURES,
        dinov2_dir=DINOV2_DIR,
        soft_label_dir=SOFT_LABEL_DIR,          # train split only
        **extra_dataset_kwargs(config),
    )
    
    val_dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir=PHRASE_LABELS_DIR,
        split_file="dataset_splits.json",
        split="val",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        tolerance_window=TOLERANCE_WINDOW,
        use_full_length=USE_FULL_LENGTH,
        base_features=BASE_FEATURES,
        kinematic_features=KINEMATIC_FEATURES,
        temporal_downsample_factor=DOWNSAMPLE_FACTOR,
        use_face_keypoints=USE_FACE_KEYPOINTS,
        face_dir=FACE_DIR,
        face_subset=FACE_SUBSET,
        face_only=FACE_ONLY,
        face_dims=FACE_DIMS,
        use_hamer_features=USE_HAMER_FEATURES,
        hamer_dir=HAMER_DIR,
        use_dinov2_features=USE_DINOV2_FEATURES,
        dinov2_dir=DINOV2_DIR,
        **extra_dataset_kwargs(config),
    )
    
    # Full-video training: batch 1 (as before) unless full_length_batch_size > 1, which pads
    # the videos of a batch to the same length and masks the padding out of the loss.
    FULL_LENGTH_BATCH = int(config.get("full_length_batch_size", 1))
    if USE_FULL_LENGTH and FULL_LENGTH_BATCH > 1:
        if LOSS_FUNCTION not in ("weighted_ce", "standard_ce", "weighted_nll"):
            raise ValueError(f"full_length_batch_size > 1 needs a loss that ignores label -100 "
                             f"(weighted_ce / standard_ce / weighted_nll), not '{LOSS_FUNCTION}'.")
        # A padded batch of 8 whole videos is large, so only ONE batch is prepared ahead
        # (1 worker, prefetch 1) and nothing is pinned -- 4 workers x 2 prefetched batches
        # (+ pinned copies) used to hold ~16 batches in RAM at once.
        FL_WORKERS = int(config.get("full_length_num_workers", 1))
        fl_kwargs = dict(batch_size=FULL_LENGTH_BATCH, num_workers=FL_WORKERS, pin_memory=False,
                         collate_fn=pad_collate)
        if FL_WORKERS > 0:
            fl_kwargs["prefetch_factor"] = 1
        train_loader = DataLoader(train_dataset, shuffle=True, **fl_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **fl_kwargs)
    else:
        loader_batch_size = 1 if USE_FULL_LENGTH else BATCH_SIZE
        train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # class_weights = "inverse_ratio": total / count per class on the training targets,
    # exactly how Moryossef et al. (2023) weight their loss. Resolved once and saved.
    if isinstance(CLASS_WEIGHTS, str):
        if CLASS_WEIGHTS != "inverse_ratio":
            raise ValueError(f"class_weights must be a list or 'inverse_ratio', got '{CLASS_WEIGHTS}'.")
        CLASS_WEIGHTS = [round(w, 4) for w in train_dataset.inverse_class_ratio()]
        print(f"⚖️  Class weights (inverse class ratio, O/I/B): {CLASS_WEIGHTS}")
        config["class_weights_resolved"] = CLASS_WEIGHTS
        with open(os.path.join(exp_dir, "hyperparameters.json"), 'w') as f:
            json.dump(config, f, indent=4)
    GRAD_CLIP = config.get("grad_clip_norm", 1.0)
    # micro_batch_size (full-video batches only): run a batch of full_length_batch_size
    # videos in chunks of this many videos, accumulating gradients -- same update as the
    # whole batch at a fraction of the memory. None = whole batch at once.
    MICRO_BATCH = config.get("micro_batch_size")
    USE_MICRO = bool(USE_FULL_LENGTH and FULL_LENGTH_BATCH > 1 and MICRO_BATCH
                     and int(MICRO_BATCH) < FULL_LENGTH_BATCH)
    if USE_MICRO:
        if LOSS_FUNCTION not in ("weighted_ce", "standard_ce"):
            raise ValueError(f"micro_batch_size needs weighted_ce or standard_ce, not '{LOSS_FUNCTION}'.")
        MICRO_BATCH = int(MICRO_BATCH)
        train_dataset.keep_half = True   # batches stay float16 in RAM; cast per video on the GPU
        val_dataset.keep_half = True
        print(f"🧩 Full-video batches of {FULL_LENGTH_BATCH} run in chunks of {MICRO_BATCH} "
              f"(gradient accumulation, identical update).")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if MODEL_NAME not in MODEL_REGISTRY:
        print(f"❌ Error: Model '{MODEL_NAME}' not found in registry. Skipping.")
        return

    # Shared with evaluate_phrase.py, so evaluation rebuilds EXACTLY this model.
    # hamer_dim / dinov2_dim come from the data (config values still override).
    model_class, model_kwargs = build_model_kwargs(
        config,
        detected_hamer_dim=train_dataset.detected_hamer_dim,
        detected_dinov2_dim=train_dataset.detected_dinov2_dim,
        detected_cue_dim=train_dataset.detected_cue_dim,
    )
    MAX_REDOS = 5
    redo_count = 0
    total_nan_this_run = 0
    
    while redo_count <= MAX_REDOS:
        if redo_count > 0:
            print(f"\n🔄 RESTARTING TRAINING (Attempt {redo_count + 1}/{MAX_REDOS + 1}) DUE TO NAN EXPLOSION...")
            
        model = model_class(**model_kwargs).to(device)
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
        aux_w = {t: torch.tensor(AUX_CLASS_W[t], dtype=torch.float, device=device) for t in AUX_TASKS} if USE_AUX else {}

        # Centralizes the "pass hamer/dinov2 or don't" branching in one place instead
        # of repeating it at every model(...) call site below.
        def call_model(feats, ham, din, cue=None):
            kwargs = {}
            if USE_HAMER_FEATURES:
                kwargs["hamer"] = ham
            if USE_DINOV2_FEATURES:
                kwargs["dinov2"] = din
            if USE_CUE_FEATURES:
                kwargs["cues"] = cue
            return model(feats, **kwargs)
        
        if LOSS_FUNCTION == "bcl":
            criterion = CombinedBoundaryLoss(focal_gamma=FOCAL_LOSS_GAMMA, contrastive_weight=config.get("contrastive_weight", 0.15))
        elif LOSS_FUNCTION == "unified_ctc":
            criterion = UnifiedCTCLoss(blank_idx=0, ctc_weight=config.get("ctc_weight", 0.5))
        elif LOSS_FUNCTION == "standard_ce":
            criterion = StandardCrossEntropyLoss()
        elif LOSS_FUNCTION == "weighted_ce":
            criterion = WeightedCrossEntropyLoss(weights=weights)
        elif LOSS_FUNCTION == "focal":
            criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA)
        elif LOSS_FUNCTION == "wce_tmse":
            criterion = WeightedCE_TMSE_Loss(
                weights=weights,
                tmse_weight=config.get("tmse_weight", 0.15),
                threshold=config.get("tmse_threshold", 0.1)
            )
        elif LOSS_FUNCTION == "weighted_nll":
            criterion = WeightedNLLLoss(weights=weights)
        else:
            raise ValueError(f"Unknown loss function '{LOSS_FUNCTION}'")

        if OPTIMIZER_NAME == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
        if SCHEDULER_NAME == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        else:
            scheduler = None
            
        metrics_log_path = os.path.join(exp_dir, "training_metrics.csv")
        with open(metrics_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'frame_f1', 'iou', 'segment_pct',
                             'segment_f1_05', 'combined_score', 'epoch_time', 'epoch_nan_count',
                             'bnd_f1_5']
                            + (['train_main_loss'] + [f'aux_loss_{t}' for t in AUX_TASKS]
                               + [f'aux_acc_{t}' for t in AUX_TASKS] if USE_AUX else []))
        
        best_combined_score = -1.0
        best_epoch = 0
        best_model_state = None
        epochs_without_improvement = 0
        actual_epochs_ran = 0
        total_nan_this_run = 0
        gpu_utilization_samples = []
        # Reset here (not once before the while loop) so total_training_time only ever
        # reflects the LATEST attempt -- a NaN-triggered restart no longer inflates it
        # with time spent on earlier, discarded attempts.
        start_train_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            
        needs_restart = False

        for epoch in range(1, EPOCHS + 1):
            actual_epochs_ran = epoch
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            epoch_nan_count = 0
            valid_batches = 0
            main_loss_sum = 0.0
            aux_loss_sum = {t: 0.0 for t in AUX_TASKS}
            aux_loss_n = {t: 0 for t in AUX_TASKS}
            aux_correct = {t: 0 for t in AUX_TASKS}
            aux_total = {t: 0 for t in AUX_TASKS}
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
            # Add the 3 underscores to absorb the vid, start, and end metadata!
            for batch in loop:
                # Generic unpacking: features, labels, [hamer], [dinov2], vid, start, end --
                # optional streams appear in that fixed order only when their flag is set
                # (matches SignSegmentationDataset.__getitem__'s exact construction), so
                # popping from the front of the "extras" slice in the same order correctly
                # recovers each one regardless of which combination is active.
                features, labels = batch[0], batch[1]
                extras = list(batch[2:-3])
                hamer = extras.pop(0) if USE_HAMER_FEATURES else None
                dinov2 = extras.pop(0) if USE_DINOV2_FEATURES else None
                cues = extras.pop(0) if USE_CUE_FEATURES else None
                aux = extras.pop(0) if USE_AUX else None      # auxiliary targets: the LAST extra
                if USE_MICRO:   # full-video batch: stays on the CPU, one video at a time on the GPU
                    optimizer.zero_grad()
                    acc = accumulated_ce(call_model, features, labels, hamer, dinov2, cues,
                                         weights if LOSS_FUNCTION == "weighted_ce" else None,
                                         MICRO_BATCH, device)
                    if acc is None:
                        optimizer.zero_grad()
                        epoch_nan_count += 1
                        total_nan_this_run += 1
                        loop.set_postfix(loss="NaN", nans=epoch_nan_count)
                        if epoch_nan_count > 50 and redo_count < MAX_REDOS:
                            print(f"\n⚠️ CRITICAL: Epoch {epoch} encountered >50 NaN losses ({epoch_nan_count}). Aborting run.")
                            needs_restart = True
                            break
                        continue
                    if GRAD_CLIP:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP))
                    optimizer.step()
                    train_loss += acc
                    valid_batches += 1
                    loop.set_postfix(loss=acc, nans=epoch_nan_count)
                    continue
                if USE_CUE_FEATURES:
                    cues = torch.nan_to_num(cues.to(device), nan=0.0, posinf=0.0, neginf=0.0)

                if USE_HAMER_FEATURES:
                    hamer = hamer.to(device)
                    hamer = torch.nan_to_num(hamer, nan=0.0, posinf=0.0, neginf=0.0)
                if USE_DINOV2_FEATURES:
                    dinov2 = dinov2.to(device)
                    dinov2 = torch.nan_to_num(dinov2, nan=0.0, posinf=0.0, neginf=0.0)

                features = features.to(device)
                labels = labels.to(device)
                
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                optimizer.zero_grad()

                if LOSS_FUNCTION == "bcl":
                    logits, embeddings = call_model(features, hamer, dinov2, cues)
                    loss, _, _ = criterion(logits, embeddings, labels)
                elif LOSS_FUNCTION == "unified_ctc":
                    logits, _ = call_model(features, hamer, dinov2, cues)
                    hard_labels = mask_padding(torch.argmax(labels, dim=1), labels)
                    loss, _, _ = criterion(logits, hard_labels)
                else:
                    # Depending on the model, it might return (logits, embeddings) or just logits
                    output = call_model(features, hamer, dinov2, cues)
                    logits = output[0] if isinstance(output, tuple) else output
                    if USE_BOUNDARY_LOSS:
                        # validation loss below stays plain weighted CE on hard labels,
                        # so val_loss remains comparable with every earlier run
                        loss = boundary_aware_loss(logits, labels, weights, soft=bool(SOFT_LABEL_DIR),
                                                   count_weight=COUNT_LOSS_WEIGHT, radius=COUNT_LOSS_RADIUS,
                                                   tau=COUNT_LOSS_TAU, mode=COUNT_LOSS_MODE)
                    else:
                        hard_labels = mask_padding(torch.argmax(labels, dim=1), labels)
                        loss = criterion(logits, hard_labels)
                    if USE_AUX:
                        main_loss_value = loss.item()
                        aux = aux.to(device)
                        for j, task in enumerate(AUX_TASKS):
                            aux_lg = model._aux_logits[task].float()
                            tgt = aux[:, j]
                            lt = F.cross_entropy(aux_lg, tgt, weight=aux_w[task], ignore_index=AUX_IGNORE)
                            if torch.isfinite(lt):            # NaN when the window has no labelled frame
                                loss = loss + AUX_LAMBDA[task] * lt
                                aux_loss_sum[task] += lt.item()
                                aux_loss_n[task] += 1
                            with torch.no_grad():
                                m = tgt != AUX_IGNORE
                                aux_correct[task] += int((aux_lg.argmax(1)[m] == tgt[m]).sum())
                                aux_total[task] += int(m.sum())
                
                if torch.isnan(loss) or torch.isinf(loss):
                    epoch_nan_count += 1
                    total_nan_this_run += 1
                    loop.set_postfix(loss="NaN", nans=epoch_nan_count)
                    
                    if epoch_nan_count > 50 and redo_count < MAX_REDOS:
                        print(f"\n⚠️ CRITICAL: Epoch {epoch} encountered >50 NaN losses ({epoch_nan_count}). Aborting run.")
                        needs_restart = True
                        break
                    continue
                
                loss.backward()
                if GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP))
                optimizer.step()
                
                train_loss += loss.item()
                valid_batches += 1
                if USE_AUX:
                    main_loss_sum += main_loss_value
                loop.set_postfix(loss=loss.item(), nans=epoch_nan_count)

                try:
                    if torch.cuda.is_available():
                        gpu_utilization_samples.append(torch.cuda.utilization(device))
                except Exception:
                    pass
            
            if needs_restart:
                break 
                
            avg_train_loss = train_loss / valid_batches if valid_batches > 0 else float('inf')
            
            if scheduler:
                scheduler.step()
                
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
                for batch in val_loop:
                    features, labels = batch[0], batch[1]
                    vids, start_indices, end_indices = batch[-3], batch[-2], batch[-1]
                    extras = list(batch[2:-3])
                    hamer = extras.pop(0) if USE_HAMER_FEATURES else None
                    dinov2 = extras.pop(0) if USE_DINOV2_FEATURES else None
                    cues = extras.pop(0) if USE_CUE_FEATURES else None
                    if USE_MICRO:
                        acc = accumulated_ce(call_model, features, labels, hamer, dinov2, cues,
                                             weights if LOSS_FUNCTION == "weighted_ce" else None,
                                             MICRO_BATCH, device, backward=False)
                        val_loss += acc if acc is not None else 0.0
                        continue
                    if USE_CUE_FEATURES:
                        cues = torch.nan_to_num(cues.to(device), nan=0.0, posinf=0.0, neginf=0.0)

                    if USE_HAMER_FEATURES:
                        hamer = hamer.to(device)
                        hamer = torch.nan_to_num(hamer, nan=0.0, posinf=0.0, neginf=0.0)
                    if USE_DINOV2_FEATURES:
                        dinov2 = dinov2.to(device)
                        dinov2 = torch.nan_to_num(dinov2, nan=0.0, posinf=0.0, neginf=0.0)

                    features = features.to(device)
                    labels = labels.to(device)
                    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                    if LOSS_FUNCTION == "bcl":
                        logits, embeddings = call_model(features, hamer, dinov2, cues)
                        loss, _, _ = criterion(logits, embeddings, labels)
                    elif LOSS_FUNCTION == "unified_ctc":
                        logits, _ = call_model(features, hamer, dinov2, cues)
                        hard_labels = mask_padding(torch.argmax(labels, dim=1), labels)
                        loss, _, _ = criterion(logits, hard_labels)
                    else:
                        output = call_model(features, hamer, dinov2, cues)
                        logits = output[0] if isinstance(output, tuple) else output
                        hard_labels = mask_padding(torch.argmax(labels, dim=1), labels)
                        loss = criterion(logits, hard_labels)
                        
                    val_loss += loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0
                    
                    try:
                        if torch.cuda.is_available():
                            gpu_utilization_samples.append(torch.cuda.utilization(device))
                    except Exception:
                        pass
                    

            avg_val_loss = val_loss / len(val_loader)
            

            # --- FULL-VIDEO VALIDATION METRICS (2023-style) ---
            # Each val video is predicted with the same windows used in training,
            # probabilities are stitched per video, and metrics are computed against the
            # HARD gold labels at the raw frame rate -- independent of tolerance_window.
            # Decoding is plain argmax (thresholds can be tuned afterwards with
            # evaluate_phrase.py --sweep).
            val_metrics = evaluate_model_on_split(model, config, val_dataset, device,
                                                  batch_size=BATCH_SIZE, decoder="argmax")
            epoch_f1 = val_metrics["Frame_F1"]
            epoch_iou = val_metrics["IoU"]
            epoch_pct = val_metrics["Pct"]
            epoch_seg_f1 = val_metrics["Segment_F1_05"]
            epoch_val_true = val_metrics["frame_true"]
            epoch_val_pred = val_metrics["frame_pred"]
            
            epoch_end_time = time.time()
            epoch_duration = round(epoch_end_time - epoch_start_time, 2)
            
            # Frame F1 + IoU + %-score, where %-score = max(0, 1 - |% - 1|) (1.0 = perfect count)
            combined_score = val_metrics["Combined"]
            # The score that decides "New Best" and early stopping (chosen at startup).
            selection_score = {"frame_f1": epoch_f1, "bnd5": val_metrics["BndF1_5"]}.get(selection_metric,
                                                                                        combined_score)
            
            if selection_score > best_combined_score:
                best_combined_score = selection_score
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                is_best = True
                
                try:
                    cm = confusion_matrix(epoch_val_true, epoch_val_pred, labels=[0, 1, 2])
                    cm_normalized = confusion_matrix(epoch_val_true, epoch_val_pred, labels=[0, 1, 2], normalize='true')
                    
                    annot_labels = np.empty_like(cm, dtype=object)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            annot_labels[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
                            
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues', 
                                xticklabels=['Outside (0)', 'Inside (1)', 'Begin (2)'], 
                                yticklabels=['Outside (0)', 'Inside (1)', 'Begin (2)'],
                                vmin=0.0, vmax=1.0)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual (Row %)')
                    plt.title(f'Validation Confusion Matrix (Best Epoch {epoch})\n{run_name}')
                    
                    cm_save_path = os.path.join(exp_dir, "best_confusion_matrix.png")
                    plt.savefig(cm_save_path, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"⚠️ Failed to generate confusion matrix: {e}")
                    
            else:
                epochs_without_improvement += 1
                is_best = False
                
            nan_string = f" | NaNs: {epoch_nan_count}" if epoch_nan_count > 0 else ""
            
            print(f"Epoch [{epoch:02d}/{EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Frame F1: {epoch_f1:.4f} | "
                  f"IoU: {epoch_iou:.4f} | "
                  f"%: {epoch_pct:.4f} | "
                  f"SegF1@0.5: {epoch_seg_f1:.4f} | "
                  f"BndF1@5: {val_metrics['BndF1_5']:.4f} | "
                  f"Time: {epoch_duration}s{nan_string}" + (" 🌟 (New Best!)" if is_best else f" (No improvement x{epochs_without_improvement})"))
                  
            aux_cols = []
            if USE_AUX:
                avg_main = main_loss_sum / max(valid_batches, 1)
                aux_l = {t: aux_loss_sum[t] / max(aux_loss_n[t], 1) for t in AUX_TASKS}
                aux_a = {t: aux_correct[t] / max(aux_total[t], 1) for t in AUX_TASKS}
                aux_cols = ([round(avg_main, 5)] + [round(aux_l[t], 5) for t in AUX_TASKS]
                            + [round(aux_a[t], 4) for t in AUX_TASKS])
                print("   aux: main loss {:.4f} | ".format(avg_main) + " | ".join(
                    f"{t}: loss {aux_l[t]:.4f} (x{AUX_LAMBDA[t]} = {AUX_LAMBDA[t] * aux_l[t]:.4f}), train acc {aux_a[t]:.3f}"
                    for t in AUX_TASKS))
                weighted_aux = sum(AUX_LAMBDA[t] * aux_l[t] for t in AUX_TASKS)
                if epoch <= 3 and weighted_aux > avg_main:
                    print(f"   ⚠️  weighted auxiliary loss ({weighted_aux:.3f}) exceeds the main loss ({avg_main:.3f}) "
                          f"in epoch {epoch} -- consider a lower aux_loss_weights (plan §4.7).")
            with open(metrics_log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, avg_train_loss, avg_val_loss, epoch_f1, epoch_iou, epoch_pct,
                                 epoch_seg_f1, combined_score, epoch_duration, epoch_nan_count,
                                 round(val_metrics["BndF1_5"], 4)] + aux_cols)

            if EARLY_STOPPING and epochs_without_improvement >= PATIENCE and epoch >= MIN_EPOCHS:
                print(f"\n🛑 Early stopping triggered! No improvement in combined score for {PATIENCE} epochs (Minimum {MIN_EPOCHS} epochs met).")
                break
                
        if not needs_restart:
            break
        
        redo_count += 1
        
    if needs_restart and redo_count > MAX_REDOS:
        print(f"\n❌ FAILED TO STABILIZE TRAINING. Maximum redos ({MAX_REDOS}) reached. Saving partial/broken run.")

    total_time = time.time() - start_train_time
    total_minutes, total_seconds = divmod(int(total_time), 60)
    
    avg_epoch_time = int(total_time / actual_epochs_ran) if actual_epochs_ran > 0 else 0
    avg_minutes, avg_seconds = divmod(avg_epoch_time, 60)
    
    avg_gpu_util = sum(gpu_utilization_samples) / len(gpu_utilization_samples) if gpu_utilization_samples else 0.0
    max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    
    hostname = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    
    hardware_summary = {
        "seed": SEED,
        "description": config.get("description", ""),
        "machine_hostname": hostname,
        "gpu_name": gpu_name,
        "total_training_time": f"{int(total_minutes)}m {int(total_seconds)}s",
        "total_training_seconds": round(total_time, 2),
        "total_epochs_ran": actual_epochs_ran,
        "early_stopping_triggered": epochs_without_improvement >= PATIENCE,
        "average_time_per_epoch": f"{int(avg_minutes)}m {int(avg_seconds)}s",
        "max_gpu_memory_used_gb": round(max_mem_gb, 4),
        "average_gpu_utilization_percent": round(avg_gpu_util, 2),
        "total_nan_loss_count_latest_run": total_nan_this_run, 
        "total_training_restarts": min(redo_count, MAX_REDOS),
        # Full job config, saved verbatim -- lets any future audit (or you, six
        # months from now) reconstruct exactly what this run was, without needing
        # to keep the description field perfectly consistent or remember to add a
        # new field here every time a new sweep parameter gets introduced.
        "config": config
    }

    summary_save_path = os.path.join(exp_dir, "hardware_summary.json")
    with open(summary_save_path, "w") as f:
        json.dump(hardware_summary, f, indent=4)
        
    print(f"📊 Hardware summary saved to {summary_save_path}")
        
    model_save_path = os.path.join(model_dir, f"{run_name}.pth")
    if best_model_state:
        torch.save(best_model_state, model_save_path)
        score_name = {"frame_f1": "Frame F1", "bnd5": "Boundary F1@5"}.get(selection_metric, "Combined Score")
        print(f"✅ Best Model (Epoch {best_epoch} | {score_name}: {best_combined_score:.4f}) saved to {model_save_path}")
    else:
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Final Model saved to {model_save_path}")
        
    print(f"🏁 Finished {run_name}\n")

if __name__ == "__main__":
    print("🚦 Starting Train Queue Manager...")
    selection_metric = ask_selection_metric()
    
    jobs_processed = 0
    while True:
        job_config = get_next_job()
        
        if job_config is None:
            if jobs_processed == 0:
                print("📭 Queue is empty. No jobs to run.")
            else:
                print(f"🎉 All {jobs_processed} queued jobs completed successfully. Shutting down.")
            break
            
        train_model(job_config, selection_metric)
        jobs_processed += 1