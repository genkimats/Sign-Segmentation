"""
train_detr.py -- trains STGCN_DETR (set-prediction segmentation) via
bipartite matching loss (DETRSegmentLoss). A genuinely different paradigm
from train.py/train_phrase.py/train_disk.py: no window size, no overlap, no
per-frame BIO labels, no chunked-vs-streaming distinction, and a DIFFERENT
evaluation metric (segment-level IoU-matched F1, NOT comparable to the
frame-level F1 everything else in this project reports).

Fully separate from your existing infrastructure: own queue file
(train_queue_detr.json), own model directory (saved_models_detr/), own
experiment directory (experiments_detr/) -- matches the isolation pattern
already used for train_phrase.py.

Batch size is always 1 (one full, unchunked video per step) -- see
dataset_segments.py's docstring for why.

SETUP CHECKLIST -- see the chat message this was delivered in for the full
list of things to verify/decide before running this.
"""
import os
import json
import time
import random
import copy
import socket
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset_segments import SignSegmentationDatasetDETR
from src.models_detr import STGCN_DETR
from src.detr_loss import DETRSegmentLoss

QUEUE_FILE = "train_queue_detr.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def compute_segment_f1(pred_segments, true_segments, iou_threshold=0.5):
    """pred_segments: list of (confidence, start_frac, end_frac).
    true_segments: list of (start_frac, end_frac). Greedy, confidence-ordered
    IoU matching -- standard detection-style evaluation, NOT the frame-level
    F1 used everywhere else in this project. See the chat message this was
    delivered in for why these numbers aren't comparable to your existing
    results."""
    sorted_preds = sorted(pred_segments, key=lambda p: -p[0])
    matched_true = set()
    tp = 0
    for conf, p_start, p_end in sorted_preds:
        best_iou, best_idx = 0, -1
        for i, (t_start, t_end) in enumerate(true_segments):
            if i in matched_true:
                continue
            inter = max(0, min(p_end, t_end) - max(p_start, t_start))
            union = (p_end - p_start) + (t_end - t_start) - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            tp += 1
            matched_true.add(best_idx)
    fp = len(pred_segments) - tp
    fn = len(true_segments) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def train_model(config):
    print(f"\n{'='*60}\n🚀 STARTING QUEUED DETR JOB\n{'='*60}")
    print(json.dumps(config, indent=4))

    SEED = config.get("seed", 42)
    set_seed(SEED)

    MODEL_NAME = config["basename"]
    EPOCHS = config.get("epochs", 100)
    LEARNING_RATE = config.get("learning_rate", 0.0001)
    NUM_QUERIES = config.get("num_queries", 100)
    D_MODEL = config.get("d_model", 256)
    NUM_ENCODER_LAYERS = config.get("num_encoder_layers", 4)
    NUM_DECODER_LAYERS = config.get("num_decoder_layers", 4)
    CONFIDENCE_THRESHOLD = config.get("confidence_threshold", 0.5)
    IOU_MATCH_THRESHOLD = config.get("iou_match_threshold", 0.5)
    BASE_FEATURES = config.get("base_features", ["x-cord", "y-cord", "z-cord"])
    KINEMATIC_FEATURES = config.get("kinematic_features", [])
    IN_CHANNELS = config.get("in_channels", 3)
    USE_HAMER_FEATURES = config.get("use_hamer_features", False)
    HAMER_DIR = config.get("hamer_dir", "processed_data/hamer_features")
    USE_DINOV2_FEATURES = config.get("use_dinov2_features", False)
    DINOV2_DIR = config.get("dinov2_dir", "processed_data/dinov2_features")
    PATIENCE = config.get("patience", 10)
    EARLY_STOPPING = config.get("early_stopping", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SignSegmentationDatasetDETR(
        keypoints_dir="processed_data/keypoints", labels_dir="processed_data/BIO_tags",
        split="train", base_features=BASE_FEATURES, kinematic_features=KINEMATIC_FEATURES,
        use_hamer_features=USE_HAMER_FEATURES, hamer_dir=HAMER_DIR,
        use_dinov2_features=USE_DINOV2_FEATURES, dinov2_dir=DINOV2_DIR,
    )
    val_dataset = SignSegmentationDatasetDETR(
        keypoints_dir="processed_data/keypoints", labels_dir="processed_data/BIO_tags",
        split="val", base_features=BASE_FEATURES, kinematic_features=KINEMATIC_FEATURES,
        use_hamer_features=USE_HAMER_FEATURES, hamer_dir=HAMER_DIR,
        use_dinov2_features=USE_DINOV2_FEATURES, dinov2_dir=DINOV2_DIR,
    )

    # Setup check: num_queries must comfortably exceed the most segments any
    # single video actually has, or those videos can never be fully predicted
    # no matter how well the model trains. Reads label files directly (cheap)
    # rather than calling __getitem__ (which would load the full kinematic/
    # HaMeR/DINOv2 data for every video just to count segments).
    from src.dataset_segments import bio_to_segments
    max_segments_train = max(
        len(bio_to_segments(np.load(os.path.join(train_dataset.labels_dir, f"{vid}.npy"))))
        for vid in train_dataset.valid_ids
    )
    print(f"[SETUP CHECK] Longest segment count in train split: {max_segments_train} "
          f"(num_queries={NUM_QUERIES}) -- {'OK' if NUM_QUERIES > max_segments_train else '⚠️ TOO LOW, RAISE num_queries'}")
    if NUM_QUERIES <= max_segments_train:
        raise ValueError(f"num_queries ({NUM_QUERIES}) must exceed the max segment count "
                          f"in the training data ({max_segments_train}) -- raise it in your config.")

    model_kwargs = dict(
        num_vertices=config.get("num_vertices", 65), in_channels=IN_CHANNELS, d_model=D_MODEL,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        num_queries=NUM_QUERIES,
    )
    if USE_HAMER_FEATURES:
        if train_dataset.detected_hamer_dim is None:
            raise RuntimeError("use_hamer_features=True but no hamer_dim was detected during validation.")
        model_kwargs["hamer_dim"] = train_dataset.detected_hamer_dim
    if USE_DINOV2_FEATURES:
        if train_dataset.detected_dinov2_dim is None:
            raise RuntimeError("use_dinov2_features=True but no dinov2_dim was detected during validation.")
        model_kwargs["dinov2_dim"] = train_dataset.detected_dinov2_dim

    model = STGCN_DETR(**model_kwargs).to(device)
    criterion = DETRSegmentLoss(
        class_weight=config.get("class_weight", 1.0),
        l1_weight=config.get("l1_weight", 5.0),
        iou_weight=config.get("iou_weight", 2.0),
        no_object_weight=config.get("no_object_weight", 0.1),
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    def run_one(features, hamer, dinov2, segments_frac):
        features = features.unsqueeze(0).to(device)
        hamer_t = hamer.unsqueeze(0).to(device) if (USE_HAMER_FEATURES and hamer is not None) else None
        dinov2_t = dinov2.unsqueeze(0).to(device) if (USE_DINOV2_FEATURES and dinov2 is not None) else None
        segments_frac = segments_frac.to(device)
        kwargs = {}
        if USE_HAMER_FEATURES:
            kwargs["hamer"] = hamer_t
        if USE_DINOV2_FEATURES:
            kwargs["dinov2"] = dinov2_t
        confidence_logits, pred_spans = model(features, **kwargs)
        return confidence_logits, pred_spans, segments_frac

    best_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)

        loop = tqdm(train_indices, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for idx in loop:
            features, hamer, dinov2, segments_frac, vid, num_frames = train_dataset[idx]
            optimizer.zero_grad()
            confidence_logits, pred_spans, segments_frac = run_one(features, hamer, dinov2, segments_frac)
            total_loss, class_loss, l1_loss, iou_loss = criterion(confidence_logits, pred_spans, segments_frac)

            if torch.isnan(total_loss):
                continue  # skip this single video rather than corrupting the whole epoch's gradient

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(total_loss.item())
            loop.set_postfix(loss=f"{total_loss.item():.3f}", cls=f"{class_loss.item():.3f}",
                              l1=f"{l1_loss.item():.3f}", iou=f"{iou_loss.item():.3f}")

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_f1s = []
        with torch.no_grad():
            for idx in range(len(val_dataset)):
                features, hamer, dinov2, segments_frac, vid, num_frames = val_dataset[idx]
                confidence_logits, pred_spans, segments_frac = run_one(features, hamer, dinov2, segments_frac)

                confidence_probs = torch.sigmoid(confidence_logits[0]).cpu().numpy()
                spans_np = pred_spans[0].cpu().numpy()
                pred_segments = [
                    (float(confidence_probs[i]), float(spans_np[i, 0]), float(spans_np[i, 1]))
                    for i in range(len(confidence_probs)) if confidence_probs[i] >= CONFIDENCE_THRESHOLD
                ]
                true_segments = [(float(s), float(e)) for s, e in segments_frac.cpu().numpy()]

                f1, _, _ = compute_segment_f1(pred_segments, true_segments, iou_threshold=IOU_MATCH_THRESHOLD)
                val_f1s.append(f1)

        mean_val_f1 = float(np.mean(val_f1s)) if val_f1s else 0.0
        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"Epoch {epoch}: train_loss={mean_train_loss:.4f}, val_segment_f1={mean_val_f1:.4f}")

        if mean_val_f1 > best_f1:
            best_f1 = mean_val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch}, F1={best_f1:.4f})")
            break

    total_time = time.time() - start_time

    prefix = config.get("prefix", "01")
    run_name = f"{MODEL_NAME}-{prefix}"

    model_dir = "saved_models_detr"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"{run_name}.pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"✅ Best model (epoch {best_epoch}, segment F1={best_f1:.4f}) saved to {model_save_path}")

    exp_dir = os.path.join("experiments_detr", run_name)
    os.makedirs(exp_dir, exist_ok=True)
    hostname = socket.gethostname()
    hardware_summary = {
        "seed": SEED,
        "machine_hostname": hostname,
        "best_epoch": best_epoch,
        "best_segment_f1": best_f1,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_training_seconds": round(total_time, 2),
        "total_training_time": f"{int(total_time // 60)}m {int(total_time % 60)}s",
        "metric_note": "segment-level IoU-matched F1 -- NOT comparable to frame-level F1 "
                        "reported by any other training script in this project",
    }
    with open(os.path.join(exp_dir, "hardware_summary.json"), "w") as f:
        json.dump(hardware_summary, f, indent=4)

    print(f"\nDone. Best segment F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    print("🚦 Starting DETR Train Queue Manager...")
    while True:
        job_config = get_next_job()
        if job_config is None:
            print("No more jobs in queue. Exiting.")
            break
        train_model(job_config)