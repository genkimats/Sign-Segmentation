"""
Evaluates INFERENCE-TIME cost and behavior across architectures -- the
deployment-time question Mamba's efficiency case actually depends on, and a
DIFFERENT question from the training-window sweep. That sweep asked "what
happens if I TRAIN on longer windows" (answer: F1 degrades for everyone).
This script asks "given a model already trained at its best window size,
what happens at INFERENCE time on a real, unchunked, arbitrary-length
sequence" -- since Mamba's O(n) vs Transformer's O(n^2) distinction is a
deployment-time property, not a training-time one.

Two separate experiments:

1. REAL VIDEOS at natural full length (via use_full_length=True), run BOTH
   chunked (safe, universal -- split into window-size pieces like training
   did) and streaming (the whole video in one forward pass). Tracks F1 for
   both, to see whether running beyond the trained context length preserves,
   degrades, or improves accuracy -- inference-only, no retraining involved.

2. SYNTHETIC, length-controlled sequences (built by tiling real feature
   windows to an exact target length) -- TIME AND MEMORY ONLY, not F1
   (concatenated synthetic sequences have no coherent labels). This gets a
   clean scaling curve at lengths well beyond whatever your corpus's natural
   video lengths happen to be, since the real payoff may only show up past
   that range.

IMPORTANT CAVEAT: this does NOT automatically separate Mamba from BiLSTM on
FLOP-complexity grounds -- both are O(n) in sequence length, unlike
Transformer's O(n^2). What SHOULD separate them is Mamba's PARALLEL scan
(processes the whole sequence at once despite being a recurrence) versus
BiLSTM's genuinely SEQUENTIAL recurrence (each step waits on the last) --
Mamba should show better WALL-CLOCK scaling than BiLSTM even at matched FLOP
complexity, not just better than Transformer. This script times actual
wall-clock (with proper CUDA synchronization), not estimated FLOPs, so that
distinction -- if real -- should be visible in the results.

Scoped to coords-only models (no HaMeR/DINOv2) deliberately: this is testing
ARCHITECTURE scaling behavior, not feature-set effects. Extend CHECKPOINTS
below with matching model_kwargs if you want to profile a feature-enabled
variant too.

FILL IN CHECKPOINTS BELOW with your actual best window=64 run prefixes
before running this -- these aren't guessable from outside your experiment
history.
"""
import os
import csv
import time
import gc
import torch
import numpy as np

from src.dataset import SignSegmentationDataset
from src.models import STGCN_Mamba, STGCN_BiMamba, STGCN_BiLSTM, STGCN_Transformer, PositionalEncoding

# ==============================================================================
# CONFIGURATION -- fill in your actual best window=64, coords-only checkpoints
# ==============================================================================

# hamer
CHECKPOINTS = {
    "stgcn_mamba":       {"path": "saved_models/stgcn_mamba-292.pth",       "class": STGCN_Mamba,       "kwargs": {}},
    "stgcn_bimamba":     {"path": "saved_models/stgcn_bimamba-12.pth",     "class": STGCN_BiMamba,     "kwargs": {}},
    "stgcn_bilstm":      {"path": "saved_models/stgcn_bilstm-15.pth",      "class": STGCN_BiLSTM,      "kwargs": {}},
    "stgcn_transformer": {"path": "saved_models/stgcn_transformer-09.pth", "class": STGCN_Transformer, "kwargs": {"nhead": 8}},
}

# keypoints only
CHECKPOINTS = {
    "stgcn_mamba":       {"path": "saved_models/stgcn_mamba-292.pth",       "class": STGCN_Mamba,       "kwargs": {}},
    "stgcn_bimamba":     {"path": "saved_models/stgcn_bimamba-12.pth",     "class": STGCN_BiMamba,     "kwargs": {}},
    "stgcn_bilstm":      {"path": "saved_models/stgcn_bilstm-15.pth",      "class": STGCN_BiLSTM,      "kwargs": {}},
    "stgcn_transformer": {"path": "saved_models/stgcn_transformer-09.pth", "class": STGCN_Transformer, "kwargs": {"nhead": 8}},
}

TRAINED_WINDOW_SIZE = 64  # the window size these checkpoints were actually trained at
# Trained checkpoints' Transformer positional encoding was built with max_len=5000
# (see models.py's PositionalEncoding default) -- fine during training (windows
# never exceeded a few hundred frames), but a hard crash on any real video or
# synthetic length beyond it (your test set includes at least one 24,068-frame
# video). Since the encoding is a pure deterministic function of position with NO
# learned parameters, it's safe to swap in a longer one AFTER loading the
# checkpoint -- this changes nothing about what the model learned, it just
# extends how far it CAN be evaluated. Must exceed your longest real video AND
# your longest synthetic sweep length.
TRANSFORMER_MAX_LEN = 50000
NUM_VERTICES = 65

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Real-video profiling: which split(s) to pull full-length videos from. Pure
# profiling (time/memory) doesn't leak test-set information the way retraining
# or model selection would, so using test+val (or even train) here is fine --
# unlike picking a final reported F1 number, this isn't a decision that could
# overfit to the test set.
SPLITS_TO_PROFILE = ["test", "val"]

# Synthetic length sweep targets -- deliberately extends well past whatever
# your corpus's natural video lengths are, since Mamba's real advantage may
# only show up beyond that range.
SYNTHETIC_LENGTHS = [1024, 2048, 4096, 8192, 16384]

OUTPUT_DIR = "./inference_scaling"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_in_channels(state_dict):
    """
    Infers in_channels directly from the checkpoint's own first-layer weight
    shape, instead of relying on a hardcoded IN_CHANNELS constant that has to
    be manually kept in sync with whatever config a given run actually used --
    the same class of bug that hit dinov2_dim/hamer_dim earlier. All four
    architectures here (STGCN_Mamba, STGCN_BiMamba, STGCN_BiLSTM,
    STGCN_Transformer) share the identical stgcn_blocks.0.gcn.conv naming,
    since they're all built on the same STGCNBlock as their first spatial
    layer, so this one key works for all of them.
    """
    key = "stgcn_blocks.0.gcn.conv.weight"
    if key not in state_dict:
        raise KeyError(f"Expected key {key!r} not found in checkpoint -- this architecture's "
                        f"first layer isn't named the way this detector assumes. Available "
                        f"keys (first 5): {list(state_dict.keys())[:5]}")
    return state_dict[key].shape[1]


def load_model(name, num_vertices):
    cfg = CHECKPOINTS[name]
    state_dict = torch.load(cfg["path"], map_location=DEVICE, weights_only=False)

    detected_in_channels = detect_in_channels(state_dict)
    print(f"  [{name}] detected in_channels={detected_in_channels} from checkpoint "
          f"(expected 3 for a coords-only run -- double check the checkpoint path if this "
          f"looks wrong for what you meant to test)")

    model = cfg["class"](num_vertices=num_vertices, in_channels=detected_in_channels, **cfg["kwargs"]).to(DEVICE)
    model.load_state_dict(state_dict)

    if hasattr(model, "pos_encoder"):
        d_model = model.pos_encoder.pe.shape[-1]
        dropout_p = model.pos_encoder.dropout.p
        model.pos_encoder = PositionalEncoding(d_model, dropout=dropout_p, max_len=TRANSFORMER_MAX_LEN).to(DEVICE)
        print(f"  [{name}] extended positional encoding max_len to {TRANSFORMER_MAX_LEN} "
              f"(checkpoint was trained with max_len=5000 -- swapped post-load since this "
              f"buffer has no learned parameters)")

    model.eval()
    return model


def reset_memory_stats():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        gc.collect()


def peak_memory_gb():
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
    return float("nan")


@torch.no_grad()
def timed_forward(model, features):
    """features: (1, C, T, V). Returns (logits, elapsed_seconds) with proper
    CUDA synchronization -- CUDA ops are asynchronous, so timing without
    synchronize() before/after would measure CPU dispatch time, not actual
    GPU compute time."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    logits, _ = model(features)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return logits, elapsed


@torch.no_grad()
def chunked_inference(model, features, window_size):
    """features: (1, C, T, V). Splits into non-overlapping window_size chunks
    (zero-padding the last one, then trimming predictions back to the real
    length), runs each chunk separately -- the 'safe, universal' deployment
    approach any architecture supports regardless of its scaling properties."""
    B, C, T, V = features.shape
    all_logits = []

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for chunk_start in range(0, T, window_size):
        chunk_end = min(chunk_start + window_size, T)
        chunk = features[:, :, chunk_start:chunk_end, :]

        if chunk.shape[2] < window_size:
            pad = torch.zeros(B, C, window_size - chunk.shape[2], V, device=features.device, dtype=features.dtype)
            chunk = torch.cat([chunk, pad], dim=2)

        logits, _ = model(chunk)  # (1, num_classes, window_size)
        all_logits.append(logits[:, :, :chunk_end - chunk_start])  # trim any padding back off

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return torch.cat(all_logits, dim=2), elapsed


def compute_f1(logits, labels):
    """logits: (1, 3, T); labels: (1, 3, T) soft/one-hot. Frame-level macro F1
    via argmax, matching the metric used throughout training."""
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()      # (T,)
    true = labels.argmax(dim=1).squeeze(0).cpu().numpy()       # (T,)

    f1s = []
    for c in range(3):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def run_real_video_experiment():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: real videos, natural full length, chunked vs streaming")
    print("=" * 70)

    results = []

    for split in SPLITS_TO_PROFILE:
        dataset = SignSegmentationDataset(
            keypoints_dir="processed_data/keypoints",
            labels_dir="processed_data/BIO_tags",
            split=split,
            use_full_length=True,
            base_features=["x-cord", "y-cord", "z-cord"],
        )

        for model_name in CHECKPOINTS:
            print(f"\n--- {model_name} ({split} split) ---")
            model = load_model(model_name, NUM_VERTICES)

            for idx in range(len(dataset)):
                features, labels, vid, _, _ = dataset[idx]
                features = features.unsqueeze(0).to(DEVICE)  # (1, C, T, V)
                labels = labels.unsqueeze(0)
                T = features.shape[2]

                for mode in ["chunked", "streaming"]:
                    reset_memory_stats()
                    try:
                        if mode == "chunked":
                            logits, elapsed = chunked_inference(model, features, TRAINED_WINDOW_SIZE)
                        else:
                            logits, elapsed = timed_forward(model, features)
                        f1 = compute_f1(logits.cpu(), labels)
                        mem = peak_memory_gb()
                        status = "ok"
                    except torch.cuda.OutOfMemoryError:
                        elapsed, f1, mem, status = float("nan"), float("nan"), float("nan"), "OOM"
                        reset_memory_stats()
                    except RuntimeError as e:
                        # Catches real, informative architectural failures too, not just OOM --
                        # e.g. Transformer exceeding its positional encoding's max_len before
                        # the fix above, or any other unexpected shape/runtime failure. Records
                        # the reason and moves on instead of losing every result collected so
                        # far in this run.
                        elapsed, f1, mem = float("nan"), float("nan"), float("nan")
                        status = f"error: {str(e)[:150]}"
                        reset_memory_stats()

                    results.append({
                        "split": split, "model": model_name, "video_id": vid,
                        "length_frames": T, "mode": mode,
                        "time_seconds": elapsed, "peak_mem_gb": mem,
                        "f1": f1, "status": status,
                    })
                    print(f"  {vid} (T={T}) [{mode}]: time={elapsed:.4f}s, mem={mem:.3f}GB, "
                          f"F1={f1:.4f}, status={status}")

            del model
            reset_memory_stats()

    out_path = os.path.join(OUTPUT_DIR, "real_video_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWritten: {out_path}")


def build_synthetic_sequence(base_features, target_length):
    """Tiles a real (C, T, V) feature window to an exact target length by
    repeating it -- purely for timing/memory profiling at controlled lengths;
    NOT meant to have coherent semantic content (no matching labels)."""
    C, T, V = base_features.shape
    repeats = (target_length + T - 1) // T
    tiled = base_features.repeat(1, repeats, 1)[:, :target_length, :]
    return tiled


def run_synthetic_length_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: synthetic length-controlled sweep (time/memory only)")
    print("=" * 70)

    dataset = SignSegmentationDataset(
        keypoints_dir="processed_data/keypoints",
        labels_dir="processed_data/BIO_tags",
        split="train",
        window_size=256,
        base_features=["x-cord", "y-cord", "z-cord"],
    )
    base_features, _, _, _, _ = dataset[0]  # (C, 256, V) -- tiling seed

    results = []
    for model_name in CHECKPOINTS:
        print(f"\n--- {model_name} ---")
        model = load_model(model_name, NUM_VERTICES)

        for target_length in SYNTHETIC_LENGTHS:
            sequence = build_synthetic_sequence(base_features, target_length).unsqueeze(0).to(DEVICE)

            for mode in ["chunked", "streaming"]:
                reset_memory_stats()
                try:
                    if mode == "chunked":
                        _, elapsed = chunked_inference(model, sequence, TRAINED_WINDOW_SIZE)
                    else:
                        _, elapsed = timed_forward(model, sequence)
                    mem = peak_memory_gb()
                    status = "ok"
                except torch.cuda.OutOfMemoryError:
                    elapsed, mem, status = float("nan"), float("nan"), "OOM"
                    reset_memory_stats()
                except RuntimeError as e:
                    elapsed, mem = float("nan"), float("nan")
                    status = f"error: {str(e)[:150]}"
                    reset_memory_stats()

                results.append({
                    "model": model_name, "length_frames": target_length, "mode": mode,
                    "time_seconds": elapsed, "peak_mem_gb": mem, "status": status,
                })
                print(f"  T={target_length} [{mode}]: time={elapsed:.4f}s, mem={mem:.3f}GB, status={status}")

                if status != "ok":
                    # No point testing longer lengths in this mode once it's failed -- it
                    # will only get worse (OOM) or fail the same way (other errors). Still
                    # tests the OTHER mode / other lengths.
                    break

        del model
        reset_memory_stats()

    out_path = os.path.join(OUTPUT_DIR, "synthetic_sweep_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    run_real_video_experiment()
    run_synthetic_length_sweep()
    print("\nDone. See experiments/inference_scaling/*.csv for full results.")