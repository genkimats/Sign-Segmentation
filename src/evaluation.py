"""
Full-video inference shared by train_phrase.py (validation each epoch) and
evaluate_phrase.py (standalone evaluation of saved runs).

How a video is predicted (identical to how the model was trained):
  - The video is cut into the SAME windows the dataset builds for training:
    length = window_size raw frames, stride = window_size - overlap, plus one
    final window aligned to the end of the video (and a single zero-padded
    window if the video is shorter than window_size).
  - temporal_downsample_factor k > 1: each window's input is subsampled [::k]
    exactly like SignSegmentationDataset.__getitem__ does, and the model's
    output is repeated k times per step to get back to the RAW frame rate.
  - Softmax probabilities of overlapping windows are averaged per frame.

Ground truth is ALWAYS the raw, hard per-frame label array at the raw frame
rate (dataset.video_cache[vid]['hard_labels']), never the tolerance-smoothed
soft labels and never downsampled. So tolerance_window (any value) and
temporal_downsample_factor change neither the ground truth nor how it is
evaluated.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.model_factory import call_model
from src.metrics import evaluate_videos


def video_windows(num_frames, window_size, overlap):
    """Replicates SignSegmentationDataset's window construction exactly."""
    windows = []
    if num_frames > window_size:
        step = window_size - overlap
        if step <= 0:
            raise ValueError(f"overlap ({overlap}) must be smaller than window_size ({window_size}).")
        start = 0
        for start in range(0, num_frames - window_size + 1, step):
            windows.append((start, start + window_size))
        if start + window_size < num_frames:
            windows.append((num_frames - window_size, num_frames))
    else:
        windows.append((0, num_frames))
    return windows


def _prepare_window(tensor_tv, start, end, k, target_T, is_graph):
    """
    tensor_tv: cached tensor with time first -- (T, V, C) for features, (T, D) for side streams.
    Returns a channel-first FP32 tensor, subsampled [::k] and zero-padded to target_T,
    i.e. exactly what __getitem__ returns for that window.
    """
    x = tensor_tv[start:end].to(torch.float32)
    if is_graph:
        x = x.permute(2, 0, 1)  # (C, T, V)
    else:
        x = x.permute(1, 0)     # (D, T)
    if k > 1:
        x = x[:, ::k, ...] if is_graph else x[:, ::k]
    T = x.shape[1]
    if T < target_T:
        pad_shape = list(x.shape)
        pad_shape[1] = target_T - T
        x = torch.cat([x, torch.zeros(pad_shape, dtype=torch.float32)], dim=1)
    return x


@torch.no_grad()
def predict_video_probs(model, config, cached, device, batch_size=16):
    """
    cached: one entry of SignSegmentationDataset.video_cache.
    Returns a (3, num_frames) numpy array of stitched softmax probabilities (O, I, B)
    at the RAW frame rate.
    """
    window_size = config["window_size"]
    overlap = config.get("overlap", 0)
    k = max(1, int(config.get("temporal_downsample_factor", 1)))
    use_hamer = config.get("use_hamer_features", False)
    use_dinov2 = config.get("use_dinov2_features", False)

    features = cached["features"]
    num_frames = features.shape[0]
    target_T = window_size // k

    windows = video_windows(num_frames, window_size, overlap)
    prob_sum = np.zeros((3, num_frames), dtype=np.float64)
    counts = np.zeros(num_frames, dtype=np.float64)

    for b0 in range(0, len(windows), batch_size):
        batch_windows = windows[b0:b0 + batch_size]
        feats = torch.stack([_prepare_window(features, s, e, k, target_T, True) for s, e in batch_windows])
        hamer = dinov2 = None
        if use_hamer:
            hamer = torch.stack([_prepare_window(cached["hamer_features"], s, e, k, target_T, False)
                                 for s, e in batch_windows])
            hamer = torch.nan_to_num(hamer.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        if use_dinov2:
            dinov2 = torch.stack([_prepare_window(cached["dinov2_features"], s, e, k, target_T, False)
                                  for s, e in batch_windows])
            dinov2 = torch.nan_to_num(dinov2.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        feats = torch.nan_to_num(feats.to(device), nan=0.0, posinf=0.0, neginf=0.0)

        logits = call_model(model, config, feats, hamer, dinov2)        # (B, 3, T_model)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()

        for (s, e), p in zip(batch_windows, probs):
            raw_len = e - s
            if k > 1:
                p = np.repeat(p, k, axis=1)                              # back to raw frame rate
            if p.shape[1] < raw_len:                                     # window not a multiple of k
                p = np.concatenate([p, np.repeat(p[:, -1:], raw_len - p.shape[1], axis=1)], axis=1)
            p = p[:, :raw_len]                                           # drop padding
            prob_sum[:, s:e] += p
            counts[s:e] += 1.0

    counts[counts == 0] = 1.0
    return (prob_sum / counts).astype(np.float32)


def predict_split(model, config, dataset, device, batch_size=16):
    """
    Runs predict_video_probs on every cached video of a dataset.
    Returns (video_probs, video_gold): dicts vid -> (3, T) probs and vid -> (T,) hard labels.
    """
    was_training = model.training
    model.eval()
    video_probs, video_gold = {}, {}
    for vid, cached in dataset.video_cache.items():
        video_probs[vid] = predict_video_probs(model, config, cached, device, batch_size)
        video_gold[vid] = np.asarray(cached["hard_labels"]).astype(np.int64)
    if was_training:
        model.train()
    return video_probs, video_gold


def evaluate_model_on_split(model, config, dataset, device, batch_size=16,
                            decoder="argmax", b_threshold=0.5, o_threshold=0.5):
    """Convenience wrapper: predict every video, then compute the full-video metrics."""
    video_probs, video_gold = predict_split(model, config, dataset, device, batch_size)
    return evaluate_videos(video_probs, video_gold, decoder=decoder,
                           b_threshold=b_threshold, o_threshold=o_threshold)