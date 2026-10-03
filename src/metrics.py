"""
Segmentation metrics for BIO tagging -- computed on FULL VIDEOS, against HARD
ground-truth labels.

Class ids used everywhere in this repo: 0 = Outside (O), 1 = Inside (I), 2 = Begin (B).

The three headline metrics follow Moryossef et al. (2023), "Linguistically
Motivated Sign Language Segmentation":

  1. Frame F1  -- macro-averaged F1 over the three BIO classes, computed on the
                  argmax prediction of every frame, pooled over all frames of
                  all videos in the split.
  2. IoU       -- per video: intersection-over-union of the binary "inside any
                  segment" masks of prediction vs gold (no one-to-one segment
                  matching), then averaged over videos.
  3. % (pct)   -- per video: (# predicted segments) / (# gold segments), then
                  averaged over videos. Optimum is 1.0; > 1 = over-segmentation,
                  < 1 = under-segmentation. It says nothing about WHERE the
                  segments are -- read it together with IoU.

Additionally reported (not in the 2023 paper, but useful):
  4. Segment F1@0.5 -- one-to-one matching of predicted to gold segments with
                  IoU >= 0.5, then F1 over the matches.

NOTE: verify the exact averaging conventions (per-video mean vs pooled counts)
against the 2023 repo's evaluation code before quoting numbers next to theirs.
"""
import numpy as np
from sklearn.metrics import f1_score

O_TAG, I_TAG, B_TAG = 0, 1, 2


# ==============================================================================
# Segment extraction
# ==============================================================================
def extract_segments(bio_sequence, start_on_i_after_o=True):
    """
    Converts a 1D BIO sequence (0=O, 1=I, 2=B) into [(start, end), ...] with
    INCLUSIVE ends.

    - B always starts a new segment (closing any open one).
    - O closes the open segment.
    - I continues the open segment. If no segment is open (i.e. I right after O,
      or I at the very start), it STARTS one when start_on_i_after_o=True.
      Without this, a predicted run that the model began with I instead of B
      would vanish entirely, which artificially lowers the segment count.
      The same rule is applied to gold and prediction, so it is symmetric.
    """
    seq = np.asarray(bio_sequence).astype(np.int64)
    segments = []
    start = -1
    for i, tag in enumerate(seq):
        if tag == B_TAG:
            if start != -1:
                segments.append((start, i - 1))
            start = i
        elif tag == O_TAG:
            if start != -1:
                segments.append((start, i - 1))
                start = -1
        else:  # I
            if start == -1 and start_on_i_after_o:
                start = i
    if start != -1:
        segments.append((start, len(seq) - 1))
    return segments


def decode_threshold_2023(probs, b_threshold=0.5, o_threshold=0.5):
    """
    Re-implementation of the 2023 repo's probability-to-segment decoding
    (probs_to_segments). Verify against their code if exact parity matters.

    probs: (3, T) array of class probabilities in THIS repo's order (O, I, B).

    - A segment starts at the first frame where P(B) > b_threshold.
    - After P(B) has dropped below b_threshold once ("passed the start"), the
      segment ends at the frame before the next frame where P(B) > b_threshold
      (a new segment starts there) or P(O) > o_threshold (no segment open).
    - A segment still open at the end runs to the last frame.

    The 2023 paper's E1s* / E4s* rows use thresholds tuned on validation; with
    default 0.5/0.5 this corresponds to their untuned rows.
    """
    p_o = probs[O_TAG]
    p_b = probs[B_TAG]
    T = probs.shape[1]
    segments = []
    start = None
    passed_start = False
    for t in range(T):
        b, o = p_b[t], p_o[t]
        if start is None:
            if b > b_threshold:
                start = t
                passed_start = False
        else:
            if passed_start:
                if b > b_threshold or o > o_threshold:
                    segments.append((start, t - 1))
                    start = None if o > o_threshold else t
                    passed_start = False
            else:
                if b < b_threshold:
                    passed_start = True
    if start is not None:
        segments.append((start, T - 1))
    return segments


# ==============================================================================
# Per-video segment metrics
# ==============================================================================
def _segments_to_mask(segments, length):
    mask = np.zeros(length, dtype=bool)
    for s, e in segments:
        mask[max(s, 0):min(e, length - 1) + 1] = True
    return mask


def segment_iou(pred_segments, gold_segments, length):
    """2023-style IoU: binary 'inside any segment' masks, no one-to-one matching."""
    pred_mask = _segments_to_mask(pred_segments, length)
    gold_mask = _segments_to_mask(gold_segments, length)
    union = np.logical_or(pred_mask, gold_mask).sum()
    if union == 0:
        return 1.0  # both empty: perfect agreement
    return float(np.logical_and(pred_mask, gold_mask).sum() / union)


def segment_percentage(pred_segments, gold_segments):
    """2023-style %: # predicted / # gold segments. NaN if the video has no gold segments."""
    if len(gold_segments) == 0:
        return float("nan")
    return len(pred_segments) / len(gold_segments)


def _interval_iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def segment_f1_at(pred_segments, gold_segments, iou_threshold=0.5):
    """One-to-one greedy matching by IoU (highest first); F1 over matched pairs."""
    if len(pred_segments) == 0 and len(gold_segments) == 0:
        return 1.0
    if len(pred_segments) == 0 or len(gold_segments) == 0:
        return 0.0
    pairs = []
    for pi, p in enumerate(pred_segments):
        for gi, g in enumerate(gold_segments):
            iou = _interval_iou(p, g)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for _, pi, gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            tp += 1
    precision = tp / len(pred_segments)
    recall = tp / len(gold_segments)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def pct_score(pct):
    """Maps % (optimum 1.0) to a [0, 1] score for model selection: 1 - |pct - 1|, floored at 0."""
    if pct is None or np.isnan(pct):
        return 0.0
    return max(0.0, 1.0 - abs(pct - 1.0))


# ==============================================================================
# Split-level evaluation
# ==============================================================================
def evaluate_videos(video_probs, video_gold, decoder="argmax", b_threshold=0.5, o_threshold=0.5):
    """
    video_probs: dict vid -> (3, T) numpy array of stitched class probabilities (O, I, B)
    video_gold:  dict vid -> (T,) numpy int array of HARD gold labels (0/1/2)
    decoder:     "argmax" or "threshold" (2023-style, uses b_threshold / o_threshold)

    Returns a dict with Frame_F1, IoU, Pct, Segment_F1_05, Combined, plus the pooled
    frame arrays (for confusion matrices) and per-video details.

    Frame F1 is always computed on the argmax prediction (decoding thresholds only
    change segments, exactly like in the 2023 paper, where tuned decoding leaves F1
    unchanged).
    """
    all_true, all_pred = [], []
    ious, pcts, seg_f1s = [], [], []
    per_video = {}

    for vid, probs in video_probs.items():
        gold = np.asarray(video_gold[vid]).astype(np.int64)
        T = min(len(gold), probs.shape[1])
        gold = gold[:T]
        probs = probs[:, :T]

        argmax_pred = probs.argmax(axis=0)
        all_true.append(gold)
        all_pred.append(argmax_pred)

        gold_segments = extract_segments(gold)
        if decoder == "argmax":
            pred_segments = extract_segments(argmax_pred)
        elif decoder == "threshold":
            pred_segments = decode_threshold_2023(probs, b_threshold, o_threshold)
        else:
            raise ValueError(f"Unknown decoder '{decoder}'")

        iou = segment_iou(pred_segments, gold_segments, T)
        pct = segment_percentage(pred_segments, gold_segments)
        sf1 = segment_f1_at(pred_segments, gold_segments, 0.5)

        ious.append(iou)
        if not np.isnan(pct):
            pcts.append(pct)
        seg_f1s.append(sf1)
        per_video[vid] = {
            "num_frames": int(T),
            "num_gold_segments": len(gold_segments),
            "num_pred_segments": len(pred_segments),
            "IoU": iou,
            "Pct": pct,
            "Segment_F1_05": sf1,
        }

    if not all_true:
        raise RuntimeError("evaluate_videos() got no videos.")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    frame_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)

    mean_iou = float(np.mean(ious))
    mean_pct = float(np.mean(pcts)) if pcts else float("nan")
    mean_seg_f1 = float(np.mean(seg_f1s))

    return {
        "Frame_F1": float(frame_f1),
        "IoU": mean_iou,
        "Pct": mean_pct,
        "Segment_F1_05": mean_seg_f1,
        # Model-selection score: all three 2023 metrics, % mapped so that 1.0 is best.
        "Combined": float(frame_f1) + mean_iou + pct_score(mean_pct),
        "frame_true": y_true,
        "frame_pred": y_pred,
        "per_video": per_video,
    }


# ==============================================================================
# DEPRECATED -- kept only so other scripts that still import it (e.g. a sign-level
# train.py) don't break. Window-level, and its third value is segment F1@0.5,
# NOT the 2023 segment percentage. Do not use for numbers you compare with papers.
# ==============================================================================
def calculate_1d_iou(seg1, seg2):
    return _interval_iou(seg1, seg2)


def calculate_segment_metrics(pred_sequence, gt_sequence, iou_threshold=0.5):
    """DEPRECATED. Segment F1 at an IoU threshold (one-to-one matching)."""
    return segment_f1_at(extract_segments(pred_sequence), extract_segments(gt_sequence), iou_threshold)


def evaluate_batch(predictions, targets):
    """DEPRECATED window-level metrics (see note above)."""
    pred_flat = np.asarray(predictions).flatten()
    target_flat = np.asarray(targets).flatten()
    frame_f1 = f1_score(target_flat, pred_flat, labels=[0, 1, 2], average="macro", zero_division=0)

    batch_size = np.asarray(predictions).shape[0]
    total_iou, total_seg_f1 = 0.0, 0.0
    for i in range(batch_size):
        p_segs = extract_segments(predictions[i])
        g_segs = extract_segments(targets[i])
        T = len(targets[i])
        total_iou += segment_iou(p_segs, g_segs, T)
        total_seg_f1 += segment_f1_at(p_segs, g_segs, 0.5)

    return {
        "Frame_F1": frame_f1,
        "Mean_IoU": total_iou / batch_size,
        "Segment_F1_05": total_seg_f1 / batch_size,
    }