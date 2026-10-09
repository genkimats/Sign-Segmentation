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

    Vectorized with numpy (no per-frame Python loop).
    """
    seq = np.asarray(bio_sequence).astype(np.int64)
    T = len(seq)
    if T == 0:
        return []
    prev = np.empty(T, dtype=np.int64)
    prev[0] = O_TAG
    prev[1:] = seq[:-1]

    is_b = seq == B_TAG
    is_i = seq == I_TAG
    if start_on_i_after_o:
        starts = is_b | (is_i & (prev == O_TAG))
        inside = seq != O_TAG
    else:
        # I with no open segment is ignored: a frame is "inside" only if a B opened
        # the current run of non-O frames.
        b_count = np.cumsum(is_b)
        last_o = np.maximum.accumulate(np.where(seq == O_TAG, np.arange(T), -1))
        b_before_run = np.where(last_o >= 0, b_count[np.clip(last_o, 0, None)], 0)
        inside = (seq != O_TAG) & ((b_count - b_before_run) > 0)
        starts = is_b & inside

    start_idx = np.flatnonzero(starts)
    if len(start_idx) == 0:
        return []
    # A segment ends at the last inside frame before the next start or the next non-inside frame.
    nxt_start = np.zeros(T, dtype=bool)
    nxt_start[:-1] = starts[1:]
    nxt_outside = np.ones(T, dtype=bool)
    nxt_outside[:-1] = ~inside[1:]
    ends = inside & (nxt_start | nxt_outside)
    end_idx = np.flatnonzero(ends)
    return list(zip(start_idx.tolist(), end_idx.tolist()))


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
    p_o = np.asarray(probs[O_TAG], dtype=np.float64).tolist()   # plain Python floats:
    p_b = np.asarray(probs[B_TAG], dtype=np.float64).tolist()   # ~10x faster loop
    T = len(p_b)
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
    """
    One-to-one greedy matching by IoU (highest first); F1 over matched pairs.

    Segments within each list are sorted and disjoint, so only overlapping
    (pred, gold) pairs can reach the threshold. They are found with a two-pointer
    sweep in O(P + G) instead of comparing every pred with every gold, which was
    the main cost when a model over-segments into thousands of short pieces.
    """
    P, G = len(pred_segments), len(gold_segments)
    if P == 0 and G == 0:
        return 1.0
    if P == 0 or G == 0:
        return 0.0
    pairs = []
    i = j = 0
    while i < P and j < G:
        p, g = pred_segments[i], gold_segments[j]
        if p[1] < g[0]:
            i += 1
            continue
        if g[1] < p[0]:
            j += 1
            continue
        # p and g overlap; also check g against later preds / p against later golds
        k = i
        while k < P and pred_segments[k][0] <= g[1]:
            iou = _interval_iou(pred_segments[k], g)
            if iou >= iou_threshold:
                pairs.append((iou, k, j))
            k += 1
        j += 1
    pairs.sort(reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for _, pi, gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            tp += 1
    precision = tp / P
    recall = tp / G
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
def evaluate_videos(video_probs, video_gold, decoder="argmax", b_threshold=0.5, o_threshold=0.5,
                    light=False, gold_segments_cache=None):
    """
    video_probs: dict vid -> (3, T) numpy array of stitched class probabilities (O, I, B)
    video_gold:  dict vid -> (T,) numpy int array of HARD gold labels (0/1/2)
    decoder:     "argmax" or "threshold" (2023-style, uses b_threshold / o_threshold)

    Returns a dict with Frame_F1, IoU, Pct, Segment_F1_05, Combined, plus the pooled
    frame arrays (for confusion matrices) and per-video details.

    Frame F1 is always computed on the argmax prediction (decoding thresholds only
    change segments, exactly like in the 2023 paper, where tuned decoding leaves F1
    unchanged).

    light=True: only IoU and % (what a threshold sweep needs); Frame F1 and
    Segment F1@0.5 are skipped (returned as NaN). Same IoU/% values as the full mode.
    gold_segments_cache: optional dict vid -> gold segments, filled/reused across calls.
    """
    all_true, all_pred = [], []
    ious, pcts, seg_f1s = [], [], []
    per_video = {}

    for vid, probs in video_probs.items():
        gold = np.asarray(video_gold[vid]).astype(np.int64)
        T = min(len(gold), probs.shape[1])
        gold = gold[:T]
        probs = probs[:, :T]

        argmax_pred = probs.argmax(axis=0) if (decoder == "argmax" or not light) else None
        if not light:
            all_true.append(gold)
            all_pred.append(argmax_pred)

        if gold_segments_cache is not None and vid in gold_segments_cache:
            gold_segments = gold_segments_cache[vid]
        else:
            gold_segments = extract_segments(gold)
            if gold_segments_cache is not None:
                gold_segments_cache[vid] = gold_segments
        if decoder == "argmax":
            pred_segments = extract_segments(argmax_pred)
        elif decoder == "threshold":
            pred_segments = decode_threshold_2023(probs, b_threshold, o_threshold)
        else:
            raise ValueError(f"Unknown decoder '{decoder}'")

        iou = segment_iou(pred_segments, gold_segments, T)
        pct = segment_percentage(pred_segments, gold_segments)
        sf1 = float("nan") if light else segment_f1_at(pred_segments, gold_segments, 0.5)

        ious.append(iou)
        if not np.isnan(pct):
            pcts.append(pct)
        seg_f1s.append(sf1)
        video_f1 = (float("nan") if light else
                    float(f1_score(gold, argmax_pred, labels=[0, 1, 2], average="macro", zero_division=0)))
        per_video[vid] = {
            "num_frames": int(T),
            "Frame_F1": video_f1,
            "num_gold_segments": len(gold_segments),
            "num_pred_segments": len(pred_segments),
            "IoU": iou,
            "Pct": pct,
            "Segment_F1_05": sf1,
        }

    if not video_probs:
        raise RuntimeError("evaluate_videos() got no videos.")

    if light:
        y_true = y_pred = None
        frame_f1 = float("nan")
        per_class_f1 = [float("nan")] * 3
        mean_seg_f1 = float("nan")
    else:
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        frame_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        # Per-class F1 (O, I, B): the macro Frame F1 is their plain average, so a weak
        # single-frame B class pulls it down even when O/I (and IoU) are very good.
        per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        mean_seg_f1 = float(np.mean(seg_f1s))

    mean_iou = float(np.mean(ious))
    mean_pct = float(np.mean(pcts)) if pcts else float("nan")

    return {
        "Frame_F1": float(frame_f1),
        "F1_O": float(per_class_f1[0]),
        "F1_I": float(per_class_f1[1]),
        "F1_B": float(per_class_f1[2]),
        "IoU": mean_iou,
        "Pct": mean_pct,
        "Segment_F1_05": mean_seg_f1,
        # Model-selection score: all three 2023 metrics, % mapped so that 1.0 is best.
        "Combined": (float("nan") if light else float(frame_f1) + mean_iou + pct_score(mean_pct)),
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


# ==============================================================================
# Boundary-type evaluation: how well are phrase STARTS found, split by context?
# ==============================================================================
BOUNDARY_TYPES = ("continuous", "short_pause", "long_pause", "video_start")
BOUNDARY_TYPE_LABELS = {
    "continuous": "continuous (no O gap)",
    "short_pause": "short pause",
    "long_pause": "long pause",
    "video_start": "video start",
}


def classify_gold_boundaries(gold, short_gap):
    """
    Every gold phrase start, classified by what precedes it:
      continuous  -- the previous frame is still inside a phrase (I/B): the new phrase
                     follows the old one with no Outside gap
      short_pause -- preceded by 1..short_gap Outside frames
      long_pause  -- preceded by more than short_gap Outside frames
      video_start -- the phrase starts at frame 0 (nothing before it)
    Returns a list of (start_frame, type, gap_frames).
    """
    gold = np.asarray(gold).astype(np.int64)
    out = []
    for s, _ in extract_segments(gold):
        if s == 0:
            out.append((s, "video_start", 0))
        elif gold[s - 1] != O_TAG:
            out.append((s, "continuous", 0))
        else:
            k = s - 1
            while k >= 0 and gold[k] == O_TAG:
                k -= 1
            gap = s - 1 - k
            out.append((s, "short_pause" if gap <= short_gap else "long_pause", gap))
    return out


def match_boundaries(gold_pos, pred_pos, tolerance):
    """
    One-to-one matching of predicted to gold boundary frames: pairs within
    +-tolerance frames, closest pairs first. Returns (gold_matched bool array,
    pred_matched bool array, signed offsets pred-gold of the matched gold boundaries
    as a float array with NaN where unmatched).
    """
    gold_pos = np.asarray(gold_pos, dtype=np.int64)
    pred_pos = np.sort(np.asarray(pred_pos, dtype=np.int64))
    g_ok = np.zeros(len(gold_pos), dtype=bool)
    p_ok = np.zeros(len(pred_pos), dtype=bool)
    offsets = np.full(len(gold_pos), np.nan)
    if len(gold_pos) == 0 or len(pred_pos) == 0:
        return g_ok, p_ok, offsets
    pairs = []
    for gi, g in enumerate(gold_pos):
        lo = np.searchsorted(pred_pos, g - tolerance, side="left")
        hi = np.searchsorted(pred_pos, g + tolerance, side="right")
        for pi in range(lo, hi):
            pairs.append((abs(int(pred_pos[pi]) - int(g)), gi, pi))
    pairs.sort()
    for _, gi, pi in pairs:
        if not g_ok[gi] and not p_ok[pi]:
            g_ok[gi] = p_ok[pi] = True
            offsets[gi] = pred_pos[pi] - gold_pos[gi]
    return g_ok, p_ok, offsets


def boundary_type_analysis(video_probs, video_gold, decoder="argmax", b_threshold=0.5, o_threshold=0.5,
                           tolerances=(2, 5, 10), short_gap=12, peak_window=5):
    """
    Phrase-start detection split by boundary type (see classify_gold_boundaries).

    Predicted boundaries = start frames of the decoded predicted segments (same decoder
    options as evaluate_videos). For each type and tolerance k it reports recall@k
    (share of gold starts with a predicted start within +-k frames, one-to-one).
    Precision@k is only defined over ALL predicted starts (a predicted start doesn't
    have a type); unmatched predicted starts are split into those inside a gold phrase
    (spurious splits) and those in gold Outside (spurious phrases in pauses).

    Decoder-independent extra: "peak P(B)" = mean over gold starts of the maximum
    P(B) within +-peak_window frames -- how strongly the model signals that boundary at
    all, before any decoding.
    """
    tolerances = sorted(int(t) for t in tolerances)
    max_tol = tolerances[-1]
    per_type = {t: {"n": 0, "gaps": [], "hits": {k: 0 for k in tolerances},
                    "offsets": [], "peak_pb": []} for t in BOUNDARY_TYPES}
    n_pred = 0
    pred_hits = {k: 0 for k in tolerances}
    fp_in_phrase = {k: 0 for k in tolerances}
    fp_in_pause = {k: 0 for k in tolerances}

    for vid, probs in video_probs.items():
        gold = np.asarray(video_gold[vid]).astype(np.int64)
        T = min(len(gold), probs.shape[1])
        gold, probs = gold[:T], probs[:, :T]

        if decoder == "argmax":
            pred_segments = extract_segments(probs.argmax(axis=0))
        elif decoder == "threshold":
            pred_segments = decode_threshold_2023(probs, b_threshold, o_threshold)
        else:
            raise ValueError(f"Unknown decoder '{decoder}'")
        pred_pos = np.array([s for s, _ in pred_segments], dtype=np.int64)
        n_pred += len(pred_pos)

        gb = classify_gold_boundaries(gold, short_gap)
        gold_pos = np.array([s for s, _, _ in gb], dtype=np.int64)
        types = [t for _, t, _ in gb]

        p_b = probs[B_TAG]
        for (s, t, gap) in gb:
            d = per_type[t]
            d["n"] += 1
            d["gaps"].append(gap)
            d["peak_pb"].append(float(p_b[max(0, s - peak_window):min(T, s + peak_window + 1)].max()))

        for k in tolerances:
            g_ok, p_ok, offsets = match_boundaries(gold_pos, pred_pos, k)
            for gi, t in enumerate(types):
                if g_ok[gi]:
                    per_type[t]["hits"][k] += 1
                    if k == max_tol:
                        per_type[t]["offsets"].append(offsets[gi])
            pred_hits[k] += int(p_ok.sum())
            sorted_pred = np.sort(pred_pos)
            unmatched = sorted_pred[~p_ok]
            inside = gold[np.clip(unmatched, 0, T - 1)] != O_TAG
            fp_in_phrase[k] += int(inside.sum())
            fp_in_pause[k] += int((~inside).sum())

    total_gold = sum(per_type[t]["n"] for t in BOUNDARY_TYPES)
    rows = {}
    for t in BOUNDARY_TYPES + ("all",):
        if t == "all":
            n = total_gold
            hits = {k: sum(per_type[x]["hits"][k] for x in BOUNDARY_TYPES) for k in tolerances}
            offsets = [o for x in BOUNDARY_TYPES for o in per_type[x]["offsets"]]
            peaks = [p for x in BOUNDARY_TYPES for p in per_type[x]["peak_pb"]]
            gaps = [g for x in BOUNDARY_TYPES for g in per_type[x]["gaps"]]
        else:
            d = per_type[t]
            n, hits, offsets, peaks, gaps = d["n"], d["hits"], d["offsets"], d["peak_pb"], d["gaps"]
        rows[t] = {
            "n": n,
            "share": n / total_gold if total_gold else float("nan"),
            "median_gap": float(np.median(gaps)) if gaps else float("nan"),
            "recall": {k: (hits[k] / n if n else float("nan")) for k in tolerances},
            "median_offset": float(np.median(offsets)) if offsets else float("nan"),
            "mean_abs_offset": float(np.mean(np.abs(offsets))) if offsets else float("nan"),
            "peak_pb": float(np.mean(peaks)) if peaks else float("nan"),
        }

    return {
        "tolerances": tolerances,
        "short_gap": short_gap,
        "peak_window": peak_window,
        "types": rows,
        "n_pred": n_pred,
        "precision": {k: (pred_hits[k] / n_pred if n_pred else float("nan")) for k in tolerances},
        "fp_in_phrase": fp_in_phrase,
        "fp_in_pause": fp_in_pause,
    }