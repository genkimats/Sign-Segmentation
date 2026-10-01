"""
hierarchy_phrase/metrics.py -- numpy-only metrics, every one computed on RAW gold at the working frame rate.

Reproduces the 2023 paper's three headline numbers and adds boundary / segment metrics:
  * frame_f1        macro-F1 over {O, I, B} frames, argmax labels. Pooled over all frames of all videos.
  * mask_iou        IoU between the 'inside any segment' frame masks, pooled. This is the closest reading of
                    the 2023 "total IoU based on all segments, no one-to-one mapping". The paper's exact formula
                    is ambiguous, so `mean_seg_iou` (each PREDICTED segment vs its best-overlapping gold segment,
                    averaged; many-to-one) is reported next to it.
  * ratio           #pred segments / #gold segments (optimum 1.0), pooled.
  * boundary F1     one-to-one matching of segment STARTS (and ENDS) within +-tol frames.
  * segment F1      one-to-one greedy matching at IoU >= thr (mF1S of Renz et al. 2021 averages thr 0.1..0.5).
"""
import numpy as np

from segments import segments_to_bio


# ------------------------------------------------------------------ primitives
def f1_from_counts(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def match_points(pred, gold, tol):
    """Maximum one-to-one matching of two sorted 1-D point sets within +-tol (greedy two-pointer is optimal
    in 1-D). Returns the number of matches."""
    i = j = tp = 0
    while i < len(pred) and j < len(gold):
        d = pred[i] - gold[j]
        if abs(d) <= tol:
            tp += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return tp


def seg_iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter)


def match_segments(pred, gold, thr):
    """One-to-one greedy matching by descending IoU; returns #matches with IoU >= thr."""
    pairs = []
    j0 = 0
    for i, p in enumerate(pred):
        while j0 < len(gold) and gold[j0][1] <= p[0]:
            j0 += 1
        j = j0
        while j < len(gold) and gold[j][0] < p[1]:
            v = seg_iou(p, gold[j])
            if v >= thr:
                pairs.append((v, i, j))
            j += 1
    pairs.sort(reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    return tp


def best_overlap_iou(pred, gold):
    """For each predicted segment: IoU with the gold segment of largest IoU (many-to-one, 'legacy')."""
    out = []
    j0 = 0
    for p in pred:
        while j0 < len(gold) and gold[j0][1] <= p[0]:
            j0 += 1
        best, j = 0.0, j0
        while j < len(gold) and gold[j][0] < p[1]:
            best = max(best, seg_iou(p, gold[j]))
            j += 1
        out.append(best)
    return out


# ------------------------------------------------------------------ dataset-level evaluation
def evaluate_videos(pred_segs_list, gold_segs_list, lengths, tols=(2, 5, 10), iou_thrs=(0.1, 0.3, 0.5, 0.7)):
    """pred_segs_list / gold_segs_list: one list of (start, end) per video; lengths: frames per video."""
    n_pred = sum(len(p) for p in pred_segs_list)
    n_gold = sum(len(g) for g in gold_segs_list)
    # frame-level (pooled 3-class macro F1 on B/I/O arrays, and mask IoU)
    cm = np.zeros((3, 3), dtype=np.int64)
    inter = union = 0
    for p, g, T in zip(pred_segs_list, gold_segs_list, lengths):
        pb, gb = segments_to_bio(p, T), segments_to_bio(g, T)
        cm += np.bincount(gb.astype(np.int64) * 3 + pb.astype(np.int64), minlength=9).reshape(3, 3)
        pm, gm = pb > 0, gb > 0
        inter += int((pm & gm).sum()); union += int((pm | gm).sum())
    f1c = []
    for c in range(3):
        tp = cm[c, c]
        f1c.append(f1_from_counts(tp, cm[:, c].sum(), cm[c, :].sum()))
    res = {"frame_f1": float(np.mean(f1c)), "frame_f1_O": f1c[0], "frame_f1_I": f1c[1], "frame_f1_B": f1c[2],
           "mask_iou": inter / union if union else 0.0,
           "ratio": n_pred / n_gold if n_gold else float("nan"), "n_pred": n_pred, "n_gold": n_gold}
    legacy = [v for p, g in zip(pred_segs_list, gold_segs_list) for v in best_overlap_iou(p, g)]
    res["mean_seg_iou"] = float(np.mean(legacy)) if legacy else 0.0
    for tol in tols:
        tps = tpe = 0
        for p, g in zip(pred_segs_list, gold_segs_list):
            tps += match_points([s for s, _ in p], [s for s, _ in g], tol)
            tpe += match_points([e for _, e in p], [e for _, e in g], tol)
        res[f"start_f1@{tol}"] = f1_from_counts(tps, n_pred, n_gold)
        res[f"end_f1@{tol}"] = f1_from_counts(tpe, n_pred, n_gold)
    for thr in iou_thrs:
        tp = sum(match_segments(p, g, thr) for p, g in zip(pred_segs_list, gold_segs_list))
        res[f"seg_f1@{thr}"] = f1_from_counts(tp, n_pred, n_gold)
    res["mF1S(0.1-0.5)"] = float(np.mean([res[f"seg_f1@{t}"] for t in iou_thrs if t <= 0.5]))
    return res