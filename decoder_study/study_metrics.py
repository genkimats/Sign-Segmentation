"""
Metrics for the decoder study (numpy only).

Why this replaces src/metrics.py for this study (details verified in
test_decoder_study.py):
  * src/metrics.py matches segments MANY-TO-ONE and divides that same TP count
    by the number of GOLD segments to get recall. Over-segmenting a long sign
    into pieces that each overlap it >=50% therefore inflates recall and can
    push "Segment F1" ABOVE 1.0. Here segments are matched ONE-TO-ONE.
  * src/metrics.py "Mean_IoU" averages over PREDICTED segments only, so it never
    penalises missed gold segments. Kept below as `legacy_*` for continuity, but
    always read it together with segment_ratio.
Everything is scored against the RAW gold labels, never the smoothed argmax.

Segments are (start, end) with END EXCLUSIVE, built with the project's rule:
B opens a segment (closing any open one), O closes it, a stray I after O is
ignored -- identical to src/metrics.py extract_segments.
"""
import numpy as np
from study_common import O, I, B


# ------------------------------------------------------------------ segments --
def bio_to_segments(bio):
    segs, start = [], -1
    for i, tag in enumerate(np.asarray(bio).tolist()):
        if tag == B:
            if start != -1:
                segs.append((start, i))
            start = i
        elif tag == O:
            if start != -1:
                segs.append((start, i))
                start = -1
    if start != -1:
        segs.append((start, len(bio)))
    return segs


def overlapping_pairs(pred, gold):
    """All (i, j, inter) with inter > 0. Both lists sorted and disjoint, so a
    two-pointer sweep is O(N + pairs)."""
    pairs, j0 = [], 0
    for i, (ps, pe) in enumerate(pred):
        while j0 < len(gold) and gold[j0][1] <= ps:
            j0 += 1
        j = j0
        while j < len(gold) and gold[j][0] < pe:
            inter = min(pe, gold[j][1]) - max(ps, gold[j][0])
            if inter > 0:
                pairs.append((i, j, inter))
            j += 1
    return pairs


def _pair_iou(pred, gold, i, j, inter):
    union = (pred[i][1] - pred[i][0]) + (gold[j][1] - gold[j][0]) - inter
    return inter / union


def match_one_to_one(pred, gold, iou_thr):
    """Greedy highest-IoU-first one-to-one matching of pairs with IoU >= thr."""
    cands = []
    for i, j, inter in overlapping_pairs(pred, gold):
        iou = _pair_iou(pred, gold, i, j, inter)
        if iou >= iou_thr:
            cands.append((iou, i, j))
    cands.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); matches.append((i, j, iou))
    return matches


def match_boundaries(pred_pos, gold_pos, tol):
    """Max-cardinality one-to-one matching of sorted 1-D positions within +-tol
    (the classic two-pointer greedy is optimal for this structure)."""
    return len(match_boundaries_indices(pred_pos, gold_pos, tol))


def match_boundaries_indices(pred_pos, gold_pos, tol):
    """Same matching as match_boundaries, but returns the set of GOLD INDICES
    (positions in gold_pos) that got matched -- needed to bucket boundary
    matches by which gold segment they belong to. pred_pos/gold_pos must be
    sorted ascending (true for segment starts/ends from bio_to_segments,
    which scans left to right)."""
    i = j = 0
    matched = set()
    while i < len(pred_pos) and j < len(gold_pos):
        d = pred_pos[i] - gold_pos[j]
        if abs(d) <= tol:
            matched.add(j); i += 1; j += 1
        elif d < -tol:
            i += 1
        else:
            j += 1
    return matched


# ------------------------------------------------------------ legacy metrics --
def legacy_metrics(pred, gold, iou_thr=0.5):
    """Faithful re-implementation of src/metrics.py's per-video Segment_F1 and
    Mean_IoU (many-to-one). NOTE recall here = (#pred matching anything) / #gold,
    which is what the original does and why it can exceed 1."""
    if len(gold) == 0 and len(pred) == 0:
        return 1.0, 1.0
    if len(gold) == 0 or len(pred) == 0:
        return 0.0, 0.0
    best = np.zeros(len(pred))
    for i, j, inter in overlapping_pairs(pred, gold):
        best[i] = max(best[i], _pair_iou(pred, gold, i, j, inter))
    tp = int((best >= iou_thr).sum())
    precision, recall = tp / len(pred), tp / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f1, float(best.mean())


# ------------------------------------------------------------ per-video stats --
def video_stats(pred_bio, gold_bio, iou_thrs=(0.3, 0.5, 0.7), tols=(2, 5), bucket_edges=None):
    pred_bio = np.asarray(pred_bio).astype(np.int64)
    gold_bio = np.asarray(gold_bio).astype(np.int64)
    assert pred_bio.shape == gold_bio.shape
    ps, gs = bio_to_segments(pred_bio), bio_to_segments(gold_bio)

    st = {"conf": np.bincount(gold_bio * 3 + pred_bio, minlength=9).reshape(3, 3).astype(np.float64),
          "n_pred": len(ps), "n_gold": len(gs), "n_frames": len(gold_bio)}
    for thr in iou_thrs:
        st[f"tp@{thr}"] = len(match_one_to_one(ps, gs, thr))
    p_starts, g_starts = [s for s, _ in ps], [s for s, _ in gs]
    p_ends, g_ends = [e for _, e in ps], [e for _, e in gs]
    for tol in tols:
        st[f"bs_tp@{tol}"] = match_boundaries(p_starts, g_starts, tol)
        st[f"be_tp@{tol}"] = match_boundaries(p_ends, g_ends, tol)

    lf1, liou = legacy_metrics(ps, gs)
    st["legacy_f1"], st["legacy_iou"] = lf1, liou

    # over-/under-segmentation counts from overlap structure
    per_gold, per_pred = np.zeros(len(gs), int), np.zeros(len(ps), int)
    for i, j, _ in overlapping_pairs(ps, gs):
        per_gold[j] += 1; per_pred[i] += 1
    st["frag_gold"] = int((per_gold >= 2).sum())    # gold split into >=2 predictions
    st["merge_pred"] = int((per_pred >= 2).sum())   # prediction spanning >=2 gold

    if bucket_edges is not None:
        nb = len(bucket_edges) - 1
        gold_bucket = np.zeros(len(gs), dtype=int)
        for j, (gs_, ge_) in enumerate(gs):
            b = int(np.searchsorted(bucket_edges, ge_ - gs_, side="right") - 1)
            gold_bucket[j] = min(max(b, 0), nb - 1)
        st["bucket_n"] = np.zeros(nb)
        for b in gold_bucket:
            st["bucket_n"][b] += 1

        # Recall at EVERY requested IoU threshold, bucketed by gold duration --
        # not just 0.5. A fixed IoU threshold is mechanically harsher on short
        # segments (a 2-frame boundary error drops a 5-frame sign's IoU to ~0.43
        # but a 30-frame sign's to ~0.88 -- same absolute error, opposite
        # verdict), so comparing bucket recall ACROSS thresholds shows how much
        # of any short-vs-long gap is that artifact versus a genuine miss.
        for thr in iou_thrs:
            matched = {j for _, j, _ in match_one_to_one(ps, gs, thr)}
            tp = np.zeros(nb)
            for j in matched:
                tp[gold_bucket[j]] += 1
            st[f"bucket_tp@{thr}"] = tp
        st["bucket_tp"] = st[f"bucket_tp@0.5"] if 0.5 in iou_thrs else st[f"bucket_tp@{iou_thrs[0]}"]

        # Bucketed START-boundary match at each tolerance: an ABSOLUTE-frame
        # criterion, not proportional to segment length -- the natural point of
        # comparison for whether the IoU-based gap above is mostly that artifact.
        for tol in tols:
            matched_starts = match_boundaries_indices(p_starts, g_starts, tol)
            bs_tp = np.zeros(nb)
            for j in matched_starts:
                bs_tp[gold_bucket[j]] += 1
            st[f"bucket_bs_tp@{tol}"] = bs_tp
    return st


_SUM_KEYS_EXCLUDE = {"legacy_f1", "legacy_iou"}


def aggregate(stats_list):
    keys = [k for k in stats_list[0] if k not in _SUM_KEYS_EXCLUDE]
    agg = {k: sum(np.asarray(s[k]) for s in stats_list) for k in keys}
    agg["legacy_f1"] = float(np.mean([s["legacy_f1"] for s in stats_list]))
    agg["legacy_iou"] = float(np.mean([s["legacy_iou"] for s in stats_list]))
    return agg


def _f1(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def summarize(agg, iou_thrs=(0.3, 0.5, 0.7), tols=(2, 5)):
    conf = agg["conf"]
    tp = np.diag(conf); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
    per_class = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    out = {"frame_macro_f1": float(per_class.mean()),
           "frame_f1_O": float(per_class[O]), "frame_f1_I": float(per_class[I]), "frame_f1_B": float(per_class[B]),
           "n_pred": float(agg["n_pred"]), "n_gold": float(agg["n_gold"]),
           "segment_ratio": float(agg["n_pred"] / max(agg["n_gold"], 1))}
    for thr in iou_thrs:
        out[f"segF1@{thr}"] = _f1(agg[f"tp@{thr}"], agg["n_pred"], agg["n_gold"])
        out[f"segP@{thr}"] = float(agg[f"tp@{thr}"] / max(agg["n_pred"], 1))
        out[f"segR@{thr}"] = float(agg[f"tp@{thr}"] / max(agg["n_gold"], 1))
    for tol in tols:
        out[f"startF1@{tol}"] = _f1(agg[f"bs_tp@{tol}"], agg["n_pred"], agg["n_gold"])
        out[f"endF1@{tol}"] = _f1(agg[f"be_tp@{tol}"], agg["n_pred"], agg["n_gold"])
    out["frag_gold_rate"] = float(agg["frag_gold"] / max(agg["n_gold"], 1))
    out["merge_pred_rate"] = float(agg["merge_pred"] / max(agg["n_pred"], 1))
    out["legacy_segF1"], out["legacy_meanIoU"] = agg["legacy_f1"], agg["legacy_iou"]
    if "bucket_n" in agg:
        n_arr = agg["bucket_n"]
        for b, n in enumerate(n_arr):
            out[f"n_gold_bucket{b}"] = float(n)
        # recall_bucket{b} kept as-is (IoU>=0.5) for backward compatibility with
        # existing dashboards/scripts; recall_bucket{b}@{thr} adds every other
        # requested threshold, and recall_bucket{b}_bs@{tol} adds the ABSOLUTE-
        # frame boundary-match view -- comparing these two families for the
        # SAME bucket is how you tell "genuinely missed" from "an artifact of
        # a fixed IoU threshold being harsher on short segments".
        for thr in iou_thrs:
            tp_arr = agg[f"bucket_tp@{thr}"]
            for b, (n, t) in enumerate(zip(n_arr, tp_arr)):
                out[f"recall_bucket{b}@{thr}"] = float(t / n) if n else float("nan")
                if thr == 0.5:
                    out[f"recall_bucket{b}"] = out[f"recall_bucket{b}@{thr}"]
        for tol in tols:
            tp_arr = agg[f"bucket_bs_tp@{tol}"]
            for b, (n, t) in enumerate(zip(n_arr, tp_arr)):
                out[f"recall_bucket{b}_bs@{tol}"] = float(t / n) if n else float("nan")
    return out


def bootstrap_ci(stats_list, n_boot=1000, seed=0, alpha=0.05, groups=None, **summ_kw):
    """Percentile CI from resampling. With `groups` (one label per video), whole
    GROUPS are resampled together. Use the recording/document id: participant A
    and B of one recording share session, topic, annotator conventions and video
    segmentation, and the dataset split itself is by document, so resampling
    A and B independently understates the uncertainty (CIs come out too narrow).
    Frames within a video are far from independent, so never resample frames."""
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [[i] for i in range(len(stats_list))]
    else:
        by_group = {}
        for i, g in enumerate(groups):
            by_group.setdefault(g, []).append(i)
        members = list(by_group.values())
    n = len(members)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = [i for k in pick for i in members[k]]
        draws.append(summarize(aggregate([stats_list[i] for i in idx]), **summ_kw))
    keys = draws[0].keys()
    return {k: (float(np.nanpercentile([d[k] for d in draws], 100 * alpha / 2)),
                float(np.nanpercentile([d[k] for d in draws], 100 * (1 - alpha / 2)))) for k in keys}


def classify_misses(pred, gold, iou_thr=0.5):
    """For every GOLD segment, classifies what the prediction did with it:
      'matched'     -- one-to-one IoU >= iou_thr (see match_one_to_one)
      'dropped'     -- no predicted segment overlaps it at all
      'merged'      -- exactly one predicted segment overlaps it, and that
                       prediction ALSO overlaps >=1 other gold segment (a
                       predicted span swallowing multiple signs)
      'fragmented'  -- >=2 predicted segments overlap it (over-segmented)
      'poor_iou'    -- exactly one overlapping prediction, one-to-one with
                       THIS gold segment, but IoU < iou_thr (imprecise, not
                       structurally wrong)
    Returns a list of labels, one per gold segment, in gold's order. This is
    diagnostic, not a scoring metric -- it explains a miss, it doesn't grade one.
    """
    matched_gold = {j for _, j, _ in match_one_to_one(pred, gold, iou_thr)}
    pred_overlaps = [[] for _ in pred]     # pred_overlaps[i] = list of gold idx i overlaps
    gold_overlaps = [[] for _ in gold]     # gold_overlaps[j] = list of pred idx overlapping j
    for i, j, _ in overlapping_pairs(pred, gold):
        pred_overlaps[i].append(j)
        gold_overlaps[j].append(i)

    labels = []
    for j in range(len(gold)):
        if j in matched_gold:
            labels.append("matched")
        elif len(gold_overlaps[j]) == 0:
            labels.append("dropped")
        elif len(gold_overlaps[j]) >= 2:
            labels.append("fragmented")
        else:
            i = gold_overlaps[j][0]
            labels.append("merged" if len(pred_overlaps[i]) >= 2 else "poor_iou")
    return labels


def begin_confidence(logp, gold, tol=2):
    """For each gold segment (start, end), the encoder's PEAK P(Begin) within
    [start-tol, start+tol] -- did the encoder have meaningful Begin signal near
    the true onset at all, independent of what any decoder then did with it.
    logp: (T, 3) log-probs. Returns one float per gold segment, gold's order."""
    logp = np.asarray(logp)
    T = len(logp)
    out = []
    for s, _ in gold:
        lo, hi = max(0, s - tol), min(T, s + tol + 1)
        out.append(float(np.exp(logp[lo:hi, B]).max()) if hi > lo else float("nan"))
    return out


def short_sign_report(pred_by_decoder, logp, gold, bucket_edges, tol=2, iou_thr=0.5):
    """Ties classify_misses + begin_confidence together for one video, bucketed
    by gold sign duration. pred_by_decoder: dict decoder_name -> BIO array (all
    scored against the SAME gold/logp). Returns {bucket_idx: {"n": int,
    "mean_peak_pB": float, decoder_name: {label: count}}}."""
    gold = list(gold)
    durs = np.array([e - s for s, e in gold])
    nb = len(bucket_edges) - 1
    buckets = np.clip(np.searchsorted(bucket_edges, durs, side="right") - 1, 0, nb - 1)
    conf = begin_confidence(logp, gold, tol)

    report = {b: {"n": 0, "peak_pB": [], **{name: {} for name in pred_by_decoder}} for b in range(nb)}
    for j, b in enumerate(buckets):
        report[b]["n"] += 1
        report[b]["peak_pB"].append(conf[j])
    for name, pred_bio in pred_by_decoder.items():
        pred_segs = bio_to_segments(pred_bio)
        labels = classify_misses(pred_segs, gold, iou_thr)
        for j, b in enumerate(buckets):
            report[b][name][labels[j]] = report[b][name].get(labels[j], 0) + 1
    for b in report:
        pB = report[b].pop("peak_pB")
        report[b]["mean_peak_pB"] = float(np.mean(pB)) if pB else float("nan")
    return report


def paired_bootstrap_diff(stats_a, stats_b, groups=None, n_boot=2000, seed=0, alpha=0.05, **summ_kw):
    """CI for summary_metric(A) - summary_metric(B), resampling the SAME document
    indices for both decoders in every draw. This is the right tool when A and B
    were scored on the SAME videos (always true here): their errors are
    correlated (same encoder, same hard/easy videos), and pairing removes that
    shared, cross-video noise instead of letting it inflate two separate marginal
    CIs that then get compared by eye. `stats_a`/`stats_b` must be per_video_stats
    lists from M.evaluate() IN THE SAME VIDEO ORDER."""
    assert len(stats_a) == len(stats_b)
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [[i] for i in range(len(stats_a))]
    else:
        by_group = {}
        for i, g in enumerate(groups):
            by_group.setdefault(g, []).append(i)
        members = list(by_group.values())
    n = len(members)
    diffs = {}
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = [i for k in pick for i in members[k]]
        sa = summarize(aggregate([stats_a[i] for i in idx]), **summ_kw)
        sb = summarize(aggregate([stats_b[i] for i in idx]), **summ_kw)
        for k in sa:
            if isinstance(sa[k], float):
                diffs.setdefault(k, []).append(sa[k] - sb[k])
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))),
                float(np.mean(np.array(v) > 0))) for k, v in diffs.items()}  # (lo, hi, P(A > B))


def evaluate(preds, golds, bucket_edges=None, iou_thrs=(0.3, 0.5, 0.7), tols=(2, 5), n_boot=0, group_fn=None):
    """preds/golds: dict vid -> BIO array. group_fn maps a video id to its
    recording/document id for the bootstrap. Returns (summary, per_video_stats, ci|None)."""
    vids = sorted(golds)
    stats = [video_stats(preds[v], golds[v], iou_thrs, tols, bucket_edges) for v in vids]
    summ = summarize(aggregate(stats), iou_thrs, tols)
    ci = None
    if n_boot:
        groups = [group_fn(v) for v in vids] if group_fn else None
        ci = bootstrap_ci(stats, n_boot, groups=groups, iou_thrs=iou_thrs, tols=tols)
    return summ, stats, ci