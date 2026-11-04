"""
trace_core.py -- numpy-only analysis core of the fusion study (tested by test_trace_core.py).

Events (all from GOLD labels, never from a model)
  phrase starts       typed by what precedes them: continuous / short_pause / long_pause
                      (src.metrics.classify_gold_boundaries; frame 0 excluded)
  sign starts         the KEY CONTROL: starts of signs (gloss labels) INSIDE a phrase, at least
                      `exclusion` frames away from every phrase start / end -- a signal that is
                      equally strong here only marks "a new sign", not "a new phrase"
  inside frames       random frames inside phrases, `exclusion` away from phrase boundaries

Analyses
  locked_average      event-locked mean (and SEM) of a per-frame signal at offsets -L..L
  lag_distribution    mean attention over lag offsets at a set of frames
  detection_matrix    which gold phrase starts each stream detects (one-to-one, +-k frames)
  complementarity     per-stream recall, union recall, exclusive recall, pairwise Jaccard
"""
import numpy as np

O_TAG, I_TAG, B_TAG = 0, 1, 2
PHRASE_TYPES = ("continuous", "short_pause", "long_pause")


def _starts(bio):
    """Segment starts with src.metrics.extract_segments semantics (B, or I after O / at frame 0)."""
    bio = np.asarray(bio).astype(np.int64)
    if len(bio) == 0:
        return np.zeros(0, np.int64)
    prev = np.r_[O_TAG, bio[:-1]]
    return np.flatnonzero((bio == B_TAG) | ((bio == I_TAG) & (prev == O_TAG)))


def _segments(bio):
    bio = np.asarray(bio).astype(np.int64)
    starts = _starts(bio)
    out = []
    T = len(bio)
    for i, s in enumerate(starts):
        e = s
        nxt = starts[i + 1] if i + 1 < len(starts) else T
        while e + 1 < nxt and bio[e + 1] != O_TAG:
            e += 1
        out.append((int(s), int(e)))
    return out


def phrase_start_events(gold, short_gap=12):
    """[(frame, type)] for every gold phrase start except one at frame 0."""
    gold = np.asarray(gold).astype(np.int64)
    out = []
    for s, _ in _segments(gold):
        if s == 0:
            continue
        if gold[s - 1] != O_TAG:
            out.append((s, "continuous"))
        else:
            k = s - 1
            while k >= 0 and gold[k] == O_TAG:
                k -= 1
            out.append((s, "short_pause" if (s - 1 - k) <= short_gap else "long_pause"))
    return out


def _boundary_distance(gold):
    """Per frame: distance (frames) to the nearest phrase start or end."""
    T = len(gold)
    marks = np.zeros(T, bool)
    for s, e in _segments(gold):
        marks[s] = True
        marks[e] = True
    if not marks.any():
        return np.full(T, np.inf)
    idx = np.flatnonzero(marks)
    t = np.arange(T)
    pos = np.searchsorted(idx, t)
    left = np.where(pos > 0, t - idx[np.clip(pos - 1, 0, len(idx) - 1)], np.inf)
    right = np.where(pos < len(idx), idx[np.clip(pos, 0, len(idx) - 1)] - t, np.inf)
    return np.minimum(left, right)


def sign_start_controls(gold_phrase, gold_sign, exclusion=10):
    """Sign starts inside a phrase, >= exclusion frames from any phrase start / end."""
    gp = np.asarray(gold_phrase).astype(np.int64)
    gs = np.asarray(gold_sign).astype(np.int64)[:len(gp)]
    dist = _boundary_distance(gp)
    starts = _starts(gs)
    return [int(s) for s in starts if gp[s] != O_TAG and dist[s] >= exclusion]


def inside_frames(gold_phrase, n, rng, exclusion=10):
    gp = np.asarray(gold_phrase).astype(np.int64)
    cand = np.flatnonzero((gp != O_TAG) & (_boundary_distance(gp) >= exclusion))
    if len(cand) == 0:
        return []
    return sorted(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())


def locked_average(signals, events, half_window):
    """
    signals: list of per-video arrays (T,) or (T, C); events: list (same order) of frame lists.
    Returns mean (2L+1, C), sem (2L+1, C), n (2L+1,) -- offsets that fall outside a video are
    simply missing for that event (not padded).
    """
    L = int(half_window)
    first = np.asarray(signals[0])
    C = 1 if first.ndim == 1 else first.shape[1]
    s1 = np.zeros((2 * L + 1, C))
    s2 = np.zeros((2 * L + 1, C))
    n = np.zeros(2 * L + 1)
    for sig, evs in zip(signals, events):
        sig = np.asarray(sig, dtype=np.float64)
        if sig.ndim == 1:
            sig = sig[:, None]
        T = sig.shape[0]
        for e in evs:
            lo, hi = max(0, e - L), min(T, e + L + 1)
            if lo >= hi:
                continue
            j0 = lo - (e - L)
            seg = sig[lo:hi]
            s1[j0:j0 + len(seg)] += seg
            s2[j0:j0 + len(seg)] += seg ** 2
            n[j0:j0 + len(seg)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n[:, None]
        var = np.maximum(s2 / n[:, None] - mean ** 2, 0.0)
        sem = np.sqrt(var / np.maximum(n[:, None] - 1, 1))
    return mean, sem, n


def lag_distribution(attn_list, events=None):
    """attn_list: per-video (T, 2K+1) attention over offsets; mean over the event frames
    (all frames if events is None). Returns (2K+1,) and the number of frames used."""
    acc, cnt = None, 0
    for i, a in enumerate(attn_list):
        a = np.asarray(a, dtype=np.float64)
        sel = a if events is None else a[[e for e in events[i] if 0 <= e < len(a)]]
        if len(sel) == 0:
            continue
        acc = sel.sum(0) if acc is None else acc + sel.sum(0)
        cnt += len(sel)
    if acc is None:
        return None, 0
    return acc / cnt, cnt


def match_one_to_one(gold_pos, pred_pos, k):
    """Gold starts matched to predicted starts within +-k frames, closest pairs first
    (the same rule as src.metrics.match_boundaries). Returns a bool array over gold_pos."""
    gold_pos = np.asarray(gold_pos, dtype=np.int64)
    pred_pos = np.asarray(pred_pos, dtype=np.int64)
    ok = np.zeros(len(gold_pos), bool)
    if len(gold_pos) == 0 or len(pred_pos) == 0:
        return ok
    d = np.abs(gold_pos[:, None] - pred_pos[None, :])
    gi, pj = np.nonzero(d <= k)
    order = np.argsort(d[gi, pj], kind="stable")
    used_p = np.zeros(len(pred_pos), bool)
    for o in order:
        g, p = gi[o], pj[o]
        if not ok[g] and not used_p[p]:
            ok[g] = True
            used_p[p] = True
    return ok


def detection_matrix(gold_starts, pred_starts_by_stream, k=5):
    """(n_gold, S) bool: does stream s detect gold start i (one-to-one within +-k)?"""
    return np.stack([match_one_to_one(gold_starts, p, k) for p in pred_starts_by_stream], axis=1) \
        if len(pred_starts_by_stream) else np.zeros((len(gold_starts), 0), bool)


def complementarity(D, names, types=None):
    """
    D: (n_gold, S) detection matrix pooled over videos; types: optional (n_gold,) boundary types.
    Returns a list of rows: per stream recall / exclusive recall, the union, and pairwise Jaccard
    of the detected sets; with types, the same per boundary type.
    """
    rows = []
    groups = [("all", np.ones(len(D), bool))]
    if types is not None:
        types = np.asarray(types)
        groups += [(t, types == t) for t in PHRASE_TYPES if (types == t).any()]
    for gname, m in groups:
        Dm = D[m]
        n = len(Dm)
        if n == 0:
            continue
        any_ = Dm.any(1)
        for j, s in enumerate(names):
            others = np.delete(Dm, j, axis=1).any(1) if Dm.shape[1] > 1 else np.zeros(n, bool)
            rows.append({"group": gname, "what": f"recall {s}", "value": float(Dm[:, j].mean()), "n": n})
            rows.append({"group": gname, "what": f"only {s}", "value": float((Dm[:, j] & ~others).mean()), "n": n})
        rows.append({"group": gname, "what": "union recall", "value": float(any_.mean()), "n": n})
        rows.append({"group": gname, "what": "missed by all", "value": float((~any_).mean()), "n": n})
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                inter = (Dm[:, a] & Dm[:, b]).sum()
                uni = (Dm[:, a] | Dm[:, b]).sum()
                rows.append({"group": gname, "what": f"jaccard {names[a]}|{names[b]}",
                             "value": float(inter / uni) if uni else float("nan"), "n": n})
    return rows
