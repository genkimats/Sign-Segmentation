"""
hierarchy_phrase/segments.py -- numpy-only segment algebra (unit-tested in test_cores.py).

Label convention of this project (confirmed against the data): 0 = Outside, 1 = Inside, 2 = Begin.
A segment is (start, end) with END EXCLUSIVE. Sign segments and phrase segments use the same type.
"""
import numpy as np

O, I, B = 0, 1, 2


# ------------------------------------------------------------------ BIO <-> segments
def bio_to_segments(bio):
    """B opens a segment (closing an open one), O closes it, a stray I after O is ignored --
    identical to the rule used by src/metrics.py and decoder_study/study_metrics.py."""
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


def segments_to_bio(segs, T):
    bio = np.zeros(T, dtype=np.int8)
    for s, e in segs:
        s, e = max(0, int(s)), min(T, int(e))
        if e <= s:
            continue
        bio[s] = B
        bio[s + 1:e] = I
    return bio


# ------------------------------------------------------------------ frame-rate change
def _rnd(x, stride):
    return int(np.floor(x / stride + 0.5))


def resample_segments(segs, stride):
    """Map segments to a frame rate `stride` times lower. The SAME rounding is applied to every
    boundary, so a phrase edge that coincided with a sign edge still coincides afterwards, and
    touching segments stay touching. Segments that collapse to zero length are dropped."""
    if stride == 1:
        return [(int(s), int(e)) for s, e in segs]
    out = []
    for s, e in segs:
        s2, e2 = _rnd(s, stride), _rnd(e, stride)
        if e2 > s2:
            out.append((s2, e2))
    return out


def working_length(T, stride):
    return -(-int(T) // int(stride))          # = len(x[::stride])


# ------------------------------------------------------------------ signs -> phrase tags
TAG_RULES = ("first_end_after", "nearest_start", "contain_or_next")


def phrase_tags_over_signs(sign_segs, phrase_segs, rule="first_end_after"):
    """Per-sign tag: 1 = this sign starts a new phrase (B), 0 = continues one (I). The first sign of a video is always B.
    Each gold phrase START p is assigned to exactly one sign. When phrase starts sit exactly on sign starts every rule gives
    the same answer; they differ when a label lands INSIDE a sign (which happens for ~40% of starts in this project's phrase
    labels, see data_audit.py):
      first_end_after  the first sign that ends after p                          (a label inside sign k -> k)
      nearest_start    the sign whose start is closest to p
      contain_or_next  p strictly inside sign k: k if p is in its first half, else k+1; p in a gap: the next sign
    data_audit.py measures how well each rule lets GOLD signs reproduce GOLD phrases (the ceiling of the hierarchy)."""
    if rule not in TAG_RULES:
        raise ValueError(f"rule must be one of {TAG_RULES}")
    K = len(sign_segs)
    tags = np.zeros(K, dtype=np.int64)
    if K == 0:
        return tags
    starts = np.array([s for s, _ in sign_segs])
    ends = np.array([e for _, e in sign_segs])
    for ps, _ in phrase_segs:
        if rule == "first_end_after":
            k = int(np.searchsorted(ends, ps, side="right"))
        elif rule == "nearest_start":
            j = int(np.searchsorted(starts, ps))
            k = min((c for c in (j - 1, j) if 0 <= c < K), key=lambda c: abs(int(starts[c]) - ps))
        else:
            j = int(np.searchsorted(starts, ps, side="right")) - 1          # last sign starting at or before ps
            if j >= 0 and starts[j] < ps < ends[j]:
                k = j if (ps - starts[j]) <= (ends[j] - starts[j]) / 2 else j + 1
            elif j >= 0 and starts[j] == ps:
                k = j
            else:
                k = j + 1
        if 0 <= k < K:
            tags[k] = 1
    tags[0] = 1
    return tags


def phrases_from_tags(sign_segs, tags):
    """B sign opens a phrase; it lasts to the end of the last sign before the next B."""
    phr, cur = [], None
    for k, (s, e) in enumerate(sign_segs):
        if tags[k] == 1 or cur is None:
            if cur is not None:
                phr.append(tuple(cur))
            cur = [s, e]
        else:
            cur[1] = e
    if cur is not None:
        phr.append(tuple(cur))
    return phr


# ------------------------------------------------------------------ jitter (train/test mismatch)
def jitter_segments(segs, T, rng, strength=1.0, max_shift=3):
    """Simulate a sign segmenter's mistakes on gold signs: boundary shifts, merges of neighbours,
    splits, drops. `strength` in [0, 1] scales every probability (0 -> unchanged = gold).
    Output is sorted, non-overlapping, every segment >= 1 frame."""
    p_shift, p_merge, p_split, p_drop = 0.5 * strength, 0.06 * strength, 0.06 * strength, 0.02 * strength
    segs = [list(s) for s in segs]
    out = []
    for s, e in segs:
        if rng.random() < p_drop:
            continue
        if out and rng.random() < p_merge:
            out[-1][1] = e
            continue
        out.append([s, e])
    res = []
    for s, e in out:
        if e - s >= 4 and rng.random() < p_split:
            m = int(rng.integers(s + 2, e - 1))
            res.append([s, m])
            res.append([m, e])
        else:
            res.append([s, e])
    for seg in res:                                  # boundary shifts
        if rng.random() < p_shift:
            seg[0] += int(rng.integers(-max_shift, max_shift + 1))
        if rng.random() < p_shift:
            seg[1] += int(rng.integers(-max_shift, max_shift + 1))
    fixed, prev_end = [], 0
    for s, e in res:                                 # enforce validity
        s = max(s, prev_end, 0)
        e = min(e, T)
        if e <= s:
            continue
        fixed.append((s, e))
        prev_end = e
    return fixed


# ------------------------------------------------------------------ windows over token sequences
def plan_windows(T, window, stride):
    if T <= window:
        return [(0, T)]
    starts = list(range(0, T - window + 1, stride))
    if starts[-1] + window < T:
        starts.append(T - window)
    return [(s, s + window) for s in starts]


def stitch_probs(T, windows, window_probs):
    """Average overlapping per-token probabilities back to one length-T array."""
    acc, cnt = np.zeros(T), np.zeros(T)
    for (s, e), p in zip(windows, window_probs):
        acc[s:e] += p[:e - s]
        cnt[s:e] += 1
    if not (cnt > 0).all():
        raise ValueError("stitch_probs: tokens not covered by any window")
    return acc / cnt