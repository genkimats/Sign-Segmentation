"""
hierarchy_phrase/sign_tokens.py -- numpy-only: greedy sign decoding and the per-sign token builder.

A token describes ONE sign k = [s_k, e_k) and its surroundings. Feature groups (selectable, so ablation 3 and the
out-of-fold variant are just different `groups`):
  h_pool     mean and max of Stage-A frame features h_t over the sign                     (2*H)
  sign_probs mean/max of the Stage-A sign B/I/O probabilities over the sign + probs at its first/last frame  (12)
  prosody    duration cues (final lengthening), hold/pause cues at the sign edges, hand-drop cues,
             head movement, and the gap before/after the sign                              (PROSODY_DIM)
`h_pool` is only meaningful when the SAME Stage-A model produced the features for training and test. Features from
different fold models live in unrelated spaces, so the out-of-fold variant must use `sign_probs` + `prosody` only.
"""
import numpy as np

SIGN_PROB_DIM = 12
PROSODY_NAMES = (
    ["log_dur", "rel_dur"]
    + ["log_gap_next", "has_gap_next", "log_gap_prev", "rel_gap_next"]
    + ["speed_first", "speed_last", "speed_mean", "speed_min", "speed_in_gap_next"]
    + ["lwrist_y_start", "lwrist_y_end", "lwrist_y_mean", "rwrist_y_start", "rwrist_y_end", "rwrist_y_mean",
       "lwrist_y_gap_next", "rwrist_y_gap_next"]
    + ["nose_speed_mean", "nose_speed_gap_next"]
)
PROSODY_DIM = len(PROSODY_NAMES)

L_WRIST, R_WRIST, NOSE = 15, 16, 0
LH, RH = slice(23, 44), slice(44, 65)


# ------------------------------------------------------------------ greedy sign decoding
def _run_peaks(mask, score):
    """Index of the maximum `score` inside each maximal run of True in `mask`."""
    peaks, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            peaks.append(i + int(np.argmax(score[i:j])))
            i = j
        else:
            i += 1
    return peaks


def greedy_decode(probs, b_thr=0.5, o_thr=0.5, min_len=1):
    """probs: (T, 3) softmax over (O, I, B). A segment OPENS at the peak of every run where P(B) > b_thr
    (a lone threshold would re-open it on every frame of the run) and CLOSES at the first frame where
    P(O) > o_thr, or where the next segment opens. Mirrors the 2023 repo's thresholded decoding in spirit;
    the exact reference implementation is not reproduced here. Returns [(start, end)] end-exclusive."""
    T = probs.shape[0]
    opens = _run_peaks(probs[:, 2] > b_thr, probs[:, 2])
    segs, open_at = [], None
    nxt = 0
    for t in range(T):
        if nxt < len(opens) and t == opens[nxt]:
            if open_at is not None:
                segs.append((open_at, t))
            open_at = t
            nxt += 1
        elif open_at is not None and probs[t, 0] > o_thr:
            segs.append((open_at, t))
            open_at = None
    if open_at is not None:
        segs.append((open_at, T))
    return [(s, e) for s, e in segs if e - s >= min_len]


# ------------------------------------------------------------------ helpers
def _softmax(x):
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def logits_to_probs(logits):
    return _softmax(np.asarray(logits, dtype=np.float64)).astype(np.float32)


def speed_signals(xyz, fps):
    """Per-frame speeds in shoulder-widths per second: dominant hand (max of the two hand centroids) and nose."""
    xyz = np.asarray(xyz, dtype=np.float32)
    def centroid_speed(sl):
        c = xyz[:, sl, :].mean(1)
        v = np.zeros(len(c), dtype=np.float32)
        v[1:] = np.linalg.norm(c[1:] - c[:-1], axis=1) * fps
        return v
    hand = np.maximum(centroid_speed(LH), centroid_speed(RH))
    n = xyz[:, NOSE, :]
    nose = np.zeros(len(n), dtype=np.float32)
    nose[1:] = np.linalg.norm(n[1:] - n[:-1], axis=1) * fps
    return hand, nose


def _local_mean(x, half=5):
    x = np.asarray(x, dtype=np.float64)
    k = np.ones(2 * half + 1)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones(len(x)), k, mode="same")
    return num / np.maximum(den, 1e-9)


def _mean_or(x, default):
    return float(np.mean(x)) if len(x) else default


# ------------------------------------------------------------------ the builder
def build_tokens(segs, T, fps, xyz=None, h=None, probs=None, groups=("h_pool", "sign_probs", "prosody"),
                 edge_frames=3):
    """segs: [(s, e)] sorted, non-overlapping, within [0, T). Returns (K, D) float32 and the column layout
    {group: (start, stop)}. Needed inputs per group: h_pool -> h (T,H); sign_probs -> probs (T,3);
    prosody -> xyz (T,65,3)."""
    K = len(segs)
    cols, parts = {}, []
    if any(not (0 <= a < b <= T) for a, b in segs):
        raise ValueError("build_tokens: every segment must satisfy 0 <= start < end <= T")
    pos = 0
    S = np.array([s for s, _ in segs], dtype=np.int64) if K else np.zeros(0, np.int64)
    E = np.array([e for _, e in segs], dtype=np.int64) if K else np.zeros(0, np.int64)

    if "h_pool" in groups:
        hf = np.asarray(h, dtype=np.float32)
        cs = np.concatenate([np.zeros((1, hf.shape[1]), np.float64), np.cumsum(hf, axis=0, dtype=np.float64)])
        mean = ((cs[E] - cs[S]) / np.maximum(E - S, 1)[:, None]).astype(np.float32)
        mx = np.stack([hf[s:e].max(0) for s, e in segs]) if K else np.zeros((0, hf.shape[1]), np.float32)
        part = np.concatenate([mean, mx], axis=1)
        cols["h_pool"] = (pos, pos + part.shape[1]); pos += part.shape[1]; parts.append(part)

    if "sign_probs" in groups:
        pr = np.asarray(probs, dtype=np.float32)
        feats = []
        for s, e in segs:
            seg = pr[s:e]
            feats.append(np.concatenate([seg.mean(0), seg.max(0), pr[s], pr[e - 1]]))
        part = np.stack(feats).astype(np.float32) if K else np.zeros((0, SIGN_PROB_DIM), np.float32)
        cols["sign_probs"] = (pos, pos + SIGN_PROB_DIM); pos += SIGN_PROB_DIM; parts.append(part)

    if "prosody" in groups:
        hand_sp, nose_sp = speed_signals(xyz, fps)
        xyz = np.asarray(xyz, dtype=np.float32)
        dur = (E - S).astype(np.float64)
        gap_next = np.zeros(K); gap_prev = np.zeros(K)
        if K:
            gap_next[:-1] = S[1:] - E[:-1]
            gap_next[-1] = max(0, T - E[-1])
            gap_prev[1:] = S[1:] - E[:-1]
            gap_prev[0] = S[0]
        rel_dur = dur / np.maximum(_local_mean(dur), 1e-6) if K else dur
        rel_gap = gap_next / np.maximum(_local_mean(gap_next), 1e-6) if K else gap_next
        lw, rw = xyz[:, L_WRIST, 1], xyz[:, R_WRIST, 1]          # y grows downwards in image coordinates
        rows = []
        for k, (s, e) in enumerate(segs):
            n = min(edge_frames, e - s)
            g0, g1 = e, int(min(T, e + max(int(gap_next[k]), 0)))
            sp = hand_sp[s:e]
            rows.append([
                np.log1p(dur[k]), rel_dur[k],
                np.log1p(gap_next[k]), float(gap_next[k] > 0), np.log1p(gap_prev[k]), rel_gap[k],
                float(hand_sp[s:s + n].mean()), float(hand_sp[e - n:e].mean()), float(sp.mean()), float(sp.min()),
                _mean_or(hand_sp[g0:g1], float(hand_sp[min(e, T - 1)])),
                float(lw[s]), float(lw[e - 1]), float(lw[s:e].mean()),
                float(rw[s]), float(rw[e - 1]), float(rw[s:e].mean()),
                _mean_or(lw[g0:g1], float(lw[e - 1])), _mean_or(rw[g0:g1], float(rw[e - 1])),
                float(nose_sp[s:e].mean()), _mean_or(nose_sp[g0:g1], float(nose_sp[min(e, T - 1)])),
            ])
        part = np.array(rows, dtype=np.float32) if K else np.zeros((0, PROSODY_DIM), np.float32)
        assert part.shape[1] == PROSODY_DIM
        cols["prosody"] = (pos, pos + PROSODY_DIM); pos += PROSODY_DIM; parts.append(part)

    X = np.concatenate(parts, axis=1) if parts else np.zeros((K, 0), np.float32)
    return X.astype(np.float32), cols