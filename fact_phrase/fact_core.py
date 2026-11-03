"""
fact_core.py -- torch-free algorithmic core of the FACT phrase pilot.

Everything here is plain numpy, so test_fact_core.py can check it against brute-force
references without torch. The torch model (fact_model.py) and the training / evaluation
scripts call these functions; nothing else in the pilot re-implements them.

Label convention (project-wide): 0 = O, 1 = I, 2 = B (frame level).
FACT token classes:              0 = O-segment, 1 = phrase, 2 = null (no segment).

Contents
  Segments     bio_to_segments       gold BIO -> (segment id per frame, class per segment)
               collapse_b_runs       B B I -> B I I (a run of B frames is ONE start)
               tdu_segments          predicted BIO -> [(start, end)] for FACT's temporal pooling
               count_segments        number of FACT segments (O runs + phrases) in a BIO window
  Matching     soft_iou              FACT's token/segment soft IoU, closed form (no T x A x S tensor)
  Decoding     token_decode          FACT's action-token decoding, adapted so that two ADJACENT
                                     phrases stay two phrases (new phrase where the token changes)
               onehot_probs          BIO labels -> (3, T) one-hot "probabilities" for src.metrics
  Windows      tile_windows          same windows as src/evaluation.py video_windows
               owned_regions         each frame is owned by exactly one window (overlap midpoints)
               stitch_probs          average of overlapping window probabilities
               stitch_labels         labels taken from each window's owned region only
  Rates        downsample_bio / upsample_bio / upsample_probs / pool_features
  Tokens       window_segment_stats  max / mean number of segments per training window
               auto_ntoken / null_weight
"""
import numpy as np

O_TAG, I_TAG, B_TAG = 0, 1, 2
TOK_O, TOK_PHRASE, TOK_NULL = 0, 1, 2


# ============================================================================ segments
def phrase_starts_mask(bio):
    """Frames that START a phrase: every B, and an I right after O (or at frame 0).
    Same rule as src.metrics.extract_segments(start_on_i_after_o=True)."""
    bio = np.asarray(bio).astype(np.int64)
    prev = np.empty_like(bio)
    if len(bio):
        prev[0] = O_TAG
        prev[1:] = bio[:-1]
    return (bio == B_TAG) | ((bio == I_TAG) & (prev == O_TAG))


def bio_to_segments(bio):
    """
    FACT segments of a BIO sequence: every phrase (src.metrics.extract_segments semantics, so
    B always starts a new phrase -- adjacent phrases stay separate) and every maximal O run.

    Returns seg_label (T,) int64 -- segment index of each frame (0..S-1, in time order),
            transcript (S,) int64 -- TOK_O or TOK_PHRASE per segment,
            segs [(start, end)] inclusive.
    """
    bio = np.asarray(bio).astype(np.int64)
    T = len(bio)
    if T == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), []
    is_o = bio == O_TAG
    starts = phrase_starts_mask(bio)
    prev_o = np.r_[False, is_o[:-1]]
    new_seg = starts | (is_o & ~prev_o)          # a phrase start, or the first O of an O run
    new_seg[0] = True
    seg_label = np.cumsum(new_seg) - 1
    first = np.flatnonzero(new_seg)
    transcript = np.where(is_o[first], TOK_O, TOK_PHRASE).astype(np.int64)
    ends = np.r_[first[1:] - 1, T - 1]
    return seg_label.astype(np.int64), transcript, list(zip(first.tolist(), ends.tolist()))


def collapse_b_runs(bio):
    """B frames directly after a B become I: a run of B frames is one phrase start."""
    bio = np.asarray(bio).astype(np.int64).copy()
    if len(bio) > 1:
        dup = (bio[1:] == B_TAG) & (bio[:-1] == B_TAG)
        bio[1:][dup] = I_TAG
    return bio


def tdu_segments(pred_bio):
    """Segments FACT's temporal down/up-sampling block pools over, from a PREDICTED BIO argmax:
    O runs and phrases, with B-runs collapsed so a stuttering B does not create 1-frame segments."""
    _, _, segs = bio_to_segments(collapse_b_runs(pred_bio))
    return segs


def count_segments(bio):
    """Number of FACT segments (O runs + phrases) in a BIO sequence."""
    return len(bio_to_segments(bio)[1])


# ============================================================================ matching
def soft_iou(a2f_attn, seg_label, n_seg):
    """
    FACT's a2f soft IoU between every token and every gold segment (FACT MatchCriterion.a2f_soft_iou),
    in closed form. a2f_attn: (T, A), each row sums to 1 (softmax over tokens), values in [0, 1].
    seg_label: (T,) segment id per frame. Returns (A, S).

    FACT computes  overlap[a,s] = sum_t attn[t,a] * 1[t in s]
                   union[a,s]   = sum_t min(attn[t,a] + 1[t in s], 1)
    and because attn <= 1 the union is  |s| + sum_t attn[t,a] - overlap[a,s]  -- no (T, A, S) tensor.
    """
    attn = np.asarray(a2f_attn, dtype=np.float64)
    seg_label = np.asarray(seg_label, dtype=np.int64)
    A = attn.shape[1]
    overlap = np.zeros((A, n_seg), dtype=np.float64)
    np.add.at(overlap.T, seg_label, attn)                        # overlap[:, s] += attn[t] for t in s
    seg_len = np.bincount(seg_label, minlength=n_seg).astype(np.float64)
    union = seg_len[None, :] + attn.sum(0)[:, None] - overlap
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, overlap / union, 0.0)
    return iou


# ============================================================================ decoding
def _softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def phrase_marginal(frame_probs_t3):
    """(T, 3) BIO probabilities -> (T, 2) [P(O), P(I) + P(B)]."""
    p = np.asarray(frame_probs_t3)
    return np.stack([p[:, O_TAG], p[:, I_TAG] + p[:, B_TAG]], axis=1)


def token_decode(action_logits, a2f_attn, frame_logits, mwt):
    """
    FACT's decoding (Block._eval) for BIO phrases.

    action_logits (A, 3) [O, phrase, null], a2f_attn (T, A) frame->token attention of the last
    block, frame_logits (T, 3) BIO logits of the last block, mwt = weight of the frame branch.

      active tokens   = tokens whose argmax class is not null
      owner of frame  = active token with the highest attention from that frame
      P(O / phrase)   = (1 - mwt) * owner's class probability + mwt * frame-branch marginal
      a frame is inside a phrase if P(phrase) > P(O); a NEW phrase starts where the owner token
      changes (or after O) -> B. Two adjacent phrases owned by different tokens therefore stay
      two phrases, which a class-run decoding (FACT's original) would merge.
    Without any active token the frame branch's own argmax is returned.
    """
    action_logits = np.asarray(action_logits, dtype=np.float64)
    frame_probs = _softmax(frame_logits, axis=1)
    tok_cls = action_logits.argmax(1)
    active = np.flatnonzero(tok_cls != TOK_NULL)
    if len(active) == 0:
        return frame_probs.argmax(1).astype(np.int64)
    tok_prob = _softmax(action_logits[:, :2], axis=1)               # drop the null logit
    owner = active[np.asarray(a2f_attn)[:, active].argmax(1)]       # (T,)
    prob = (1.0 - mwt) * tok_prob[owner] + mwt * phrase_marginal(frame_probs)
    inside = prob[:, 1] > prob[:, 0]
    T = len(owner)
    bio = np.zeros(T, dtype=np.int64)
    bio[inside] = I_TAG
    prev_inside = np.r_[False, inside[:-1]]
    prev_owner = np.r_[-1, owner[:-1]]
    new_phrase = inside & (~prev_inside | (owner != prev_owner))
    bio[new_phrase] = B_TAG
    return bio


def onehot_probs(bio):
    """BIO labels (T,) -> (3, T) float32 one-hot, so decoded labels go through src.metrics unchanged."""
    bio = np.asarray(bio).astype(np.int64)
    out = np.zeros((3, len(bio)), dtype=np.float32)
    out[bio, np.arange(len(bio))] = 1.0
    return out


# ============================================================================ windows
def tile_windows(T, W, step):
    """Identical to src.evaluation.video_windows(T, W, overlap=W-step)."""
    if T <= W:
        return [(0, T)]
    if step <= 0:
        raise ValueError("step must be positive")
    wins = [(s, s + W) for s in range(0, T - W + 1, step)]
    if wins[-1][1] < T:
        wins.append((T - W, T))
    return wins


def owned_regions(windows, T):
    """Splits [0, T) into one region per window: the boundary between consecutive windows is the
    midpoint of their overlap (or their meeting point). Every region lies inside its window."""
    starts = [s for s, _ in windows]
    ends = [e for _, e in windows]
    cuts = [0]
    for i in range(len(windows) - 1):
        cuts.append((starts[i + 1] + ends[i]) // 2)
    cuts.append(T)
    return [(cuts[i], cuts[i + 1]) for i in range(len(windows))]


def stitch_probs(window_probs, windows, T):
    """window_probs: list of (C, e - s) arrays -> (C, T) average over overlapping windows."""
    C = window_probs[0].shape[0]
    acc = np.zeros((C, T), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    for p, (s, e) in zip(window_probs, windows):
        acc[:, s:e] += p[:, : e - s]
        cnt[s:e] += 1.0
    cnt[cnt == 0] = 1.0
    return (acc / cnt).astype(np.float32)


def stitch_labels(window_labels, windows, T):
    """window_labels: list of (e - s,) label arrays; each frame is taken from its owning window."""
    out = np.zeros(T, dtype=np.int64)
    for lab, (s, _), (a, b) in zip(window_labels, windows, owned_regions(windows, T)):
        out[a:b] = np.asarray(lab)[a - s: b - s]
    return out


# ============================================================================ frame rates
def downsample_bio(bio, k):
    """Groups of k frames -> one label: B if the group holds a phrase start, else I if it holds any
    I (or B), else O. Phrase starts survive (each in its own group unless two starts share a group)."""
    bio = np.asarray(bio).astype(np.int64)
    if k <= 1:
        return bio.copy()
    T = len(bio)
    n = -(-T // k)
    starts = phrase_starts_mask(bio)
    pad = n * k - T
    st = np.r_[starts, np.zeros(pad, bool)].reshape(n, k).any(1)
    ins = np.r_[bio != O_TAG, np.zeros(pad, bool)].reshape(n, k).any(1)
    out = np.where(st, B_TAG, np.where(ins, I_TAG, O_TAG))
    return out.astype(np.int64)


def upsample_bio(bio_low, k, T):
    """Repeat k times; a B becomes B followed by I's (one start, not k starts). Cropped to T."""
    bio_low = np.asarray(bio_low).astype(np.int64)
    if k <= 1:
        return bio_low[:T].copy()
    up = np.repeat(bio_low, k)
    rep_b = np.repeat(bio_low == B_TAG, k)
    first = np.zeros_like(rep_b)
    first[::k] = True
    up[rep_b & ~first] = I_TAG
    return up[:T]


def upsample_probs(probs_low, k, T):
    """(3, T_low) -> (3, T): repeat k times; inside each repeated block only the FIRST frame keeps
    P(B). On the other frames P(B) is moved to P(I) where B was the argmax (the phrase that just
    started continues), else split between O and I in proportion (their order is kept). So the
    argmax of the result equals upsample_bio(argmax of the input) and every column still sums to 1."""
    p = np.asarray(probs_low, dtype=np.float32)
    if k <= 1:
        return p[:, :T].copy()
    up = np.repeat(p, k, axis=1)
    not_first = np.ones(up.shape[1], dtype=bool)
    not_first[::k] = False
    b_won = np.repeat(p.argmax(0) == B_TAG, k)
    to_i = not_first & b_won
    split = not_first & ~b_won
    up[I_TAG, to_i] += up[B_TAG, to_i]
    oi = up[O_TAG, split] + up[I_TAG, split]
    share_o = np.where(oi > 0, up[O_TAG, split] / np.maximum(oi, 1e-12), 0.5)
    up[O_TAG, split] += up[B_TAG, split] * share_o
    up[I_TAG, split] += up[B_TAG, split] * (1.0 - share_o)
    up[B_TAG, not_first] = 0.0
    if up.shape[1] < T:                                          # cannot happen with ceil, kept for safety
        up = np.concatenate([up, np.repeat(up[:, -1:], T - up.shape[1], axis=1)], axis=1)
    return up[:, :T]


def pool_features(x, k):
    """(T, D) -> (ceil(T/k), D) mean over groups of k frames (last group may be shorter)."""
    x = np.asarray(x, dtype=np.float32)
    if k <= 1:
        return x
    T, D = x.shape
    n = -(-T // k)
    pad = n * k - T
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)], axis=0)
    return x.reshape(n, k, D).mean(1)


# ============================================================================ tokens
def window_segment_stats(label_list, W, step):
    """Segments per training window over every label sequence, windows tiled with `step`.
    Returns (max, mean, n_windows)."""
    counts = []
    for bio in label_list:
        for s, e in tile_windows(len(bio), W, step):
            counts.append(count_segments(bio[s:e]))
    if not counts:
        return 0, 0.0, 0
    return int(max(counts)), float(np.mean(counts)), len(counts)


def auto_ntoken(max_segments, margin=1.25, multiple=8, minimum=16):
    """Enough tokens for the busiest window (+margin), rounded up to a multiple of 8."""
    n = int(np.ceil(max(max_segments, 1) * margin / multiple) * multiple)
    return max(minimum, n)


def null_weight(ntoken, mean_segments, n_classes=2):
    """FACT's compute_null_weight: the null class gets the frequency weight 1 / n_classes."""
    num_null = max(ntoken - mean_segments, 1e-3)
    return float(ntoken / (num_null * n_classes))
