"""
Torch-free algorithmic cores. Every function takes an array namespace `xp`
(seg_ops.NP_OPS in the tests, seg_ops.get_torch_ops() in training) as its first
argument, so the code the unit tests run against brute-force references is the
very same code that trains the model.

Contents
  CRF            crf_log_partition / crf_gold_score / crf_viterbi
  xLSTM          mlstm_parallel (matrix memory, stabilised parallel form)
                 slstm_scan     (scalar memory, exponential gating, stabilised)
  Similarity     cosine_ssm / row_similarity / make_novelty_bands /
                 checkerboard_novelty  (Foote-style novelty on a self-similarity
                 matrix, plus TransNetV2-style per-frame similarity rows)

Label convention (project-wide): 0 = Outside, 1 = Inside, 2 = Begin.
"""
import numpy as np

O, I, B = 0, 1, 2
NEG = -1e30   # finite stand-in for -inf: keeps one-hot products NaN-free


# ===================================================================== CRF ==
def forbid_mask(num_tags=3, forbidden=((O, I),)):
    m = np.zeros((num_tags, num_tags))
    for a, b in forbidden:
        m[a, b] = 1.0
    return m


def crf_log_partition(xp, E, trans, start, end, mask=None):
    """log Z(x) of a linear-chain CRF by the forward algorithm.
    E (B,T,K) emission scores, trans (K,K) [from,to], start/end (K,).
    mask: optional static numpy bool (B,T), True on real frames, prefix-shaped.
    Returns (B,)."""
    T = E.shape[1]
    alpha = start[None, :] + E[:, 0]
    for t in range(1, T):
        nxt = xp.logsumexp(alpha[:, :, None] + trans[None, :, :], 1) + E[:, t]
        alpha = nxt if mask is None else xp.where(mask[:, t][:, None], nxt, alpha)
    return xp.logsumexp(alpha + end[None, :], 1)


def crf_gold_score(xp, E, y, trans, start, end, mask=None):
    """Unnormalised score of the gold path y (B,T) ints. Returns (B,)."""
    B_, T, K = E.shape
    oh = xp.onehot(y, K, E)                                    # (B,T,K)
    m = np.ones((B_, T)) if mask is None else np.asarray(mask, dtype=np.float64)
    mt = xp.asarray(m, E)                                      # (B,T)
    emit = xp.sum(xp.sum(E * oh, 2) * mt, 1)
    first = oh[:, 0] @ start
    if T > 1:
        pair = xp.sum((oh[:, :-1] @ trans) * oh[:, 1:], 2)     # (B,T-1)
        tr = xp.sum(pair * mt[:, 1:], 1)
    else:
        tr = 0.0 * first
    nxt = np.concatenate([m[:, 1:], np.zeros((B_, 1))], axis=1)
    last = xp.asarray(m - nxt, E)                              # 1 at last real frame
    endsc = xp.sum((oh @ end) * last, 1)
    return emit + first + tr + endsc


def crf_nll_per_frame(xp, E, y, trans, start, end, mask=None):
    """Mean negative log-likelihood per real frame (scalar)."""
    nll = crf_log_partition(xp, E, trans, start, end, mask) - crf_gold_score(xp, E, y, trans, start, end, mask)
    n_frames = E.shape[0] * E.shape[1] if mask is None else float(np.asarray(mask).sum())
    return xp.sum(nll, 0) / n_frames


def crf_viterbi(xp, E, trans, start, end):
    """Best path (B,T) under the CRF scores. Full-length sequences only."""
    T = E.shape[1]
    delta = start[None, :] + E[:, 0]
    bps = []
    for t in range(1, T):
        cand = delta[:, :, None] + trans[None, :, :]           # (B,from,to)
        bps.append(xp.argmax(cand, 1))                         # (B,to)
        delta = xp.amax(cand, 1) + E[:, t]
    state = xp.argmax(delta + end[None, :], 1)
    path = [state]
    for bp in reversed(bps):
        state = xp.take_along_row(bp, state)
        path.append(state)
    path.reverse()
    return xp.stack(path, 1)


# ================================================================== mLSTM ==
def mlstm_parallel(xp, q, k, v, i_pre, f_pre):
    """Stabilised parallel form of the xLSTM matrix-memory cell (causal).
    q,k,v (B,H,T,dh); i_pre,f_pre (B,H,T) gate pre-activations. Returns
    (B,H,T,dh). Mathematically identical to the recurrence
        C_t = f_t C_{t-1} + i_t v_t k_t^T,  n_t = f_t n_{t-1} + i_t k_t,
        h_t = C_t q_t / max(|n_t . q_t|, 1)     with i_t = exp(i_pre), f_t = sigmoid(f_pre)
    (k pre-scaled by 1/sqrt(dh)), but computed with a log-space stabiliser m_t
    so large exponential input gates cannot overflow."""
    T, dh = q.shape[2], q.shape[3]
    logf = xp.logsigmoid(f_pre)
    F = xp.cumsum(logf, 2)                                     # (B,H,T)
    logD = F[:, :, :, None] - F[:, :, None, :] + i_pre[:, :, None, :]   # [t, s]
    causal = np.tril(np.ones((T, T), dtype=bool))
    logD = xp.where(causal, logD, xp.full_like(logD, NEG))
    m = xp.amax(logD, 3, keepdims=True)                        # (B,H,T,1)
    D = xp.exp(logD - m)
    S = (q @ xp.swap_last(k)) / float(np.sqrt(dh)) * D
    b = xp.sum(S, 3, keepdims=True)
    denom = xp.maximum(xp.abs(b), xp.exp(-m))
    return (S / denom) @ v


# ================================================================== sLSTM ==
def slstm_scan(xp, pre, R):
    """Stabilised sLSTM recurrence (scalar memory, exponential input gate,
    recurrent memory mixing within heads). Causal.
    pre: (B,T,4,H,dh) input pre-activations for gates (z, i, f, o).
    R:   (4,H,dh,dh) per-head recurrent weights.
    Returns hidden states (B,T,H,dh)."""
    B_, T, _, H, dh = pre.shape
    h = xp.zeros((B_, H, dh), pre)
    c = xp.zeros((B_, H, dh), pre)
    n = xp.zeros((B_, H, dh), pre)
    m = xp.zeros((B_, H, dh), pre)
    outs = []
    for t in range(T):
        a = pre[:, t] + xp.einsum('bhd,ghde->bghe', h, R)      # (B,4,H,dh)
        z = xp.tanh(a[:, 0])
        i_t = a[:, 1]
        f_t = a[:, 2]
        o = xp.sigmoid(a[:, 3])
        logf = xp.logsigmoid(f_t)
        m_new = xp.maximum(logf + m, i_t)
        i_p = xp.exp(i_t - m_new)
        f_p = xp.exp(logf + m - m_new)
        c = f_p * c + i_p * z
        n = f_p * n + i_p
        h = o * c / n
        m = m_new
        outs.append(h)
    return xp.stack(outs, 1)


# =============================================================== similarity ==
def cosine_ssm(xp, e, eps=1e-8):
    """e (B,T,D) -> cosine self-similarity matrix (B,T,T)."""
    en = e / xp.sqrt(xp.sum(e * e, 2, keepdims=True) + eps)
    return en @ xp.swap_last(en)


def row_similarity(xp, S, K):
    """TransNetV2-style per-frame similarity vector: for frame t, the cosine
    similarity to frames t-K..t+K (offset axis). Out-of-range offsets are 0.
    S (B,T,T) -> (B,T,2K+1)."""
    T = S.shape[1]
    raw = np.arange(T)[:, None] + np.arange(-K, K + 1)[None, :]
    valid = ((raw >= 0) & (raw < T)).astype(np.float64)
    idx = np.clip(raw, 0, T - 1)
    return xp.gather_last(S, idx) * xp.asarray(valid, S)[None]


def make_novelty_bands(T, L):
    """(T,T) banded matrix A_L with A[t, t+i] = a(i) for 1<=|i|<=L,
    a(i) = sign(i) * Gaussian(i; sigma=L/2), normalised so sum|a| = 1.
    Then novelty_t = sum_{p,q} A[t,p] S[p,q] A[t,q] is the checkerboard-kernel
    correlation of Foote (2000) along the diagonal of S: it is large when the
    frames just before t are mutually similar, the frames just after t are
    mutually similar, and the two groups are dissimilar to each other.

    Rows within L frames of either end of the sequence are left at zero. A
    truncated kernel has lost one side, which would otherwise produce a
    spurious positive response at the start and end of every window (0.25 for a
    perfectly uniform signal); zero means "no evidence" instead."""
    offs = np.array([i for i in range(-L, L + 1) if i != 0])
    a = np.sign(offs) * np.exp(-0.5 * (offs / (L / 2.0)) ** 2)
    a = a / np.abs(a).sum()
    A = np.zeros((T, T))
    for t in range(T):
        if t - L < 0 or t + L > T - 1:
            continue    # kernel not fully supported here: leave the row 0 so novelty is exactly 0
        for off, val in zip(offs, a):
            A[t, t + off] = val
    return A


def checkerboard_novelty(xp, S, bands):
    """S (B,T,T); bands: list of (T,T) arrays from make_novelty_bands.
    Returns (B,T,len(bands))."""
    outs = []
    for Aband in bands:
        A = xp.asarray(Aband, S)
        outs.append(xp.sum((A @ S) * A, 2))                    # (B,T)
    return xp.stack(outs, 2)