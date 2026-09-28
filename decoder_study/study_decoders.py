"""
Decoders for the decoder study (numpy only, all operate on ONE video at a time).

Input everywhere is `logp`: (T, 3) log-probabilities over [O, I, B] (use
to_logp() to get it from exported logits, optionally temperature-scaled).
Output is an int8 BIO array (T,) with 0=O, 1=I, 2=B.

  decode_argmax            baseline 0a
  decode_threshold         baseline 0b: B if p_B >= t_b, elif O if p_O >= t_o, else I
  collapse_b_runs          fairness fix for encoders trained on dilated Begin
                           targets (see below) -- applied on top of any greedy
  decode_hysteresis        port of the project's existing "linguistic" decoder
  decode_viterbi           1: constrained Viterbi (no O->I, cannot start with I)
  decode_semi_markov       1: semi-Markov Viterbi with a learned duration prior

WHY collapse_b_runs EXISTS. This project's encoders are trained against
argmax(Gaussian-smoothed labels), which turns every 1-frame Begin into a
~3-frame Begin run. A greedy decode of such a model emits B B B, and the
BIO->segment rule turns each extra B into its own 1-frame segment. Any decoder
that penalises 1-frame segments (the semi-Markov one does, via the duration
prior) would get credit merely for undoing that training artifact, so the
greedy baseline must get the same courtesy to keep the comparison fair.
"""
import numpy as np
from study_common import O, I, B
from study_metrics import bio_to_segments

NEG = -np.inf


def to_logp(logits, temperature=1.0, class_bias=None):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    if class_bias is not None:
        z = z + np.asarray(class_bias, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


# ------------------------------------------------------------------- greedy --
def decode_argmax(logp):
    return np.asarray(logp).argmax(axis=1).astype(np.int8)


def decode_threshold(logp, t_b=0.5, t_o=0.5):
    p = np.exp(logp)
    return np.where(p[:, B] >= t_b, B, np.where(p[:, O] >= t_o, O, I)).astype(np.int8)


def collapse_b_runs(labels, logp):
    """Keep only the max-p_B frame of every run of consecutive B labels as the
    Begin; frames after it become I; frames before it become O when the run
    started right after O (or at the video start, so no illegal O->I is created)
    and otherwise take the better of O/I."""
    out = np.asarray(labels).astype(np.int8).copy()
    isb = (out == B).astype(np.int8)
    d = np.diff(np.concatenate([[0], isb, [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s < 2:
            continue
        peak = s + int(np.argmax(logp[s:e, B]))
        if peak > s:
            if s == 0 or out[s - 1] == O:
                out[s:peak] = O
            else:
                out[s:peak] = np.where(logp[s:peak, O] >= logp[s:peak, I], O, I)
        out[peak + 1:e] = I
    return out


def decode_hysteresis(logp, threshold=0.60):
    """Faithful numpy port of src/decoder.py strategy='linguistic'."""
    p = np.exp(logp)
    raw = p.argmax(axis=1)
    out = np.zeros(len(raw), dtype=np.int8)
    cur = 0
    for t in range(len(raw)):
        tgt = int(raw[t]); tprob = p[t, tgt]; pB = p[t, B]
        if cur == O and tgt == I:
            nxt = B if pB > (1.0 - threshold) else O
        elif tgt != cur and tprob < threshold:
            nxt = cur
        else:
            nxt = tgt
        out[t] = nxt
        cur = nxt
    return out


# -------------------------------------------------------- constrained Viterbi --
def decode_viterbi(logp, trans_logp=None):
    """Frame-level Viterbi over {O,I,B}. Forbids O->I and starting in I; every
    other transition is allowed at zero cost unless `trans_logp` (3x3,
    [from, to]) supplies learned costs."""
    logp = np.asarray(logp, dtype=np.float64)
    T = len(logp)
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    trans = np.zeros((3, 3)) if trans_logp is None else np.array(trans_logp, dtype=np.float64)
    trans[O, I] = NEG
    delta = logp[0].copy(); delta[I] = NEG
    bp = np.zeros((T, 3), dtype=np.int8)
    for t in range(1, T):
        cand = delta[:, None] + trans
        bp[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + logp[t]
    out = np.zeros(T, dtype=np.int8)
    state = int(delta.argmax())
    for t in range(T - 1, -1, -1):
        out[t] = state
        if t > 0:
            state = int(bp[t, state])
    return out


# ---------------------------------------------------------------- semi-Markov --
def fit_duration_prior(gold_list, dmax_quantile=0.999, dmax_margin=1.25, dmax_cap=400,
                       smooth_sigma=1.5, alpha=0.5):
    """Duration prior P(d) over sign durations (in FRAMES), from RAW gold labels
    of the TRAIN split only -- this uses labels, not encoder outputs, so it is
    not affected by the encoder's in-sample bias. Smoothed empirical histogram.
    If your videos mix frame rates, fit one prior per fps group."""
    durs = np.array([e - s for g in gold_list for s, e in bio_to_segments(g)])
    if len(durs) == 0:
        raise ValueError("no gold segments to fit a duration prior from")
    dmax = int(min(dmax_cap, max(10, np.quantile(durs, dmax_quantile) * dmax_margin)))
    kept = durs[durs <= dmax]
    counts = np.bincount(kept, minlength=dmax + 1).astype(np.float64)
    counts[0] = 0.0
    if smooth_sigma > 0:
        half = int(np.ceil(4 * smooth_sigma))
        k = np.exp(-0.5 * (np.arange(-half, half + 1) / smooth_sigma) ** 2)
        counts = np.convolve(counts, k / k.sum(), mode="same")
        counts[0] = 0.0
    counts[1:] += alpha
    prob = counts / counts.sum()
    dur_logp = np.full(dmax + 1, NEG)
    dur_logp[1:] = np.log(prob[1:])
    return {"dur_logp": dur_logp, "dmax": dmax, "n_segments": int(len(durs)),
            "median": float(np.median(durs)), "p05": float(np.quantile(durs, 0.05)),
            "p95": float(np.quantile(durs, 0.95)), "frac_beyond_dmax": float((durs > dmax).mean())}


def decode_semi_markov(logp, dur_logp, duration_weight=1.0, seg_penalty=0.0):
    """Exact semi-Markov Viterbi. A labelling is a sequence of O frames and sign
    segments; a segment of duration d starting at s scores
        logp_B[s] + sum_{u=s+1}^{s+d-1} logp_I[u] + duration_weight*log P(d) + seg_penalty
    and O frames score logp_O. Every such labelling is a legal BIO sequence by
    construction (I only ever follows B/I; signs may be adjacent).

    duration_weight rescales the prior (an overconfident encoder usually needs
    it < 1..., tune on the selection split); seg_penalty is a per-segment
    log-cost trading precision against recall. With a uniform prior, weight
    irrelevant and penalty 0 this is EXACTLY decode_viterbi (tested).
    Cost O(T * dmax)."""
    logp = np.asarray(logp, dtype=np.float64)
    T = len(logp)
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    dmax = len(dur_logp) - 1
    lpO, lpI, lpB = logp[:, O], logp[:, I], logp[:, B]
    cumI = np.concatenate([[0.0], np.cumsum(lpI)])
    wdur = duration_weight * np.asarray(dur_logp, dtype=np.float64)

    A = np.full(T + 1, NEG); Z = np.full(T + 1, NEG); G = np.full(T + 1, NEG)
    Gsrc = np.zeros(T + 1, dtype=np.int8)            # 0: A, 1: Z, 2: start
    bpd = np.zeros(T + 1, dtype=np.int32)
    Bs = np.full(T + 1, NEG)                         # Bs[s] = G[s] + lpB[s] - cumI[s+1]
    G[0], Gsrc[0] = 0.0, 2
    Bs[0] = lpB[0] - cumI[1]
    for t in range(1, T + 1):
        Z[t] = lpO[t - 1] + G[t - 1]
        dm = min(dmax, t)
        cand = Bs[t - dm:t][::-1] + wdur[1:dm + 1]   # position k  <->  d = k+1, s = t-d
        k = int(np.argmax(cand))
        A[t] = cumI[t] + seg_penalty + cand[k]
        bpd[t] = k + 1
        if A[t] >= Z[t]:
            G[t], Gsrc[t] = A[t], 0
        else:
            G[t], Gsrc[t] = Z[t], 1
        if t < T:
            Bs[t] = G[t] + lpB[t] - cumI[t + 1]

    out = np.zeros(T, dtype=np.int8)
    t, state = T, (0 if A[T] >= Z[T] else 1)
    while t > 0 and state != 2:
        if state == 1:
            out[t - 1] = O
            state = int(Gsrc[t - 1]); t -= 1
        else:
            s = t - int(bpd[t])
            out[s] = B
            out[s + 1:t] = I
            state = int(Gsrc[s]); t = s
    return out


# ------------------------------------------------------------------- oracles --
def oracle_logp(labels, eps=1e-3):
    """Near-one-hot log-probs from RAW gold: the pipeline sanity check. A correct
    decoding pipeline must reproduce gold from these."""
    lab = np.asarray(labels).astype(np.int64)
    p = np.full((len(lab), 3), eps / 2.0)
    p[np.arange(len(lab)), lab] = 1.0 - eps
    return np.log(p)


def oracle_soft_logp(soft_labels, floor=1e-6):
    """Log-probs equal to the smoothed TRAINING targets: what a PERFECT model of
    the training objective would output. Isolates how much each decoder loses
    to the Begin-dilation alone, independent of any encoder error."""
    return np.log(np.clip(np.asarray(soft_labels, dtype=np.float64), floor, 1.0))