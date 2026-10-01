"""
src/handson_ctc_core.py -- numpy-only pieces of the Hands-On CTC loss (unit-tested without torch).

WHAT THE CTC SUPERVISES
The paper adds a gloss-level CTC loss. This project has no gloss identities in its labels, only BIO tags, so the
CTC here is SIGN-LEVEL: the target for a window is "one token per sign, in order" (a single non-blank class), and the
loss asks a dedicated CTC head to emit exactly that many sign peaks, in the right places, with blanks elsewhere. It
therefore supplies a sequence-level count/ordering constraint on top of the frame-level BIO cross-entropy -- the same
role the paper gives CTC -- but it cannot supply gloss identity. (To use real gloss ids you would extend the token
ids; not implemented.)

TARGET DERIVATION from hard BIO tags (0=Outside, 1=Inside, 2=Begin):
  a sign starts at a Begin frame that is not itself preceded by Begin (collapses the Begin runs that label dilation
  creates), or at an Inside frame preceded by Outside (a sign whose Begin tag is missing / cut by the window edge).

FEASIBILITY
CTC needs a blank between two identical consecutive tokens, so L identical tokens need at least 2L-1 output steps.
Windows that cannot satisfy this are EXCLUDED from the CTC term (and counted), rather than letting them produce an
infinite loss that is silently zeroed.
"""
import numpy as np


def sign_starts(bio):
    bio = np.asarray(bio)
    prev = np.concatenate([[0], bio[:-1]])
    return ((bio == 2) & (prev != 2)) | ((bio == 1) & (prev == 0))


def count_signs(bio):
    return int(sign_starts(bio).sum())


def ctc_min_steps(n_tokens):
    return 0 if n_tokens == 0 else 2 * n_tokens - 1


def ctc_feasible(n_tokens, n_steps):
    return n_steps >= ctc_min_steps(n_tokens)


def _logsumexp(a, b):
    m = max(a, b)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_nll_numpy(log_probs, n_tokens, blank=0, token=1):
    """Reference CTC negative log-likelihood (log-space forward algorithm) for a target of `n_tokens` identical
    tokens. log_probs: (T, K) log-softmax outputs. Returns +inf when the alignment is impossible."""
    T = log_probs.shape[0]
    ext = [blank]
    for _ in range(n_tokens):
        ext += [token, blank]
    S = len(ext)
    alpha = np.full(S, -np.inf)
    alpha[0] = log_probs[0, blank]
    if S > 1:
        alpha[1] = log_probs[0, ext[1]]
    for t in range(1, T):
        new = np.full(S, -np.inf)
        for s in range(S):
            v = alpha[s]
            if s >= 1:
                v = _logsumexp(v, alpha[s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                v = _logsumexp(v, alpha[s - 2])
            new[s] = v + log_probs[t, ext[s]]
        alpha = new
    total = alpha[S - 1] if S == 1 else _logsumexp(alpha[S - 1], alpha[S - 2])
    return float(-total)