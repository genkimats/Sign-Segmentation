"""
Torch-free core of the modality-imbalance methods (balance_study). Plain Python / numpy, so it is
unit-tested without PyTorch (test_balance_core.py). balancer.py feeds it scalars computed in torch.

Streams are indexed 0..M-1 in the model's stream order. "Scores" follow each paper's own convention;
every function says which.
"""
import itertools
import math

import numpy as np


# ==============================================================================
# Shapley decomposition (AGM, ICCV 2023; also the stream_shapley.py diagnostic)
# ==============================================================================
def coalitions(n):
    """All non-empty subsets of range(n), as sorted tuples, smallest first."""
    out = []
    for k in range(1, n + 1):
        out.extend(itertools.combinations(range(n), k))
    return out


def shapley_matrix(n):
    """
    Returns (coals, W) with W of shape (n, len(coals)) such that, for coalition values v(c) (v(empty) = 0),
        phi_m = sum_c W[m, c] * v(c)
    is the Shapley value of player m. Because v(empty) = 0, sum_m phi_m = v(all) (efficiency), which is
    what AGM relies on: the mono-modal responses add up to the model output. (AGM's Eq. 1 writes the sum
    over non-empty S, but its two-modality example keeps the phi({m}) term with phi(empty) = 0, which is
    the standard Shapley value with v(empty) = 0; that is what this implements.)
    """
    coals = coalitions(n)
    index = {c: i for i, c in enumerate(coals)}
    W = np.zeros((n, len(coals)))
    for m in range(n):
        others = [j for j in range(n) if j != m]
        for k in range(len(others) + 1):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for S in itertools.combinations(others, k):
                with_m = tuple(sorted(S + (m,)))
                W[m, index[with_m]] += weight
                if S:                                      # v(empty) = 0 contributes nothing
                    W[m, index[tuple(sorted(S))]] -= weight
    return coals, W


# ==============================================================================
# OPM -- On-the-fly Prediction Modulation (Wei et al., TPAMI 2024)
# ==============================================================================
def opm_ratios(scores):
    """
    scores: per-stream unimodal score s_m (higher = better; OPM: softmax probability of the true class,
    summed over the batch). rho_m = mean over j != m of s_m / s_j (Eq. 7). rho_m > 1: m dominates.
    """
    s = np.asarray(scores, dtype=float)
    M = len(s)
    if M < 2:
        return np.ones(M)
    s = np.maximum(s, 1e-12)
    return np.array([np.mean([s[m] / s[j] for j in range(M) if j != m]) for m in range(M)])


def opm_drop_prob(rho, q_base=0.3, lam=0.5):
    """Eq. 8: q = q_base * (1 + lam * tanh(rho - 1)) if rho > 1 else 0, capped at 1."""
    rho = np.asarray(rho, dtype=float)
    q = np.where(rho > 1.0, q_base * (1.0 + lam * np.tanh(rho - 1.0)), 0.0)
    return np.clip(q, 0.0, 1.0)


def opm_keep_mask(q, rng_uniform, fallback):
    """
    q: (M,) drop probabilities; rng_uniform: (B, M) uniform [0,1) draws; fallback: stream index kept when a
    sample would lose every stream (OPM never drops all modalities). Returns a (B, M) boolean keep mask.
    """
    keep = np.asarray(rng_uniform) >= np.asarray(q)[None, :]
    none = ~keep.any(axis=1)
    keep[none, fallback] = True
    return keep


# ==============================================================================
# CGGM -- Classifier-guided Gradient Modulation (Guo et al., NeurIPS 2024), magnitude term
# ==============================================================================
def cggm_balance(delta_eps, rho=None, eps=1e-4, lo=0.1, hi=3.0):
    """
    Eq. 7: B_m = rho * sum_{k != m} d_k / sum_k d_k, where d_k = improvement of stream k's unimodal head.
    A stream that improves slowly gets a large B (its encoder is sped up). Improvements are floored at
    eps (negative or zero improvements would make the ratio meaningless); if every stream is below eps
    they count as equal. rho defaults to M / (M - 1), so equal improvement gives B = 1 for every stream.
    The result is clipped to [lo, hi].
    """
    d = np.maximum(np.asarray(delta_eps, dtype=float), eps)
    M = len(d)
    if M < 2:
        return np.ones(M)
    rho = M / (M - 1) if rho is None else rho
    total = d.sum()
    B = rho * (total - d) / total
    return np.clip(B, lo, hi)


# ==============================================================================
# AGM -- Adaptive Gradient Modulation (Li et al., ICCV 2023)
# ==============================================================================
def agm_kappa(s, s_hat, alpha=1.0):
    """
    s: per-stream mono-modal score at this step, s_hat: its running average. AGM's Algorithm 1 uses the
    mean LOG-probability of the true class (higher = stronger; its Eq. 4 has the opposite sign, but only
    the algorithm's sign makes "r > tau = too strong -> kappa < 1" consistent). Eqs. 7-9:
        r_m   = exp(mean_{m' != m} (s_m - s_m'))
        tau_m = exp(mean_{m' != m} (s_hat_m - s_hat_m'))
        kappa_m = exp(-alpha * (r_m - tau_m))
    Returns (kappa, r, tau).
    """
    s = np.asarray(s, dtype=float)
    sh = np.asarray(s_hat, dtype=float)
    M = len(s)
    if M < 2:
        return np.ones(M), np.ones(M), np.ones(M)
    r = np.array([math.exp(np.mean([s[m] - s[j] for j in range(M) if j != m])) for m in range(M)])
    tau = np.array([math.exp(np.mean([sh[m] - sh[j] for j in range(M) if j != m])) for m in range(M)])
    kappa = np.exp(-alpha * (r - tau))
    return kappa, r, tau


def running_mean(prev, value, t):
    """AGM Eq. 10: s_hat(t) = s_hat(t-1) * (t-1)/t + s(t)/t, t = 1, 2, ..."""
    return value if t <= 1 or prev is None else prev * (t - 1) / t + value / t


# ==============================================================================
# MMPareto (Wei & Hu, ICML 2024)
# ==============================================================================
def mmpareto_coeffs(gm_gm, gu_gu, gm_gu, gamma=1.5):
    """
    For one encoder: g_m = gradient of the multimodal loss, g_u = gradient of its unimodal loss, given by
    their dot products. Returns (k_m, k_u, conflict) such that the final gradient is k_m * g_m + k_u * g_u:
      - cos(g_m, g_u) >= 0: 2*alpha_m = 2*alpha_u = 1 (plain sum direction);
      - otherwise: the two-objective Pareto (min-norm) weights
            alpha_m = ((g_u - g_m) . g_u) / ||g_m - g_u||^2, clipped to [0, 1], alpha_u = 1 - alpha_m,
        direction 2*alpha_m*g_m + 2*alpha_u*g_u;
      - the direction is rescaled to gamma * ||g_m + g_u|| (Algorithm 1).
    """
    norm_m, norm_u = math.sqrt(max(gm_gm, 0.0)), math.sqrt(max(gu_gu, 0.0))
    sum_sq = gm_gm + gu_gu + 2.0 * gm_gu
    if norm_m == 0.0 or norm_u == 0.0 or sum_sq <= 0.0:
        return gamma, gamma, False                       # degenerate: plain (rescaled) sum
    cos = gm_gu / (norm_m * norm_u)
    conflict = cos < 0.0
    if conflict:
        diff_sq = gm_gm + gu_gu - 2.0 * gm_gu
        a_m = (gu_gu - gm_gu) / diff_sq if diff_sq > 0 else 0.5
        a_m = min(max(a_m, 0.0), 1.0)
        c_m, c_u = 2.0 * a_m, 2.0 * (1.0 - a_m)
    else:
        c_m, c_u = 1.0, 1.0
    dir_sq = c_m * c_m * gm_gm + c_u * c_u * gu_gu + 2.0 * c_m * c_u * gm_gu
    if dir_sq <= 0.0:
        return 0.0, 0.0, conflict                        # Pareto-stationary: no common descent direction
    scale = gamma * math.sqrt(sum_sq) / math.sqrt(dir_sq)
    return c_m * scale, c_u * scale, conflict


# ==============================================================================
# InfoReg (Huang et al., CVPR 2025)
# ==============================================================================
def inforeg_delta(scores):
    """
    scores: per-stream unimodal loss (Eq. 8: cross-entropy, lower = better). Delta_m = mean over the
    streams that are WORSE than m of (s_m' - s_m); 0 if no stream is worse. Delta_m > 0 marks an
    information-sufficient stream (one that is ahead of some other stream), which gets regulated; the
    weakest stream gets 0, as the paper's text says. (The paper's Eq. 10 is written with the opposite
    comparison, which would regulate the weakest stream and contradict its own text.)
    """
    s = np.asarray(scores, dtype=float)
    out = np.zeros(len(s))
    for m in range(len(s)):
        worse = [s[j] - s[m] for j in range(len(s)) if j != m and s[j] > s[m]]
        out[m] = float(np.mean(worse)) if worse else 0.0
    return out


def inforeg_alpha(delta, beta=1.0):
    """Eq. 16: alpha = exp(beta * tanh(Delta))."""
    return np.exp(beta * np.tanh(np.asarray(delta, dtype=float)))


def inforeg_window(tr_prev, tr_prev2, K=0.0):
    """Eq. 11: the prime learning window is open when (Tr(F^{t-1}) - Tr(F^{t-2})) / Tr(F^{t-1}) > K."""
    if tr_prev is None or tr_prev2 is None or tr_prev <= 0:
        return False
    return (tr_prev - tr_prev2) / tr_prev > K


# ==============================================================================
# Wu et al. (ICML 2022): conditional learning speed and the guided re-balancing schedule
# ==============================================================================
def effective_update(grad_sq, param_sq):
    """mu(theta; i) = ||G||^2 / ||theta||^2 (Definition 5.1)."""
    return grad_sq / param_sq if param_sq > 0 else 0.0


def wu_speeds(sum_mu_fusion, sum_mu_branch):
    """
    Conditional learning speed of each stream, adapted to concat fusion. Wu et al. define
    s(m1 | m0) = log(sum mu(theta'_0) / sum mu(theta_0)): theta_0 = the branch of m0, theta'_0 = the fusion
    parameters that carry m1's information to m0's output. Here there is one output, so theta'_m = the
    fusion-projection columns of every OTHER stream and theta_m = stream m's encoder:
        s_m = log( sum_{j != m} sum_mu_fusion[j] / sum_mu_branch[m] ).
    With two streams this is exactly their s(m1|m0) and s(m0|m1). A large s_m: the model learns from the
    other streams fast relative to m (m is being out-learned).
    """
    f = np.asarray(sum_mu_fusion, dtype=float)
    b = np.asarray(sum_mu_branch, dtype=float)
    M = len(f)
    out = np.zeros(M)
    for m in range(M):
        num = f.sum() - f[m]
        out[m] = math.log(num / b[m]) if num > 0 and b[m] > 0 else 0.0
    return out


def wu_decide(speeds, alpha):
    """d_speed = max - min of the conditional speeds; if |d| > alpha, accelerate the most out-learned
    stream (argmax). Returns (trigger, target_index, d)."""
    s = np.asarray(speeds, dtype=float)
    d = float(s.max() - s.min()) if len(s) > 1 else 0.0
    return (d > alpha), int(np.argmax(s)), d


class WuSchedule:
    """
    Algorithm 1 of Wu et al.: regular steps during warm-up; afterwards, at every check (q == Q) take a
    regular step and test the imbalance; if it exceeds alpha, the next Q - 1 steps are re-balancing steps
    for the chosen stream, then check again. next_step() says what the coming step is.
    """

    def __init__(self, Q=5):
        self.Q = int(Q)
        self.q = self.Q
        self.target = None

    def next_step(self, warm):
        """Returns ("regular", None) / ("check", None) / ("rebalance", target)."""
        if warm:
            return "regular", None
        if self.q == self.Q:
            return "check", None
        self.q += 1
        return "rebalance", self.target

    def after_check(self, trigger, target):
        if trigger:
            self.q = 1
            self.target = target


# ==============================================================================
# Smoothed improvement tracker (CGGM's per-stream delta-epsilon)
# ==============================================================================
class ImprovementTracker:
    """
    Keeps an exponential moving average of a per-stream loss and returns its decrease over the last
    `window` steps (loss decrease = improvement; CGGM's epsilon is "higher is better", so the decrease of a
    loss plays its role). Per-iteration differences on random batches are mostly batch noise, hence the
    EMA and the window. Returns None until `window` steps of history exist.
    """

    def __init__(self, n, beta=0.98, window=100):
        self.beta, self.window = beta, int(window)
        self.ema = None
        self.hist = []
        self.n = n

    def update(self, losses):
        x = np.asarray(losses, dtype=float)
        self.ema = x.copy() if self.ema is None else self.beta * self.ema + (1 - self.beta) * x
        self.hist.append(self.ema.copy())
        if len(self.hist) > self.window + 1:
            self.hist.pop(0)
        if len(self.hist) <= self.window:
            return None
        return self.hist[0] - self.hist[-1]


# ==============================================================================
# Class-weighted per-frame scores (all methods): BIO is O-dominated, so plain means would mostly measure
# "is the person signing" -- every score is weighted by the training class weights.
# ==============================================================================
def weighted_gold_prob(probs, gold, class_weights, ignore=-100):
    """probs (N, 3) or (3, T) handled by caller as (N, 3); gold (N,). Class-weighted mean of p(gold)."""
    probs = np.asarray(probs, dtype=float)
    gold = np.asarray(gold)
    ok = gold != ignore
    if not ok.any():
        return float("nan")
    w = np.asarray(class_weights, dtype=float)[gold[ok]]
    p = probs[ok, gold[ok]]
    return float((w * p).sum() / w.sum())
