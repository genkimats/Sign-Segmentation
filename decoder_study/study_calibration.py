"""
Calibration checks and temperature scaling (numpy + scipy).

Structured decoders (Viterbi, semi-Markov) multiply per-frame probabilities
across many frames, so they are far more sensitive to over/under-confidence
than thresholding is. An encoder trained with class-weighted CE (here
[0.6, 0.8, 1.0]) is also shifted toward the up-weighted classes by construction.
Fit T on an OUT-OF-SAMPLE split (val), never on frames the encoder trained on.

WHICH TARGETS: this encoder was trained to predict the dilated-Begin targets
(study_common.training_targets), so calibration is fit against THOSE by default
-- that is the quantity the network actually estimates. Calibration against the
raw gold is reported separately: the gap between the two is the dilation effect,
not miscalibration.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from study_decoders import to_logp


def mean_nll(logits_list, targets_list, temperature=1.0):
    tot, n = 0.0, 0
    for lg, tg in zip(logits_list, targets_list):
        lp = to_logp(lg, temperature)
        tot -= lp[np.arange(len(tg)), np.asarray(tg, dtype=np.int64)].sum()
        n += len(tg)
    return tot / max(n, 1)


def fit_temperature(logits_list, targets_list, bounds=(0.25, 6.0)):
    res = minimize_scalar(lambda t: mean_nll(logits_list, targets_list, t),
                          bounds=bounds, method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def ece_binary(prob, hit, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf, acc = float(prob[m].mean()), float(hit[m].mean())
            ece += float(m.mean()) * abs(conf - acc)
            rows.append((conf, acc, int(m.sum())))
    return float(ece), rows


def calibration_report(logits_list, targets_list, temperature=1.0, n_bins=15):
    """Per-class one-vs-rest ECE (O, I, B) and top-label ECE, frame-level."""
    lp = np.concatenate([to_logp(l, temperature) for l in logits_list])
    tg = np.concatenate([np.asarray(t, dtype=np.int64) for t in targets_list])
    p = np.exp(lp)
    rep = {"nll": float(-lp[np.arange(len(tg)), tg].mean())}
    for c, name in enumerate("OIB"):
        rep[f"ece_{name}"], rows = ece_binary(p[:, c], (tg == c).astype(float), n_bins)
        rep[f"rows_{name}"] = rows
    top = p.max(axis=1)
    rep["ece_top"], _ = ece_binary(top, (p.argmax(axis=1) == tg).astype(float), n_bins)
    rep["mean_p_B"], rep["freq_B"] = float(p[:, 2].mean()), float((tg == 2).mean())
    return rep