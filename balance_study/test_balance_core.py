"""
Numpy tests for balance_study/core.py (no PyTorch needed). Run from the repo root:
    python balance_study/test_balance_core.py
"""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from balance_study import core  # noqa: E402


def brute_shapley(v, n):
    """Textbook Shapley value of v (dict frozenset -> value, v(empty) given)."""
    phi = np.zeros(n)
    for m in range(n):
        others = [j for j in range(n) if j != m]
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                phi[m] += w * (v[frozenset(S + (m,))] - v[frozenset(S)])
    return phi


def test_shapley():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4):
        coals, W = core.shapley_matrix(n)
        assert len(coals) == 2 ** n - 1 and coals[-1] == tuple(range(n))
        for _ in range(20):
            vals = rng.normal(size=len(coals))
            v = {frozenset(): 0.0}
            v.update({frozenset(c): x for c, x in zip(coals, vals)})
            phi = W @ vals
            assert np.allclose(phi, brute_shapley(v, n)), n
            assert np.isclose(phi.sum(), v[frozenset(range(n))])          # efficiency (v(empty) = 0)
        # with a non-zero v(empty): subtracting it from every coalition gives the same Shapley values
        vals = rng.normal(size=len(coals))
        v0 = 0.7
        v = {frozenset(): v0}
        v.update({frozenset(c): x for c, x in zip(coals, vals)})
        assert np.allclose(W @ (vals - v0), brute_shapley(v, n))
    # additive game: phi_m = own value; dummy player gets 0
    coals, W = core.shapley_matrix(3)
    vals = np.array([sum([1.0, 2.0, 0.0][i] for i in c) for c in coals])
    assert np.allclose(W @ vals, [1.0, 2.0, 0.0])
    # AGM's two-modality example: phi1 = 1/2 [v12 - v2 + v1]
    coals, W = core.shapley_matrix(2)
    v1, v2, v12 = 0.3, -0.4, 1.1
    vals = np.array([{(0,): v1, (1,): v2, (0, 1): v12}[c] for c in coals])
    assert np.isclose((W @ vals)[0], 0.5 * (v12 - v2 + v1))


def test_opm():
    rho = core.opm_ratios([0.6, 0.6, 0.2])
    assert np.allclose(rho, [(1 + 3) / 2, (1 + 3) / 2, (1 / 3 + 1 / 3) / 2])
    q = core.opm_drop_prob(rho, q_base=0.3, lam=0.5)
    assert q[2] == 0.0 and np.isclose(q[0], 0.3 * (1 + 0.5 * math.tanh(1.0)))
    assert core.opm_drop_prob([1.0], 0.5, 0.5)[0] == 0.0                    # rho = 1: not dominant
    assert core.opm_drop_prob([50.0], 0.9, 0.5)[0] == 1.0                   # capped
    u = np.array([[0.0, 0.0, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9]])
    keep = core.opm_keep_mask([0.5, 0.5, 0.5], u, fallback=2)
    assert keep.tolist() == [[False, False, True], [False, False, True], [True, True, True]]


def test_cggm():
    assert np.allclose(core.cggm_balance([1.0, 1.0, 1.0]), 1.0)              # equal improvement -> 1
    B = core.cggm_balance([1.0, 1.0, 0.0])                                  # stream 3 stalls
    assert B[2] > 1.0 > B[0] and np.isclose(B[0], B[1])
    assert np.allclose(core.cggm_balance([-1.0, -2.0, 0.0]), 1.0)            # all below eps -> equal
    assert np.allclose(core.cggm_balance([2.0, 1.0], rho=1.0), [1 / 3, 2 / 3])  # paper's formula, rho 1
    assert core.cggm_balance([1.0, 0.0, 0.0], hi=3.0).max() <= 3.0


def test_agm():
    k, r, tau = core.agm_kappa([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert np.allclose(k, 1.0)
    # stream 0 suddenly stronger than its usual relative strength -> suppressed (kappa < 1)
    k, r, tau = core.agm_kappa([-0.2, -0.8, -1.0], [-0.5, -0.8, -1.0], alpha=1.0)
    assert k[0] < 1.0 and r[0] > tau[0]
    s = None
    for t, x in enumerate([1.0, 2.0, 3.0], start=1):
        s = core.running_mean(s, np.array([x]), t)
    assert np.isclose(s[0], 2.0)


def brute_min_norm(gm, gu):
    best, best_a = None, None
    for a in np.linspace(0, 1, 100001):
        n = np.linalg.norm(a * gm + (1 - a) * gu)
        if best is None or n < best:
            best, best_a = n, a
    return best_a


def test_mmpareto():
    rng = np.random.default_rng(1)
    for _ in range(30):
        gm, gu = rng.normal(size=20), rng.normal(size=20)
        k_m, k_u, conflict = core.mmpareto_coeffs(gm @ gm, gu @ gu, gm @ gu, gamma=1.5)
        h = k_m * gm + k_u * gu
        assert conflict == (gm @ gu < 0)
        assert np.isclose(np.linalg.norm(h), 1.5 * np.linalg.norm(gm + gu))   # magnitude rule
        if conflict:
            a = brute_min_norm(gm, gu)
            d = 2 * a * gm + 2 * (1 - a) * gu
            cos = h @ d / (np.linalg.norm(h) * np.linalg.norm(d))
            assert cos > 0.9999                                              # Pareto direction
            assert h @ gm >= -1e-9 and h @ gu >= -1e-9                       # common descent direction
        else:
            assert np.isclose(k_m, k_u) and np.isclose(k_m, 1.5)
    k_m, k_u, c = core.mmpareto_coeffs(0.0, 1.0, 0.0)
    assert (k_m, k_u) == (1.5, 1.5)


def test_inforeg():
    d = core.inforeg_delta([0.3, 0.5, 1.0])                                 # losses: stream 0 best
    assert np.isclose(d[0], np.mean([0.2, 0.7])) and np.isclose(d[1], 0.5) and d[2] == 0.0
    assert np.allclose(core.inforeg_alpha([0.0]), 1.0)
    assert core.inforeg_window(1.2, 1.0, K=0.1) and not core.inforeg_window(1.0, 1.2, K=0.0)
    assert not core.inforeg_window(None, 1.0)


def test_wu():
    s = core.wu_speeds([1.0, 1.0], [1.0, 1.0])                              # identical streams
    assert np.allclose(s, 0.0)
    # two streams: s_0 = log(mu_fus_1 / mu_branch_0), as Wu et al.'s s(m1 | m0)
    s = core.wu_speeds([2.0, 4.0], [1.0, 8.0])
    assert np.allclose(s, [math.log(4.0 / 1.0), math.log(2.0 / 8.0)])
    trig, tgt, d = core.wu_decide(s, alpha=0.1)
    assert trig and tgt == 0 and np.isclose(d, s[0] - s[1])
    sched = core.WuSchedule(Q=3)
    assert sched.next_step(warm=True) == ("regular", None)
    assert sched.next_step(warm=False) == ("check", None)
    sched.after_check(True, 1)
    kinds = [sched.next_step(False) for _ in range(3)]
    assert kinds == [("rebalance", 1), ("rebalance", 1), ("check", None)]   # Q - 1 re-balancing steps
    sched.after_check(False, 0)
    assert sched.next_step(False) == ("check", None)


def test_tracker():
    tr = core.ImprovementTracker(2, beta=0.0, window=3)
    out = [tr.update([10 - t, 5.0]) for t in range(5)]
    assert out[:3] == [None, None, None]
    assert np.allclose(out[3], [3.0, 0.0]) and np.allclose(out[4], [3.0, 0.0])


def test_weighted_gold_prob():
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1]])
    gold = np.array([0, 2, -100])
    v = core.weighted_gold_prob(probs, gold, [1.0, 1.1, 8.5])
    assert np.isclose(v, (1.0 * 0.8 + 8.5 * 0.8) / 9.5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all core tests passed")
