"""
Self-tests for trace_core.py (numpy only). Run from the repo root:  python fusion_study/test_trace_core.py
Where src.metrics is importable (numpy + scikit-learn) the event / matching rules are also checked
against the project's own implementations.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fusion_study import trace_core as tc  # noqa: E402

try:
    from src.metrics import classify_gold_boundaries, extract_segments, match_boundaries  # noqa: E402
except Exception:  # pragma: no cover
    classify_gold_boundaries = extract_segments = match_boundaries = None

RNG = np.random.default_rng(1)
O, I, B = 0, 1, 2


def random_bio(T, p_gap=0.4):
    out = np.zeros(T, np.int64)
    t = int(RNG.integers(0, 30))
    while t < T:
        L = int(RNG.integers(15, 120))
        out[t:t + L] = I
        out[t] = B
        t += L
        if RNG.random() < p_gap:
            t += int(RNG.integers(1, 40))
    return out


def test_segments_and_events():
    for _ in range(200):
        bio = random_bio(int(RNG.integers(1, 1500)))
        if RNG.random() < 0.3:
            bio[RNG.random(len(bio)) < 0.02] = RNG.integers(0, 3)
        if extract_segments is not None:
            assert tc._segments(bio) == extract_segments(bio)
            ref = [(s, t) for s, t, _ in classify_gold_boundaries(bio, 12) if t != "video_start"]
            assert tc.phrase_start_events(bio, 12) == ref


def test_controls():
    phrase = np.array([O] * 5 + [B] + [I] * 39 + [O] * 5 + [B] + [I] * 49)      # phrases 5-44, 50-99
    sign = np.zeros_like(phrase)
    for s in (5, 12, 20, 40, 50, 70, 95):
        sign[s] = B
        sign[s + 1:s + 4] = I
    got = tc.sign_start_controls(phrase, sign, exclusion=10)
    # 5 and 50 are phrase starts, 40 and 95 are within 10 frames of a phrase end
    assert got == [20, 70], got
    rng = np.random.default_rng(0)
    fr = tc.inside_frames(phrase, 1000, rng, exclusion=10)
    d = tc._boundary_distance(phrase)
    assert all(phrase[f] != O and d[f] >= 10 for f in fr) and len(fr) == len(set(fr))


def test_boundary_distance():
    for _ in range(50):
        bio = random_bio(int(RNG.integers(5, 400)))
        d = tc._boundary_distance(bio)
        marks = sorted({x for s, e in tc._segments(bio) for x in (s, e)})
        for t in range(len(bio)):
            ref = min(abs(t - m) for m in marks) if marks else np.inf
            assert d[t] == ref


def test_locked_average():
    sig = [np.arange(100, dtype=float), np.arange(50, dtype=float) * 2]
    ev = [[10, 98], [3]]
    mean, sem, n = tc.locked_average(sig, ev, 5)
    # offset 0: (10 + 98 + 6) / 3 ; offset +3: only events 10 and 3 (98+3 is outside) -> (13 + 12) / 2
    assert np.isclose(mean[5, 0], (10 + 98 + 6) / 3) and n[5] == 3
    assert np.isclose(mean[8, 0], (13 + 12) / 2) and n[8] == 2
    assert n[0] == 2 and np.isclose(mean[0, 0], (5 + 93) / 2)        # offset -5: event 3 -> frame -2 missing
    m2, _, _ = tc.locked_average([np.stack([s, -s], 1) for s in sig], ev, 5)
    assert np.allclose(m2[:, 0], mean[:, 0]) and np.allclose(m2[:, 1], -mean[:, 0])


def test_lag_distribution():
    a = [np.tile(np.array([0.1, 0.8, 0.1]), (20, 1)), np.tile(np.array([0.5, 0.5, 0.0]), (10, 1))]
    d, n = tc.lag_distribution(a)
    assert n == 30 and np.allclose(d, (20 * np.array([.1, .8, .1]) + 10 * np.array([.5, .5, 0])) / 30)
    d, n = tc.lag_distribution(a, [[0, 1], []])
    assert n == 2 and np.allclose(d, [.1, .8, .1])


def test_matching_and_complementarity():
    for _ in range(300):
        g = np.sort(RNG.choice(1000, size=int(RNG.integers(0, 30)), replace=False))
        p = np.sort(RNG.choice(1000, size=int(RNG.integers(0, 30)), replace=False))
        k = int(RNG.integers(0, 8))
        got = tc.match_one_to_one(g, p, k)
        if match_boundaries is not None:
            ref = match_boundaries(g, p, k)[0]
            assert got.sum() == np.asarray(ref).sum()
    D = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], bool)
    rows = {r["what"]: r["value"] for r in tc.complementarity(D, ["body", "face", "hamer"]) if r["group"] == "all"}
    assert rows["recall body"] == .5 and rows["only body"] == .25 and rows["only face"] == 0
    assert rows["union recall"] == .75 and rows["missed by all"] == .25
    assert np.isclose(rows["jaccard body|face"], 0.5) and rows["jaccard body|hamer"] == 0
    typed = tc.complementarity(D, ["body", "face", "hamer"], types=["continuous", "long_pause", "continuous", "long_pause"])
    assert any(r["group"] == "continuous" and r["what"] == "union recall" and r["value"] == 1.0 for r in typed)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nAll {len(tests)} trace_core tests passed.")
