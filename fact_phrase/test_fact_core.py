"""
Self-tests for fact_core.py (numpy only, no torch). Run from the repo root:

    python fact_phrase/test_fact_core.py

Each test compares the real code against a brute-force reference or an exact expectation.
Where src.metrics is importable (numpy + scikit-learn), decoded labels are also pushed through
the project's own evaluate_videos to check the end-to-end path the evaluation uses.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fact_phrase import fact_core as core  # noqa: E402

try:
    from src.metrics import evaluate_videos, extract_segments  # noqa: E402
except Exception:  # pragma: no cover - only when scikit-learn is missing
    evaluate_videos = extract_segments = None

O, I, B = 0, 1, 2
RNG = np.random.default_rng(0)


def random_bio(T, rng=RNG, p_gap=0.35):
    """Random gold-like BIO: phrases of 20-150 frames, often adjacent (no O between)."""
    out = np.zeros(T, np.int64)
    t = int(rng.integers(0, 30))
    while t < T:
        L = int(rng.integers(20, 150))
        out[t:t + L] = I
        out[t] = B
        t += L
        if rng.random() < p_gap:
            t += int(rng.integers(1, 40))
    return out


def brute_segments(bio):
    """Reference: phrases from extract_segments semantics + maximal O runs, in time order."""
    segs = []
    T = len(bio)
    t = 0
    while t < T:
        if bio[t] == O:
            s = t
            while t < T and bio[t] == O:
                t += 1
            segs.append((s, t - 1, core.TOK_O))
        else:
            s = t
            t += 1
            while t < T and bio[t] == I:
                t += 1
            segs.append((s, t - 1, core.TOK_PHRASE))
    return segs


def test_bio_to_segments():
    for _ in range(200):
        bio = random_bio(int(RNG.integers(1, 800)))
        if RNG.random() < 0.3:                       # also I-after-O starts and stray Bs
            bio[RNG.random(len(bio)) < 0.02] = RNG.integers(0, 3)
        seg_label, transcript, segs = core.bio_to_segments(bio)
        ref = brute_segments(bio)
        assert [(s, e) for s, e, _ in ref] == segs, (bio, segs, ref)
        assert transcript.tolist() == [c for _, _, c in ref]
        for k, (s, e) in enumerate(segs):
            assert (seg_label[s:e + 1] == k).all()
        if extract_segments is not None:
            phrases = [(s, e) for (s, e), c in zip(segs, transcript) if c == core.TOK_PHRASE]
            assert phrases == extract_segments(bio), "phrases must equal src.metrics.extract_segments"
    # adjacent phrases stay separate
    bio = np.array([B, I, I, B, I, O, O, B, I])
    _, tr, segs = core.bio_to_segments(bio)
    assert segs == [(0, 2), (3, 4), (5, 6), (7, 8)] and tr.tolist() == [1, 1, 0, 1]


def test_collapse_and_tdu():
    bio = np.array([O, B, B, B, I, I, B, I, O, B])
    assert core.collapse_b_runs(bio).tolist() == [O, B, I, I, I, I, B, I, O, B]
    assert core.tdu_segments(bio) == [(0, 0), (1, 5), (6, 7), (8, 8), (9, 9)]


def test_soft_iou():
    for _ in range(50):
        T, A = int(RNG.integers(5, 120)), int(RNG.integers(1, 12))
        bio = random_bio(T)
        seg_label, transcript, _ = core.bio_to_segments(bio)
        S = len(transcript)
        logits = RNG.normal(size=(T, A)) * 3
        attn = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        onehot = np.zeros((T, S))
        onehot[np.arange(T), seg_label] = 1
        a, o = attn[:, :, None], onehot[:, None, :]
        overlap = (a * o).sum(0)
        union = np.minimum(a + o, 1.0).sum(0)
        ref = np.nan_to_num(overlap / union)                  # exactly FACT's computation
        got = core.soft_iou(attn, seg_label, S)
        assert np.allclose(got, ref, atol=1e-10), np.abs(got - ref).max()


def test_windows_and_stitching():
    for _ in range(200):
        T = int(RNG.integers(1, 5000))
        W = int(RNG.integers(10, 1600))
        step = int(RNG.integers(1, W + 1))
        wins = core.tile_windows(T, W, step)
        # identical to src/evaluation.py video_windows (re-implemented here to avoid torch)
        ref = []
        if T > W:
            s = 0
            for s in range(0, T - W + 1, step):
                ref.append((s, s + W))
            if s + W < T:
                ref.append((T - W, T))
        else:
            ref.append((0, T))
        assert wins == ref
        regs = core.owned_regions(wins, T)
        assert regs[0][0] == 0 and regs[-1][1] == T
        for (a, b), (s, e) in zip(regs, wins):
            assert s <= a <= b <= e
        for (_, b0), (a1, _) in zip(regs[:-1], regs[1:]):
            assert b0 == a1                                   # partition, no gap, no overlap
        # stitching each window's slice of a known signal reproduces it
        bio = random_bio(T)
        assert (core.stitch_labels([bio[s:e] for s, e in wins], wins, T) == bio).all()
        p = core.onehot_probs(bio)
        assert np.allclose(core.stitch_probs([p[:, s:e] for s, e in wins], wins, T), p)


def test_token_decode_oracle():
    """Tokens that attend exactly to the gold segments decode back to the gold BIO."""
    for _ in range(100):
        T = int(RNG.integers(30, 1500))
        bio = random_bio(T)
        seg_label, transcript, _ = core.bio_to_segments(bio)
        S = len(transcript)
        A = S + int(RNG.integers(0, 10))
        perm = RNG.permutation(A)                              # token k <- segment order scrambled
        action = np.full((A, 3), -5.0)
        action[:, core.TOK_NULL] = 5.0                         # unused tokens: null
        attn = np.full((T, A), 1e-3)
        for s in range(S):
            k = perm[s]
            action[k] = -5.0
            action[k, transcript[s]] = 5.0
            attn[seg_label == s, k] = 1.0
        attn /= attn.sum(1, keepdims=True)
        frame_logits = np.zeros((T, 3))                        # uninformative frame branch
        got = core.token_decode(action, attn, frame_logits, mwt=0.1)
        assert (got == bio).all()
        if extract_segments is not None:
            assert extract_segments(got) == extract_segments(bio)
    # no active token -> frame branch argmax
    fl = np.array([[0, 5, 0], [0, 0, 5.0]])
    assert core.token_decode(np.array([[0, 0, 9.0]]), np.ones((2, 1)), fl, 0.1).tolist() == [I, B]


def test_rates():
    for _ in range(200):
        T = int(RNG.integers(1, 3000))
        k = int(RNG.integers(1, 5))
        bio = random_bio(T)
        low = core.downsample_bio(bio, k)
        assert len(low) == -(-T // k)
        up = core.upsample_bio(low, k, T)
        assert len(up) == T
        g_starts = np.flatnonzero(core.phrase_starts_mask(bio))
        u_starts = np.flatnonzero(core.phrase_starts_mask(up))
        # every gold start has an upsampled start within k-1 frames (starts never vanish ...
        for s in g_starts:
            assert np.min(np.abs(u_starts - s)) <= k - 1
        # ... and no extra starts appear: at most one start per group of k frames
        assert len(u_starts) <= len(g_starts)
        if k == 1:
            assert (up == bio).all()
        # probabilities: argmax of upsampled probs == upsampled argmax labels
        p = RNG.random((3, len(low))).astype(np.float32)
        p /= p.sum(0, keepdims=True)
        pu = core.upsample_probs(p, k, T)
        assert pu.shape == (3, T) and np.allclose(pu.sum(0), 1.0, atol=1e-5)
        assert (pu.argmax(0) == core.upsample_bio(p.argmax(0), k, T)).all()
    x = RNG.normal(size=(7, 3)).astype(np.float32)
    px = core.pool_features(x, 3)
    assert px.shape == (3, 3) and np.allclose(px[0], x[:3].mean(0)) and np.allclose(px[2], x[6])


def test_token_stats():
    labs = [random_bio(int(RNG.integers(100, 4000))) for _ in range(20)]
    mx, mean, n = core.window_segment_stats(labs, 1500, 375)
    brute = [core.count_segments(b[s:e]) for b in labs for s, e in core.tile_windows(len(b), 1500, 375)]
    assert mx == max(brute) and abs(mean - np.mean(brute)) < 1e-9 and n == len(brute)
    assert core.auto_ntoken(29) == 40 and core.auto_ntoken(3) == 16
    w = core.null_weight(64, 30.0)
    assert abs(w - 64 / (34 * 2)) < 1e-12


def test_metrics_path():
    """Decoded labels through src.metrics: one-hot probs of the gold give perfect scores."""
    if evaluate_videos is None:
        print("  (src.metrics not importable here -- skipped)")
        return
    gold = {f"v{i}": random_bio(int(RNG.integers(500, 3000))) for i in range(4)}
    m = evaluate_videos({v: core.onehot_probs(g) for v, g in gold.items()}, gold)
    assert abs(m["BndF1_5"] - 1) < 1e-9 and abs(m["IoU"] - 1) < 1e-9 and abs(m["Pct"] - 1) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nAll {len(tests)} fact_core tests passed.")
