"""
hierarchy_phrase/window_inference.py -- numpy-only: chunked inference with overlap and CENTER-KEEP stitching (unit-tested).

A video of T frames is covered by windows of length W whose starts are `round(W * keep)` frames apart. Every frame is taken from the
window in which it is closest to the window centre, so each window contributes its central `keep` fraction (keep = 1.0 is plain
concatenation without overlap; keep = 0.75 discards the outer 12.5% on each side, which the neighbouring windows cover from THEIR
centres). The first window keeps its left edge and the last window its right edge (video boundaries have no neighbour). All windows have
exactly length W, so they can be batched. Videos with T <= W are processed whole.
"""
import numpy as np


def plan_center_windows(T, W, keep=1.0):
    """[(s, e, lo, hi)]: window [s, e) and the kept frame range [lo, hi) inside it; kept ranges partition [0, T)."""
    if W <= 0 or T <= W:
        return [(0, T, 0, T)]
    if not 0.0 < keep <= 1.0:
        raise ValueError("keep must be in (0, 1]")
    stride = max(1, int(round(W * keep)))
    starts = list(range(0, T - W + 1, stride))
    if starts[-1] + W < T:
        starts.append(T - W)
    centers = [s + W / 2 for s in starts]
    out = []
    for k, s in enumerate(starts):
        lo = 0 if k == 0 else int(np.ceil((centers[k - 1] + centers[k]) / 2))
        hi = T if k == len(starts) - 1 else int(np.ceil((centers[k] + centers[k + 1]) / 2))
        out.append((s, s + W, lo, hi))
    return out


def stitch_center(T, plan, outs):
    """outs[i]: array of shape (e-s, ...) for window i. Returns the (T, ...) array assembled from the kept ranges."""
    first = np.asarray(outs[0])
    res = np.zeros((T,) + first.shape[1:], dtype=first.dtype)
    for (s, e, lo, hi), o in zip(plan, outs):
        res[lo:hi] = np.asarray(o)[lo - s:hi - s]
    return res