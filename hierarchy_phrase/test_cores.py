"""test_cores.py -- numpy-only tests of segments / metrics / sign_tokens (no torch, no data needed)."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segments as S
import metrics as M
import sign_tokens as ST

rng = np.random.default_rng(0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def random_signs(T, rng, mean_len=12, gap_p=0.4):
    segs, t = [], int(rng.integers(0, 6))
    while True:
        L = int(rng.integers(3, 2 * mean_len))
        if t + L > T:
            break
        segs.append((t, t + L))
        t += L
        if rng.random() < gap_p:
            t += int(rng.integers(1, 10))
    return segs


# ------------------------------------------------------------------ segments
bio = np.array([0, 2, 1, 1, 0, 0, 2, 1, 2, 2, 1, 0])
check("bio_to_segments follows the project rule", S.bio_to_segments(bio) == [(1, 4), (6, 8), (8, 9), (9, 11)],
      str(S.bio_to_segments(bio)))
segs = random_signs(500, rng)
check("segments -> bio -> segments round trip", S.bio_to_segments(S.segments_to_bio(segs, 500)) == segs)

ok = True
for stride in (2, 3):
    for _ in range(50):
        T = int(rng.integers(200, 400))
        sg = random_signs(T, rng)
        ph = [(sg[i][0], sg[j][1]) for i, j in [(0, 3), (4, 8), (9, min(14, len(sg) - 1))] if j < len(sg)]
        rs, rp = S.resample_segments(sg, stride), S.resample_segments(ph, stride)
        starts_s = {s for s, _ in rs}
        ends_s = {e for _, e in rs}
        # a phrase edge that equalled a sign edge must still equal one (same rounding everywhere)
        ok &= all((s2 in starts_s) for (s, _), (s2, _) in zip(ph, rp) if any(s == a for a, _ in sg))
        ok &= all((e2 in ends_s) for (_, e), (_, e2) in zip(ph, rp) if any(e == b for _, b in sg))
        ok &= all(rs[i][1] <= rs[i + 1][0] for i in range(len(rs) - 1))
check("resampling keeps phrase/sign edge alignment and segment order", ok)
check("working_length equals len(x[::stride])", all(S.working_length(T, s) == len(np.arange(T)[::s])
                                                     for T in range(1, 40) for s in (1, 2, 3)))

# ------------------------------------------------------------------ phrase tags
sg = random_signs(600, rng)
cuts = sorted(rng.choice(np.arange(1, len(sg)), size=len(sg) // 5, replace=False).tolist())
bounds = [0] + cuts + [len(sg)]
ph = [(sg[a][0], sg[b - 1][1]) for a, b in zip(bounds[:-1], bounds[1:])]
tags = S.phrase_tags_over_signs(sg, ph)
check("gold signs: tags are 1 exactly at phrase starts", [i for i, t in enumerate(tags) if t] == bounds[:-1])
check("tags -> phrases reproduces gold phrases", S.phrases_from_tags(sg, tags) == ph)
shifted = [(s + int(rng.integers(-1, 2)), e) for s, e in sg]
shifted = [(max(s, 0), e) for s, e in shifted if e > s]
tags2 = S.phrase_tags_over_signs(shifted, ph)
check("shifted signs: #B tags still equals #phrases (+- first sign)", abs(int(tags2.sum()) - len(ph)) <= 1)
check("first sign is always B; empty input safe", S.phrase_tags_over_signs([(0, 3), (3, 6)], [(4, 6)])[0] == 1
      and len(S.phrase_tags_over_signs([], [])) == 0)

# ------------------------------------------------------------------ jitter
ok = True
for seed in range(200):
    r = np.random.default_rng(seed)
    T = 300
    sg = random_signs(T, r)
    j = S.jitter_segments(sg, T, r, strength=float(r.random() * 2))
    ok &= all(0 <= s < e <= T for s, e in j) and all(j[i][1] <= j[i + 1][0] for i in range(len(j) - 1))
check("jitter output always valid (sorted, disjoint, in range)", ok)
check("jitter strength 0 is the identity", S.jitter_segments(sg, 600, rng, strength=0.0) == sg)
chg = np.mean([S.jitter_segments(sg, 600, np.random.default_rng(i), strength=1.0) != sg for i in range(50)])
check("jitter strength 1 changes the segmentation", chg > 0.9, f"changed in {chg:.0%} of draws")

# ------------------------------------------------------------------ metrics
T = 400
gold = random_signs(T, rng)
r = M.evaluate_videos([gold], [gold], [T])
check("perfect prediction -> all metrics 1", all(abs(r[k] - 1) < 1e-9 for k in
      ("frame_f1", "mask_iou", "ratio", "start_f1@2", "end_f1@5", "seg_f1@0.7", "mean_seg_iou")))

from sklearn.metrics import f1_score
pred = S.jitter_segments(gold, T, np.random.default_rng(3), strength=1.5)
r = M.evaluate_videos([pred], [gold], [T])
ref = f1_score(S.segments_to_bio(gold, T), S.segments_to_bio(pred, T), average="macro", labels=[0, 1, 2])
check("frame macro-F1 equals sklearn", abs(r["frame_f1"] - ref) < 1e-9, f"{r['frame_f1']:.4f} vs {ref:.4f}")


def brute_match(a, b, tol):
    best = 0
    # exhaustive maximum matching on small sets
    for k in range(min(len(a), len(b)), -1, -1):
        for ia in itertools.combinations(range(len(a)), k):
            for ib in itertools.permutations(range(len(b)), k):
                if all(abs(a[i] - b[j]) <= tol for i, j in zip(ia, ib)):
                    return k
    return best


ok = True
for _ in range(60):
    a = sorted(rng.integers(0, 30, size=int(rng.integers(0, 6))).tolist())
    b = sorted(rng.integers(0, 30, size=int(rng.integers(0, 6))).tolist())
    tol = int(rng.integers(0, 4))
    ok &= M.match_points(a, b, tol) == brute_match(a, b, tol)
check("point matching equals exhaustive maximum matching", ok)

check("segment matching is one-to-one (a long gold cannot be hit twice)",
      M.match_segments([(0, 5), (5, 10)], [(0, 10)], 0.4) == 1 and M.match_segments([(0, 10)], [(0, 10)], 0.5) == 1)
r = M.evaluate_videos([[(0, 10), (10, 20), (20, 30)]], [[(0, 30)]], [30])
check("over-segmentation: ratio 3, mask IoU 1, boundary/segment F1 penalised",
      abs(r["ratio"] - 3) < 1e-9 and abs(r["mask_iou"] - 1) < 1e-9 and r["seg_f1@0.5"] == 0.0)

# ------------------------------------------------------------------ greedy decode
ok, T = True, 300
for seed in range(40):
    r_ = np.random.default_rng(seed)
    sg = random_signs(T, r_)
    bio = S.segments_to_bio(sg, T)
    P = np.full((T, 3), 0.05)
    P[np.arange(T), bio] = 0.9
    P += r_.random((T, 3)) * 0.02
    P /= P.sum(1, keepdims=True)
    dec = ST.greedy_decode(P)
    ok &= dec == sg
check("greedy decode recovers clean segments (touching and gapped)", ok)
P = np.zeros((10, 3)); P[:, 0] = 0.9; P[3:6, 2] = [0.6, 0.9, 0.7]; P[3:6, 0] = 0.1; P[3:6, 1] = 0.2
check("a run above the B threshold opens ONE segment at its peak", ST.greedy_decode(P) == [(4, 10)] or
      ST.greedy_decode(P)[0][0] == 4, str(ST.greedy_decode(P)))
check("decode on empty / all-outside is empty", ST.greedy_decode(np.tile([0.9, 0.05, 0.05], (20, 1))) == [])

# ------------------------------------------------------------------ tokens
T, H, fps = 400, 16, 25.0
xyz = rng.normal(size=(T, 65, 3)).astype(np.float32)
h = rng.normal(size=(T, H)).astype(np.float32)
P = ST.logits_to_probs(rng.normal(size=(T, 3)))
sg = random_signs(T, rng)
X, cols = ST.build_tokens(sg, T, fps, xyz=xyz, h=h, probs=P)
check("token matrix shape and finiteness", X.shape == (len(sg), 2 * H + ST.SIGN_PROB_DIM + ST.PROSODY_DIM)
      and np.isfinite(X).all(), f"{X.shape} cols={cols}")
a, b = cols["h_pool"]
ref_mean = np.stack([h[s:e].mean(0) for s, e in sg]); ref_max = np.stack([h[s:e].max(0) for s, e in sg])
check("h_pool = mean and max over the sign", np.allclose(X[:, a:a + H], ref_mean, atol=1e-5)
      and np.allclose(X[:, a + H:b], ref_max))
a, b = cols["sign_probs"]
check("sign_probs = mean/max/first/last", np.allclose(X[0, a:a + 3], P[sg[0][0]:sg[0][1]].mean(0), atol=1e-6)
      and np.allclose(X[0, a + 9:b], P[sg[0][1] - 1]))
a, b = cols["prosody"]
pro = X[:, a:b]
names = ST.PROSODY_NAMES
gaps = [sg[i + 1][0] - sg[i][1] for i in range(len(sg) - 1)]
check("prosody: duration and gap features", abs(pro[0, names.index("log_dur")] - np.log1p(sg[0][1] - sg[0][0])) < 1e-6
      and abs(pro[0, names.index("log_gap_next")] - np.log1p(gaps[0])) < 1e-6
      and all(pro[i, names.index("has_gap_next")] == float(gaps[i] > 0) for i in range(len(gaps))))
try:
    ST.build_tokens([(0, T + 5)], T, fps, xyz=xyz, h=h, probs=P)
    check("out-of-range segment is rejected with a clear error", False)
except ValueError:
    check("out-of-range segment is rejected with a clear error", True)
ok = True
for n in range(0, 31):                     # regression: videos with fewer than 11 signs used to crash
    v = rng.random(n) + 0.1
    lm = ST._local_mean(v)
    ref = np.array([v[max(0, i - 5):i + 6].mean() for i in range(n)])
    ok &= lm.shape == (n,) and np.allclose(lm, ref)
check("local mean is correct for every length 0..30 (short-video regression)", ok)
ok = True
for K in (1, 2, 3, 10, 11, 12):
    sgk = random_signs(60 * K, rng, mean_len=10, gap_p=0.0)[:K]
    Tk = sgk[-1][1] + 5
    Xk, _ = ST.build_tokens(sgk, Tk, fps, xyz=rng.normal(size=(Tk, 65, 3)).astype(np.float32), h=rng.normal(size=(Tk, H)).astype(np.float32),
                            probs=ST.logits_to_probs(rng.normal(size=(Tk, 3))))
    ok &= Xk.shape[0] == len(sgk) and np.isfinite(Xk).all()
check("build_tokens works for videos with very few signs", ok)
Xe, _ = ST.build_tokens([], T, fps, xyz=xyz, h=h, probs=P)
check("empty segmentation gives an empty (0, D) matrix", Xe.shape == (0, X.shape[1]))
Xp, cp = ST.build_tokens(sg, T, fps, probs=P, groups=("sign_probs",))
check("groups are independent (sign_probs only needs probs)", Xp.shape == (len(sg), ST.SIGN_PROB_DIM))
still = np.zeros((T, 65, 3), np.float32)
Xs, cs_ = ST.build_tokens(sg, T, fps, xyz=still, groups=("prosody",))
check("a motionless signer has zero speed features", np.allclose(Xs[:, names.index("speed_mean")], 0))

# ------------------------------------------------------------------ windows
ok = True
for K in (1, 5, 127, 128, 129, 300, 1000):
    w = S.plan_windows(K, 128, 64)
    cov = np.zeros(K, bool)
    for s, e in w:
        cov[s:e] = True
    ok &= cov.all() and all(e - s <= 128 for s, e in w)
    pr = [np.full(e - s, 0.3) for s, e in w]
    ok &= np.allclose(S.stitch_probs(K, w, pr), 0.3)
check("token windows cover every token and stitching averages overlaps", ok)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)