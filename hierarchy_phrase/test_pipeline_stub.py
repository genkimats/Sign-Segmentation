"""
test_pipeline_stub.py -- exercises the NON-torch logic of stage_a.py / stage_c.py (cache I/O, training-source policies,
token + tag assembly, fold splitting, class weights) on a synthetic cache with the real on-disk layout.
Uses the real torch if installed; otherwise a stub that only provides what the module bodies touch at import time.
"""
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import torch  # noqa: F401
    REAL_TORCH = True
except ImportError:
    REAL_TORCH = False

    class _Mod:                                    # base class stand-in for nn.Module
        pass
    t = types.ModuleType("torch"); nn = types.ModuleType("torch.nn"); fn = types.ModuleType("torch.nn.functional")
    nn.Module = _Mod; nn.functional = fn; t.nn = nn
    t.no_grad = lambda: (lambda f: f)
    for name, m in (("torch", t), ("torch.nn", nn), ("torch.nn.functional", fn)):
        sys.modules[name] = m

import common as C
import segments as S
from sign_tokens import PROSODY_DIM, SIGN_PROB_DIM
import stage_a as A
import stage_c as SC

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


rng = np.random.default_rng(0)
tmp = tempfile.mkdtemp()
cache = os.path.join(tmp, "sa", "cache", "train")


def make_video(i, T=900):
    t, sg = 6, []
    while t + 20 < T:
        L = int(rng.integers(6, 30)); sg.append((t, t + L)); t += L + (int(rng.integers(1, 6)) if rng.random() < .5 else 0)
    sg = [(s, e) for s, e in sg if e <= T]
    cuts = [0] + sorted(rng.choice(np.arange(1, len(sg)), size=len(sg) // 5, replace=False).tolist()) + [len(sg)]
    ph = [(sg[a][0], sg[b - 1][1]) for a, b in zip(cuts[:-1], cuts[1:])]
    bio = S.segments_to_bio(sg, T)
    logit = np.log(np.full((T, 3), 0.05)); logit[np.arange(T), bio] = np.log(0.9)
    logit += rng.normal(scale=0.3, size=(T, 3))
    return dict(xyz=rng.normal(size=(T, 65, 3)).astype(np.float16), h=rng.normal(size=(T, 32)).astype(np.float16),
                sign_logits=logit.astype(np.float32), phrase_logits=rng.normal(size=(T, 3)).astype(np.float32),
                gold_sign=C.segs_to_arr(sg), gold_phrase=C.segs_to_arr(ph), T=np.int64(T), stride=np.int64(2),
                has_phrase_head=np.int64(1)), sg, ph


gold = {}
for i in range(6):
    d, sg, ph = make_video(i)
    C.save_cache_video(os.path.join(cache, f"v{i}_A.npz"), **d)
    gold[f"v{i}_A"] = (sg, ph)

items = SC.load_items([cache])
check("load_items reads the cache with the right fields", len(items) == 6 and items[0]["xyz"].shape == (900, 65, 3)
      and items[0]["sign_probs"].shape == (900, 3) and np.allclose(items[0]["sign_probs"].sum(1), 1, atol=1e-5))
check("gold segments round-trip through the npz cache", all(it["gold_sign"] == gold[it["vid"]][0]
                                                           and it["gold_phrase"] == gold[it["vid"]][1] for it in items))

G = ("h_pool", "sign_probs", "prosody")
D = 2 * 32 + SIGN_PROB_DIM + PROSODY_DIM
for src in ("gold", "jitter", "schedule", "mix", "pred"):
    pol = SC.SourcePolicy(src, ramp=5, jitter_max=1.0, b_thr=0.5, o_thr=0.5, seed=1)
    ok = True
    for epoch in (1, 3, 6):
        ex = SC.make_examples(items, pol, epoch, G)
        ok &= len(ex) == len(items) and all(x.shape[1] == D and len(t) == len(x) and np.isfinite(x).all() for x, t in ex)
        ok &= all(t[0] == 1 for _, t in ex)
    check(f"source '{src}': examples are well-formed at every epoch", ok)

pol_g = SC.SourcePolicy("gold", 5, 1.0, .5, .5, 0)
ex_g = SC.make_examples(items, pol_g, 1, G)
check("gold source: tags equal phrase starts exactly", all(
    [i for i, t in enumerate(tg) if t] == [k for k, (s, _) in enumerate(it["gold_sign"]) if any(s == p for p, _ in it["gold_phrase"])]
    for (x, tg), it in zip(ex_g, items)))
pol_j = SC.SourcePolicy("jitter", 5, 1.0, .5, .5, 0)
check("jitter source starts at gold (strength ramps from 0)", pol_j.segs(items[0], 0) == items[0]["gold_sign"]
      and pol_j.segs(items[0], 5) != items[0]["gold_sign"])
pol_p = SC.SourcePolicy("pred", 5, 1.0, .5, .5, 0)
pr = pol_p.segs(items[0], 1)
check("pred source decodes the Stage-A sign head (close to gold on this clean synthetic cache)",
      abs(len(pr) - len(items[0]["gold_sign"])) <= 0.1 * len(items[0]["gold_sign"]), f"{len(pr)} vs {len(items[0]['gold_sign'])}")
check("predicted-sign decoding is cached per video", pol_p.pred(items[0]) is pol_p.pred(items[0]))

segs = items[0]["gold_sign"]
check("phrases_from_pB reproduces gold phrases from oracle probabilities",
      SC.phrases_from_pB(segs, S.phrase_tags_over_signs(segs, items[0]["gold_phrase"]).astype(float), 0.5) == items[0]["gold_phrase"])
check("phrases_from_pB on empty segmentation is empty", SC.phrases_from_pB([], np.zeros(0), 0.5) == [])

# ---- stage A helpers
ids = [f"{d}_{s}" for d in range(40) for s in "AB"]
tr, held = A.fold_split(ids, 1, 4)
check("fold_split keeps both signers of a document together and partitions the data",
      set(tr) | set(held) == set(ids) and not set(tr) & set(held) and
      all((v.rsplit('_', 1)[0] + '_A' in held) == (v.rsplit('_', 1)[0] + '_B' in held) for v in held))
folds = [set(A.fold_split(ids, f, 4)[1]) for f in range(4)]
check("the K held-out folds are disjoint and cover every video", sum(len(f) for f in folds) == len(ids) and len(set().union(*folds)) == len(ids))
labs = [np.array([0] * 90 + [1] * 8 + [2] * 2)]
w = A.class_weights(labs, 0.5)
check("class weights: O = 1, rarer classes up-weighted, monotone in rarity", w[0] == 1 and w[2] > w[1] > w[0], str(np.round(w, 2)))

print(f"\n{sum(results)}/{len(results)} checks passed   (torch: {'real' if REAL_TORCH else 'stub'})")
import shutil; shutil.rmtree(tmp)
sys.exit(0 if all(results) else 1)