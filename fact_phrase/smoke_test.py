"""
smoke_test.py -- end-to-end check of the FACT pilot on SYNTHETIC data (no exported features needed).
Run from the repo root on the training machine (CPU is fine, ~2-5 min; GPU faster):

    python fact_phrase/smoke_test.py

What it checks
  1. fact_core self-tests (numpy).
  2. Parameter counts of the default FACT and frozen-BiLSTM configs at the real input size (256).
  3. One FACT window: forward, loss finite, gradients reach the frame branch AND the action tokens,
     predict() shapes; works with a window that has more segments than tokens (unmatched segments).
  4. The real data path (FeatureStore -> TrainWindows -> DataLoader -> loss -> predict_store ->
     src.metrics) on a synthetic store where phrase starts are visible in the features:
     a small FACT and a small frozen BiLSTM must both LEARN it (val boundary F1@5 well above chance),
     with feature_stride 1 and 2.
If any step fails the script stops with the reason. Passing does not say anything about real data;
it says the plumbing, losses and decoding work.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fact_phrase import fact_core as core  # noqa: E402
from fact_phrase.data import (FeatureStore, TrainWindows, decoded_probs, list_collate, pad_batch,  # noqa: E402
                              predict_store)
from fact_phrase.fact_model import BILSTM_DEFAULTS, FACT_DEFAULTS, build_model, count_parameters  # noqa: E402
from fact_phrase.train_fact import TRAIN_DEFAULTS  # noqa: E402
from src.metrics import evaluate_videos  # noqa: E402

RNG = np.random.default_rng(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def step(msg):
    print(f"\n== {msg}")


def synth_bio(T):
    out = np.zeros(T, np.int64)
    t = int(RNG.integers(0, 40))
    while t < T:
        L = int(RNG.integers(40, 160))
        out[t:t + L] = 1
        out[t] = 2
        t += L
        if RNG.random() < 0.5:                       # half of the boundaries are continuous (no O gap)
            t += int(RNG.integers(5, 50))
    return out


def synth_features(bio, D=16):
    """Phrase interior on channel 0, a short bump at every phrase start on channel 1, noise elsewhere."""
    T = len(bio)
    x = RNG.normal(0, 0.3, size=(T, D)).astype(np.float32)
    x[:, 0] += (bio != 0) * 1.0
    starts = np.flatnonzero(core.phrase_starts_mask(bio))
    for s in starts:
        lo, hi = max(0, s - 2), min(T, s + 3)
        x[lo:hi, 1] += 2.0
    return x


def make_store(tmp, n_train=24, n_val=6, D=16):
    src = "synthetic-01"
    labels = os.path.join(tmp, "labels")
    os.makedirs(labels)
    meta = {"source_run": src, "labels_dir": labels, "dim": D, "splits": {}}
    for split, n in (("train", n_train), ("val", n_val)):
        d = os.path.join(tmp, "features", src, split)
        os.makedirs(d)
        vids, frames = [], {}
        for i in range(n):
            vid = f"{split}{i:03d}_a"
            bio = synth_bio(int(RNG.integers(1500, 4000)))
            np.save(os.path.join(labels, f"{vid}.npy"), bio)
            np.save(os.path.join(d, f"{vid}.npy"), synth_features(bio, D).astype(np.float16))
            vids.append(vid)
            frames[vid] = len(bio)
        meta["splits"][split] = {"videos": vids, "frames": frames}
    with open(os.path.join(tmp, "features", src, "meta.json"), "w") as f:
        json.dump(meta, f)
    return os.path.join(tmp, "features"), src


def small_cfg(model, W):
    if model == "fact":
        return {**FACT_DEFAULTS, **TRAIN_DEFAULTS["fact"], "hid_dim": 64, "f_dim": 32, "a_dim": 32, "a_ffdim": 64,
                "a_nhead": 4, "a_layers": 2, "f_layers": 8, "u_f_layers": 6, "window_frames": W, "eval_step_frac": 0.5}
    return {**BILSTM_DEFAULTS, **TRAIN_DEFAULTS["frozen_bilstm"], "d_model": 32, "n_layers": 2, "window_frames": W,
            "eval_step_frac": 1.0}


def train_small(model_name, root, src, stride, steps, W=600):
    cfg = small_cfg(model_name, W)
    cfg["feature_stride"] = stride
    tr = FeatureStore(src, "train", stride, root=root)
    va = FeatureStore(src, "val", stride, root=root)
    Wm = W // stride
    if model_name == "fact":
        mx, mean, _ = core.window_segment_stats(tr.model_rate_labels(), Wm, Wm // 4)
        cfg["ntoken"] = core.auto_ntoken(mx)
        cfg["nullw_resolved"] = core.null_weight(cfg["ntoken"], mean)
    model = build_model(cfg, tr.dim).to(DEV)
    if model_name == "fact":
        model.attach_criterion(cfg["nullw_resolved"], DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 2 if model_name == "fact" else 8
    loader = DataLoader(TrainWindows(tr, Wm, steps * bs), batch_size=bs, collate_fn=list_collate, num_workers=0)
    losses = []
    model.train()
    for batch in loader:
        opt.zero_grad()
        if model_name == "fact":
            loss = sum(model.loss(x.to(DEV), y.to(DEV)) for x, y in batch) / len(batch)
        else:
            X, Y = pad_batch(batch, DEV)
            loss = model.loss_batch(X, Y)
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        losses.append(float(loss))
    preds = predict_store(model, cfg, va, DEV)
    gold = {v: va.videos[v]["gold"] for v in preds}
    for v, p in preds.items():
        assert p["head"].shape == (3, len(gold[v])), "head predictions must be at the raw frame rate"
        if p["token"] is not None:
            assert p["token"].shape == (len(gold[v]),)
    res = {}
    for dec in (["token", "head"] if model_name == "fact" else ["head"]):
        m = evaluate_videos(decoded_probs(preds, dec), gold, light=True)
        res[dec] = m["BndF1_5"]
    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    return first, last, res, cfg


def main():
    step("1. fact_core self-tests")
    r = subprocess.run([sys.executable, os.path.join("fact_phrase", "test_fact_core.py")], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr)
    if r.returncode != 0:
        raise SystemExit(r.stdout + r.stderr)

    step("2. parameter counts at the real input size (256)")
    fact = build_model({**FACT_DEFAULTS, "ntoken": 64}, 256)
    bil = build_model(dict(BILSTM_DEFAULTS), 256)
    print(f"   FACT default (iuUU, hid 256, ntoken 64): {count_parameters(fact) / 1e6:.2f} M")
    print(f"   frozen BiLSTM (4 x 256, bidirectional):   {count_parameters(bil) / 1e6:.2f} M   (R0's encoder: 5.78 M + classifier)")

    step("3. one FACT window: forward / loss / gradients / predict")
    cfg = {**FACT_DEFAULTS, "ntoken": 8, "hid_dim": 64, "f_dim": 32, "a_dim": 32, "a_ffdim": 64, "a_layers": 2,
           "f_layers": 6, "u_f_layers": 4}
    model = build_model(cfg, 16).to(DEV)
    model.attach_criterion(core.null_weight(8, 4.0), DEV)
    bio = synth_bio(700)                                       # usually > 8 segments: some stay unmatched
    x = torch.from_numpy(synth_features(bio)).to(DEV)
    y = torch.from_numpy(bio).to(DEV)
    model.train()
    loss = model.loss(x, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_query.grad is not None and model.action_query.grad.abs().sum() > 0, "no gradient to tokens"
    g_frame = model.block_list[0].frame_branch.conv_1x1_in.weight.grad
    assert g_frame is not None and g_frame.abs().sum() > 0, "no gradient to the frame branch"
    model.eval()
    head, token = model.predict(x)
    assert head.shape == (3, 700) and token.shape == (700,) and np.allclose(head.sum(0), 1, atol=1e-4)
    print(f"   loss {float(loss):.3f}, {core.count_segments(bio)} segments vs 8 tokens -- ok")

    tmp = tempfile.mkdtemp(prefix="fact_smoke_")
    try:
        root, src = make_store(tmp)
        for stride in (1, 2):
            step(f"4. learn a synthetic signal through the real data path (feature_stride {stride})")
            for name, steps in (("frozen_bilstm", 150), ("fact", 150)):
                first, last, res, _ = train_small(name, root, src, stride, steps)
                print(f"   {name:14s} loss {first:.3f} -> {last:.3f}   val BndF1@5 " +
                      ", ".join(f"{d} {v:.3f}" for d, v in res.items()))
                assert last < first, f"{name}: loss did not decrease"
                assert max(res.values()) > 0.5, f"{name}: did not learn the synthetic boundaries"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
