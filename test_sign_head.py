"""
Smoke tests for the sign-head study (queue_train_phrase.py SIGN_HEAD_STUDY). Run from the repo root:
    python test_sign_head.py
Synthetic data in a temporary folder only.

  1. label helpers: sign starts survive frame-rate resampling and downsampling
  2. soft pause features (torch): on a hard 0/1 track they equal the exact pause length
  3. sign-start extraction and sign-level targets
  4. dataset: sign targets are the last extra on train (also with frame-rate augmentation), absent on val
  5. every SIGN_HEAD_STUDY arm: one training step exactly as train_phrase.py computes the losses,
     then evaluation mode (valid probabilities; sign-level runs put P(B) only on predicted sign starts)
  6. old checkpoints: the model without a sign head is unchanged
  7. predict_video_probs(with_sign=True) and the decoding-study selftest
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import queue_train_phrase as q                                                  # noqa: E402
from src.aux_labels import inv_sqrt_weights                                     # noqa: E402
from src.dataset import (SignSegmentationDataset, downsample_hard_labels,       # noqa: E402
                         extra_dataset_kwargs, resample_hard_labels)
from src.evaluation import predict_video_probs                                  # noqa: E402
from src.model_factory import build_model_kwargs                                # noqa: E402
from src.sign_cascade import (segment_targets, soft_tempo_features,             # noqa: E402
                              starts_from_labels, starts_from_logits)
from test_temporal_context import _full_config, _make_data                      # noqa: E402

OK = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    OK.append(bool(cond))


def test_label_helpers():
    print("1. label helpers")
    lab = np.ones(1000, dtype=np.int64)
    lab[::25] = 2
    lab[:5] = 0
    idx = np.arange(0, 1000, 2)
    out = resample_hard_labels(lab, idx, 1000)
    check("resampling keeps every sign start", (out == 2).sum() == (lab == 2).sum(), f"{(out == 2).sum()} starts")
    d = downsample_hard_labels(torch.from_numpy(lab), 3)
    check("downsampling keeps every sign start", int((d == 2).sum()) == int((lab == 2).sum()))


def test_tempo():
    print("2. soft pause features")
    rng = np.random.default_rng(0)
    T = 300
    o = (rng.random(T) < 0.3).astype(np.float32)
    o[50:90] = 1
    probs = torch.zeros(1, 3, T)
    probs[0, 0] = torch.from_numpy(o)
    probs[0, 1] = torch.from_numpy(1 - o)
    f = soft_tempo_features(probs)
    exact = np.zeros(T)
    for t in range(1, T):
        n, j = 0, t - 1
        while j >= 0 and o[j] == 1 and n < 100:
            n, j = n + 1, j - 1
        exact[t] = n
    err = np.abs(np.expm1(f[0, 3].numpy()) - exact).max()
    check("log pause == exact pause length", err < 0.05, f"max error {err:.4f} frames")
    check("6 channels, finite", f.shape == (1, 6, T) and bool(torch.isfinite(f).all()))


def test_starts():
    print("3. sign starts and sign-level targets")
    sign = torch.tensor([[0, 0, 2, 1, 1, 2, 1, 0, 0, 1, 1, 2, 2, 1, -100, -100]])
    st = starts_from_labels(sign)[0].tolist()
    check("starts from labels", st == [2, 5, 9, 11], str(st))
    logits = F.one_hot(sign.clamp_min(0), 3).permute(0, 2, 1).float() * 5
    check("starts from logits (merge < 3 frames)", starts_from_logits(logits, 3)[0].tolist() == [2, 5, 9], "")
    phrase = torch.tensor([[0, 0, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, -100, -100]])
    y = segment_targets([torch.tensor([2, 5, 9, 11])], phrase, tol=1)[0].tolist()
    check("one positive per gold phrase start", y == [1.0, 0.0, 1.0, 0.0], str(y))


def test_dataset(d):
    print("4. dataset")
    base = {**q.MAMBA_DEFAULTS, **q._MS_BEST, **q._SH, "sign_labels_dir": f"{d}/sign",
            "kinematic_dir": f"{d}/kin1"}

    def ds(cfg, split):
        return SignSegmentationDataset(
            keypoints_dir=f"{d}/kp", labels_dir=f"{d}/labels", split_file=f"{d}/splits.json", split=split,
            window_size=512, overlap=0, tolerance_window=1, base_features=cfg["base_features"],
            kinematic_features=[], use_face_keypoints=True, face_dir=f"{d}/face", face_subset="compact",
            face_dims=2, use_hamer_features=True, hamer_dir=f"{d}/hamer", **extra_dataset_kwargs(cfg))
    tr, va = ds(base, "train"), ds(base, "val")
    item = tr[0]
    check("train: sign targets are the last extra", item[-4].dtype == torch.int64 and item[-4].shape == (512,)
          and int((item[-4] == 2).sum()) > 0, f"{int((item[-4] == 2).sum())} sign starts in window 0")
    check("val: no sign targets", len(va[0]) == len(item) - 1)
    check("sign class counts", tr.sign_class_counts.sum() > 0, str(tr.sign_class_counts.tolist()))
    fa = ds({**base, "fps_augment_range": [0.5, 0.5]}, "train")
    it = fa[1]
    check("frame-rate augmentation: sign targets resampled", it[-4].shape == (512,), str(tuple(it[-4].shape)))
    return tr


def test_models(train_ds):
    print("5. every SIGN_HEAD_STUDY arm: one training step + evaluation mode")
    torch.manual_seed(0)
    T = 128
    sign_w = torch.tensor(inv_sqrt_weights([300, 600, 40]), dtype=torch.float)
    for exp in q.SIGN_HEAD_STUDY:
        c = _full_config(q, exp)
        cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
        model = cls(**kw)
        model.train()
        x, ham = torch.randn(2, c["in_channels"], T, c["num_vertices"]), torch.randn(2, 288, T)
        phrase = torch.zeros(2, T, dtype=torch.long)
        sign = torch.ones(2, T, dtype=torch.long)
        sign[:, ::16] = 2
        phrase[:, ::64] = 2
        phrase[:, 1:] = torch.where(phrase[:, 1:] == 2, phrase[:, 1:], torch.ones_like(phrase[:, 1:]))
        sign[:, -5:] = -100
        if c.get("sign_level"):
            model.set_targets(phrase=phrase, sign=sign)
        logits = model(x, hamer=ham)[0]
        loss = F.cross_entropy(logits, phrase, weight=torch.tensor([1.0, 1.1, 8.5]))
        loss = loss + c["sign_loss_weight"] * F.cross_entropy(model._sign_logits, sign, weight=sign_w,
                                                              ignore_index=-100)
        if c.get("sign_level") and "sign_level" in model._extra_losses:
            loss = loss + model._extra_losses["sign_level"]
        for task, lg in model._aux_logits.items():
            n_cls = lg.shape[1]
            loss = loss + 0.3 * F.cross_entropy(lg, torch.randint(0, n_cls, (2, T)))
        loss.backward()
        skip = () if "sign_level" in model._extra_losses else ("sign_level_decoder",)
        grads = all(p.grad is not None for n, p in model.named_parameters()
                    if n.startswith(("sign_", "classifier")) and not n.startswith(skip) and p.requires_grad)
        model.eval()
        with torch.no_grad():
            out = model(x, hamer=ham)[0]
        p = out.float().softmax(1)
        valid = bool(torch.isfinite(out).all()) and torch.allclose(p.sum(1), torch.ones(2, T), atol=1e-4)
        extra = ""
        if c.get("sign_level"):
            st = starts_from_logits(model._sign_logits)
            off = torch.ones(2, T, dtype=torch.bool)
            for b, s in enumerate(st):
                off[b, s] = False
            valid = valid and float(p[:, 2][off].max()) < 1e-3
            extra = f", P(B) only at {sum(len(s) for s in st)} predicted sign starts"
        n = sum(p_.numel() for p_ in model.parameters())
        check(exp["description"][:62], valid and grads and bool(torch.isfinite(loss)),
              f"loss {loss.item():.3f}, {n / 1e6:.2f}M params{extra}")


def test_compat():
    print("6. old checkpoints")
    c = _full_config(q, q._MS_BEST)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    check("model without sign head gets no new arguments", not any(k.startswith("sign") for k in kw))
    keys = set(cls(**kw).state_dict())
    check("no new parameters without the sign head", not any(k.startswith(("sign_", "sign_level")) for k in keys))


def test_predict(train_ds):
    print("7. evaluation helpers")
    c = _full_config(q, {**q._MS_BEST, **q._SH, "window_size": 512})
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    model = cls(**kw).eval()
    cached = next(iter(train_ds.video_cache.values()))
    ph, sg = predict_video_probs(model, c, cached, torch.device("cpu"), 4, with_sign=True)
    check("phrase and sign probabilities per frame", ph.shape == sg.shape == (3, cached["features"].shape[0])
          and np.allclose(sg.sum(0), 1, atol=1e-3))
    import sign_decoding_study
    check("sign_decoding_study selftest", sign_decoding_study.selftest())


if __name__ == "__main__":
    os.chdir(ROOT)
    test_label_helpers()
    test_tempo()
    test_starts()
    with tempfile.TemporaryDirectory() as d:
        vids = _make_data(d)
        os.makedirs(f"{d}/sign")
        for v in vids:
            T = len(np.load(f"{d}/labels/{v}.npy"))
            s = np.ones(T, dtype=np.int64)
            s[::20] = 2
            np.save(f"{d}/sign/{v}.npy", s)
        tr = test_dataset(d)
        test_models(tr)
        test_compat()
        test_predict(tr)
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)