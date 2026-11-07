"""
Smoke tests for the Moryossef 2026 architecture study (queue_train_phrase.py MORYOSSEF2026_STUDY).
Run from the repo root:
    python test_moryossef2026.py
Synthetic data in a temporary folder only.

  1. UNet blocks: output length == input length for even, odd and non-multiple-of-8 lengths
  2. RoPE layer: depends only on relative position (shifting every position changes nothing)
  3. every MORYOSSEF2026_STUDY arm, with its config built as queue_train_phrase.py builds it:
     one training step with the losses of train_phrase.py (phrase CE [+ sign CE]), every parameter
     gets a gradient, evaluation mode gives valid probabilities; sign logits exist iff sign_head
  4. old checkpoints: the multi-stream BiLSTM gets no new arguments / parameters
  5. dataset + predict_video_probs(with_sign=True) on the 2026 model (keypoints only, no HaMeR)
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
from src.dataset import SignSegmentationDataset, extra_dataset_kwargs           # noqa: E402
from src.evaluation import predict_video_probs                                  # noqa: E402
from src.model_factory import build_model_kwargs                                # noqa: E402
from src.moryossef2026 import (STAGE1_CONVS, STAGE2_CONVS, PoseEncoderUNetBlock,  # noqa: E402
                               RoPETransformerEncoderLayer)
from test_temporal_context import _make_data                                    # noqa: E402

OK = []
MULTISTREAM = ("multistream_bilstm", "multistream_transformer", "multistream_bilstm_attn", "multistream_fusion",
               "multistream_unet_rope")


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    OK.append(bool(cond))


def queue_config(exp):
    """The config exactly as queue_train_phrase.py's __main__ builds it."""
    c = dict(q.MAMBA_DEFAULTS)
    c.update(exp)
    if c["basename"] in MULTISTREAM:
        c["use_face_keypoints"] = bool(c.get("use_stream_face", False))
        c["use_hamer_features"] = bool(c.get("use_stream_hamer", False))
        c["use_cue_features"] = bool(c.get("use_stream_cues", False))
    c["in_channels"] = q.calculate_in_channels(c)
    c["num_vertices"] = q.calculate_num_vertices(c)
    return c


def test_unet():
    print("1. UNet blocks keep the length")
    torch.manual_seed(0)
    s1 = PoseEncoderUNetBlock(7, 16, STAGE1_CONVS(3)).eval()
    s2 = PoseEncoderUNetBlock(16, 16, STAGE2_CONVS).eval()
    for T in (512, 500, 97, 33):
        h = s1(torch.randn(2, T, 7, 3))
        out = s2(h[..., None])
        check(f"T = {T}", h.shape == (2, T, 16) and out.shape == (2, T, 16), f"{tuple(h.shape)} -> {tuple(out.shape)}")


def test_rope():
    print("2. RoPE: relative positions only")
    torch.manual_seed(0)
    layer = RoPETransformerEncoderLayer(64, 4, 128, dropout=0.0).eval()
    x = torch.randn(2, 50, 64)
    with torch.no_grad():
        a = layer(x, torch.arange(50, dtype=torch.float32))
        b = layer(x, torch.arange(50, dtype=torch.float32) + 37)
        c = layer(x, torch.arange(50, dtype=torch.float32) * 2)
    err = float((a - b).abs().max())
    check("shifting every position changes nothing", err < 1e-4, f"max diff {err:.1e}")
    check("changing the spacing does change the output", float((a - c).abs().max()) > 1e-3)


def test_arms():
    print("3. every MORYOSSEF2026_STUDY arm: one training step + evaluation mode")
    T = 128
    for exp in q.MORYOSSEF2026_STUDY:
        torch.manual_seed(0)
        c = queue_config(exp)
        cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
        model = cls(**kw)
        model.train()
        x = torch.randn(2, c["in_channels"], T, c["num_vertices"])
        ham = torch.randn(2, 288, T)
        phrase = torch.ones(2, T, dtype=torch.long)
        phrase[:, ::40] = 2
        phrase[:, -6:] = -100
        sign = torch.ones(2, T, dtype=torch.long)
        sign[:, ::12] = 2
        kwargs = {"hamer": ham} if c.get("use_hamer_features") else {}
        logits, emb = model(x, **kwargs)
        loss = F.cross_entropy(logits, phrase, weight=torch.tensor([1.0, 1.1, 8.5]), ignore_index=-100)
        has_sign = model._sign_logits is not None
        if c.get("sign_head"):
            loss = loss + c["sign_loss_weight"] * F.cross_entropy(model._sign_logits, sign, ignore_index=-100)
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        model.eval()
        with torch.no_grad():
            out = model(x, **kwargs)[0]
        p = out.float().softmax(1)
        valid = bool(torch.isfinite(out).all()) and float((p.sum(1) - 1).abs().max()) < 1e-4
        n = sum(p_.numel() for p_ in model.parameters())
        check(exp["description"][:60],
              valid and not no_grad and out.shape == (2, 3, T) and has_sign == bool(c.get("sign_head"))
              and bool(torch.isfinite(loss)),
              f"{cls.__name__}, d_model {kw['d_model']}, hamer {'yes' if 'hamer_dim' in kw else 'no'}, "
              f"sign head {'yes' if has_sign else 'no'}, loss {loss.item():.3f}, {n / 1e6:.2f}M params"
              + (f", NO GRAD: {no_grad[:4]}" if no_grad else ""))


def test_compat():
    print("4. old checkpoints")
    c = queue_config(q._MS_BEST)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    check("multi-stream BiLSTM gets no new arguments",
          not any(k in kw for k in ("attn_ff_mult", "attn_dropout", "unet_stage")))
    keys = set(cls(**kw).state_dict())
    check("no new parameters in the multi-stream BiLSTM", not any("unet_rope" in k for k in keys))


def test_predict(d):
    print("5. dataset + evaluation helper on the 2026 model")
    exp = next(e for e in q.MORYOSSEF2026_STUDY if e["description"].startswith("M26a"))
    c = {**queue_config(exp), "sign_labels_dir": f"{d}/sign"}
    ds = SignSegmentationDataset(
        keypoints_dir=f"{d}/kp", labels_dir=f"{d}/labels", split_file=f"{d}/splits.json", split="train",
        window_size=512, overlap=0, tolerance_window=1, base_features=c["base_features"], kinematic_features=[],
        use_face_keypoints=c["use_face_keypoints"], face_dir=f"{d}/face", face_subset=c["face_subset"],
        face_dims=2, use_hamer_features=c["use_hamer_features"], hamer_dir=f"{d}/hamer",
        **{**extra_dataset_kwargs(c), "kinematic_dir": f"{d}/kin1"})
    item = ds[0]
    check("train window: keypoints (C, T, V) match the model", tuple(item[0].shape) == (c["in_channels"], 512,
                                                                                     c["num_vertices"]),
          str(tuple(item[0].shape)))
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    model = cls(**kw).eval()
    cached = next(iter(ds.video_cache.values()))
    ph, sg = predict_video_probs(model, c, cached, torch.device("cpu"), 4, with_sign=True)
    check("phrase and sign probabilities per frame", ph.shape == sg.shape == (3, cached["features"].shape[0])
          and np.allclose(ph.sum(0), 1, atol=1e-3) and np.allclose(sg.sum(0), 1, atol=1e-3))


if __name__ == "__main__":
    os.chdir(ROOT)
    test_unet()
    test_rope()
    test_arms()
    test_compat()
    with tempfile.TemporaryDirectory() as d:
        vids = _make_data(d)
        os.makedirs(f"{d}/sign")
        for v in vids:
            T = len(np.load(f"{d}/labels/{v}.npy"))
            s = np.ones(T, dtype=np.int64)
            s[::20] = 2
            np.save(f"{d}/sign/{v}.npy", s)
        test_predict(d)
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)