"""
End-to-end check of balance_study on random data (CPU is fine, ~1-3 minutes). Run from the repo root BEFORE
queueing the study:
    python balance_study/smoke_test.py

Checks:
  1. the split forward (_encode_streams + _decode_streams) gives exactly the original forward;
  2. the balance heads do not change the prediction, and a model with heads reloads strictly through
     build_model_kwargs (what evaluate_phrase.py does);
  3. every method of balance_study/experiments.py takes real optimizer steps (finite loss, parameters move,
     the learning rates are restored, balance_log.csv is written), including Wu's re-balancing steps and
     InfoReg's regulated steps;
  4. AGM's Shapley responses add up to the full logits;
  5. invalid combinations are rejected.
"""
import copy
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_factory import build_model_kwargs  # noqa: E402
from balance_study.balancer import Balancer, validate_balance_config  # noqa: E402
from balance_study.experiments import imbalance_arms  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"
B, T, V = 2, 48, 96
W = [1.0, 1.1, 8.5]

BASE = {
    "basename": "multistream_bilstm", "in_channels": 3, "num_vertices": V, "d_model": 32, "n_layers": 2,
    "base_features": ["x-cord", "y-cord", "z-cord"], "kinematic_features": [],
    "use_stream_body_hands": True, "use_stream_face": True, "use_stream_hamer": True,
    "use_face_keypoints": True, "use_hamer_features": True, "face_subset": "compact", "face_dims": 2,
    "fusion": "concat", "body_hands_proj_dim": 32, "face_proj_dim": 32, "hamer_proj_dim": 32,
    "hamer_dim": 288, "loss_function": "weighted_ce", "class_weights": W, "window_size": T,
    "use_full_length": False, "temporal_downsample_factor": 1,
}


def build(cfg):
    cls, kw = build_model_kwargs(cfg)
    return cls(**kw).to(DEV)


def batch():
    x = torch.randn(B, 3, T, V)
    hamer = torch.randn(B, 288, T)
    y = torch.zeros(B, T, dtype=torch.long)
    y[:, 5:20] = 1
    y[:, 5] = 2
    y[:, 25:40] = 1
    y[:, 25] = 2
    labels = torch.nn.functional.one_hot(y, 3).permute(0, 2, 1).float()
    labels[1, :, -4:] = 0.0                                   # padding frames
    return x, hamer, labels


def check_forward():
    m = build(BASE).eval()
    x, h, _ = batch()
    with torch.no_grad():
        a, ea = m(x, hamer=h)
        b, eb = m._decode_streams(*m._encode_streams(x, hamer=h))
    assert torch.equal(a, b) and torch.equal(ea, eb), "split forward differs from forward"

    cfg_h = {**BASE, "balance_heads": ["body_hands", "face", "hamer"], "balance_head_detach": True}
    mh = build(cfg_h).eval()
    missing, unexpected = mh.load_state_dict(m.state_dict(), strict=False)
    assert unexpected == [] and all(k.startswith("bal_heads.") for k in missing)
    with torch.no_grad():
        c, _ = mh(x, hamer=h)
    assert torch.allclose(a, c), "balance heads changed the prediction"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pth")
        torch.save(mh.state_dict(), p)
        build(cfg_h).load_state_dict(torch.load(p), strict=True)
    # stream-mean ablation of several streams at once runs
    zero = {s: torch.zeros(m.stream_dims[s]) for s in m.streams}
    m.set_ablations([("face", zero["face"]), ("hamer", zero["hamer"])])
    with torch.no_grad():
        m(x, hamer=h)
    m.set_ablations(None)
    print("ok  split forward, heads, strict reload, multi-stream ablation")


def teachers():
    t = {}
    for s, flags in (("body_hands", (True, False, False)), ("face", (False, True, False)),
                     ("hamer", (False, False, True))):
        cfg = {**BASE, "use_stream_body_hands": flags[0], "use_stream_face": flags[1], "use_stream_hamer": flags[2]}
        m = build(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        t[s] = m
    return t


def run_arm(arm, epochs=3, steps=3, tweak=None):
    cfg = {**BASE, **{k: v for k, v in arm.items() if k.startswith("balance")}}
    cfg["balance"] = dict(cfg["balance"])
    if tweak:
        cfg["balance"].update(tweak)
    validate_balance_config(cfg)
    model = build(cfg).train()
    with tempfile.TemporaryDirectory() as d:
        t = teachers() if cfg["balance"]["method"] == "umt" else None
        if t is not None:
            t = {s: t[s] for s in cfg["balance"]["teachers"]}
        bal = Balancer(cfg, model, DEV, W, exp_dir=d, teachers=t)
        opt = torch.optim.AdamW(bal.optimizer_param_groups(), lr=1e-3, weight_decay=0.01)
        before = copy.deepcopy(model.state_dict())
        lrs = [g["lr"] for g in opt.param_groups]
        for ep in range(1, epochs + 1):
            if cfg["balance"]["method"] == "inforeg" and ep == 3:
                bal.tr_hist = {1: [1.0] * bal.M, 2: [2.0] * bal.M}     # force the prime window open
            bal.start_epoch(ep)
            for _ in range(steps):
                x, h, y = batch()
                loss = bal.step(x, y, h, None, opt, 1.0)
                assert loss is not None and np.isfinite(loss), f"{cfg['balance']}: loss {loss}"
            bal.end_epoch(ep, {"Frame_F1": 0.5, "BndF1_5": 0.3})
        assert [g["lr"] for g in opt.param_groups] == lrs, "learning rates not restored"
        moved = [k for k, v in model.state_dict().items()
                 if v.dtype.is_floating_point and not torch.equal(v, before[k])]
        assert any(k.startswith("lstm.") for k in moved), "BiLSTM did not move"
        for s in model.streams:
            pre = {"body_hands": "branches.body_hands.", "face": "branches.face.", "hamer": "branches.hamer."}[s]
            assert any(k.startswith(pre) for k in moved), f"stream {s} did not move"
        log = os.path.join(d, "balance_log.csv")
        assert os.path.exists(log)
        with open(log) as f:
            rows = f.read().strip().splitlines()
        assert len(rows) == epochs + 1
        return bal, rows


def main():
    check_forward()
    for arm in imbalance_arms(BASE):
        method = arm["balance"]["method"]
        bal, rows = run_arm(arm)
        print(f"ok  {method:9s} {arm['description'][:70]}")
        if method == "agm":
            model = bal.model.eval()
            x, h, y = batch()
            with torch.no_grad():
                z, ci, aux = model._encode_streams(x, hamer=h)
                phi, full = bal.shapley_phi(z, ci, aux)
            assert torch.allclose(phi.sum(0), full, atol=1e-5), "Shapley responses do not add up"
            print("ok  agm: mono-modal responses sum to the logits")
        if method == "wu":
            bal, _ = run_arm(arm, tweak={"alpha": -1.0, "warm_epochs": 0})   # always re-balance
            assert bal.ep["rebal"].sum() > 0, "Wu took no re-balancing step"
            print("ok  wu: re-balancing steps taken")
        if method == "inforeg":
            assert bal.ir_active.all() and bal.ep["x1"].sum() > 0, "InfoReg never regulated"
            print("ok  inforeg: regulated in the forced window")
    # invalid combinations
    bad = {**BASE, "balance": {"method": "monitor"}}
    try:
        validate_balance_config(bad)
        raise AssertionError("monitor without heads was accepted")
    except ValueError:
        pass
    bad = {**BASE, "balance_heads": ["face"], "balance_head_detach": True, "balance": {"method": "monitor"},
           "count_loss_weight": 2.0}
    try:
        validate_balance_config(bad)
        raise AssertionError("count loss was accepted")
    except ValueError:
        pass
    print("ok  invalid configs rejected")
    print("\nAll balance_study smoke tests passed.")


if __name__ == "__main__":
    main()
