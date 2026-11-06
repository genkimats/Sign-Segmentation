"""
Smoke tests for the temporal-context study (queue_train_phrase.py TEMPORAL_CONTEXT_STUDY).
Run from the repo root before queueing:   python test_temporal_context.py
Uses only synthetic data in a temporary folder; touches nothing in processed_data/.

  1. preprocess_kinematics: per-second velocity vs the analytic derivative; positions == version 1
  2. frame-rate resampling helpers: every phrase start survives, labels stay distributions
  3. dataset end to end (synthetic videos): v2 velocity channels, frame-rate augmentation on train
     only, the per-frame-units guard
  4. every TEMPORAL_CONTEXT_STUDY arm: model builds, forward + backward, output length = input length
  5. old checkpoints still load: the default model's state_dict keys are unchanged
  6. LR warmup schedule
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import preprocess_kinematics as pk                                    # noqa: E402
from src.dataset import (SignSegmentationDataset, extra_dataset_kwargs,  # noqa: E402
                         fps_resample_indices, resample_soft_labels)
from src.model_factory import build_model_kwargs                      # noqa: E402
from src.face_subsets import FACE_LAYOUTS                             # noqa: E402

FPS = 50
OK = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    OK.append(bool(cond))


def test_kinematics():
    print("1. preprocess_kinematics")
    T = 600
    t = np.arange(T) / FPS
    raw = np.zeros((T, 65, 3), np.float32)
    raw[:, 30, 0] = 0.5 * np.sin(2 * np.pi * t) + 1.0
    raw[:, 30, 1] = 0.3 * t
    raw += np.random.RandomState(0).normal(0, 0.005, raw.shape).astype(np.float32)
    k = pk.compute_kinematics(raw, FPS)
    true_v = np.pi * np.cos(2 * np.pi * t)
    rms = np.sqrt(np.mean((k["velocity"][5:-5, 30, 0] - true_v[5:-5]) ** 2))
    check("velocity in shoulder widths per second", rms < 0.15, f"rms error {rms:.3f} on amplitude 3.14")
    from scipy.signal import medfilt
    check("positions identical to version 1", np.allclose(k["base"], medfilt(raw, (5, 1, 1))))
    check("all channels finite", all(np.isfinite(v).all() for v in k.values()))


def test_resampling():
    print("2. frame-rate resampling helpers")
    rng = np.random.RandomState(1)
    T = 6000
    lab = np.zeros(T, int)
    starts = np.sort(rng.choice(np.arange(200, 5800, 10), 80, replace=False))   # >= 10 frames apart
    for s in starts:
        lab[s], lab[s + 1:s + 8] = 2, 1
    soft = np.eye(3, dtype=np.float32)[lab]
    good = True
    for rate in (0.5, 0.63, 0.8, 1.0):
        for start in (0, 2500, 5600):
            idx, end = fps_resample_indices(T, start, 512, rate)
            out = resample_soft_labels(soft, idx, end)
            good &= len(idx) == 512 and np.all(np.diff(idx) > 0)
            good &= int((soft[idx[0]:end, 2] > 0).sum()) == int((out[:, 2] > 0).sum())
            good &= np.allclose(out.sum(1), 1.0)
    check("512 frames, every phrase start kept, rows sum to 1", good)
    # videos only slightly shorter than the stretched window (rounding used to give 513 frames)
    worst = max(len(fps_resample_indices(n, 0, 512, r)[0])
                for r in np.linspace(0.5, 1.0, 501) for n in range(int(512 / r) - 3, int(512 / r) + 4))
    check("never more frames than the window", worst <= 512, f"max {worst}")


def _make_data(d, n_videos=4, T=1500):
    rng = np.random.RandomState(2)
    os.makedirs(f"{d}/kp", exist_ok=True)
    for sub in ("labels", "kin1", "kin2", "face", "hamer"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    vids = [f"v{i}" for i in range(n_videos)]
    for v in vids:
        kp = rng.normal(0, 0.3, (T, 65, 3)).cumsum(0).astype(np.float32) * 0.01
        lab = np.zeros(T, np.int64)
        for s in range(100, T - 100, 150):
            lab[s], lab[s + 1:s + 120] = 2, 1
        np.save(f"{d}/labels/{v}.npy", lab)
        k = pk.compute_kinematics(kp, FPS)
        v2 = {n: torch.from_numpy(x).half() for n, x in k.items()}
        v2["meta"] = {"version": 2, "units": "per_second", "fps": FPS}
        torch.save(v2, f"{d}/kin2/{v}.pt")
        torch.save({n: x for n, x in v2.items() if n != "meta"}, f"{d}/kin1/{v}.pt")   # no units = v1
        np.save(f"{d}/face/{v}.npy", rng.normal(0, 1, (T, FACE_LAYOUTS["saved83"], 2)).astype(np.float32))
        torch.save({"hand_pose": torch.randn(T, 2, 15, 3, 3), "global_orient": torch.randn(T, 2, 3, 3),
                    "temporal_downsample_factor": 1}, f"{d}/hamer/{v}_hamer.pt")
    with open(f"{d}/splits.json", "w") as f:
        json.dump({"train": vids[:3], "val": vids[3:], "test": vids[3:]}, f)
    return vids


def _dataset(d, config, split):
    return SignSegmentationDataset(
        keypoints_dir=f"{d}/kp", labels_dir=f"{d}/labels", split_file=f"{d}/splits.json", split=split,
        window_size=config["window_size"], overlap=0, tolerance_window=1,
        base_features=config["base_features"], kinematic_features=config["kinematic_features"],
        use_face_keypoints=True, face_dir=f"{d}/face", face_subset="compact", face_dims=2,
        use_hamer_features=True, hamer_dir=f"{d}/hamer", **extra_dataset_kwargs(config))


def test_dataset():
    print("3. dataset (synthetic videos)")
    import queue_train_phrase as q
    with tempfile.TemporaryDirectory() as d:
        _make_data(d)
        base = {**q.MAMBA_DEFAULTS, **q._MS_BEST}
        vel = {**base, "kinematic_features": ["velocity"], "kinematic_dir": f"{d}/kin2"}
        ds = _dataset(d, vel, "train")
        x, y = ds[0][0], ds[0][1]
        check("velocity adds 3 channels", x.shape[0] == q.calculate_in_channels(vel) == 6, str(tuple(x.shape)))
        fps = {**base, "kinematic_dir": f"{d}/kin1", "fps_augment_range": [0.5, 0.5]}
        tr, va = _dataset(d, fps, "train"), _dataset(d, fps, "val")
        b_out = sum(int((tr[i][1][2] > 0).sum()) for i in range(len(tr)))
        check("frame-rate augmentation only on train", tr.fps_augment_range == (0.5, 0.5)
              and va.fps_augment_range is None)
        x, y, h = tr[1][0], tr[1][1], tr[1][2]
        check("resampled window keeps window length", x.shape[1] == y.shape[1] == h.shape[1] == 512)
        check("resampled windows contain phrase starts", b_out > 0, f"{b_out} B frames over {len(tr)} windows")
        try:
            _dataset(d, {**fps, "kinematic_features": ["velocity"]}, "train")
            check("per-frame derivatives + frame-rate augmentation refused", False)
        except ValueError:
            check("per-frame derivatives + frame-rate augmentation refused", True)


def _full_config(q, exp):
    c = dict(q.MAMBA_DEFAULTS)
    c.update(exp)
    c["use_face_keypoints"] = bool(c.get("use_stream_face"))
    c["use_hamer_features"] = bool(c.get("use_stream_hamer"))
    c["use_cue_features"] = bool(c.get("use_stream_cues"))
    c["in_channels"] = q.calculate_in_channels(c)
    c["num_vertices"] = q.calculate_num_vertices(c)
    return c


def test_models():
    print("4. every TEMPORAL_CONTEXT_STUDY arm builds and trains one step")
    import queue_train_phrase as q
    for exp in q.TEMPORAL_CONTEXT_STUDY:
        c = _full_config(q, exp)
        cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
        model = cls(**kw)
        T = 96
        x = torch.randn(2, c["in_channels"], T, c["num_vertices"])
        ham = torch.randn(2, 288, T)
        logits = model(x, hamer=ham)[0]
        logits.sum().backward()
        n = sum(p.numel() for p in model.parameters())
        check(exp["description"][:58], logits.shape == (2, 3, T), f"{n / 1e6:.2f}M params")


def test_checkpoint_compat():
    print("5. old checkpoints")
    import queue_train_phrase as q
    c = _full_config(q, q._MS_BEST)
    cls, kw = build_model_kwargs(c, detected_hamer_dim=288)
    keys_new = set(cls(**kw).state_dict())
    kw_old = {k: v for k, v in kw.items()
              if k not in ("stgcn_temporal_dilations", "hamer_temporal_layers", "hamer_temporal_dilations")}
    check("default model has no new arguments", kw == kw_old)
    check("state_dict keys unchanged", keys_new == set(cls(**kw_old).state_dict()))
    c1 = _full_config(q, {**q._MS_BEST, "hamer_temporal_kernel": 9})
    cls, kw = build_model_kwargs(c1, detected_hamer_dim=288)
    check("single HaMeR temporal block keeps its old name",
          any(k.startswith("hamer_temporal.conv") for k in cls(**kw).state_dict()))


def test_warmup():
    print("6. LR warmup")
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=95)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[5])
    lrs = []
    for _ in range(8):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    check("10% -> 100% over 5 epochs, then cosine", abs(lrs[0] - 1e-5) < 1e-9 and abs(lrs[5] - 1e-4) < 1e-7
          and lrs[7] < lrs[6] <= 1e-4, " ".join(f"{v:.1e}" for v in lrs))


if __name__ == "__main__":
    os.chdir(ROOT)
    for fn in (test_kinematics, test_resampling, test_dataset, test_models, test_checkpoint_compat, test_warmup):
        fn()
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)