"""
data.py -- frozen-feature data for the FACT pilot: per-video feature store, random training windows,
and full-video inference (window tiling + stitching) shared by training (validation) and evaluation.

Frame rates: features are stored at 50 fps. feature_stride k > 1 mean-pools k frames (labels:
fact_core.downsample_bio); predictions are brought back to 50 fps with fact_core.upsample_probs /
upsample_bio, so every metric is computed against the raw gold labels at the raw frame rate.
"""
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from fact_phrase import fact_core as core

FEATURES_ROOT = os.path.join("fact_phrase", "features")


def load_feature_meta(source_run, root=FEATURES_ROOT):
    path = os.path.join(root, source_run, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} -- run: python fact_phrase/export_features.py --run {source_run}")
    with open(path) as f:
        return json.load(f)


class FeatureStore:
    """All videos of one split: memory-mapped features + gold labels (raw and model rate).
    Holds only paths and small label arrays, so it pickles cheaply into DataLoader workers; each
    process opens its own memory maps on first access."""

    def __init__(self, source_run, split, stride=1, root=FEATURES_ROOT, labels_dir=None):
        meta = load_feature_meta(source_run, root)
        if split not in meta.get("splits", {}):
            raise FileNotFoundError(f"split {split!r} not exported for {source_run} -- run export_features.py "
                                    f"--run {source_run} --splits {split}")
        self.source_run, self.split, self.stride = source_run, split, int(stride)
        self.labels_dir = labels_dir or meta["labels_dir"]
        self.dim = int(meta["dim"])
        self.dir = os.path.join(root, source_run, split)
        self.videos = {}
        missing = []
        for vid in meta["splits"][split]["videos"]:
            fp, gp = os.path.join(self.dir, f"{vid}.npy"), os.path.join(self.labels_dir, f"{vid}.npy")
            if not (os.path.exists(fp) and os.path.exists(gp)):
                missing.append(vid)
                continue
            gold = np.load(gp).astype(np.int64)
            T = min(len(gold), int(meta["splits"][split]["frames"][vid]))
            gold = gold[:T]
            self.videos[vid] = {"path": fp, "gold": gold, "T": T, "Tm": -(-T // self.stride),
                                "bio_m": core.downsample_bio(gold, self.stride)}
        if missing:
            print(f"  [{split}] {len(missing)} exported videos without features/labels skipped, e.g. {missing[:3]}")
        if not self.videos:
            raise RuntimeError(f"no usable {split} videos for {source_run}")
        self._mm = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_mm"] = {}
        return state

    def _features(self, vid):
        mm = self._mm.get(vid)
        if mm is None:
            mm = self._mm[vid] = np.load(self.videos[vid]["path"], mmap_mode="r")
        return mm

    def window(self, vid, s, e):
        """Model-rate window [s, e): features (e - s, D) float32 and labels (e - s,) int64."""
        v = self.videos[vid]
        k = self.stride
        if k == 1:
            x = np.asarray(self._features(vid)[s:e], dtype=np.float32)
        else:
            x = core.pool_features(np.asarray(self._features(vid)[s * k:min(e * k, v["T"])], dtype=np.float32), k)
        return x, v["bio_m"][s:e]

    def model_rate_labels(self):
        return [v["bio_m"] for v in self.videos.values()]

    def total_model_frames(self):
        return int(sum(v["Tm"] for v in self.videos.values()))


class TrainWindows(Dataset):
    """n_samples random windows of W model-rate frames per epoch; videos drawn in proportion to the
    number of window positions they offer (shorter-than-W videos are used whole)."""

    def __init__(self, store, W, n_samples):
        self.store, self.W, self.n = store, int(W), int(n_samples)
        self.vids = list(store.videos)
        Tm = np.array([store.videos[v]["Tm"] for v in self.vids], dtype=np.float64)
        self.Tm = Tm.astype(np.int64)
        self.cum = np.cumsum(np.maximum(Tm - self.W + 1, 1.0))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = int(np.searchsorted(self.cum, torch.rand(1).item() * self.cum[-1], side="right"))
        j = min(j, len(self.vids) - 1)
        Tm = int(self.Tm[j])
        if Tm <= self.W:
            s, e = 0, Tm
        else:
            s = int(torch.randint(0, Tm - self.W + 1, (1,)).item())
            e = s + self.W
        x, y = self.store.window(self.vids[j], s, e)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def list_collate(batch):
    return batch


def samples_per_epoch(store, W, coverage):
    return max(1, int(math.ceil(coverage * store.total_model_frames() / W)))


def pad_batch(items, device):
    """[(x (T, D), y (T,))] -> x (B, Tmax, D), one-hot targets (B, 3, Tmax) with zero padding."""
    Tmax = max(x.shape[0] for x, _ in items)
    D = items[0][0].shape[1]
    X = torch.zeros(len(items), Tmax, D)
    Y = torch.zeros(len(items), 3, Tmax)
    for b, (x, y) in enumerate(items):
        X[b, :x.shape[0]] = x
        Y[b, y, torch.arange(len(y))] = 1.0
    return X.to(device, non_blocking=True), Y.to(device, non_blocking=True)


@torch.no_grad()
def predict_store(model, cfg, store, device, batch_size=16):
    """
    Full-video predictions for every video of the store.
    Returns {vid: {"head": (3, T) float32 probabilities, "token": (T,) int64 BIO or None}} at 50 fps.
      head  = argmax-able B/I/O probabilities (FACT: last block's frame branch; BiLSTM: its output),
              overlapping windows averaged.
      token = FACT's token decoding, each frame taken from the window that owns it
              (fact_core.owned_regions), FACT only.
    """
    was_training = model.training
    model.eval()
    W = int(cfg["window_frames"]) // store.stride
    step = max(1, int(round(W * float(cfg["eval_step_frac"]))))
    is_fact = cfg["model"] == "fact"
    out = {}
    for vid, v in store.videos.items():
        Tm = v["Tm"]
        wins = core.tile_windows(Tm, W, step)
        heads, toks = [], []
        if is_fact:
            for s, e in wins:
                x, _ = store.window(vid, s, e)
                h, t = model.predict(torch.from_numpy(x).to(device))
                heads.append(h)
                toks.append(t)
        else:
            for b0 in range(0, len(wins), batch_size):
                ws = wins[b0:b0 + batch_size]
                xs = [store.window(vid, s, e)[0] for s, e in ws]
                if len({len(x) for x in xs}) == 1:
                    heads.extend(model.predict_batch(torch.from_numpy(np.stack(xs)).to(device)))
                else:
                    heads.extend(model.predict_batch(torch.from_numpy(x[None]).to(device))[0] for x in xs)
        head = core.upsample_probs(core.stitch_probs(heads, wins, Tm), store.stride, v["T"])
        tok = core.upsample_bio(core.stitch_labels(toks, wins, Tm), store.stride, v["T"]) if is_fact else None
        out[vid] = {"head": head, "token": tok}
    if was_training:
        model.train()
    return out


def decoded_probs(preds, decoder):
    """{vid: (3, T)} for src.metrics: 'head' -> the probabilities, 'token' -> one-hot of the decoded BIO."""
    if decoder == "head":
        return {v: p["head"] for v, p in preds.items()}
    if decoder == "token":
        return {v: core.onehot_probs(p["token"]) for v, p in preds.items()}
    raise ValueError(decoder)
