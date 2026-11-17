"""
balance_study/laziness_probe.py -- is a stream's encoder under-trained inside the joint model?

The test of Du et al. (ICML 2023, "modality laziness") and of AGM's competition strength: freeze a trained
model, read ONE stream's branch output (before its dropout), and train the same small probe on it to predict
phrase B/I/O. Compare the probe on the joint model's branch (e.g. the face branch of B+H+Fc 373) with the probe
on the single-stream model's branch (the face-only model 136, or the SS-face rerun):
  - joint ~ single-stream: the joint model keeps what the stream can carry alone -> no laziness gap;
  - joint << single-stream: the stream's encoder is under-trained in the joint model (greedy learning).

Every probe sees exactly the same training windows (same videos, same random window starts, fixed seed) and
is selected on val Frame F1, then reported on test (argmax decoder).

Usage (repo root):
    python balance_study/laziness_probe.py --pairs multistream_bilstm-373:face multistream_bilstm-136:face \\
        multistream_bilstm-373:hamer multistream_bilstm-133:hamer \\
        multistream_bilstm-373:body_hands multistream_bilstm-82:body_hands --probe conv9
Output: evaluation_phrase/balance/probe_<probe>.csv (rows appended / replaced per run, stream, split)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate_phrase  # noqa: E402
from evaluate_phrase import get_dataset, EXP_DIR, MODEL_DIR  # noqa: E402
from src.model_factory import build_model_kwargs  # noqa: E402
from src.evaluation import video_windows, _prepare_window  # noqa: E402
from src.metrics import evaluate_videos  # noqa: E402

OUT_DIR = os.path.join("evaluation_phrase", "balance")


def load(run, split, device):
    with open(os.path.join(EXP_DIR, run, "hyperparameters.json")) as f:
        config = json.load(f)
    ds = get_dataset(config, split)
    cls, kw = build_model_kwargs(config, detected_hamer_dim=ds.detected_hamer_dim,
                                 detected_dinov2_dim=ds.detected_dinov2_dim,
                                 detected_cue_dim=getattr(ds, "detected_cue_dim", None))
    model = cls(**kw).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{run}.pth"), map_location=device), strict=True)
    model.eval()
    return config, model


def make_inputs(config, cached, wins, T, device):
    feats = torch.stack([_prepare_window(cached["features"], s, e, 1, T, True) for s, e in wins])
    feats = torch.nan_to_num(feats.to(device), nan=0.0, posinf=0.0, neginf=0.0)
    hamer = cues = None
    if config.get("use_hamer_features"):
        hamer = torch.stack([_prepare_window(cached["hamer_features"], s, e, 1, T, False) for s, e in wins])
        hamer = torch.nan_to_num(hamer.to(device), nan=0.0, posinf=0.0, neginf=0.0)
    if config.get("use_cue_features"):
        cues = torch.stack([_prepare_window(cached["cue_features"], s, e, 1, T, False) for s, e in wins])
        cues = torch.nan_to_num(cues.to(device), nan=0.0, posinf=0.0, neginf=0.0)
    return feats, hamer, cues


@torch.no_grad()
def branch_features(model, config, stream, cached, wins, T, device):
    store = {}
    h = model.branches[stream][-1].register_forward_pre_hook(lambda m, inp: store.__setitem__("z", inp[0]))
    try:
        feats, hamer, cues = make_inputs(config, cached, wins, T, device)
        model._encode_streams(feats, hamer=hamer, cues=cues)
    finally:
        h.remove()
    return store["z"].float().cpu()                                      # (n, T, D)


def extract(run, stream, device, windows_per_video, seed, batch):
    if int(json.load(open(os.path.join(EXP_DIR, run, "hyperparameters.json"))).get(
            "temporal_downsample_factor", 1)) != 1:
        raise SystemExit("laziness_probe assumes temporal_downsample_factor 1")
    out = {}
    for split in ("train", "val", "test"):
        config, model = load(run, split, device)
        if stream not in model.streams:
            raise SystemExit(f"{run} has no stream '{stream}' (streams: {model.streams})")
        ds = get_dataset(config, split)
        Tw = int(config["window_size"])
        if split == "train":
            rng = np.random.default_rng(seed)
            X, Y = [], []
            for vid in sorted(ds.video_cache):
                cached = ds.video_cache[vid]
                n = cached["features"].shape[0]
                starts = rng.integers(0, max(n - Tw, 0) + 1, size=windows_per_video)
                wins = [(int(s), int(min(s + Tw, n))) for s in starts]
                X.append(branch_features(model, config, stream, cached, wins, Tw, device).half())
                gold = np.asarray(cached["hard_labels"]).astype(np.int64)
                y = np.full((len(wins), Tw), -100, dtype=np.int64)
                for i, (s, e) in enumerate(wins):
                    y[i, :e - s] = gold[s:e]
                Y.append(torch.from_numpy(y))
            out[split] = (torch.cat(X), torch.cat(Y))
        else:
            feats, gold = {}, {}
            for vid in sorted(ds.video_cache):
                cached = ds.video_cache[vid]
                n = cached["features"].shape[0]
                wins = video_windows(n, Tw, int(config.get("overlap", 0)))
                acc, cnt = None, np.zeros(n)
                for b0 in range(0, len(wins), batch):
                    bw = wins[b0:b0 + batch]
                    zb = branch_features(model, config, stream, cached, bw, Tw, device).numpy()
                    if acc is None:
                        acc = np.zeros((n, zb.shape[-1]), dtype=np.float32)
                    for (s, e), zz in zip(bw, zb):
                        acc[s:e] += zz[:e - s]
                        cnt[s:e] += 1
                feats[vid] = torch.from_numpy((acc / np.maximum(cnt, 1)[:, None]).astype(np.float16))
                gold[vid] = np.asarray(cached["hard_labels"]).astype(np.int64)
            out[split] = (feats, gold)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    evaluate_phrase._DATASET_CACHE.clear()                               # free the train split before the next run
    return config, out


class Probe(nn.Module):
    def __init__(self, dim, kind):
        super().__init__()
        self.net = nn.Conv1d(dim, 3, 9, padding=4) if kind == "conv9" else nn.Conv1d(dim, 3, 1)

    def forward(self, x):                                                # (n, T, D) -> (n, 3, T)
        return self.net(x.transpose(1, 2))


@torch.no_grad()
def predict(probe, feats, mu, sd, device):
    out = {}
    for vid, x in feats.items():
        z = ((x.float() - mu) / sd).to(device).unsqueeze(0)
        out[vid] = F.softmax(probe(z)[0].float(), dim=0).cpu().numpy()
    return out


def train_probe(data, class_weights, kind, epochs, lr, device, seed):
    torch.manual_seed(seed)
    Xtr, Ytr = data["train"]
    flat = Xtr.float().reshape(-1, Xtr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-4)
    probe = Probe(Xtr.shape[-1], kind).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    w = torch.tensor(class_weights, dtype=torch.float, device=device)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            x = ((Xtr[idx].float() - mu) / sd).to(device)
            y = Ytr[idx].to(device)
            loss = F.cross_entropy(probe(x), y, weight=w, ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        m = evaluate_videos(predict(probe, data["val"][0], mu, sd, device), data["val"][1], decoder="argmax")
        if m["Frame_F1"] > best:
            best, best_ep = m["Frame_F1"], ep
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    probe.eval()
    res = {}
    for split in ("val", "test"):
        m = evaluate_videos(predict(probe, data[split][0], mu, sd, device), data[split][1], decoder="argmax")
        res[split] = {k: m[k] for k in ("Frame_F1", "F1_O", "F1_I", "F1_B", "IoU", "Pct", "BndF1_5")}
    return res, best_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True, help="run:stream, e.g. multistream_bilstm-373:face")
    ap.add_argument("--probe", choices=["linear", "conv9"], default="conv9",
                    help="linear = per-frame; conv9 = 9-frame temporal conv (the branch has no BiLSTM context)")
    ap.add_argument("--windows-per-video", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"probe_{args.probe}.csv")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    new = []
    for pair in args.pairs:
        run, stream = pair.split(":")
        print(f"\n=== {run} : {stream}")
        config, data = extract(run, stream, device, args.windows_per_video, args.seed, args.batch_size)
        res, best_ep = train_probe(data, config.get("class_weights", [1.0, 1.1, 8.5]), args.probe, args.epochs,
                                   args.lr, device, args.seed)
        for split, m in res.items():
            new.append({"run": run, "stream": stream, "split": split, "probe": args.probe, "best_epoch": best_ep,
                        "description": config.get("description", ""), **{k: round(v, 5) for k, v in m.items()}})
            print(f"  {split}: Frame F1 {m['Frame_F1']:.4f} | B F1 {m['F1_B']:.4f} | BndF1@5 {m['BndF1_5']:.4f} "
                  f"| IoU {m['IoU']:.4f} | % {m['Pct']:.2f} (probe epoch {best_ep})")
    keys = {(r["run"], r["stream"], r["split"]) for r in new}
    rows = [r for r in rows if (r["run"], r["stream"], r["split"]) not in keys] + new
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(new[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in new[0].keys()})
    print(f"\nsaved {path}")
    print("\nTest, by stream (joint vs single-stream branches):")
    for stream in sorted({r["stream"] for r in new}):
        for r in new:
            if r["stream"] == stream and r["split"] == "test":
                print(f"  {stream:10s} {r['run']:24s} Frame F1 {float(r['Frame_F1']):.4f}  B F1 {float(r['F1_B']):.4f}  "
                      f"BndF1@5 {float(r['BndF1_5']):.4f}  ({r['description'][:50]})")


if __name__ == "__main__":
    main()
