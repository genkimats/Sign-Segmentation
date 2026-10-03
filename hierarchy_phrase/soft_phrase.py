"""
hierarchy_phrase/soft_phrase.py -- frame-level phrase segmentation that uses a sign model's boundary probabilities as SOFT INPUT.

WHY. The hierarchy showed that sign structure carries phrase information (gold signs + Stage C: about +0.2 frame F1 over a flat
head) but that committing hard to predicted sign segments throws it away. Here nothing is committed: a frame-level BiLSTM phrase
segmenter sees the sign model's per-frame P(O), P(I), P(B) next to the pose (and optional prosody), and learns how much to trust them.
No cascade, works on any signer/dataset for which a sign segmenter exists.

    # data = an existing hierarchy cache (25 fps keypoints, sign probabilities, gold sign/phrase spans), e.g. the imported HaMeR sign model
    python soft_phrase.py train --cache-run al_imp_sb44m --tag pose        --inputs pose                       # control (flat, same regime)
    python soft_phrase.py train --cache-run al_imp_sb44m --tag soft        --inputs pose signprobs             # the method
    python soft_phrase.py train --cache-run al_imp_sb44m --tag soft_pros   --inputs pose signprobs prosody
    python soft_phrase.py train --cache-run al_imp_sb44m --tag oracle      --inputs pose signprobs --oracle-signs   # ceiling
    python soft_phrase.py eval  --cache-run al_imp_sb44m --tag soft                                          # val-tuned thresholds -> test
    python compare_results.py results/al_imp_sb44m__soft_pose_s42.json results/al_imp_sb44m__soft_soft_s42.json --split val

Inputs (concatenated per frame, then standardised with train statistics):
  pose       xyz of the 65 landmarks (195)
  signprobs  the cache's sign-model softmax P(O, I, B) (3); with --oracle-signs: one-hot GOLD sign tags (upper bound of this idea)
  prosody    log hand speed, log nose speed, left/right wrist height (4)
Train-time robustness for signprobs (train-split sign probabilities are IN-SAMPLE for the sign model, i.e. too clean):
  --sp-noise  Gaussian noise added to the sign log-probabilities before re-normalising (default 0.5)
  --sp-drop   probability that a training crop gets UNINFORMATIVE sign channels (1/3 each) (default 0.2)
Model: 4-layer BiLSTM, hidden 256 (E1s size), weighted CE on phrase B/I/O, random 1024-frame crops, whole-video inference.
Model selection: best val mF1S over a small decoding grid. eval tunes the greedy decoding thresholds on val over the full grid and
applies them unchanged to test; rows 'soft_default' (0.5/0.5) and 'soft_tuned' are written in the evaluate.py JSON format.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HERE, RUNS_DIR, fps_of
from metrics import evaluate_videos
from segments import segments_to_bio
from sign_tokens import L_WRIST, R_WRIST, greedy_decode, speed_signals
from stage_c import load_items

INPUT_GROUPS = ("pose", "signprobs", "prosody")
GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SELECT_GRID = [0.3, 0.5, 0.7]
PROSODY_FRAME_DIM = 4


# ------------------------------------------------------------------ features (numpy)
def prosody_frames(xyz, fps):
    hand, nose = speed_signals(xyz, fps)
    x = np.asarray(xyz, dtype=np.float32)
    return np.stack([np.log1p(hand), np.log1p(nose), x[:, L_WRIST, 1], x[:, R_WRIST, 1]], axis=1).astype(np.float32)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class VideoData:
    """Per-video raw inputs + phrase labels. features(s, e, rng=...) returns the (e-s, D) raw feature block; augmentation of the
    sign channels happens only when an rng is passed (training)."""

    def __init__(self, item, groups, oracle=False):
        self.vid, self.T, self.groups = item["vid"], item["T"], tuple(g for g in INPUT_GROUPS if g in groups)
        T = self.T
        self.pose = item["xyz"].reshape(T, -1) if "pose" in self.groups else None
        if "signprobs" in self.groups:
            self.sp = (np.eye(3, dtype=np.float32)[segments_to_bio(item["gold_sign"], T).astype(np.int64)] if oracle
                       else np.asarray(item["sign_probs"], dtype=np.float32))
        else:
            self.sp = None
        self.pros = prosody_frames(item["xyz"], fps_of(item["stride"])) if "prosody" in self.groups else None
        self.y = segments_to_bio(item["gold_phrase"], T).astype(np.int64)
        self.gold = item["gold_phrase"]

    def features(self, s=0, e=None, rng=None, sp_noise=0.0, sp_drop=0.0):
        e = self.T if e is None else e
        parts = []
        if self.pose is not None:
            parts.append(self.pose[s:e].astype(np.float32))
        if self.sp is not None:
            p = self.sp[s:e]
            if rng is not None:
                if sp_noise > 0:
                    p = _softmax(np.log(p + 1e-6) + rng.normal(0.0, sp_noise, p.shape)).astype(np.float32)
                if sp_drop > 0 and rng.random() < sp_drop:
                    p = np.full_like(p, 1.0 / 3.0)
            parts.append(p)
        if self.pros is not None:
            parts.append(self.pros[s:e])
        return np.concatenate(parts, axis=1)


def feature_stats(vds):
    """Streaming mean / std over every frame of the given videos (no augmentation)."""
    s1 = s2 = None
    n = 0
    for vd in vds:
        f = vd.features().astype(np.float64)
        s1 = f.sum(0) if s1 is None else s1 + f.sum(0)
        s2 = (f * f).sum(0) if s2 is None else s2 + (f * f).sum(0)
        n += len(f)
    mean = s1 / max(n, 1)
    std = np.sqrt(np.maximum(s2 / max(n, 1) - mean * mean, 0.0)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def class_weights(vds, power=0.5):
    cnt = np.bincount(np.concatenate([vd.y for vd in vds]), minlength=3).astype(np.float64)
    w = (cnt.sum() / (3.0 * np.maximum(cnt, 1))) ** power
    return (w / w[0]).astype(np.float32)


def metric_of(res, key):
    return res["mF1S(0.1-0.5)"] if key == "mF1S" else res[key]


def best_decoding(probs_list, vds, grid, key):
    """Greedy-decoding thresholds (b, o) maximising `key` on the given videos -> (score, b, o)."""
    best = (-1.0, 0.5, 0.5)
    for b in grid:
        for o in grid:
            r = evaluate_videos([greedy_decode(p, b, o) for p in probs_list], [vd.gold for vd in vds], [vd.T for vd in vds],
                                tols=(2, 5, 10), iou_thrs=(0.1, 0.3, 0.5, 0.7))
            v = metric_of(r, key)
            if v > best[0]:
                best = (v, b, o)
    return best


def decode_eval(probs_list, vds, b, o):
    return evaluate_videos([greedy_decode(p, b, o) for p in probs_list], [vd.gold for vd in vds], [vd.T for vd in vds],
                           tols=(2, 5, 10), iou_thrs=(0.1, 0.3, 0.5, 0.7))


# ------------------------------------------------------------------ model (torch)
class SoftPhraseBiLSTM(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=4, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, 3)

    def forward(self, x, lengths=None):
        z = self.inp(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        else:
            h, _ = self.lstm(z)
        return self.head(h)


@torch.no_grad()
def predict_probs(model, vd, mean, std, device):
    x = torch.from_numpy(((vd.features() - mean) / std).astype(np.float32)).unsqueeze(0).to(device)
    return F.softmax(model(x)[0], -1).float().cpu().numpy()


def soft_dir(cache_run, tag, seed):
    d = os.path.join(RUNS_DIR, cache_run, "soft", f"{tag}_s{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def load_vds(cache_run, split, groups, oracle, limit=None):
    items = load_items([os.path.join(RUNS_DIR, cache_run, "cache", split)], limit)
    return [VideoData(it, groups, oracle) for it in items]


# ------------------------------------------------------------------ train
def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = tuple(g for g in INPUT_GROUPS if g in a.inputs)
    if a.oracle_signs and "signprobs" not in groups:
        raise SystemExit("--oracle-signs needs 'signprobs' in --inputs")
    out = soft_dir(a.cache_run, a.tag, a.seed)
    print("loading caches ...")
    tr = load_vds(a.cache_run, "train", groups, a.oracle_signs, a.limit)
    va = load_vds(a.cache_run, "val", groups, a.oracle_signs, a.limit)
    mean, std = feature_stats(tr)
    w = class_weights(tr, a.weight_power)
    D = len(mean)
    model = SoftPhraseBiLSTM(D, a.hidden, a.layers, a.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"inputs {groups}{' (ORACLE gold sign tags)' if a.oracle_signs else ''}  dim {D}  train {len(tr)} val {len(va)} videos  "
          f"params {n_par:,}  phrase class weights (O,I,B) {np.round(w, 2).tolist()}")
    cfg = {**vars(a), "groups": list(groups), "in_dim": D, "n_params": n_par}
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    cw = torch.tensor(w, device=device)
    steps = max(1, (len(tr) * a.crops_per_video) // a.batch)
    best, bad, log = -1.0, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(steps):
            idx = rng.integers(0, len(tr), size=a.batch)
            blocks, labels, lens = [], [], []
            for i in idx:
                vd = tr[int(i)]
                s = int(rng.integers(0, max(vd.T - a.crop, 0) + 1))
                e = min(vd.T, s + a.crop)
                blocks.append((vd.features(s, e, rng, a.sp_noise, a.sp_drop) - mean) / std)
                labels.append(vd.y[s:e]); lens.append(e - s)
            L = max(lens)
            X = np.zeros((len(idx), L, D), np.float32)
            Y = np.full((len(idx), L), -100, np.int64)
            for k, (f, y) in enumerate(zip(blocks, labels)):
                X[k, :len(f)] = f; Y[k, :len(y)] = y
            logits = model(torch.from_numpy(X).to(device), torch.tensor(lens))
            loss = F.cross_entropy(logits.reshape(-1, 3), torch.from_numpy(Y).to(device).reshape(-1), weight=cw, ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        model.eval()
        P = [predict_probs(model, vd, mean, std, device) for vd in va]
        score, b, o = best_decoding(P, va, SELECT_GRID, "mF1S")
        r = decode_eval(P, va, b, o)
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"state": copy.deepcopy(model.state_dict()), "config": cfg, "mean": mean, "std": std}, os.path.join(out, "model.pt"))
        else:
            bad += 1
        log.append({"epoch": epoch, "loss": tot / steps, "val_mF1S": score, "b": b, "o": o, "val_frame_f1": r["frame_f1"],
                    "val_ratio": r["ratio"], "val_start_f1@5": r["start_f1@5"], "val_seg_f1@0.5": r["seg_f1@0.5"]})
        print(f"epoch {epoch:03d} loss {tot / steps:.4f} | val mF1S {score:.4f} (b/o {b}/{o}) frameF1 {r['frame_f1']:.4f} ratio {r['ratio']:.2f} "
              f"startF1@5 {r['start_f1@5']:.3f} segF1@.5 {r['seg_f1@0.5']:.3f} | {time.time() - t0:.0f}s" + (" *" if improved else f" (x{bad})"))
        json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
        if bad >= a.patience:
            print(f"early stopping after {epoch} epochs")
            break
    print(f"best val mF1S {best:.4f} -> {out}/model.pt")


# ------------------------------------------------------------------ eval
def evaluate(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(soft_dir(a.cache_run, a.tag, a.seed), "model.pt"), map_location=device, weights_only=False)
    c, mean, std = ck["config"], ck["mean"], ck["std"]
    model = SoftPhraseBiLSTM(c["in_dim"], c["hidden"], c["layers"], c["dropout"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    groups, oracle = tuple(c["groups"]), c["oracle_signs"]
    va = load_vds(a.cache_run, "val", groups, oracle, a.limit)
    te = load_vds(a.cache_run, "test", groups, oracle, a.limit)
    Pv = [predict_probs(model, vd, mean, std, device) for vd in va]
    Pt = [predict_probs(model, vd, mean, std, device) for vd in te]
    _, b, o = best_decoding(Pv, va, GRID, a.tune_metric)
    print(f"soft phrase model: inputs {groups}{' (ORACLE)' if oracle else ''}  tuned on val ({a.tune_metric}): b/o = {b}/{o}")
    out = {"cache_run": a.cache_run, "tag": a.tag, "seed": a.seed, "tune_metric": a.tune_metric, "inputs": list(groups),
           "oracle_signs": oracle, "thresholds": {"soft": [b, o]}, "n_params": c["n_params"]}
    for name, P, vds in (("val", Pv, va), ("test", Pt, te)):
        out[name] = {"soft_default": decode_eval(P, vds, 0.5, 0.5), "soft_tuned": decode_eval(P, vds, b, o)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{a.cache_run}__soft_{a.tag}_s{a.seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    cols = ["frame_f1", "frame_f1_B", "mask_iou", "ratio", "start_f1@2", "start_f1@5", "start_f1@10", "seg_f1@0.5", "mF1S(0.1-0.5)"]
    print(f"\nTEST  (thresholds from validation)\n{'row':<16}" + "".join(f"{x:>14}" for x in cols))
    for k, r in out["test"].items():
        print(f"{k:<16}" + "".join(f"{r[x]:>14.3f}" for x in cols))
    print(f"\nsaved {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cache-run", required=True, help="hierarchy run whose cache/ to use (e.g. al_imp_sb44m)")
    t.add_argument("--tag", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--inputs", nargs="+", default=["pose", "signprobs"], choices=list(INPUT_GROUPS))
    t.add_argument("--oracle-signs", action="store_true", help="feed one-hot GOLD sign tags instead of the sign model's probabilities")
    t.add_argument("--sp-noise", type=float, default=0.5)
    t.add_argument("--sp-drop", type=float, default=0.2)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--crop", type=int, default=1024)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--crops-per-video", type=int, default=2)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--weight-power", type=float, default=0.5)
    t.add_argument("--limit", type=int, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--cache-run", required=True)
    e.add_argument("--tag", required=True)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--tune-metric", default="mF1S", choices=["frame_f1", "start_f1@5", "seg_f1@0.5", "mF1S"])
    e.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()