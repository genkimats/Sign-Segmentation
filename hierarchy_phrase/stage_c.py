"""
hierarchy_phrase/stage_c.py -- Stage C of the hierarchy: a Transformer over SIGN TOKENS predicting, for every sign,
whether it starts a new phrase (B) or continues one (I).

    python stage_c.py train --stage-a sa_s42 --tag main --seed 42 --source mix
    python stage_c.py train --stage-a sa_s42 --tag oof  --seed 42 --source pred --groups sign_probs prosody \
                            --oof-runs sa_f0 sa_f1 sa_f2 sa_f3
    python stage_c.py train --stage-a sa_s42 --tag bilstm --arch bilstm --source mix           # ablation 5 (sanity check)

Input windows: 128 sign tokens, stride 64 (merged by averaging at inference). Position: learned within the window.
Output: P(B) per sign. Phrases are rebuilt from the tags (start of a B sign -> end of the last sign before the next B).

TRAIN / TEST MISMATCH (brief, section 6) -- `--source`:
  gold      train on gold sign segments only                                      (the naive baseline)
  jitter    gold boundaries randomly shifted / merged / split / dropped; strength ramps 0 -> --jitter over --ramp epochs
  schedule  gold -> predicted: per video and epoch, P(use predicted signs) ramps 0 -> 1 over --ramp epochs
  mix       per video and epoch, uniformly one of {gold, jittered gold, predicted}   (the brief's option c)
  pred      always the predicted signs of the cache used for training. With --oof-runs those are OUT-OF-FOLD predictions
            (option b). Fold models do not share a feature space, so --oof-runs requires --groups without h_pool.
Predicted signs = greedy decoding of the Stage-A sign head (--b-thr / --o-thr, default 0.5 / 0.5; evaluate.py tunes them
on the validation split for the reported numbers).
Validation (model selection) always uses PREDICTED signs, i.e. the test-time condition, and logs the gold-sign oracle too.
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

from common import RUNS_DIR, arr_to_segs, fps_of, load_cache_video
from metrics import evaluate_videos
from segments import (jitter_segments, phrase_tags_over_signs, phrases_from_tags, plan_windows, stitch_probs)
from sign_tokens import build_tokens, greedy_decode, logits_to_probs

ALL_GROUPS = ("h_pool", "sign_probs", "prosody")


# ------------------------------------------------------------------ models
class TokenTransformer(nn.Module):
    def __init__(self, in_dim, d_model=256, layers=4, heads=4, dropout=0.1, max_len=128, feat_dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(feat_dropout), nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x, pad=None):
        z = self.proj(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.head(self.enc(z, src_key_padding_mask=pad))


class TokenBiLSTM(nn.Module):
    def __init__(self, in_dim, d_model=256, layers=2, hidden=256, dropout=0.1, feat_dropout=0.1, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(feat_dropout), nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))
        self.lstm = nn.LSTM(d_model, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x, pad=None):
        out, _ = self.lstm(self.proj(x))
        return self.head(out)


def build_model(arch, in_dim, cfg):
    if arch == "transformer":
        return TokenTransformer(in_dim, cfg["d_model"], cfg["layers"], cfg["heads"], cfg["dropout"], cfg["window"], cfg["feat_dropout"])
    if arch == "bilstm":
        return TokenBiLSTM(in_dim, cfg["d_model"], cfg["lstm_layers"], cfg["lstm_hidden"], cfg["dropout"], cfg["feat_dropout"])
    raise ValueError(arch)


def focal_ce(logits, target, weight, gamma, ignore_index=-100):
    logp = F.log_softmax(logits, -1)
    ce = F.nll_loss(logp, target, weight=weight, ignore_index=ignore_index, reduction="none")
    if gamma > 0:
        pt = logp.gather(1, target.clamp(min=0)[:, None]).squeeze(1).exp()
        ce = ce * (1 - pt) ** gamma
    valid = (target != ignore_index).float()
    return (ce * valid).sum() / valid.sum().clamp(min=1)


# ------------------------------------------------------------------ cache items
def load_items(cache_dirs, limit=None):
    items = []
    for d in cache_dirs:
        names = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
        for f in names[:limit]:
            z = load_cache_video(os.path.join(d, f))
            items.append({"vid": f[:-4], "xyz": z["xyz"], "h": z["h"], "sign_probs": logits_to_probs(z["sign_logits"]),
                          "phrase_probs": logits_to_probs(z["phrase_logits"]), "gold_sign": arr_to_segs(z["gold_sign"]),
                          "gold_phrase": arr_to_segs(z["gold_phrase"]), "T": int(z["T"]), "stride": int(z["stride"])})
    return items


def predicted_signs(item, b_thr=0.5, o_thr=0.5):
    return greedy_decode(item["sign_probs"], b_thr, o_thr)


def tokens_of(item, segs, groups):
    fps = fps_of(item["stride"])
    return build_tokens(segs, item["T"], fps, xyz=item["xyz"] if "prosody" in groups else None,
                        h=item["h"] if "h_pool" in groups else None,
                        probs=item["sign_probs"] if "sign_probs" in groups else None, groups=groups)[0]


# ------------------------------------------------------------------ training-source policy
class SourcePolicy:
    def __init__(self, source, ramp, jitter_max, b_thr, o_thr, seed):
        self.source, self.ramp, self.jitter_max = source, max(ramp, 1), jitter_max
        self.b_thr, self.o_thr = b_thr, o_thr
        self.rng = np.random.default_rng(seed)
        self._pred = {}

    def pred(self, item):
        if item["vid"] not in self._pred:
            self._pred[item["vid"]] = predicted_signs(item, self.b_thr, self.o_thr)
        return self._pred[item["vid"]]

    def segs(self, item, epoch):
        gold, f = item["gold_sign"], min(1.0, epoch / self.ramp)
        if self.source == "gold":
            return gold
        if self.source == "jitter":
            return jitter_segments(gold, item["T"], self.rng, strength=f * self.jitter_max)
        if self.source == "schedule":
            return self.pred(item) if self.rng.random() < f else gold
        if self.source == "mix":
            c = int(self.rng.integers(0, 3))
            return gold if c == 0 else (jitter_segments(gold, item["T"], self.rng, strength=self.jitter_max) if c == 1 else self.pred(item))
        if self.source == "pred":
            return self.pred(item)
        raise ValueError(self.source)

    @property
    def static(self):
        return self.source in ("gold", "pred")


def make_examples(items, policy, epoch, groups):
    ex = []
    for it in items:
        segs = policy.segs(it, epoch)
        if len(segs) == 0:
            continue
        ex.append((tokens_of(it, segs, groups), phrase_tags_over_signs(segs, it["gold_phrase"])))
    return ex


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict_pB(model, X, mean, std, window, stride, device, batch=64):
    """X (K, D) raw token features -> P(B) per sign (K,), windows of `window` tokens merged by averaging."""
    K = len(X)
    if K == 0:
        return np.zeros(0)
    Z = torch.from_numpy(((X - mean) / std).astype(np.float32))
    wins = plan_windows(K, window, stride)
    outs = []
    for k in range(0, len(wins), batch):
        chunk = wins[k:k + batch]
        L = max(e - s for s, e in chunk)
        xb = torch.zeros(len(chunk), L, Z.shape[1])
        pad = torch.ones(len(chunk), L, dtype=torch.bool)
        for i, (s, e) in enumerate(chunk):
            xb[i, :e - s] = Z[s:e]
            pad[i, :e - s] = False
        pr = F.softmax(model(xb.to(device), pad.to(device)), -1)[..., 1].cpu().numpy()
        outs.extend(pr[i, :e - s] for i, (s, e) in enumerate(chunk))
    return stitch_probs(K, wins, outs)


def phrases_from_pB(segs, pB, thr):
    if len(segs) == 0:
        return []
    return phrases_from_tags(segs, (np.asarray(pB) >= thr).astype(np.int64))


def evaluate_items(model, items, groups, mean, std, window, device, signs_of, thr=0.5, tols=(2, 5, 10)):
    P, G, L = [], [], []
    for it in items:
        segs = signs_of(it)
        X = tokens_of(it, segs, groups) if len(segs) else np.zeros((0, len(mean)), np.float32)
        P.append(phrases_from_pB(segs, predict_pB(model, X, mean, std, window, window // 2, device), thr))
        G.append(it["gold_phrase"]); L.append(it["T"])
    return evaluate_videos(P, G, L, tols=tols, iou_thrs=(0.1, 0.3, 0.5, 0.7))


# ------------------------------------------------------------------ bundle for evaluate.py
class Bundle:
    def __init__(self, path, device):
        ck = torch.load(os.path.join(path, "stage_c.pt"), map_location=device, weights_only=False)
        self.cfg, self.groups = ck["config"], tuple(ck["config"]["groups"])
        self.mean, self.std = ck["mean"], ck["std"]
        self.device = device
        self.model = build_model(self.cfg["arch"], len(self.mean), self.cfg).to(device)
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def pB(self, item, segs):
        X = tokens_of(item, segs, self.groups) if len(segs) else np.zeros((0, len(self.mean)), np.float32)
        return predict_pB(self.model, X, self.mean, self.std, self.cfg["window"], self.cfg["window"] // 2, self.device)


# ------------------------------------------------------------------ train
def stage_c_dir(stage_a, tag, seed):
    d = os.path.join(RUNS_DIR, stage_a, "stage_c", f"{tag}_s{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = tuple(a.groups)
    if a.oof_runs and "h_pool" in groups:
        raise SystemExit("--oof-runs requires --groups without h_pool: fold models do not share a feature space.")
    if a.source == "pred" and not a.oof_runs and "h_pool" in groups:
        print("WARNING: --source pred without --oof-runs trains on IN-SAMPLE Stage-A predictions (too clean). "
              "This is the naive cascade; use --oof-runs for the out-of-fold variant.")
    out = stage_c_dir(a.stage_a, a.tag, a.seed)
    base = os.path.join(RUNS_DIR, a.stage_a, "cache")
    train_dirs = [os.path.join(RUNS_DIR, r, "cache", "oof") for r in a.oof_runs] if a.oof_runs else [os.path.join(base, "train")]
    print("loading caches ...")
    train_items = load_items(train_dirs, a.limit)
    val_items = load_items([os.path.join(base, "val")], a.limit)
    print(f"train videos {len(train_items)}  val videos {len(val_items)}  groups {groups}")

    policy = SourcePolicy(a.source, a.ramp, a.jitter, a.b_thr, a.o_thr, a.seed)
    gold_ex = make_examples(train_items, SourcePolicy("gold", 1, 0, 0.5, 0.5, 0), 1, groups)
    allX = np.concatenate([x for x, _ in gold_ex])
    mean, std = allX.mean(0), allX.std(0) + 1e-6
    tags_all = np.concatenate([t for _, t in gold_ex])
    cnt = np.bincount(tags_all, minlength=2).astype(np.float64)
    w = (cnt.sum() / (2.0 * cnt)) ** a.weight_power
    print(f"sign-level tags: I {int(cnt[0])}  B {int(cnt[1])}  (1:{cnt[0] / cnt[1]:.1f})  class weights (I,B) = {np.round(w, 2).tolist()}")
    cfg = {**vars(a), "groups": list(groups), "in_dim": int(allX.shape[1]), "window": a.window}
    model = build_model(a.arch, allX.shape[1], cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Stage C ({a.arch}) parameters: {n_par:,}   input dim {allX.shape[1]}")
    cfg["n_params"] = n_par
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    examples = gold_ex if policy.static and a.source == "gold" else None
    best, best_state, bad, log = -1.0, None, 0, []
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        if examples is None or not policy.static:
            examples = make_examples(train_items, policy, epoch, groups)
        sizes = np.array([len(x) for x, _ in examples], dtype=np.float64)
        n_win = max(1, int(sizes.sum() // (a.window // 2)))
        pvid = sizes / sizes.sum()
        model.train()
        tot, steps = 0.0, 0
        for _ in range(0, n_win, a.batch):
            xb = np.zeros((a.batch, a.window, allX.shape[1]), np.float32)
            yb = np.full((a.batch, a.window), -100, np.int64)
            pad = np.ones((a.batch, a.window), bool)
            for i in range(a.batch):
                x, t = examples[int(rng.choice(len(examples), p=pvid))]
                s = int(rng.integers(0, max(len(x) - a.window, 0) + 1))
                e = min(len(x), s + a.window)
                xb[i, :e - s] = (x[s:e] - mean) / std
                yb[i, :e - s] = t[s:e]
                pad[i, :e - s] = False
            logits = model(torch.from_numpy(xb).to(device), torch.from_numpy(pad).to(device))
            loss = focal_ce(logits.reshape(-1, 2), torch.from_numpy(yb).to(device).reshape(-1), wt, a.focal_gamma)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); steps += 1
        sched.step()
        model.eval()
        r_pred = evaluate_items(model, val_items, groups, mean, std, a.window, device, lambda it: predicted_signs(it, a.b_thr, a.o_thr))
        r_gold = evaluate_items(model, val_items, groups, mean, std, a.window, device, lambda it: it["gold_sign"])
        score = r_pred["frame_f1"]
        improved = score > best
        if improved:
            best, bad, best_state = score, 0, copy.deepcopy(model.state_dict())
            torch.save({"state": best_state, "config": cfg, "mean": mean, "std": std}, os.path.join(out, "stage_c.pt"))
        else:
            bad += 1
        row = {"epoch": epoch, "loss": tot / max(steps, 1), "val_pred_signs_frame_f1": score, "val_pred_ratio": r_pred["ratio"],
               "val_pred_start_f1@5": r_pred["start_f1@5"], "val_gold_signs_frame_f1": r_gold["frame_f1"], "val_gold_ratio": r_gold["ratio"]}
        log.append(row)
        print(f"epoch {epoch:03d} loss {row['loss']:.4f} | val[pred signs] F1 {score:.4f} ratio {r_pred['ratio']:.2f} startF1@5 {r_pred['start_f1@5']:.3f}"
              f" | val[gold signs] F1 {r_gold['frame_f1']:.4f} ratio {r_gold['ratio']:.2f} | {time.time() - t0:.0f}s" + (" *" if improved else f" (x{bad})"))
        json.dump(log, open(os.path.join(out, "stage_c_log.json"), "w"), indent=1)
        if bad >= a.patience:
            print(f"early stopping after {epoch} epochs")
            break
    print(f"best val frame F1 (predicted signs) {best:.4f} -> {out}/stage_c.pt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--stage-a", required=True, help="Stage-A run name (its cache/ supplies val/test and, unless --oof-runs, train)")
    t.add_argument("--tag", default="main")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--source", choices=["gold", "jitter", "schedule", "mix", "pred"], default="mix")
    t.add_argument("--groups", nargs="+", default=list(ALL_GROUPS), choices=list(ALL_GROUPS))
    t.add_argument("--oof-runs", nargs="*", default=[])
    t.add_argument("--arch", choices=["transformer", "bilstm"], default="transformer")
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--lstm-layers", type=int, default=2)
    t.add_argument("--lstm-hidden", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--feat-dropout", type=float, default=0.1)
    t.add_argument("--window", type=int, default=128)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--weight-power", type=float, default=1.0, help="class weight = (N / (2 n_c))^power from the train B:I ratio")
    t.add_argument("--focal-gamma", type=float, default=0.0, help="0 = weighted CE; 2 = focal loss")
    t.add_argument("--ramp", type=int, default=20, help="epochs over which the gold->noisy schedule ramps")
    t.add_argument("--jitter", type=float, default=1.0)
    t.add_argument("--b-thr", type=float, default=0.5)
    t.add_argument("--o-thr", type=float, default=0.5)
    t.add_argument("--limit", type=int, default=None, help="debug: only the first N cached videos per directory")
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()