"""
hierarchy_phrase/stage_a.py -- Stage A of the hierarchy: an E1s-style BiLSTM over MediaPipe keypoints.

    python stage_a.py train  --name sa_s42 --seed 42                       # full model (used for val/test + gold/jitter training)
    python stage_a.py train  --name sa_f0  --seed 42 --fold 0 --n-folds 4  # one out-of-fold model (held-out fold exported as OOF)
    python stage_a.py export --name sa_s42                                 # cache features / logits for Stage C and evaluate.py

Model (2023 E1s configuration): 4-layer bidirectional LSTM, hidden 256, Adam lr 1e-3, weighted cross-entropy with B up-weighted.
Inputs are the 65 body+hand landmarks (x, y, z), 195 values per frame, shoulder-normalised as in the rest of this project.
Two heads read the same BiLSTM output: a SIGN BIO head (what the hierarchy uses) and a flat frame-level PHRASE BIO head --
the head of the E1s baseline, so "flat vs hierarchical" (ablation 1) compares heads on the SAME encoder. Training it jointly
(as in 2023) lets the phrase loss shape the shared features; `--no-phrase-head` trains a sign-only encoder for ablation 6.

Training uses random crops of --crop frames (default 1024); inference runs on whole videos. Model selection: mean of the
argmax frame macro-F1 of the sign head and (if present) the phrase head on the validation split, the paper's primary metric.
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

from common import (DEFAULT_STRIDE, RUNS_DIR, SPLIT_FILE, have_all_files, load_cache_video, load_gold_segments,
                    load_keypoints, run_dir, save_cache_video, segs_to_arr, split_ids)
from metrics import evaluate_videos
from segments import bio_to_segments, segments_to_bio


# ------------------------------------------------------------------ model
class StageA(nn.Module):
    def __init__(self, in_dim=195, hidden=256, layers=4, dropout=0.2, phrase_head=True):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.sign_head = nn.Linear(2 * hidden, 3)
        self.phrase_head = nn.Linear(2 * hidden, 3) if phrase_head else None

    def forward(self, x, lengths=None):
        """x (B, T, 195) -> sign logits (B, T, 3), phrase logits (B, T, 3) or None, features h (B, T, 2*hidden)."""
        z = self.inp(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        else:
            h, _ = self.lstm(z)
        return self.sign_head(h), (self.phrase_head(h) if self.phrase_head is not None else None), h


# ------------------------------------------------------------------ data
def doc_of(vid):
    return vid.rsplit("_", 1)[0]


def fold_split(train_ids, fold, n_folds, seed=0):
    """Videos of one document (both signers) always land in the same fold -- they share content."""
    docs = sorted({doc_of(v) for v in train_ids})
    r = np.random.default_rng(seed)
    r.shuffle(docs)
    fold_of = {d: i % n_folds for i, d in enumerate(docs)}
    held = [v for v in train_ids if fold_of[doc_of(v)] == fold]
    return [v for v in train_ids if fold_of[doc_of(v)] != fold], held


class Store:
    """Keypoints (float16) and raw BIO label arrays for one split, held in RAM."""

    def __init__(self, split, stride, ids=None, verbose=True):
        ids = ids if ids is not None else split_ids(split)
        self.ids, self.x, self.sign, self.phrase, self.gold = [], [], [], [], []
        skipped = 0
        for vid in ids:
            if not have_all_files(vid):
                skipped += 1
                continue
            g = load_gold_segments(vid, stride)
            if g is None or not g["sign"]:
                skipped += 1
                continue
            kp = load_keypoints(vid, stride)
            T = min(len(kp), g["T"])
            self.ids.append(vid)
            self.x.append(kp[:T].reshape(T, -1).astype(np.float16))
            self.sign.append(segments_to_bio(g["sign"], g["T"])[:T])
            self.phrase.append(segments_to_bio(g["phrase"], g["T"])[:T])
            self.gold.append({"sign": [(s, e) for s, e in g["sign"] if e <= T], "phrase": [(s, e) for s, e in g["phrase"] if e <= T], "T": T})
        if verbose:
            print(f"[{split}] {len(self.ids)} videos loaded ({skipped} skipped: missing label/keypoint file)")

    def __len__(self):
        return len(self.ids)


def class_weights(label_list, power=0.5):
    cnt = np.bincount(np.concatenate(label_list).astype(np.int64), minlength=3).astype(np.float64)
    w = (cnt.sum() / (3.0 * np.maximum(cnt, 1))) ** power
    return (w / w[0]).astype(np.float32)               # O = 1, so B carries the up-weighting


def make_batch(store, idxs, crop, rng, device):
    lens, xs, ys, yp = [], [], [], []
    for i in idxs:
        T = len(store.x[i])
        s = int(rng.integers(0, max(T - crop, 0) + 1))
        e = min(T, s + crop)
        xs.append(store.x[i][s:e]); ys.append(store.sign[i][s:e]); yp.append(store.phrase[i][s:e]); lens.append(e - s)
    L = max(lens)
    X = np.zeros((len(idxs), L, store.x[0].shape[1]), np.float32)
    Ys = np.full((len(idxs), L), -100, np.int64)
    Yp = np.full((len(idxs), L), -100, np.int64)
    for k, (x, a, b) in enumerate(zip(xs, ys, yp)):
        X[k, :len(x)] = x; Ys[k, :len(x)] = a; Yp[k, :len(x)] = b
    t = lambda a: torch.from_numpy(a).to(device)                    # noqa: E731
    return t(X), t(Ys), t(Yp), torch.tensor(lens)


@torch.no_grad()
def infer_video(model, x_np, device):
    x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(device)
    sg, ph, h = model(x)
    return (sg[0].float().cpu().numpy(), None if ph is None else ph[0].float().cpu().numpy(), h[0].float().cpu().numpy())


def validate(model, store, device):
    """Argmax frame macro-F1 (the 2023 paper's primary metric) of the sign head and, if present, the phrase head."""
    model.eval()
    ps, pp, gs, gp, lens = [], [], [], [], []
    for i in range(len(store)):
        sg, ph, _ = infer_video(model, store.x[i], device)
        ps.append(bio_to_segments(sg.argmax(1)))
        gs.append(store.gold[i]["sign"]); lens.append(store.gold[i]["T"])
        if ph is not None:
            pp.append(bio_to_segments(ph.argmax(1))); gp.append(store.gold[i]["phrase"])
    r_sign = evaluate_videos(ps, gs, lens, tols=(5,), iou_thrs=(0.5,))
    out = {"sign_frame_f1": r_sign["frame_f1"], "sign_ratio": r_sign["ratio"]}
    score = r_sign["frame_f1"]
    if pp:
        r_ph = evaluate_videos(pp, gp, lens, tols=(5,), iou_thrs=(0.5,))
        out.update({"phrase_frame_f1": r_ph["frame_f1"], "phrase_ratio": r_ph["ratio"]})
        score = 0.5 * (r_sign["frame_f1"] + r_ph["frame_f1"])
    out["score"] = score
    return out


# ------------------------------------------------------------------ train
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(args.name)
    ids = split_ids("train")
    held = []
    if args.fold is not None:
        ids, held = fold_split(ids, args.fold, args.n_folds)
        with open(os.path.join(out, "heldout.json"), "w") as f:
            json.dump(held, f)
        print(f"fold {args.fold}/{args.n_folds}: training on {len(ids)} videos, holding out {len(held)}")
    train_s, val_s = Store("train", args.stride, ids), Store("val", args.stride)
    w_sign = class_weights(train_s.sign, args.weight_power)
    w_phr = class_weights(train_s.phrase, args.weight_power)
    print(f"class weights (O,I,B): sign {np.round(w_sign, 2).tolist()}  phrase {np.round(w_phr, 2).tolist()}")
    model = StageA(in_dim=train_s.x[0].shape[1], hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                   phrase_head=not args.no_phrase_head).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"StageA parameters: {n_par:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    cw_s, cw_p = torch.tensor(w_sign, device=device), torch.tensor(w_phr, device=device)
    cfg = {**vars(args), "n_params": n_par, "weights_sign": w_sign.tolist(), "weights_phrase": w_phr.tolist(),
           "in_dim": train_s.x[0].shape[1], "train_videos": train_s.ids}
    json.dump(cfg, open(os.path.join(out, "stage_a_config.json"), "w"), indent=1)

    best, best_state, bad, log = -1.0, None, 0, []
    steps_per_epoch = max(1, (len(train_s) * args.crops_per_video) // args.batch)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(steps_per_epoch):
            idxs = rng.integers(0, len(train_s), size=args.batch)
            X, Ys, Yp, lens = make_batch(train_s, idxs, args.crop, rng, device)
            sg, ph, _ = model(X, lens)
            loss = F.cross_entropy(sg.reshape(-1, 3), Ys.reshape(-1), weight=cw_s, ignore_index=-100)
            if ph is not None:
                loss = loss + F.cross_entropy(ph.reshape(-1, 3), Yp.reshape(-1), weight=cw_p, ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        v = validate(model, val_s, device)
        improved = v["score"] > best
        if improved:
            best, bad, best_state = v["score"], 0, copy.deepcopy(model.state_dict())
            torch.save({"state": best_state, "config": cfg}, os.path.join(out, "stage_a.pt"))
        else:
            bad += 1
        log.append({"epoch": epoch, "train_loss": tot / steps_per_epoch, **v})
        print(f"epoch {epoch:03d} loss {tot / steps_per_epoch:.4f} | " +
              " ".join(f"{k} {x:.4f}" for k, x in v.items()) + f" | {time.time() - t0:.0f}s" + (" *" if improved else f" (no improvement x{bad})"))
        json.dump(log, open(os.path.join(out, "stage_a_log.json"), "w"), indent=1)
        if bad >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break
    print(f"best validation score {best:.4f}  ->  {out}/stage_a.pt")


# ------------------------------------------------------------------ export
def load_model(name, device):
    ck = torch.load(os.path.join(RUNS_DIR, name, "stage_a.pt"), map_location=device, weights_only=False)
    c = ck["config"]
    m = StageA(in_dim=c["in_dim"], hidden=c["hidden"], layers=c["layers"], dropout=c["dropout"],
               phrase_head=not c["no_phrase_head"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, c


def cache_dir(name, split):
    return os.path.join(RUNS_DIR, name, "cache", split)


def export(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.name, device)
    jobs = []                                              # (split label, ids)
    heldout_path = os.path.join(RUNS_DIR, args.name, "heldout.json")
    if os.path.exists(heldout_path):                       # a fold model: only its held-out videos are out-of-fold
        jobs.append(("oof", json.load(open(heldout_path))))
        splits = [s for s in args.splits if s != "train"]
    else:
        splits = args.splits
    for sp in splits:
        jobs.append((sp, split_ids(sp)))
    for label, ids in jobs:
        n = 0
        for vid in ids:
            if not have_all_files(vid):
                continue
            g = load_gold_segments(vid, args.stride)
            if g is None or not g["sign"]:
                continue
            kp = load_keypoints(vid, args.stride)
            T = min(len(kp), g["T"])
            sg, ph, h = infer_video(model, kp[:T].reshape(T, -1), device)
            save_cache_video(os.path.join(cache_dir(args.name, label), f"{vid}.npz"),
                             xyz=kp[:T].astype(np.float16), h=h.astype(np.float16), sign_logits=sg.astype(np.float32),
                             phrase_logits=(ph if ph is not None else np.zeros_like(sg)).astype(np.float32),
                             gold_sign=segs_to_arr([s for s in g["sign"] if s[1] <= T]),
                             gold_phrase=segs_to_arr([s for s in g["phrase"] if s[1] <= T]),
                             T=np.int64(T), stride=np.int64(args.stride), has_phrase_head=np.int64(ph is not None))
            n += 1
        print(f"cached {n} videos -> {cache_dir(args.name, label)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--crop", type=int, default=1024)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--crops-per-video", type=int, default=2)
    t.add_argument("--weight-power", type=float, default=0.5, help="class weight = (inverse frequency)^power, O=1")
    t.add_argument("--no-phrase-head", action="store_true")
    t.add_argument("--fold", type=int, default=None)
    t.add_argument("--n-folds", type=int, default=4)
    e = sub.add_parser("export")
    e.add_argument("--name", required=True)
    e.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    e.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    train(a) if a.cmd == "train" else export(a)


if __name__ == "__main__":
    main()