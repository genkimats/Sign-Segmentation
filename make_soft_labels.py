"""
Ambiguity-aware soft training targets for phrase segmentation (AMBIGUITY study).

Why: the error analysis showed that many remaining errors sit where the ANNOTATION itself is
inconsistent -- after a palm-up gesture a phrase boundary follows only ~61% of the time, so
the same visual situation is labelled "boundary" in one place and "no boundary" in another.
Hard 0/1 targets force the model to be fully confident on both. Here each sign transition's
target is softened towards how the annotation USUALLY treats that kind of transition:

    p_hat = P(boundary | annotation cues of this transition)      (out-of-fold, see below)
    gold phrase start   (frame is B):  q_B = (1 - alpha) + alpha * p_hat,  q_I = 1 - q_B
    within-phrase sign  (frame is I):  q_B =       alpha  * p_hat,         q_I = 1 - q_B

Clear cases (p_hat close to the gold label) stay almost hard; contradictory cases (a start
where starts are rare, or a non-start where starts are usual) are softened. Only the first
frame of each sign is touched; every other frame keeps its one-hot target. With alpha < 0.5
the gold class always stays the most probable one, so argmax(target) = gold label (class
weights, the hard validation loss and the count loss keep working unchanged).

p_hat comes from the same logistic-regression oracle as phrase_oracle_analysis.py (feature
groups "all" = sign timing + hands + mouthing + glosses, or "lexical"), fitted OUT-OF-FOLD over
the training videos (GroupKFold by video): a transition's p_hat never comes from a model that
saw that transition, so it reflects the annotation's consistency, not memorisation.
Only TRAIN videos get soft targets; validation / test always use the gold labels.

Run once from the repo root (needs the .eaf annotations, the sign BIO tags and sklearn):
  python make_soft_labels.py                          # alpha 0.2 and 0.4 -> processed_data/BIO_soft_ambig_a20 / _a40
  python make_soft_labels.py --alphas 0.3 --features lexical

Outputs: processed_data/BIO_soft_ambig_a<alpha*100>[_lex]/<video>.npy, float32 (T, 3) [O, I, B],
plus soft_labels_summary.json in each folder.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from phrase_oracle_analysis import GROUPS, Featurizer, gloss_base, video_transitions

O_TAG, I_TAG, B_TAG = 0, 1, 2


def one_hot(labels):
    out = np.zeros((len(labels), 3), dtype=np.float32)
    out[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return out


def apply_soft(phrase, rows, p_hat, alpha):
    """phrase: (T,) gold BIO; rows: this video's transitions; p_hat: their P(boundary)."""
    q = one_hot(phrase)
    n_used = 0
    for r, p in zip(rows, p_hat):
        f = int(r["frame"])
        if not 0 <= f < len(phrase):
            continue
        gold = int(phrase[f])
        if r["y"] == 1 and gold == B_TAG:
            q_b = (1.0 - alpha) + alpha * p
        elif r["y"] == 0 and gold == I_TAG:
            q_b = alpha * p
        else:
            continue
        q[f] = (0.0, 1.0 - q_b, q_b)
        n_used += 1
    return q, n_used


def out_of_fold_p(rows, blocks, folds, C, min_count):
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["video"] for r in rows])
    p = np.full(len(rows), np.nan)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(rows, y, groups)):
        rtr = [rows[i] for i in tr]
        rte = [rows[i] for i in te]
        F = Featurizer(rtr, min_count)
        Xtr, _ = F.matrix(rtr, blocks)
        Xte, _ = F.matrix(rte, blocks)
        lr = LogisticRegression(C=C, max_iter=3000)
        lr.fit(Xtr, y[tr])
        p[te] = lr.predict_proba(Xte)[:, 1]
        print(f"  fold {k + 1}/{folds}: {len(tr)} train / {len(te)} held-out transitions")
    return p, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=os.path.join("raw_data", "annotations"))
    ap.add_argument("--gloss-labels", default=os.path.join("processed_data", "BIO_tags"))
    ap.add_argument("--phrase-labels", default=os.path.join("processed_data", "BIO_tags_phrase_v2"))
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--short-gap", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--features", choices=["all", "lexical"], default="all",
                    help="oracle feature group for p_hat (default all = timing + hands + mouth + glosses)")
    ap.add_argument("--C", type=float, default=0.1, help="logistic-regression C (0.1 = best 'all' oracle)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--out-root", default="processed_data")
    args = ap.parse_args()

    for a in args.alphas:
        if not 0 < a < 0.5:
            raise SystemExit(f"alpha must be in (0, 0.5) so the gold class stays on top; got {a}")

    with open(args.split_file) as f:
        splits = json.load(f)
    vids = sorted(re.sub(r"\.(npy|pt)$", "", v) for v in splits["train"])
    rows, by_vid, skipped, eaf_cache = [], defaultdict(list), defaultdict(list), {}
    for vid in tqdm(vids, desc="train transitions"):
        r, why = video_transitions(vid, args, eaf_cache)
        if r is None:
            skipped[why].append(vid)
            continue
        for x in r:
            by_vid[vid].append(len(rows))
            rows.append(x)
    y_all = np.array([r["y"] for r in rows])
    print(f"train: {len(vids)} videos, {len(rows)} sign transitions, {int(y_all.sum())} phrase starts; "
          f"videos without transitions (keep hard targets): { {k: len(v) for k, v in skipped.items()} }")

    blocks = GROUPS["all"] if args.features == "all" else GROUPS["lexical"]
    print(f"out-of-fold P(boundary) from the '{args.features}' oracle ({args.folds} folds by video, C={args.C})")
    p, y = out_of_fold_p(rows, blocks, args.folds, args.C, args.min_count)
    summary = {
        "features": args.features, "C": args.C, "folds": args.folds,
        "n_transitions": len(rows), "n_starts": int(y.sum()),
        "oof_auc": round(float(roc_auc_score(y, p)), 4),
        "oof_ap": round(float(average_precision_score(y, p)), 4),
        "mean_p_at_starts": round(float(p[y == 1].mean()), 4),
        "mean_p_at_non_starts": round(float(p[y == 0].mean()), 4),
    }
    ctx = defaultdict(list)
    for r, pi in zip(rows, p):
        ctx[gloss_base(r["gloss_prev"])].append((r["y"], pi))
    frequent = sorted(((g, v) for g, v in ctx.items() if len(v) >= 100), key=lambda kv: -len(kv[1]))[:15]
    summary["contexts_after_gloss"] = {g: {"n": len(v), "boundary_rate": round(float(np.mean([a for a, _ in v])), 3),
                                           "mean_p_hat": round(float(np.mean([b for _, b in v])), 3)}
                                       for g, v in frequent}
    print(f"  out-of-fold AUC {summary['oof_auc']}, AP {summary['oof_ap']}; mean p_hat at starts "
          f"{summary['mean_p_at_starts']}, at non-starts {summary['mean_p_at_non_starts']}")

    for alpha in args.alphas:
        name = f"BIO_soft_ambig_a{int(round(alpha * 100)):02d}" + ("_lex" if args.features == "lexical" else "")
        out = os.path.join(args.out_root, name)
        os.makedirs(out, exist_ok=True)
        dev, n_soft, n_hard = [], 0, 0
        for vid in tqdm(vids, desc=f"writing {name}"):
            pp = os.path.join(args.phrase_labels, f"{vid}.npy")
            if not os.path.exists(pp):
                continue
            phrase = np.load(pp)
            idx = by_vid.get(vid, [])
            q, used = apply_soft(phrase, [rows[i] for i in idx], p[idx] if idx else [], alpha)
            assert (q.argmax(1) == phrase).all(), f"{vid}: soft target changed the gold class"
            np.save(os.path.join(out, f"{vid}.npy"), q)
            n_soft += used
            n_hard += int(used == 0)
            if idx:
                yy = np.array([rows[i]["y"] for i in idx])
                qb = np.array([q[rows[i]["frame"], B_TAG] for i in idx])
                dev += list(np.abs(qb - yy))
        s = dict(summary, alpha=alpha, softened_transitions=n_soft, videos_all_hard=n_hard,
                 mean_abs_change_at_transitions=round(float(np.mean(dev)), 4) if dev else 0.0)
        with open(os.path.join(out, "soft_labels_summary.json"), "w") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        print(f"  {name}: {n_soft} transitions softened, mean |q_B - gold| {s['mean_abs_change_at_transitions']}"
              f" -> {out}/")


if __name__ == "__main__":
    main()