"""
phrase_oracle_analysis.py -- how much of a phrase boundary is visible, and how much is linguistic?

Oracle decomposition on the GOLD annotation (no video). The unit is a sign-to-sign
transition: for every sign that lies inside a phrase, "does a new phrase start here?"
(y = 1 if the sign is the first sign of a phrase). Signs come from the verified gloss
labels (processed_data/BIO_tags), phrases from the v2 phrase labels, gloss identities and
mouthings from the ELAN files. Classifiers trained on TRAIN, tuned on VAL, reported on
TEST, using one feature group at a time:

  gap only     the pause before the sign (a single-number "pause rule")
  timing       pauses before/after, sign durations (final lengthening), local signing rate,
               whether the pause is the largest in its neighbourhood, signs since the last
               long pause -- all normalized per video (signing speed)
  hands        right / left / both-handed for the previous and current sign, hand switch
  mouth        mouthing (spoken-word mouth picture) vs mouth gesture vs none for both signs,
               and whether ONE mouthing spreads across the transition (a known within-
               phrase cue in DGS)
  lexical      gloss identity (lexeme) of the two signs before and after the transition,
               plus gloss categories ($GEST, $GEST-OFF palm-up, $INDEX pointing, $PROD
               depicting, $ALPHA fingerspelling, $NUM, ...)
  non-lexical  timing + hands + mouth  (roughly what a perfect pose model could see)
  all          everything

Reading the results:
  * non-lexical ~ your model    -> the model already uses what the visible rhythm offers
  * lexical / all >> non-lexical -> phrase boundaries need language knowledge (recognition-
                                    informed segmentation), not better pose encoders
  * even "all" is low           -> the labels (translation units) are partly arbitrary:
                                    that, not the model, is the ceiling

With --predictions (from evaluate_phrase.py --save-predictions, val AND test) your trained
models are scored on the SAME transitions (score = max P(B) within +-k frames of the sign
start), so they sit in the same table. "model + lexical" stacks the model score with the
lexical oracle (fit on val, reported on test): how much a perfect gloss recognizer would
add to the current model.

Note: every number here is CONDITIONAL ON GOLD SIGN SEGMENTATION (candidate positions are
the true sign starts), for the oracles and the models alike.

Outputs (evaluation_phrase/oracle/):
  oracle_summary.csv / .txt     one row per feature group (and model)
  oracle_lexical_weights.csv    glosses / categories most indicative of a phrase boundary
  transitions_<split>.csv       every transition with features, label and oracle scores
                                (input for boundary_error_analysis.py)

Usage (repo root):
  python phrase_oracle_analysis.py
  python phrase_oracle_analysis.py --predictions evaluation_phrase/predictions \
      --runs multistream_bilstm-124 multistream_bilstm-125 multistream_bilstm-126
"""
import argparse
import csv
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

O_TAG, I_TAG, B_TAG = 0, 1, 2
GLOSS_CATEGORIES = ["$GEST-OFF", "$GEST", "$INDEX", "$PROD", "$ALPHA", "$NUM", "$ORAL", "$LIST", "$PMS", "$"]
HANDS = ["r", "l", "rl", "none"]
MOUTH = ["word", "gesture", "none"]


# ============================================================================ ELAN parsing
def read_eaf(path):
    """{tier_id: sorted [(start_ms, end_ms, value)]} for time-aligned annotations."""
    root = ET.parse(path).getroot()
    ts = {s.get("TIME_SLOT_ID"): s.get("TIME_VALUE") for s in root.iter("TIME_SLOT")}
    tiers = {}
    for tier in root.iter("TIER"):
        anns = []
        for a in tier.iter("ALIGNABLE_ANNOTATION"):
            t1, t2 = ts.get(a.get("TIME_SLOT_REF1")), ts.get(a.get("TIME_SLOT_REF2"))
            if t1 is None or t2 is None:
                continue
            v = a.find("ANNOTATION_VALUE")
            anns.append((int(t1), int(t2), (v.text or "").strip() if v is not None else ""))
        if anns:
            tiers[tier.get("TIER_ID")] = sorted(anns)
    return tiers


def find_tier(tiers, prefix, suffix):
    for tid in tiers:
        if tid.startswith(prefix) and tid.endswith(suffix):
            return tiers[tid]
    return []


def to_frames(anns, fps):
    if not anns:
        return np.zeros((0, 2), dtype=np.int64), []
    a = np.array([[int(round(s / 1000.0 * fps)), int(round(e / 1000.0 * fps)) - 1] for s, e, _ in anns])
    return a, [v for _, _, v in anns]


def gloss_base(g):
    """'GEHEN3^*' -> 'GEHEN', '$INDEX1' -> '$INDEX', '$GEST-OFF^' -> '$GEST-OFF'."""
    g = g.strip().upper()
    g = re.sub(r"[\^\*]+", "", g)
    g = re.sub(r"\d+[A-Z]?$", "", g)
    return g or "?"


def gloss_category(g):
    g = g.strip().upper()
    for c in GLOSS_CATEGORIES:
        if g.startswith(c):
            return c
    return "lexical"


def mouth_kind(v):
    if not v:
        return "none"
    return "gesture" if v.startswith("[") else "word"


# ============================================================================ signs & features
def sign_segments(gloss_labels):
    """Same rule as create_phrase_bio_tags_v2.py: B starts a sign, O ends it (inclusive ends)."""
    segs, start = [], -1
    for i, tag in enumerate(gloss_labels):
        if tag == B_TAG:
            if start != -1:
                segs.append((start, i - 1))
            start = i
        elif tag == O_TAG:
            if start != -1:
                segs.append((start, i - 1))
                start = -1
        elif start == -1:
            start = i
    if start != -1:
        segs.append((start, len(gloss_labels) - 1))
    return np.array(segs, dtype=np.int64).reshape(-1, 2)


def overlap_matrix(signs, anns):
    if len(anns) == 0 or len(signs) == 0:
        return np.zeros((len(signs), len(anns)))
    lo = np.maximum(signs[:, None, 0], anns[None, :, 0])
    hi = np.minimum(signs[:, None, 1], anns[None, :, 1])
    return np.clip(hi - lo + 1, 0, None).astype(np.float64)


def video_transitions(vid, args, eaf_cache):
    transcript, participant = vid.rsplit("_", 1)
    gp = os.path.join(args.gloss_labels, f"{vid}.npy")
    pp = os.path.join(args.phrase_labels, f"{vid}.npy")
    ep = os.path.join(args.annotations, f"{transcript}.eaf")
    if not (os.path.exists(gp) and os.path.exists(pp)):
        return None, "missing_labels"
    if not os.path.exists(ep):
        return None, "missing_eaf"
    gloss, phrase = np.load(gp), np.load(pp)
    if len(gloss) != len(phrase):
        return None, "length_mismatch"
    if transcript not in eaf_cache:
        eaf_cache.clear()
        eaf_cache[transcript] = read_eaf(ep)
    tiers = eaf_cache[transcript]
    signs = sign_segments(gloss)
    if len(signs) < 3:
        return None, "too_few_signs"

    r_fr, r_val = to_frames(find_tier(tiers, "Lexem_Geb", f"_r_{participant}"), args.fps)
    l_fr, l_val = to_frames(find_tier(tiers, "Lexem_Geb", f"_l_{participant}"), args.fps)
    m_fr, m_val = to_frames(find_tier(tiers, "Mundbild", f"_{participant}"), args.fps)
    if len(r_fr) + len(l_fr) == 0:
        return None, "no_gloss_tier"

    dur = signs[:, 1] - signs[:, 0] + 1
    ov_r, ov_l, ov_m = overlap_matrix(signs, r_fr), overlap_matrix(signs, l_fr), overlap_matrix(signs, m_fr)
    best_r = ov_r.max(1) if ov_r.shape[1] else np.zeros(len(signs))
    best_l = ov_l.max(1) if ov_l.shape[1] else np.zeros(len(signs))
    gl, hand, mouth, mouth_id = [], [], [], []
    for i in range(len(signs)):
        if best_r[i] >= best_l[i] and best_r[i] > 0:
            g = r_val[int(ov_r[i].argmax())]
        elif best_l[i] > 0:
            g = l_val[int(ov_l[i].argmax())]
        else:
            g = "?"
        gl.append(g)
        if best_r[i] >= 0.5 * dur[i] and best_l[i] >= 0.5 * dur[i]:
            hand.append("rl")
        elif max(best_r[i], best_l[i]) > 0:
            hand.append("r" if best_r[i] >= best_l[i] else "l")
        else:
            hand.append("none")
        if ov_m.shape[1] and ov_m[i].max() > 0:
            j = int(ov_m[i].argmax())
            mouth.append(mouth_kind(m_val[j]))
            mouth_id.append(j)
        else:
            mouth.append("none")
            mouth_id.append(-1)

    gaps = np.r_[np.nan, signs[1:, 0] - signs[:-1, 1] - 1].astype(np.float64)   # gap BEFORE sign i
    pos_gaps = gaps[1:][gaps[1:] > 0]
    med_gap = float(np.median(pos_gaps)) if len(pos_gaps) else 1.0
    long_gap = float(np.percentile(pos_gaps, 90)) if len(pos_gaps) else 25.0
    med_dur = float(np.median(dur))
    starts_s = signs[:, 0] / args.fps

    rows = []
    since_long = 0
    for i in range(1, len(signs)):
        g = gaps[i]
        since_long = 0 if g >= long_gap else since_long + 1
        lab = phrase[signs[i, 0]]
        if lab == O_TAG:            # sign not assigned to any phrase
            continue
        nb = gaps[max(1, i - 5):min(len(signs), i + 6)]
        nb = nb[np.isfinite(nb)]
        others = np.delete(nb, np.where(nb == g)[0][:1]) if len(nb) > 1 else np.array([0.0])
        g_next = gaps[i + 1] if i + 1 < len(signs) else np.nan
        rate = np.sum(np.abs(starts_s - starts_s[i]) <= 2.0) / 4.0
        gtype = "continuous" if g <= 0 else ("short_pause" if g <= args.short_gap else "long_pause")
        rows.append({
            "video": vid, "sign_index": i, "frame": int(signs[i, 0]), "y": int(lab == B_TAG),
            "boundary_gap_type": gtype,
            # timing
            "gap_prev": float(g), "gap_next": float(g_next) if np.isfinite(g_next) else -1.0,
            "log_gap_prev": float(np.log1p(max(g, 0))), "log_gap_next": float(np.log1p(max(g_next, 0))) if np.isfinite(g_next) else 0.0,
            "gap_prev_rel": float(max(g, 0) / max(med_gap, 1.0)),
            "gap_prev_local_max": float(max(g, 0) / max(np.max(others), 1.0)),
            "dur_prev": float(dur[i - 1]), "dur_cur": float(dur[i]),
            "dur_prev_rel": float(dur[i - 1] / med_dur), "dur_cur_rel": float(dur[i] / med_dur),
            "signs_since_long_gap": float(min(since_long, 30)), "local_rate": float(rate),
            # hands
            "hand_prev": hand[i - 1], "hand_cur": hand[i], "hand_switch": int(hand[i - 1] != hand[i]),
            # mouth
            "mouth_prev": mouth[i - 1], "mouth_cur": mouth[i],
            "mouth_spread": int(mouth_id[i] >= 0 and mouth_id[i] == mouth_id[i - 1]),
            # lexical
            "gloss_prev2": gl[i - 2] if i >= 2 else "<s>", "gloss_prev": gl[i - 1], "gloss_cur": gl[i],
            "gloss_next": gl[i + 1] if i + 1 < len(gl) else "</s>",
        })
    return rows, None


TIMING = ["log_gap_prev", "log_gap_next", "gap_prev_rel", "gap_prev_local_max", "dur_prev_rel",
          "dur_cur_rel", "signs_since_long_gap", "local_rate"]
GROUPS = {
    "gap only": ["gap"],
    "timing": ["timing"],
    "hands": ["hands"],
    "mouth": ["mouth"],
    "lexical": ["lexical"],
    "non-lexical (timing+hands+mouth)": ["timing", "hands", "mouth"],
    "all": ["timing", "hands", "mouth", "lexical"],
}


class Featurizer:
    """Builds a sparse design matrix for a set of feature blocks; vocabulary from TRAIN."""

    def __init__(self, train_rows, min_count=5, n_bins=10):
        self.edges = {}
        for f in TIMING:
            x = np.array([r[f] for r in train_rows])
            self.edges[f] = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
        self.mu = {f: np.mean([r[f] for r in train_rows]) for f in TIMING}
        self.sd = {f: np.std([r[f] for r in train_rows]) + 1e-8 for f in TIMING}
        cnt = Counter()
        for r in train_rows:
            for k in ("gloss_prev2", "gloss_prev", "gloss_cur", "gloss_next"):
                cnt[(k, gloss_base(r[k]))] += 1
        self.vocab = {key: i for i, key in enumerate(sorted(k for k, c in cnt.items() if c >= min_count))}

    def block_names(self, block):
        if block == "gap":
            return ["log_gap_prev"] + [f"log_gap_prev_bin{j}" for j in range(len(self.edges["log_gap_prev"]) + 1)]
        if block == "timing":
            out = list(TIMING)
            for f in TIMING:
                out += [f"{f}_bin{j}" for j in range(len(self.edges[f]) + 1)]
            return out
        if block == "hands":
            return [f"hand_prev={h}" for h in HANDS] + [f"hand_cur={h}" for h in HANDS] + ["hand_switch"]
        if block == "mouth":
            return [f"mouth_prev={m}" for m in MOUTH] + [f"mouth_cur={m}" for m in MOUTH] + ["mouth_spread"]
        if block == "lexical":
            cats = [f"{k}_cat={c}" for k in ("gloss_prev2", "gloss_prev", "gloss_cur", "gloss_next")
                    for c in GLOSS_CATEGORIES + ["lexical"]]
            return cats + [f"{k}={g}" for (k, g) in sorted(self.vocab, key=self.vocab.get)]
        raise KeyError(block)

    def block(self, rows, block):
        names = self.block_names(block)
        idx = {n: j for j, n in enumerate(names)}
        data, ri, ci = [], [], []

        def put(i, name, v=1.0):
            j = idx.get(name)
            if j is not None:
                ri.append(i)
                ci.append(j)
                data.append(v)

        for i, r in enumerate(rows):
            if block in ("gap", "timing"):
                fs = ["log_gap_prev"] if block == "gap" else TIMING
                for f in fs:
                    put(i, f, (r[f] - self.mu[f]) / self.sd[f])
                    put(i, f"{f}_bin{int(np.searchsorted(self.edges[f], r[f], side='right'))}")
            elif block == "hands":
                put(i, f"hand_prev={r['hand_prev']}")
                put(i, f"hand_cur={r['hand_cur']}")
                put(i, "hand_switch", float(r["hand_switch"]))
            elif block == "mouth":
                put(i, f"mouth_prev={r['mouth_prev']}")
                put(i, f"mouth_cur={r['mouth_cur']}")
                put(i, "mouth_spread", float(r["mouth_spread"]))
            elif block == "lexical":
                for k in ("gloss_prev2", "gloss_prev", "gloss_cur", "gloss_next"):
                    g = r[k]
                    if g not in ("<s>", "</s>"):
                        put(i, f"{k}_cat={gloss_category(g)}")
                    j = self.vocab.get((k, gloss_base(g)))
                    if j is not None:
                        put(i, f"{k}={gloss_base(g)}")
        return sparse.csr_matrix((data, (ri, ci)), shape=(len(rows), len(names))), names

    def matrix(self, rows, blocks):
        mats, names = [], []
        for b in blocks:
            m, n = self.block(rows, b)
            mats.append(m)
            names += [f"{b}:{x}" for x in n]
        return sparse.hstack(mats).tocsr(), names


# ============================================================================ evaluation
def best_f1_threshold(y, s):
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    j = int(np.argmax(f1))
    return float(s[order][j]), float(f1[j])


def scores_at(y, s, thr, types=None):
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(y.sum()), 1)
    out = {"auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
           "ap": float(average_precision_score(y, s)) if y.sum() else np.nan,
           "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}
    if types is not None:
        for t in ("continuous", "short_pause", "long_pause"):
            m = (types == t) & (y == 1)
            out[f"recall_{t}"] = float(np.mean(pred[m])) if m.any() else np.nan
    return out


def fit_group(F, rows, blocks, y, Cs=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0), use_gbm=True):
    Xtr, names = F.matrix(rows["train"], blocks)
    Xva, _ = F.matrix(rows["val"], blocks)
    Xte, _ = F.matrix(rows["test"], blocks)
    best = None
    for C in Cs:
        lr = LogisticRegression(C=C, max_iter=3000)
        lr.fit(Xtr, y["train"])
        ap = average_precision_score(y["val"], lr.decision_function(Xva))
        if best is None or ap > best[0]:
            best = (ap, "logreg", C, lr)
    if use_gbm and "lexical" not in blocks:
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                             random_state=0)
        gbm.fit(Xtr.toarray(), y["train"])
        ap = average_precision_score(y["val"], gbm.predict_proba(Xva.toarray())[:, 1])
        if ap > best[0]:
            best = (ap, "gbm", None, gbm)
    _, kind, C, model = best

    def score(X):
        if kind == "logreg":
            return model.decision_function(X)
        return logit(model.predict_proba(X.toarray())[:, 1])
    s = {"train": score(Xtr), "val": score(Xva), "test": score(Xte)}
    return s, kind, C, model, names


# ============================================================================ model scores
def model_scores(pred_dir, runs, split, rows, k):
    """Mean over runs of max P(B) in [frame-k, frame+k] at each transition."""
    acc = np.zeros(len(rows))
    used = 0
    by_vid = defaultdict(list)
    for i, r in enumerate(rows):
        by_vid[r["video"]].append(i)
    for run in runs:
        path = os.path.join(pred_dir, run, f"{split}.npz")
        if not os.path.exists(path):
            print(f"  [model] missing {path} -- run evaluate_phrase.py --split {split} --save-predictions")
            continue
        z = np.load(path)
        s = np.full(len(rows), np.nan)
        for vid, idx in by_vid.items():
            if vid not in z:
                continue
            pb = z[vid][B_TAG].astype(np.float32)
            T = len(pb)
            for i in idx:
                f = rows[i]["frame"]
                s[i] = pb[max(0, f - k):min(T, f + k + 1)].max()
        if np.isnan(s).all():
            continue
        acc += np.nan_to_num(s, nan=0.0)
        used += 1
    return acc / used if used else None


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Oracle decomposition of phrase boundaries on gold annotation.")
    ap.add_argument("--annotations", default=os.path.join("raw_data", "annotations"))
    ap.add_argument("--gloss-labels", default=os.path.join("processed_data", "BIO_tags"))
    ap.add_argument("--phrase-labels", default=os.path.join("processed_data", "BIO_tags_phrase_v2"))
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--short-gap", type=int, default=12, help="max gap (frames) of a SHORT pause (as evaluate_phrase.py)")
    ap.add_argument("--min-count", type=int, default=5, help="min train count of a gloss to get its own feature")
    ap.add_argument("--predictions", default=None, help="evaluation_phrase/predictions (from --save-predictions)")
    ap.add_argument("--runs", nargs="*", default=[], help="runs to score (averaged = one 'model'); "
                                                          "use --run-groups for several models")
    ap.add_argument("--run-groups", nargs="*", default=[],
                    help="several models as NAME=run1,run2,... (e.g. MS=multistream_bilstm-124,...)")
    ap.add_argument("--peak-window", type=int, default=5, help="+-frames for the model score (default 5)")
    ap.add_argument("--max-videos", type=int, default=None, help="per split, for a quick test")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "oracle"))
    args = ap.parse_args()

    with open(args.split_file) as f:
        splits = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    rows, skips, eaf_cache = {}, Counter(), {}
    for split in ("train", "val", "test"):
        vids = sorted(re.sub(r"\.(npy|pt)$", "", v) for v in splits[split])
        if args.max_videos:
            vids = vids[:args.max_videos]
        rows[split] = []
        for vid in tqdm(sorted(vids), desc=f"transitions {split}"):
            r, why = video_transitions(vid, args, eaf_cache)
            if r is None:
                skips[(split, why)] += 1
                continue
            rows[split] += r
    for split in rows:
        y = np.array([r["y"] for r in rows[split]])
        print(f"{split}: {len(rows[split])} transitions, {int(y.sum())} phrase starts "
              f"({y.mean():.1%}); skipped videos: "
              f"{ {w: c for (s, w), c in skips.items() if s == split} }")
    y = {s: np.array([r["y"] for r in rows[s]]) for s in rows}
    types = {s: np.array([r["boundary_gap_type"] for r in rows[s]]) for s in rows}

    F = Featurizer(rows["train"], args.min_count)
    results, oracle_scores = [], {}
    for name, blocks in GROUPS.items():
        s, kind, C, model, names = fit_group(F, rows, blocks, y)
        thr, _ = best_f1_threshold(y["val"], s["val"])
        res = scores_at(y["test"], s["test"], thr, types["test"])
        res.update({"group": name, "classifier": kind if kind == "gbm" else f"logreg C={C}",
                    "n_features": len(names)})
        results.append(res)
        oracle_scores[name] = s
        print(f"  {name:<34s} AUC {res['auc']:.3f}  AP {res['ap']:.3f}  F1 {res['f1']:.3f}  ({res['classifier']})")
        if name == "lexical" and kind == "logreg":
            coef = model.coef_[0]
            order = np.argsort(-coef)
            with open(os.path.join(args.out_dir, "oracle_lexical_weights.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["feature", "weight", "direction"])
                for j in list(order[:60]) + list(order[-60:]):
                    w.writerow([names[j], round(float(coef[j]), 4),
                                "phrase boundary" if coef[j] > 0 else "within phrase"])

    # ---- trained models on the same transitions
    groups = []
    if args.runs:
        groups.append(("model", args.runs))
    for g in args.run_groups:
        n, runs = g.split("=", 1)
        groups.append((n, [x for x in runs.split(",") if x]))
    if args.predictions and groups:
        for gname, runs in groups:
            sv = model_scores(args.predictions, runs, "val", rows["val"], args.peak_window)
            st = model_scores(args.predictions, runs, "test", rows["test"], args.peak_window)
            if sv is None or st is None:
                continue
            thr, _ = best_f1_threshold(y["val"], sv)
            res = scores_at(y["test"], st, thr, types["test"])
            res.update({"group": f"{gname} ({len(runs)} runs, max P(B) +-{args.peak_window})",
                        "classifier": "trained segmentation model", "n_features": 0})
            results.append(res)
            print(f"  {res['group']:<34s} AUC {res['auc']:.3f}  AP {res['ap']:.3f}  F1 {res['f1']:.3f}")
            for oname in ("lexical", "mouth", "all"):
                Xv = np.c_[logit(sv), oracle_scores[oname]["val"]]
                Xt = np.c_[logit(st), oracle_scores[oname]["test"]]
                stack = LogisticRegression(max_iter=1000).fit(Xv, y["val"])
                sv2, st2 = stack.decision_function(Xv), stack.decision_function(Xt)
                thr2, _ = best_f1_threshold(y["val"], sv2)
                res2 = scores_at(y["test"], st2, thr2, types["test"])
                res2.update({"group": f"{gname} + {oname} oracle", "classifier": "stack (fit on val)",
                             "n_features": 2})
                results.append(res2)
                print(f"  {res2['group']:<34s} AUC {res2['auc']:.3f}  AP {res2['ap']:.3f}  F1 {res2['f1']:.3f}")
            for r, a in zip(rows["test"], st):
                r[f"model_score:{gname}"] = round(float(a), 5)
            for r, a in zip(rows["val"], sv):
                r[f"model_score:{gname}"] = round(float(a), 5)

    base = {s: float(y[s].mean()) for s in y}
    cols = ["group", "classifier", "n_features", "auc", "ap", "f1", "precision", "recall",
            "recall_continuous", "recall_short_pause", "recall_long_pause"]
    with open(os.path.join(args.out_dir, "oracle_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(args.out_dir, "oracle_summary.txt"), "w") as f:
        f.write(f"TEST transitions: {len(y['test'])}, phrase starts {int(y['test'].sum())} "
                f"(base rate {base['test']:.3f} = AP of a random scorer)\n")
        f.write("Conditional on gold sign segmentation. Thresholds for F1 tuned on val.\n\n")
        f.write(f"{'group':<44s}{'AUC':>7s}{'AP':>7s}{'F1':>7s}{'P':>7s}{'R':>7s}"
                f"{'R cont':>8s}{'R short':>8s}{'R long':>8s}\n")
        for r in results:
            f.write(f"{r['group'][:44]:<44s}{r['auc']:7.3f}{r['ap']:7.3f}{r['f1']:7.3f}{r['precision']:7.3f}"
                    f"{r['recall']:7.3f}{r.get('recall_continuous', np.nan):8.3f}"
                    f"{r.get('recall_short_pause', np.nan):8.3f}{r.get('recall_long_pause', np.nan):8.3f}\n")
        f.write("\nRecall by the pause before the boundary: continuous = no gap, short <= "
                f"{args.short_gap} frames, long > {args.short_gap}.\n")
    print(open(os.path.join(args.out_dir, "oracle_summary.txt")).read())

    for split in ("val", "test"):
        for name, s in oracle_scores.items():
            for r, v in zip(rows[split], s[split]):
                r[f"oracle:{name}"] = round(float(v), 4)
        keys = list(rows[split][0].keys()) if rows[split] else []
        for r in rows[split]:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, f"transitions_{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows[split])
    print(f"Saved to {args.out_dir}/ (oracle_summary, oracle_lexical_weights, transitions_val/test)")


if __name__ == "__main__":
    main()