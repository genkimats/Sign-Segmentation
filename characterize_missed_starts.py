"""
characterize_missed_starts.py -- what is different about the phrase starts our model never finds?

Per gold phrase start of the TEST split:
  detection   found by how many seeds of the proposed model (default multistream_bilstm-115..117,
              Body + HaMeR + Face), each decoded by peak picking tuned on val (count_oracle_decoding.py,
              the decoder of the results slides); found = a predicted start within +-5 frames, one-to-one
  groups      MISSED = found by no seed, FOUND = found by every seed (the rest is reported, not compared)
  features    ~60 characteristics in 8 families (below), from files that already exist:
                - gold phrase labels (meta labels_dir) and gold sign labels (processed_data/BIO_tags)
                - evaluation_phrase/oracle/transitions_test.csv  (phrase_oracle_analysis.py: sign timing,
                  hands, mouthing, glosses)
                - processed_data/cue_features/<video>.npy        (extract_cue_features.py: per-video robust-z
                  movement / face / head cues and detection flags)
  comparison  per feature: AUC = P(value at a MISSED start > value at a FOUND start) (0.5 = no difference;
              > 0.5 = larger at missed starts), 95% CI from a bootstrap over test videos (starts of one
              video stay together), p from the bootstrap standard error, Holm over all features; plus the same AUC WITHIN PAUSE STRATA
              (pause before the start: 0 / 1-12 / 13-25 / 26-50 / > 50 frames) -- does the feature still
              differ once pause length is held fixed?
  robustness  with --runs-169 (gold_boundaries_test.csv of boundary_error_analysis.py) the same AUCs for
              the starts missed (<= 10 %) vs found (>= 90 %) by the 169-run pool, as an extra column

FEATURE FAMILIES (s = phrase start frame, 50 fps; "before" = [s-25, s), "after" = [s, s+25),
"around" = [s-12, s+12])
  timing       pause before (s), pause / signer's median sign gap, pause / largest nearby gap,
               pause after the first sign (s), time since the previous phrase start (s)
  tempo        previous / new phrase duration (s), signs in previous / new phrase, signing rate
               (signs/s within +-2 s), rate relative to the video median, signs since the last long pause
  lengthening  last sign before / first sign after the start, duration relative to the signer's median
  hands        hand switch, previous / first sign two-handed (gloss tiers); wrist speed before / after /
               change, minimum wrist speed around (hold), hold share before, wrist height before and
               change (hands dropping), hand distance change, handshape change and HaMeR articulation
               speed around, whole-body motion around
  head/body    head speed around, head pitch change (nod), head vertical velocity min (moving down),
               head rotation speed around, shoulder width change (lean / turn)
  face         blink / eyes narrowed within +-5 frames, inner brow height change (abs), brow velocity
               around (max abs), mouth motion before / after, face deformation around, mouthing spreads
               over the transition, mouthing on previous / first sign
  lexical      previous sign / first sign: palm-up ($GEST-OFF), pointing ($INDEX), depicting ($PROD),
               fingerspelling or number ($ALPHA/$NUM), other gesture or non-manual ($GEST/$ORAL/...),
               first sign is a connective (ABER, WENN, DANN, UND, ...)
  technical    distance to the nearest 512-frame inference-window edge (frames), position in the video,
               face / MediaPipe dominant hand / HaMeR dominant hand detected (share of frames around)
  signer       the video's miss rate computed WITHOUT this start (leave-one-out): are misses a signer effect?

Outputs (evaluation_phrase/missed_starts/):
  per_start.csv      every start: detection, group, all features
  features.csv       per feature: n, median / share per group, AUC (95% CI), p, Holm p, stratified AUC
  summary.txt        ranked table + per-video miss rates
  ranked_auc.png     top features (AUC with CI, stratified AUC as an open marker)
  top_features.png   distributions of the most different features, missed vs found

    python characterize_missed_starts.py
    python characterize_missed_starts.py --runs multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117
    python characterize_missed_starts.py --runs-169 evaluation_phrase/error_analysis/gold_boundaries_test.csv
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

from src.metrics import extract_segments, match_boundaries

FPS = 50
TOL = 5
BEFORE, AROUND = 25, 12
WINDOW = 512
PAUSE_BINS = [(-10 ** 9, 0), (1, 12), (13, 25), (26, 50), (51, 10 ** 9)]
DEFAULT_RUNS = [f"multistream_bilstm-{i}" for i in (115, 116, 117)]
CONNECTIVES = ["ABER", "WENN", "DANN", "UND", "DARUM", "UND-DANN", "SPÄTER", "GRUND", "PLÖTZLICH",
               "DESHALB", "JETZT", "BEISPIEL", "JA", "FRÜHER", "HEUTE", "MEISTENS", "ACHTUNG"]

# feature name -> (family, label for tables / figures)
FEATURES = {}


def feat(name, family, label):
    FEATURES[name] = (family, label)


for _n, _f, _l in [
    ("pause_s", "timing", "pause before the start (s)"),
    ("pause_rel", "timing", "pause / signer's median sign gap"),
    ("pause_local_max", "timing", "pause / largest gap nearby"),
    ("pause_after_first_s", "timing", "pause after the first sign (s)"),
    ("since_prev_start_s", "timing", "time since previous phrase start (s)"),
    ("prev_phrase_s", "tempo", "previous phrase duration (s)"),
    ("cur_phrase_s", "tempo", "new phrase duration (s)"),
    ("prev_phrase_signs", "tempo", "signs in previous phrase"),
    ("cur_phrase_signs", "tempo", "signs in new phrase"),
    ("rate", "tempo", "signing rate around (signs/s)"),
    ("rate_rel", "tempo", "signing rate / video median"),
    ("signs_since_long_pause", "tempo", "signs since last long pause"),
    ("dur_prev_rel", "lengthening", "last sign before: duration / median"),
    ("dur_cur_rel", "lengthening", "first sign after: duration / median"),
    ("hand_switch", "hands", "hand switch at the transition"),
    ("prev_two_handed", "hands", "last sign before is two-handed"),
    ("cur_two_handed", "hands", "first sign is two-handed"),
    ("wrist_speed_before", "hands", "dom wrist speed before"),
    ("wrist_speed_after", "hands", "dom wrist speed after"),
    ("wrist_speed_change", "hands", "dom wrist speed change (after - before)"),
    ("wrist_speed_min", "hands", "min dom wrist speed around (hold)"),
    ("nondom_speed_before", "hands", "nondom wrist speed before"),
    ("hold_share_before", "hands", "dom hold share before"),
    ("wrist_height_before", "hands", "dom wrist height before"),
    ("wrist_height_min", "hands", "min dom wrist height around (hands dropped)"),
    ("wrist_height_change", "hands", "dom wrist height change (after - before)"),
    ("hand_distance_change", "hands", "hand distance change (abs)"),
    ("handshape_change", "hands", "dom handshape change around"),
    ("hamer_artic", "hands", "HaMeR dom articulation speed around"),
    ("body_motion_min", "hands", "min whole-body motion around"),
    ("head_speed", "head/body", "max head speed around"),
    ("head_pitch_change", "head/body", "head pitch change (abs)"),
    ("head_down_velocity", "head/body", "min head vertical velocity around (down)"),
    ("head_rotation", "head/body", "max head rotation speed around"),
    ("shoulder_change", "head/body", "shoulder width change (abs)"),
    ("blink", "face", "blink within +-5 frames"),
    ("eyes_narrowed", "face", "eyes narrowed within +-5 frames"),
    ("inner_brow_change", "face", "inner brow height change (abs)"),
    ("brow_velocity", "face", "max |brow velocity| around"),
    ("mouth_motion_before", "face", "mouth motion before"),
    ("mouth_motion_after", "face", "mouth motion after"),
    ("face_deformation", "face", "face deformation around"),
    ("mouth_spread", "face", "one mouthing spreads over the transition"),
    ("mouth_prev_word", "face", "mouthing on the last sign before"),
    ("mouth_cur_word", "face", "mouthing on the first sign"),
    ("prev_palm_up", "lexical", "last sign before is palm-up ($GEST-OFF)"),
    ("cur_palm_up", "lexical", "first sign is palm-up ($GEST-OFF)"),
    ("prev_pointing", "lexical", "last sign before is pointing ($INDEX)"),
    ("cur_pointing", "lexical", "first sign is pointing ($INDEX)"),
    ("cur_depicting", "lexical", "first sign is depicting ($PROD)"),
    ("cur_alpha_num", "lexical", "first sign is fingerspelling / number"),
    ("cur_other_gesture", "lexical", "first sign is another gesture / non-manual"),
    ("prev_other_gesture", "lexical", "last sign before is another gesture / non-manual"),
    ("cur_connective", "lexical", "first sign is a connective (ABER, WENN, ...)"),
    ("window_edge_dist", "technical", "distance to an inference-window edge (frames)"),
    ("video_position", "technical", "position in the video (0-1)"),
    ("face_detected", "technical", "face detected around (share)"),
    ("dom_hand_detected", "technical", "MediaPipe dom hand detected around (share)"),
    ("hamer_detected", "technical", "HaMeR dom hand detected around (share)"),
    ("video_miss_rate_loo", "signer", "video's miss rate without this start"),
]:
    feat(_n, _f, _l)

FAMILY_COLORS = {"timing": "#1F5FA8", "tempo": "#4C8BD6", "lengthening": "#7FA9DF", "hands": "#C0392B",
                 "head/body": "#8C564B", "face": "#2A9D8F", "lexical": "#D9822B", "technical": "#7F7F7F",
                 "signer": "#6A4C93"}


# ============================================================================ helpers
def gloss_category(g):
    g = (g or "").strip().upper()
    for c in ["$GEST-OFF", "$GEST", "$INDEX", "$PROD", "$ALPHA", "$NUM", "$ORAL", "$LIST", "$PMS", "$"]:
        if g.startswith(c):
            return c
    return "lexical"


def gloss_base(g):
    g = (g or "").strip().upper()
    g = re.sub(r"[\^\*]+", "", g)
    g = re.sub(r"\d+[A-Z]?$", "", g)
    return g or "?"


def sign_segments(gloss_labels):
    """Same rule as phrase_oracle_analysis.sign_segments (B starts a sign, O ends it)."""
    segs, start = [], -1
    for i, tag in enumerate(gloss_labels):
        if tag == 2:
            if start != -1:
                segs.append((start, i - 1))
            start = i
        elif tag == 0:
            if start != -1:
                segs.append((start, i - 1))
                start = -1
        elif start == -1:
            start = i
    if start != -1:
        segs.append((start, len(gloss_labels) - 1))
    return np.array(segs, dtype=np.int64).reshape(-1, 2)


def window_edges(T, w=WINDOW):
    """Frame indices where an inference window starts (overlap 0, last window end-aligned)."""
    if T <= w:
        return np.array([0, T])
    starts = list(range(0, T - w + 1, w))
    if starts[-1] + w < T:
        starts.append(T - w)
    return np.array(sorted(set(starts + [s + w for s in starts])))


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), NaN-free inputs."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_auc(pos, neg, spos, sneg, min_n=5):
    """Pause-stratified AUC: within-stratum AUCs weighted by n1*n0 (van Elteren style); strata with
    fewer than min_n missed or found starts are skipped (too small to compare within)."""
    num = den = 0.0
    for k in np.unique(np.r_[spos, sneg]):
        a, b = pos[spos == k], neg[sneg == k]
        if len(a) >= min_n and len(b) >= min_n:
            w = len(a) * len(b)
            num += w * auc(a, b)
            den += w
    return num / den if den else np.nan


def pause_stratum(p_frames):
    for k, (lo, hi) in enumerate(PAUSE_BINS):
        if lo <= p_frames <= hi:
            return k
    return len(PAUSE_BINS) - 1


# ============================================================================ detection
def detection_by_model(runs, pred_root, decoder):
    from phrase_stats import gold_starts_of, load_predictions
    det, gold, labels_dir = {}, None, None
    for run in runs:
        probs, meta, g = load_predictions(pred_root, run, "test")
        labels_dir = labels_dir or meta.get("labels_dir")
        gold = gold or g
        if decoder == "peak":
            from count_oracle_decoding import peaks_with_heights, tune_peaks_on_val
            try:
                t = tune_peaks_on_val(load_predictions(pred_root, run, "val"))
            except FileNotFoundError:
                raise SystemExit(f"{run}: no val predictions -- evaluate_phrase.py --split val --save-predictions, "
                                 "or use --decoder argmax")
            print(f"  {run}: peak picking tuned on val (min_dist {t['min_dist']}, sigma {t['sigma']}, thr {t['thr']})")
        for v in probs:
            gs = gold_starts_of(g[v])
            if decoder == "peak":
                idx, h = peaks_with_heights(probs[v][2], t["min_dist"], t["sigma"])
                ps = idx[h >= t["thr"]]
            else:
                ps = np.array([s for s, _ in extract_segments(probs[v].argmax(axis=0))], dtype=np.int64)
            ok, _, _ = match_boundaries(gs, ps, TOL)
            for s, o in zip(gs, ok):
                det.setdefault((v, int(s)), []).append(int(o))
    return det, gold, labels_dir


# ============================================================================ features
def cue_window(cues, names, ch, a, b):
    if cues is None or ch not in names:
        return None
    T = cues.shape[0]
    a, b = max(0, a), min(T, b)
    if b <= a:
        return None
    return cues[a:b, names.index(ch)].astype(np.float64)


def start_features(v, s, segs_phrase, k, signs, trans_row, cues, cue_names, T, vid_rate_median):
    f = {}
    # ---------- timing / tempo from phrase labels
    prev_end = segs_phrase[k - 1][1] if k > 0 else None
    f["pause_s"] = (s - prev_end - 1) / FPS if prev_end is not None else np.nan
    f["since_prev_start_s"] = (s - segs_phrase[k - 1][0]) / FPS if k > 0 else np.nan
    f["prev_phrase_s"] = (segs_phrase[k - 1][1] - segs_phrase[k - 1][0] + 1) / FPS if k > 0 else np.nan
    f["cur_phrase_s"] = (segs_phrase[k][1] - segs_phrase[k][0] + 1) / FPS
    if signs is not None and len(signs):
        cs = signs[:, 0]
        f["cur_phrase_signs"] = float(np.sum((cs >= segs_phrase[k][0]) & (cs <= segs_phrase[k][1])))
        f["prev_phrase_signs"] = (float(np.sum((cs >= segs_phrase[k - 1][0]) & (cs <= segs_phrase[k - 1][1])))
                                  if k > 0 else np.nan)
    # ---------- from the transitions file
    t = trans_row or {}

    def num(key, scale=1.0):
        try:
            return float(t[key]) * scale
        except (KeyError, ValueError, TypeError):
            return np.nan
    f["pause_rel"] = num("gap_prev_rel")
    f["pause_local_max"] = num("gap_prev_local_max")
    gn = num("gap_next")
    f["pause_after_first_s"] = gn / FPS if np.isfinite(gn) and gn >= 0 else np.nan
    f["rate"] = num("local_rate")
    f["rate_rel"] = f["rate"] / vid_rate_median if np.isfinite(f["rate"]) and vid_rate_median else np.nan
    f["signs_since_long_pause"] = num("signs_since_long_gap")
    f["dur_prev_rel"] = num("dur_prev_rel")
    f["dur_cur_rel"] = num("dur_cur_rel")
    if t:
        f["hand_switch"] = num("hand_switch")
        f["prev_two_handed"] = float(t.get("hand_prev") == "rl")
        f["cur_two_handed"] = float(t.get("hand_cur") == "rl")
        f["mouth_spread"] = num("mouth_spread")
        f["mouth_prev_word"] = float(t.get("mouth_prev") == "word")
        f["mouth_cur_word"] = float(t.get("mouth_cur") == "word")
        pc, cc = gloss_category(t.get("gloss_prev")), gloss_category(t.get("gloss_cur"))
        other = {"$GEST", "$ORAL", "$LIST", "$PMS", "$"}
        f["prev_palm_up"] = float(pc == "$GEST-OFF")
        f["cur_palm_up"] = float(cc == "$GEST-OFF")
        f["prev_pointing"] = float(pc == "$INDEX")
        f["cur_pointing"] = float(cc == "$INDEX")
        f["cur_depicting"] = float(cc == "$PROD")
        f["cur_alpha_num"] = float(cc in ("$ALPHA", "$NUM"))
        f["cur_other_gesture"] = float(cc in other)
        f["prev_other_gesture"] = float(pc in other)
        f["cur_connective"] = float(cc == "lexical" and gloss_base(t.get("gloss_cur")) in CONNECTIVES)
    # ---------- movement / face / detection from the cue features
    def w(ch, a, b):
        return cue_window(cues, cue_names, ch, s + a, s + b)

    def mean(x):
        return float(np.mean(x)) if x is not None and len(x) else np.nan

    def mx(x, fn=np.max):
        return float(fn(x)) if x is not None and len(x) else np.nan
    wb, wa = w("dom wrist speed", -BEFORE, 0), w("dom wrist speed", 0, BEFORE)
    f["wrist_speed_before"], f["wrist_speed_after"] = mean(wb), mean(wa)
    f["wrist_speed_change"] = f["wrist_speed_after"] - f["wrist_speed_before"]
    f["wrist_speed_min"] = mx(w("dom wrist speed", -AROUND, AROUND), np.min)
    f["nondom_speed_before"] = mean(w("nondom wrist speed", -BEFORE, 0))
    f["hold_share_before"] = mean(w("dom hold", -BEFORE, 0))
    hb, ha = w("dom wrist height", -BEFORE, 0), w("dom wrist height", 0, BEFORE)
    f["wrist_height_before"] = mean(hb)
    f["wrist_height_change"] = mean(ha) - mean(hb)
    f["wrist_height_min"] = mx(w("dom wrist height", -AROUND, AROUND), np.min)
    f["hand_distance_change"] = abs(mean(w("hand distance", 0, BEFORE)) - mean(w("hand distance", -BEFORE, 0)))
    f["handshape_change"] = mean(w("dom handshape change", -AROUND, AROUND))
    f["hamer_artic"] = mean(w("HaMeR dom articulation speed", -AROUND, AROUND))
    f["body_motion_min"] = mx(w("whole-body motion energy", -AROUND, AROUND), np.min)
    f["head_speed"] = mx(w("head speed", -AROUND, AROUND))
    f["head_pitch_change"] = abs(mean(w("head pitch (nod)", 0, BEFORE)) - mean(w("head pitch (nod)", -BEFORE, 0)))
    f["head_down_velocity"] = mx(w("head vertical velocity", -AROUND, AROUND), np.min)
    f["head_rotation"] = mx(w("head rotation speed", -AROUND, AROUND))
    f["shoulder_change"] = abs(mean(w("shoulder width change", 0, BEFORE)) -
                               mean(w("shoulder width change", -BEFORE, 0)))
    f["blink"] = mx(w("blink", -TOL, TOL + 1))
    f["eyes_narrowed"] = mx(w("eyes narrowed (long)", -TOL, TOL + 1))
    f["inner_brow_change"] = abs(mean(w("inner brow height", 0, BEFORE)) - mean(w("inner brow height", -BEFORE, 0)))
    bv = w("brow vertical velocity", -AROUND, AROUND)
    f["brow_velocity"] = mx(np.abs(bv) if bv is not None else None)
    f["mouth_motion_before"] = mean(w("mouth motion energy", -BEFORE, 0))
    f["mouth_motion_after"] = mean(w("mouth motion energy", 0, BEFORE))
    f["face_deformation"] = mean(w("face deformation energy", -AROUND, AROUND))
    f["face_detected"] = mean(w("face detected", -BEFORE, BEFORE))
    f["dom_hand_detected"] = mean(w("MP dom hand detected", -BEFORE, BEFORE))
    f["hamer_detected"] = mean(w("HaMeR dom detected", -BEFORE, BEFORE))
    # ---------- technical
    edges = window_edges(T)
    f["window_edge_dist"] = float(np.min(np.abs(edges - s)))
    f["video_position"] = s / max(T - 1, 1)
    return f


# ============================================================================ statistics
class VideoBoot:
    def __init__(self, vids, n_boot, seed=0):
        rng = np.random.default_rng(seed)
        self.vids = list(vids)
        self.draws = [rng.integers(0, len(self.vids), len(self.vids)) for _ in range(n_boot)]


def feature_stats(rows, name, boot, gkey="group"):
    by_vid_pos, by_vid_neg = defaultdict(list), defaultdict(list)
    for r in rows:
        x = r.get(name, np.nan)
        if x is None or not np.isfinite(x):
            continue
        if r[gkey] == "missed":
            by_vid_pos[r["video"]].append((x, r["pause_stratum"]))
        elif r[gkey] == "found":
            by_vid_neg[r["video"]].append((x, r["pause_stratum"]))

    def arrays(vid_idx):
        P = [t for i in vid_idx for t in by_vid_pos.get(boot.vids[i], [])]
        N = [t for i in vid_idx for t in by_vid_neg.get(boot.vids[i], [])]
        P = np.array(P, dtype=float).reshape(-1, 2)
        N = np.array(N, dtype=float).reshape(-1, 2)
        return P, N
    P, N = arrays(range(len(boot.vids)))
    if len(P) < 5 or len(N) < 5:
        return None
    a = auc(P[:, 0], N[:, 0])
    sa = strat_auc(P[:, 0], N[:, 0], P[:, 1], N[:, 1])
    draws, sdraws = [], []
    for d in boot.draws:
        p, n = arrays(d)
        if len(p) and len(n):
            draws.append(auc(p[:, 0], n[:, 0]))
            sdraws.append(strat_auc(p[:, 0], n[:, 0], p[:, 1], n[:, 1]))
    draws, sdraws = np.array(draws), np.array([x for x in sdraws if np.isfinite(x)])
    if len(draws) == 0:
        draws = np.array([np.nan])
    binary = set(np.unique(np.r_[P[:, 0], N[:, 0]])) <= {0.0, 1.0}
    out = {"feature": name, "family": FEATURES[name][0], "label": FEATURES[name][1], "binary": int(binary),
           "n_missed": len(P), "n_found": len(N),
           "missed": float(P[:, 0].mean() if binary else np.median(P[:, 0])),
           "found": float(N[:, 0].mean() if binary else np.median(N[:, 0])),
           "auc": a, "auc_lo": float(np.nanquantile(draws, 0.025)) if np.isfinite(draws).any() else np.nan,
           "auc_hi": float(np.nanquantile(draws, 0.975)) if np.isfinite(draws).any() else np.nan,
           "p": boot_p(draws, a),
           "auc_strat": sa,
           "auc_strat_lo": float(np.quantile(sdraws, 0.025)) if len(sdraws) else np.nan,
           "auc_strat_hi": float(np.quantile(sdraws, 0.975)) if len(sdraws) else np.nan}
    out["p_strat"] = boot_p(sdraws, sa) if len(sdraws) else np.nan
    return out


def boot_p(draws, est=None):
    """Two-sided p for AUC = 0.5 from the bootstrap standard error (normal approximation), so the
    p-value is not limited by the number of draws (needed for Holm over ~60 features)."""
    from scipy.stats import norm
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    se = d.std(ddof=1)
    centre = np.mean(d) if est is None else est
    if se <= 0:
        return 0.0 if centre != 0.5 else 1.0
    return float(2 * norm.sf(abs(centre - 0.5) / se))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    out = np.empty_like(ps)
    m, running = len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--decoder", choices=("peak", "argmax"), default="peak")
    ap.add_argument("--pred-root", default=os.path.join("evaluation_phrase", "predictions"))
    ap.add_argument("--transitions", default=os.path.join("evaluation_phrase", "oracle", "transitions_test.csv"))
    ap.add_argument("--gloss-labels", default=os.path.join("processed_data", "BIO_tags"))
    ap.add_argument("--cue-dir", default=os.path.join("processed_data", "cue_features"))
    ap.add_argument("--runs-169", default=None,
                    help="gold_boundaries_test.csv of boundary_error_analysis.py (169-run detection rates)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "missed_starts"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("decoding ...")
    det, gold, labels_dir = detection_by_model(args.runs, args.pred_root, args.decoder)
    trans = {}
    if os.path.exists(args.transitions):
        with open(args.transitions) as f:
            for row in csv.DictReader(f):
                trans[(row["video"], int(row["frame"]))] = row
    else:
        print(f"  (no transitions file at {args.transitions}: run phrase_oracle_analysis.py; "
              "sign-timing, hand, mouthing and gloss features will be empty)")
    r169 = {}
    if args.runs_169 and os.path.exists(args.runs_169):
        with open(args.runs_169) as f:
            for row in csv.DictReader(f):
                r169[(row["video"], int(row["frame"]))] = float(row["detection_rate"])
    cue_names = None
    names_path = os.path.join(args.cue_dir, "_names.json")
    if os.path.exists(names_path):
        with open(names_path) as f:
            cue_names = json.load(f)
    else:
        print(f"  (no cue features at {args.cue_dir}: movement / face / detection features will be empty)")

    rows = []
    n_seeds = len(args.runs)
    vids = sorted({v for v, _ in det})
    for v in vids:
        g = gold[v]
        T = len(g)
        segs = extract_segments(g)
        gp = os.path.join(args.gloss_labels, f"{v}.npy")
        signs = sign_segments(np.load(gp)) if os.path.exists(gp) else None
        cues = None
        if cue_names is not None and os.path.exists(os.path.join(args.cue_dir, f"{v}.npy")):
            cues = np.load(os.path.join(args.cue_dir, f"{v}.npy"))
            if cues.shape[0] != T:
                cues = cues[:T] if cues.shape[0] > T else None
        vr = [float(r["local_rate"]) for (vv, _), r in trans.items() if vv == v and r.get("local_rate")]
        vid_rate_median = float(np.median(vr)) if vr else None
        for k, (s, _) in enumerate(segs):
            d = det.get((v, int(s)))
            if d is None:
                continue
            nfound = int(sum(d))
            row = {"video": v, "frame": int(s), "found_by_seeds": nfound,
                   "group": "missed" if nfound == 0 else ("found" if nfound == n_seeds else "partly")}
            if r169:
                rt = r169.get((v, int(s)))
                row["rate_169"] = rt
                row["group_169"] = ("missed" if rt is not None and rt <= 0.1 else
                                    "found" if rt is not None and rt >= 0.9 else "partly")
            row.update(start_features(v, int(s), segs, k, signs, trans.get((v, int(s))), cues, cue_names, T,
                                      vid_rate_median))
            pf = row["pause_s"] * FPS if np.isfinite(row.get("pause_s", np.nan)) else 10 ** 6
            row["pause_stratum"] = pause_stratum(pf)
            rows.append(row)
    # leave-one-out video miss rate
    by_v = defaultdict(list)
    for r in rows:
        by_v[r["video"]].append(r)
    for v, rs in by_v.items():
        m = np.array([r["group"] == "missed" for r in rs], dtype=float)
        for i, r in enumerate(rs):
            r["video_miss_rate_loo"] = float((m.sum() - m[i]) / max(len(m) - 1, 1))

    n = len(rows)
    cnt = {gname: sum(r["group"] == gname for r in rows) for gname in ("missed", "partly", "found")}
    print(f"{n} test phrase starts: missed by all {n_seeds} seeds {cnt['missed']} ({cnt['missed'] / n:.1%}), "
          f"found by all {cnt['found']} ({cnt['found'] / n:.1%}), partly {cnt['partly']}")

    boot = VideoBoot(vids, args.n_boot)
    stats = []
    for name in FEATURES:
        st = feature_stats(rows, name, boot)
        if st is None:
            continue
        if r169:
            s2 = feature_stats(rows, name, VideoBoot(vids, 0), gkey="group_169")
            st["auc_169"] = s2["auc"] if s2 else np.nan
        stats.append(st)
    hp = holm([s["p"] for s in stats])
    hps = holm([s["p_strat"] if np.isfinite(s["p_strat"]) else 1.0 for s in stats])
    for s, h, h2 in zip(stats, hp, hps):
        s["p_holm"], s["p_strat_holm"] = float(h), float(h2)
    stats.sort(key=lambda s: -abs(s["auc"] - 0.5))

    # ---------------------------------------------------------------- outputs
    def write(path, rs):
        keys = []
        for r in rs:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
    write(os.path.join(args.out_dir, "per_start.csv"), rows)
    write(os.path.join(args.out_dir, "features.csv"), stats)

    L = [f"Phrase starts missed vs found by the proposed model ({', '.join(args.runs)}; decoder: "
         f"{'peak picking tuned on val' if args.decoder == 'peak' else 'argmax'}; +-{TOL} frames)",
         f"{n} test starts: MISSED by all seeds {cnt['missed']} ({cnt['missed'] / n:.1%}), "
         f"FOUND by all seeds {cnt['found']} ({cnt['found'] / n:.1%}), partly {cnt['partly']}",
         "AUC = P(value at a missed start > value at a found start); 0.5 = no difference. 95% CI: percentile "
         f"bootstrap over the {len(vids)} test videos; p: bootstrap SE (normal approx.); Holm over {len(stats)} features.",
         "strat = the same AUC within pause strata (0 / 1-12 / 13-25 / 26-50 / >50 frames before the start)."
         + ("  169 = AUC for the 169-run pool (<= 10 % vs >= 90 %)." if r169 else ""), "",
         f"{'feature':<48s}{'family':<12s}{'missed':>9s}{'found':>9s}{'AUC (95% CI)':>22s}{'Holm p':>8s}"
         f"{'strat AUC (95% CI)':>24s}{'Holm p':>8s}" + ("{:>7s}".format("169") if r169 else "")]
    for s in stats:
        fm = "{:.2f}" if s["binary"] else "{:.3g}"
        L.append(f"{s['label'][:47]:<48s}{s['family']:<12s}{fm.format(s['missed']):>9s}{fm.format(s['found']):>9s}"
                 f"{s['auc']:>8.2f} ({s['auc_lo']:.2f} to {s['auc_hi']:.2f}){s['p_holm']:>8.3f}"
                 f"{s['auc_strat']:>10.2f} ({s['auc_strat_lo']:.2f} to {s['auc_strat_hi']:.2f}){s['p_strat_holm']:>8.3f}"
                 + (f"{s.get('auc_169', np.nan):>7.2f}" if r169 else ""))
    L += ["", "(binary features: missed / found columns are shares; others are medians)", "",
          "Per-video share of starts missed by all seeds:"]
    for v, rs in sorted(by_v.items(), key=lambda kv: -np.mean([r['group'] == 'missed' for r in kv[1]])):
        L.append(f"  {v:<14s} {np.mean([r['group'] == 'missed' for r in rs]):.2f}  (n = {len(rs)})")
    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    try:
        make_figures(stats, rows, args.out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"(figures skipped: {e})")
    print(f"\nwrote {args.out_dir}/ (per_start.csv, features.csv, summary.txt, ranked_auc.png, top_features.png)")


def make_figures(stats, rows, out_dir, top=18):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    S = stats[:top][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(S) + 1.2), dpi=180)
    for i, s in enumerate(S):
        c = FAMILY_COLORS.get(s["family"], "#555555")
        ax.hlines(i, s["auc_lo"], s["auc_hi"], color=c, lw=3, alpha=0.45)
        ax.plot(s["auc"], i, "o", color=c, ms=7, zorder=3)
        if np.isfinite(s["auc_strat"]):
            ax.plot(s["auc_strat"], i, "o", mfc="none", mec=c, ms=7, zorder=3)
    ax.axvline(0.5, color="#888888", ls="--", lw=1)
    ax.set_yticks(range(len(S)))
    ax.set_yticklabels([s["label"] for s in S], fontsize=9)
    ax.set_xlabel("AUC: missed vs found starts (filled = raw, open = within pause strata; 0.5 = no difference)")
    fams = []
    for s in S:
        if s["family"] not in fams:
            fams.append(s["family"])
    for fam in fams:
        ax.plot([], [], "o", color=FAMILY_COLORS.get(fam, "#555555"), label=fam)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ranked_auc.png"), bbox_inches="tight")
    plt.close(fig)

    cont = [s for s in stats if not s["binary"]][:6]
    if cont:
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), dpi=180)
        for ax, s in zip(axes.ravel(), cont):
            m = [r[s["feature"]] for r in rows if r["group"] == "missed" and np.isfinite(r.get(s["feature"], np.nan))]
            fnd = [r[s["feature"]] for r in rows if r["group"] == "found" and np.isfinite(r.get(s["feature"], np.nan))]
            lo, hi = np.nanpercentile(m + fnd, [1, 99])
            bins = np.linspace(lo, hi, 30)
            ax.hist(fnd, bins=bins, color="#9AA5B1", alpha=0.7, density=True, label="found by all seeds")
            ax.hist(m, bins=bins, color="#C0392B", alpha=0.55, density=True, label="missed by all seeds")
            ax.set_title(f"{s['label']}\nAUC {s['auc']:.2f}", fontsize=9)
            ax.set_yticks([])
        axes.ravel()[0].legend(frameon=False, fontsize=8)
        for ax in axes.ravel()[len(cont):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "top_features.png"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())