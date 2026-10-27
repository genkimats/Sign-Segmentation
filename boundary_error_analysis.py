"""
boundary_error_analysis.py -- are the errors in the models or in the data?

Uses the saved per-frame probabilities of MANY trained runs (evaluate_phrase.py
--save-predictions) and asks, per gold phrase start and per false alarm:

  A. Error consistency   For every gold phrase start: the share of runs that detect it
                         (a predicted start within +-k frames, one-to-one per run).
                         - "missed by every run" vs what independent errors would give
                           (binomial expectation at the same mean recall)
                         - boundary ICC: share of the variance in "detected?" that belongs to
                           the BOUNDARY rather than to the run (1 = all models fail on the same
                           boundaries, 0 = errors are model-specific)
                         - union recall (detected by >= 1 run) = what better model selection /
                           ensembling could reach at most
  B. Hard boundaries     Missed by (almost) every run vs found by (almost) every run, described
                         with the transition features of phrase_oracle_analysis.py (pause before,
                         sign durations, mouthing spread, hand switch, gloss categories, and the
                         timing-only and lexical oracle scores): what makes a boundary hard?
  C. False alarms        Predicted starts made by many runs where the gold has none. Where are
                         they (between phrases / at a sign transition inside a phrase / inside a
                         sign), and do they look like boundaries (pause, oracle score)? Many
                         consistent false alarms at long pauses inside translation units point to
                         the LABEL definition, not to the models.
  D. Timing              Offsets (predicted - gold) of matched starts: systematic shift vs scatter.
                         The shift is estimated on val and applied on test: if boundary F1 at
                         +-1/+-2 frames jumps, part of the "timing error" is an annotation
                         convention (e.g. onset = hand starts moving vs hand reaches the sign).

Inputs: evaluation_phrase/predictions/<run>/<split>.npz + <split>_meta.json for every run
(evaluate the runs with --save-predictions on test, and on val for D's shift estimate), and
optionally evaluation_phrase/oracle/transitions_<split>.csv (phrase_oracle_analysis.py) for B/C.

Usage (repo root):
  python boundary_error_analysis.py                                   # every run with saved test predictions
  python boundary_error_analysis.py --runs-glob "multistream_bilstm-*" --decoder tuned
  python boundary_error_analysis.py --tolerance 5 --transitions evaluation_phrase/oracle/transitions_test.csv
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from src.metrics import extract_segments, decode_threshold_2023, match_boundaries

B_TAG = 2


def config_of(meta):
    desc = re.sub(r"\s*\[seed=\d+\]\s*$", "", meta.get("description", "")).strip()
    return f"{meta.get('model', meta['run'].rsplit('-', 1)[0])} | {desc or meta['run']}"


def assign_seed_blocks(runs):
    """Same description is reused by different experiments (e.g. "MS: Body" in several
    studies). A configuration = runs with the same description AND consecutive prefixes with
    distinct seeds (how the queue expands one experiment into its seeds)."""
    by_desc = {}
    for r in runs:
        by_desc.setdefault(r["config"], []).append(r)
    for desc, rs in by_desc.items():
        rs.sort(key=lambda r: int(r["meta"].get("prefix") or r["run"].rsplit("-", 1)[-1]))
        block, last, seeds = 0, None, set()
        for r in rs:
            pref = int(r["meta"].get("prefix") or r["run"].rsplit("-", 1)[-1])
            seed = str(r["meta"].get("seed", ""))
            if last is not None and (pref - last > 1 or seed in seeds):
                block, seeds = block + 1, set()
            seeds.add(seed)
            last = pref
            r["config"] = f"{desc} #{block}" if block else desc


def load_runs(pred_dir, split, runs_glob, decoder):
    runs = []
    for d in sorted(glob.glob(os.path.join(pred_dir, runs_glob))):
        p, m = os.path.join(d, f"{split}.npz"), os.path.join(d, f"{split}_meta.json")
        if not (os.path.exists(p) and os.path.exists(m)):
            continue
        with open(m) as f:
            meta = json.load(f)
        thr = None
        if decoder == "tuned":
            thr = meta.get("tuned_thresholds")
            if not thr:
                print(f"  {meta['run']}: no tuned thresholds saved (evaluate with --tune-on val); using fixed.")
                thr = meta.get("fixed_thresholds", [0.5, 0.5])
        elif decoder == "fixed":
            thr = meta.get("fixed_thresholds", [0.5, 0.5])
        runs.append({"run": meta["run"], "path": p, "meta": meta, "thr": thr, "config": config_of(meta)})
    return runs


def predicted_starts(probs, thr):
    if thr is None:
        segs = extract_segments(probs.argmax(axis=0))
    else:
        segs = decode_threshold_2023(probs, thr[0], thr[1])
    return np.array([s for s, _ in segs], dtype=np.int64)


def gold_info(labels_dir, vids):
    out = {}
    for v in vids:
        g = np.load(os.path.join(labels_dir, f"{v}.npy"))
        segs = extract_segments(g)
        out[v] = {"labels": g, "starts": np.array([s for s, _ in segs], dtype=np.int64), "segs": segs}
    return out


def boundary_icc(det):
    """det: (n_boundaries, n_runs) 0/1. One-way ICC over boundaries (runs as repeated measures)."""
    n_b, n_r = det.shape
    p_b = det.mean(1)
    grand = det.mean()
    msb = n_r * np.sum((p_b - grand) ** 2) / (n_b - 1)
    msw = np.sum((det - p_b[:, None]) ** 2) / (n_b * (n_r - 1))
    return float((msb - msw) / (msb + (n_r - 1) * msw))


def binom_pmf0(n, p):
    return (1 - p) ** n


def main():
    ap = argparse.ArgumentParser(description="Error consistency, hard boundaries, false alarms, timing.")
    ap.add_argument("--predictions", default=os.path.join("evaluation_phrase", "predictions"))
    ap.add_argument("--runs-glob", default="*", help="run directories to include (default: all)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--shift-split", default="val", help="split the timing shift is estimated on")
    ap.add_argument("--decoder", default="argmax", choices=["argmax", "tuned", "fixed"])
    ap.add_argument("--tolerance", type=int, default=5, help="+-frames for a detection (default 5)")
    ap.add_argument("--offset-tolerance", type=int, default=25,
                    help="+-frames when collecting offsets for the timing analysis (default 25)")
    ap.add_argument("--max-pred-ratio", type=float, default=3.0,
                    help="drop runs predicting more than this x the gold starts (heavy over-segmenters, "
                         "e.g. old label versions); they inflate the union recall (default 3)")
    ap.add_argument("--include-unseeded", action="store_true",
                    help="also use runs without a recorded seed (older experiments)")
    ap.add_argument("--dup-window", type=int, default=10,
                    help="a false alarm within this many frames of a gold start = duplicate start "
                         "(fragmentation at a real boundary), reported separately (default 10)")
    ap.add_argument("--min-recall", type=float, default=0.1,
                    help="drop degenerate runs (e.g. face only) with recall below this (default 0.1)")
    ap.add_argument("--transitions", default=None,
                    help="transitions_<split>.csv from phrase_oracle_analysis.py (default: "
                         "evaluation_phrase/oracle/transitions_<split>.csv if it exists)")
    ap.add_argument("--hard", type=float, default=0.1, help="'hard' = detected by <= this share of runs")
    ap.add_argument("--easy", type=float, default=0.9, help="'easy' = detected by >= this share of runs")
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "error_analysis"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    runs = load_runs(args.predictions, args.split, args.runs_glob, args.decoder)
    if not runs:
        raise SystemExit(f"No runs with {args.split}.npz in {args.predictions} (evaluate_phrase.py --save-predictions).")
    labels_dirs = Counter(r["meta"]["labels_dir"] for r in runs)
    labels_dir = labels_dirs.most_common(1)[0][0]
    runs = [r for r in runs if r["meta"]["labels_dir"] == labels_dir]

    # ---------------------------------------------------------------- decode every run
    vids = None
    preds = {}
    for r in runs:
        z = np.load(r["path"])
        vids = sorted(z.files) if vids is None else sorted(set(vids) & set(z.files))
        preds[r["run"]] = {v: predicted_starts(z[v].astype(np.float32), r["thr"]) for v in z.files}
    gold = gold_info(labels_dir, vids)
    n_gold = sum(len(gold[v]["starts"]) for v in vids)

    # drop degenerate runs (e.g. face only): almost no recall
    keep = []
    for r in runs:
        hit = sum(match_boundaries(gold[v]["starts"], preds[r["run"]][v], args.tolerance)[0].sum() for v in vids)
        r["recall"] = hit / max(n_gold, 1)
        r["n_pred"] = sum(len(preds[r["run"]][v]) for v in vids)
        seeded = str(r["meta"].get("seed", "")) != ""
        if r["recall"] < args.min_recall:
            print(f"  dropped {r['run']} (recall {r['recall']:.3f}: degenerate)")
        elif r["n_pred"] > args.max_pred_ratio * n_gold:
            print(f"  dropped {r['run']} ({r['n_pred'] / n_gold:.1f}x gold starts: over-segmenting)")
        elif not (seeded or args.include_unseeded):
            print(f"  dropped {r['run']} (no seed recorded; --include-unseeded keeps it)")
        else:
            keep.append(r)
    runs = keep
    assign_seed_blocks(runs)
    R = len(runs)
    out(f"{R} runs, {len(vids)} {args.split} videos, {n_gold} gold phrase starts, decoder={args.decoder}, "
        f"tolerance +-{args.tolerance} frames, labels {labels_dir}")

    # ---------------------------------------------------------------- A. consistency
    keys = [(v, int(s)) for v in vids for s in gold[v]["starts"]]
    det = np.zeros((len(keys), R), dtype=np.int8)
    offsets = defaultdict(list)                       # run -> offsets at offset_tolerance
    fp_events = []                                    # (video, frame, run)
    row0 = 0
    for v in vids:
        gs = gold[v]["starts"]
        for j, r in enumerate(runs):
            ps = preds[r["run"]][v]
            g_ok, p_ok, _ = match_boundaries(gs, ps, args.tolerance)
            det[row0:row0 + len(gs), j] = g_ok
            for f in ps[~p_ok]:
                fp_events.append((v, int(f), r["run"]))
            _, _, off = match_boundaries(gs, ps, args.offset_tolerance)
            offsets[r["run"]] += [float(o) for o in off[np.isfinite(off)]]
        row0 += len(gs)
    rate = det.mean(1)
    mean_recall = det.mean()
    exp_never = binom_pmf0(R, mean_recall)
    exp_always = mean_recall ** R
    icc = boundary_icc(det.astype(float))
    configs = sorted({r["config"] for r in runs})
    out("\nA. ERROR CONSISTENCY")
    out(f"  mean single-run recall@{args.tolerance}: {mean_recall:.3f}   (runs: "
        f"{min(r['recall'] for r in runs):.3f}-{max(r['recall'] for r in runs):.3f})")
    out(f"  detected by NO run:   {np.mean(rate == 0):.3f} of gold starts   (independent errors would give {exp_never:.2e})")
    out(f"  detected by EVERY run:{np.mean(rate == 1):.3f}                  (independent errors would give {exp_always:.2e})")
    out(f"  detected by >= half:  {np.mean(rate >= 0.5):.3f}")
    out(f"  union recall (>= 1 run detects it): {np.mean(rate > 0):.3f}  -> upper bound for ensembling / selection")
    out(f"  boundary ICC: {icc:.2f}  (share of variance in 'detected?' due to the boundary, not the run)")
    hist = np.histogram(rate, bins=np.linspace(0, 1, 11))[0]
    out("  detection-rate histogram (0-10%, ..., 90-100% of runs): " + " ".join(str(int(h)) for h in hist))
    if len(configs) > 1:
        cfg_rate = {c: det[:, [j for j, r in enumerate(runs) if r["config"] == c]].mean(1) for c in configs}
        out(f"  {len(configs)} configurations; mean pairwise correlation of per-boundary detection rates: "
            f"{np.nanmean([np.corrcoef(cfg_rate[a], cfg_rate[b])[0, 1] for i, a in enumerate(configs) for b in configs[i + 1:]]):.2f}")
    vid_rate = {v: rate[[i for i, k in enumerate(keys) if k[0] == v]].mean() for v in vids}
    out("  per-video mean detection rate: min {:.2f} / median {:.2f} / max {:.2f}".format(
        min(vid_rate.values()), float(np.median(list(vid_rate.values()))), max(vid_rate.values())))

    # ---------------------------------------------------------------- B. hard boundaries
    tpath = args.transitions or os.path.join("evaluation_phrase", "oracle", f"transitions_{args.split}.csv")
    trans = {}
    trans_frames = defaultdict(list)
    if os.path.exists(tpath):
        with open(tpath) as f:
            for row in csv.DictReader(f):
                trans[(row["video"], int(row["frame"]))] = row
                trans_frames[row["video"]].append(int(row["frame"]))
    trans_frames = {v: np.array(sorted(fs)) for v, fs in trans_frames.items()}
    gold_rows = []
    for i, (v, s) in enumerate(keys):
        segs = gold[v]["segs"]
        k = [j for j, (a, _) in enumerate(segs) if a == s][0]
        prev_gap = s - segs[k - 1][1] - 1 if k > 0 else -1
        row = {"video": v, "frame": s, "detection_rate": round(float(rate[i]), 4),
               "phrase_len": segs[k][1] - segs[k][0] + 1,
               "prev_phrase_len": segs[k - 1][1] - segs[k - 1][0] + 1 if k > 0 else -1,
               "phrase_gap_before": prev_gap}
        t = trans.get((v, s))
        if t:
            for col in ("gap_prev", "dur_prev_rel", "dur_cur_rel", "gap_prev_local_max", "mouth_spread",
                        "hand_switch", "mouth_prev", "mouth_cur", "gloss_prev", "gloss_cur",
                        "oracle:non-lexical (timing+hands+mouth)", "oracle:lexical", "oracle:all"):
                if col in t:
                    row[col] = t[col]
        gold_rows.append(row)

    def describe(rows, label):
        def med(col):
            x = [float(r[col]) for r in rows if col in r and r[col] not in ("", None)]
            return f"{np.median(x):.2f}" if x else "-"

        def share(col, val):
            x = [r[col] for r in rows if col in r]
            return f"{np.mean([str(v) == str(val) for v in x]):.2f}" if x else "-"
        out(f"  {label:<10s} n={len(rows):5d}  pause before phrase (fr) {med('phrase_gap_before'):>7s}  "
            f"phrase len {med('phrase_len'):>6s}  prev phrase len {med('prev_phrase_len'):>6s}  "
            f"prev-sign dur (rel) {med('dur_prev_rel'):>5s}  local-max pause {med('gap_prev_local_max'):>5s}  "
            f"mouthing spreads {share('mouth_spread', 1):>5s}  hand switch {share('hand_switch', 1):>5s}  "
            f"oracle timing {med('oracle:non-lexical (timing+hands+mouth)'):>6s}  "
            f"oracle lexical {med('oracle:lexical'):>6s}")

    hard = [r for r in gold_rows if r["detection_rate"] <= args.hard]
    easy = [r for r in gold_rows if r["detection_rate"] >= args.easy]
    mid = [r for r in gold_rows if args.hard < r["detection_rate"] < args.easy]
    out(f"\nB. HARD vs EASY BOUNDARIES (hard: <= {args.hard:.0%} of runs, easy: >= {args.easy:.0%})")
    for rows, lab in ((hard, "hard"), (mid, "medium"), (easy, "easy")):
        describe(rows, lab)
    if trans:
        for col in ("oracle:non-lexical (timing+hands+mouth)", "oracle:lexical", "phrase_gap_before"):
            x = np.array([float(r[col]) for r in gold_rows if col in r])
            yv = np.array([r["detection_rate"] for r in gold_rows if col in r])
            if len(x) > 10:
                rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(yv))
                out(f"  Spearman(detection rate, {col}): {np.corrcoef(rx, ry)[0, 1]:+.2f}")
        cats = Counter()
        for r in hard:
            for c in ("gloss_prev", "gloss_cur"):
                if c in r:
                    g = r[c].upper()
                    cats[(c, "$GEST-OFF" if g.startswith("$GEST-OFF") else g.split("^")[0][:12])] += 1
        out("  most frequent glosses around HARD boundaries: " +
            ", ".join(f"{c.split('_')[1]}={g} ({n})" for (c, g), n in cats.most_common(12)))
    else:
        out(f"  (no transitions file at {tpath}: run phrase_oracle_analysis.py for the feature columns)")

    # ---------------------------------------------------------------- C. false alarms
    out("\nC. FALSE ALARMS (unmatched predicted starts)")
    by_vid = defaultdict(list)
    for v, f_, run in fp_events:
        by_vid[v].append((f_, run))
    clusters = []
    for v, evs in by_vid.items():
        evs.sort()
        cur = [evs[0]]
        for e in evs[1:]:
            if e[0] - cur[-1][0] <= args.tolerance:
                cur.append(e)
            else:
                clusters.append((v, cur))
                cur = [e]
        clusters.append((v, cur))
    fa_rows = []
    for v, evs in clusters:
        frames = [e[0] for e in evs]
        share = len({e[1] for e in evs}) / R
        c = int(np.median(frames))
        g = gold[v]["labels"]
        where = "between phrases" if g[min(c, len(g) - 1)] == 0 else "inside phrase"
        t = None
        if trans:
            fr = trans_frames.get(v)
            if fr is not None and len(fr):
                j = int(np.clip(np.searchsorted(fr, c), 1, len(fr) - 1)) if len(fr) > 1 else 0
                cand = fr[max(0, j - 1):j + 1]
                best = cand[np.argmin(np.abs(cand - c))]
                if abs(int(best) - c) <= args.tolerance:
                    t = trans[(v, int(best))]
        gsv = gold[v]["starts"]
        near_gold = int(np.min(np.abs(gsv - c))) if len(gsv) else 10 ** 9
        if near_gold <= args.dup_window:
            where = "duplicate start near a gold start"
            t = None          # its transition IS the true boundary -- don't describe it as a false one
        elif where == "inside phrase":
            where = "within-phrase sign transition" if t else "inside a sign / no transition"
        fa_rows.append({"video": v, "frame": c, "share_of_runs": round(share, 3), "where": where,
                        **({k: t[k] for k in ("gap_prev", "gap_prev_local_max", "mouth_spread",
                                                "gloss_prev", "gloss_cur",
                                                "oracle:non-lexical (timing+hands+mouth)") if k in t} if t else {})})
    consistent = [r for r in fa_rows if r["share_of_runs"] >= 0.5]
    out(f"  {len(fp_events)} false alarms over {R} runs ({len(fp_events) / R:.0f} per run) in "
        f"{len(clusters)} positions; {len(consistent)} positions are false alarms of >= 50% of runs")
    for lab, rows in (("all false alarms (weighted by runs)", fa_rows), ("consistent (>= 50% of runs)", consistent)):
        w = Counter()
        for r in rows:
            w[r["where"]] += r["share_of_runs"] if lab.startswith("all") else 1
        tot = sum(w.values()) or 1
        out(f"  {lab}: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in w.most_common()))
    if trans:
        neg = [r for r in trans.values() if r["y"] == "0"]
        pos = [r for r in trans.values() if r["y"] == "1"]
        cons_t = [r for r in consistent if "gap_prev" in r and r["where"] == "within-phrase sign transition"]
        long_neg = [(r["video"], int(r["frame"])) for r in neg if float(r["gap_prev"]) >= 25]
        flagged = {(r["video"], int(r["frame"])) for r in fa_rows if r["share_of_runs"] >= 0.5}
        fl = 0
        for v_, fr_ in long_neg:
            fl += any((v_, fr_ + d) in flagged for d in range(-args.tolerance, args.tolerance + 1))

        def mgap(rows):
            x = [float(r["gap_prev"]) for r in rows if "gap_prev" in r]
            return f"{np.median(x):.1f} fr (share > 12 fr: {np.mean(np.array(x) > 12):.2f})" if x else "-"

        def morc(rows):
            x = [float(r["oracle:non-lexical (timing+hands+mouth)"]) for r in rows
                 if r.get("oracle:non-lexical (timing+hands+mouth)") not in (None, "")]
            return f"{np.median(x):+.2f}" if x else "-"
        out(f"  within-phrase pauses >= 25 fr: {len(long_neg)}, of which {fl} ({fl / max(len(long_neg), 1):.0%}) are "
            f"consistent false alarms (candidate prosodic boundaries not marked by the translation units)")
        out(f"  pause before the sign:  true boundaries {mgap(pos)} | consistent within-phrase false alarms {mgap(cons_t)} | "
            f"all within-phrase transitions {mgap(neg)}")
        out(f"  timing-oracle score:     true boundaries {morc(pos)} | consistent within-phrase false alarms {morc(cons_t)} | "
            f"all within-phrase transitions {morc(neg)}")
        out("  (false alarms that look like true boundaries on these measures are candidates for prosodic "
            "boundaries the translation segmentation does not mark)")

    # ---------------------------------------------------------------- D. timing
    out(f"\nD. TIMING (offsets predicted - gold of matched starts, within +-{args.offset_tolerance} frames)")
    allo = np.concatenate([np.array(offsets[r["run"]]) for r in runs if offsets[r["run"]]])
    mean_off = float(np.mean(allo))
    out(f"  pooled: mean {mean_off:+.2f}, median {np.median(allo):+.1f}, SD {np.std(allo):.2f}, "
        f"MAD {np.median(np.abs(allo - np.median(allo))):.2f} frames; "
        f"robust bias share median^2 / (median^2 + (1.48 MAD)^2): "
        f"{np.median(allo) ** 2 / max(np.median(allo) ** 2 + (1.4826 * np.median(np.abs(allo - np.median(allo)))) ** 2, 1e-9):.2f}")
    per_cfg = defaultdict(list)
    for r in runs:
        if offsets[r["run"]]:
            per_cfg[r["config"]].append(np.median(offsets[r["run"]]))
    out("  median offset per configuration: " + "; ".join(
        f"{c.split('|')[-1].strip()[:30]} {np.mean(v):+.1f}" for c, v in sorted(per_cfg.items())[:12]))
    # shift estimated on the other split, applied here
    shift_rows = []
    for r in runs:
        mp = os.path.join(args.predictions, r["run"], f"{args.shift_split}.npz")
        if not os.path.exists(mp) or args.shift_split == args.split:
            continue
        zv = np.load(mp)
        gv = gold_info(labels_dir, zv.files)
        ov = []
        for v in zv.files:
            ps = predicted_starts(zv[v].astype(np.float32), r["thr"])
            _, _, o = match_boundaries(gv[v]["starts"], ps, args.offset_tolerance)
            ov += list(o[np.isfinite(o)])
        if not ov:
            continue
        shift = int(round(np.median(ov)))
        res = {"run": r["run"], "shift": shift}
        for k in (1, 2, 5):
            for lab, sh in (("before", 0), ("after", shift)):
                tp = npred = 0
                for v in vids:
                    ps = preds[r["run"]][v] - sh
                    g_ok, _, _ = match_boundaries(gold[v]["starts"], ps, k)
                    tp += g_ok.sum()
                    npred += len(ps)
                p, rc = tp / max(npred, 1), tp / max(n_gold, 1)
                res[f"F1@{k}_{lab}"] = 2 * p * rc / max(p + rc, 1e-12)
        shift_rows.append(res)
    if shift_rows:
        out(f"  shift estimated on {args.shift_split} (median offset per run), applied on {args.split}:")
        for k in (1, 2, 5):
            b = np.mean([r[f"F1@{k}_before"] for r in shift_rows])
            a = np.mean([r[f"F1@{k}_after"] for r in shift_rows])
            out(f"    boundary F1@{k}: {b:.3f} -> {a:.3f} ({a - b:+.3f})  (mean over {len(shift_rows)} runs, "
                f"shifts {Counter(r['shift'] for r in shift_rows).most_common(3)})")
    else:
        out(f"  (save {args.shift_split} predictions too to test a calibrated shift)")

    # ---------------------------------------------------------------- outputs
    def write(path, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    write(os.path.join(args.out_dir, f"gold_boundaries_{args.split}.csv"), gold_rows)
    # per-configuration detection rate of every gold boundary (long format), e.g. to compare
    # face vs no-face models on phrases that open with a non-manual sign ($ORAL, $GEST-NM)
    cfg_names = sorted({r["config"] for r in runs})
    cols = {c: [j for j, r in enumerate(runs) if r["config"] == c] for c in cfg_names}
    long_rows = []
    for c, js in cols.items():
        rates = det[:, js].mean(1)
        for (v, s_), rt in zip(keys, rates):
            long_rows.append({"config": c, "n_runs": len(js), "video": v, "frame": s_, "rate": round(float(rt), 4)})
    write(os.path.join(args.out_dir, f"boundary_detection_by_config_{args.split}.csv"), long_rows)
    write(os.path.join(args.out_dir, f"false_alarms_{args.split}.csv"), sorted(fa_rows, key=lambda r: -r["share_of_runs"]))
    write(os.path.join(args.out_dir, f"timing_shift_{args.split}.csv"), shift_rows)
    write(os.path.join(args.out_dir, f"runs_{args.split}.csv"),
          [{"run": r["run"], "config": r["config"], "recall": round(r["recall"], 4), "n_pred": r["n_pred"],
            "median_offset": float(np.median(offsets[r["run"]])) if offsets[r["run"]] else ""} for r in runs])
    with open(os.path.join(args.out_dir, f"summary_{args.split}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        bins = np.arange(R + 2) - 0.5
        ax[0].hist(det.sum(1), bins=bins, color="#3b6fb6", label="observed")
        from math import comb
        expct = [len(keys) * comb(R, k) * mean_recall ** k * (1 - mean_recall) ** (R - k) for k in range(R + 1)]
        ax[0].plot(range(R + 1), expct, color="#d6336c", label="independent errors")
        ax[0].set_xlabel(f"number of runs (of {R}) detecting the boundary")
        ax[0].set_ylabel("gold phrase starts")
        ax[0].legend(frameon=False)
        ax[0].set_title(f"Error consistency (ICC {icc:.2f})")
        lim = args.offset_tolerance
        ax[1].hist(allo, bins=np.arange(-lim, lim + 2) - 0.5, color="#17a589")
        ax[1].axvline(0, color="k", lw=0.8)
        ax[1].set_xlabel("predicted - gold start (frames)")
        ax[1].set_title(f"Timing offsets (median {np.median(allo):+.1f}, mean {mean_off:+.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"error_analysis_{args.split}.png"), dpi=180)
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()