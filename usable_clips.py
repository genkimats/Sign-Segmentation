"""
usable_clips.py -- downstream-oriented evaluation: how many predicted phrases would give a usable
(clip, sentence) pair for building a sentence-level dataset or feeding a translation model?

Every predicted phrase is treated as a clip that would be paired with one sentence translation, and is
classified against the annotated phrases of its video:

  usable    IoU >= --iou with exactly one annotated phrase, and no OTHER annotated phrase has more than
            --other of its length inside the clip  -> a correct (clip, sentence) pair
  merged    covers at least --cover of two or more annotated phrases  -> one clip, several sentences
  fragment  lies mostly (> 50% of the clip) inside one annotated phrase but its IoU with every phrase is
            below the threshold  -> an incomplete sentence
  spurious  lies mostly (> 50% of the clip) outside every annotated phrase  -> no sentence
  other     the rest (shifted or partly overlapping clips)

Matching is one-to-one: an annotated phrase can make at most one clip usable. So
  clip precision = usable / predicted clips       (how much of the output is directly usable)
  phrase recovery = usable / annotated phrases    (how much of the corpus becomes training pairs)
  usable-clip F1  = their harmonic mean
These are counted at two IoU thresholds (default 0.5 and 0.7: a translation clip should hold nearly
the whole sentence).

Decoding: peak picking + threshold with settings tuned on validation per run (tune_peaks_on_val), pause
end o_min = 5 -- the decoder used for the main results. Annotated phrases from the labels in the
predictions' meta (BIO_tags_phrase_v2). Intervals: paired bootstrap over test videos, runs averaged
(phrase_stats.paired_bootstrap).

    python usable_clips.py
    python usable_clips.py --a multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117 \
                           --b bilstm_2023-244 bilstm_2023-245 bilstm_2023-246 \
                           --extra bilstm_2023-302 bilstm_2023-303 bilstm_2023-304
    python usable_clips.py --selftest        # checks the clip classification on hand-made cases
Output: evaluation_phrase/usable_clips/{summary.txt, per_run.csv}
"""
import argparse
import csv
import os
import sys

import numpy as np

ARM_A = ["multistream_bilstm-115", "multistream_bilstm-116", "multistream_bilstm-117"]
ARM_B = ["bilstm_2023-244", "bilstm_2023-245", "bilstm_2023-246"]
NAMES = {"a": "Proposed (feature-specific encoders)", "b": "Moryossef et al. (2023), re-implemented",
         "extra": "Early fusion, same inputs"}
CLASSES = ("usable", "merged", "fragment", "spurious", "other")


# ----------------------------------------------------------------------------- segments
def to_half_open(segs, end_inclusive):
    """(start, end) pairs -> half-open [start, end) arrays, whatever convention extract_segments uses."""
    if not len(segs):
        return np.zeros((0, 2), dtype=np.int64)
    a = np.asarray(segs, dtype=np.int64).reshape(-1, 2).copy()
    if end_inclusive:
        a[:, 1] += 1
    return a


def end_is_inclusive():
    """Probe src.metrics.extract_segments once: labels O B I I O -> (1, 3) inclusive or (1, 4) half-open."""
    from src.metrics import extract_segments
    s = extract_segments(np.array([0, 2, 1, 1, 0]))
    if len(s) != 1 or s[0][0] != 1 or s[0][1] not in (3, 4):
        raise RuntimeError(f"unexpected extract_segments output on a probe: {s}")
    return s[0][1] == 3


def classify(pred, gold, iou_thr=0.5, cover=0.5, other=0.2):
    """
    pred, gold: (n, 2) half-open segments of one video.
    Returns (labels per predicted clip, number of usable clips after one-to-one matching).
    """
    labels, cand = [], []                      # cand: (iou, clip index, gold index) for usable candidates
    glen = gold[:, 1] - gold[:, 0] if len(gold) else np.zeros(0)
    for i, (s, e) in enumerate(pred):
        plen = max(e - s, 1)
        if len(gold):
            ov = np.clip(np.minimum(e, gold[:, 1]) - np.maximum(s, gold[:, 0]), 0, None)
            iou = ov / (plen + glen - ov)
            cov_g = ov / np.maximum(glen, 1)        # share of each annotated phrase inside the clip
            share_p = ov / plen                     # share of the clip inside each annotated phrase
        else:
            ov = iou = cov_g = share_p = np.zeros(0)
        hits = np.where(iou >= iou_thr)[0]
        if len(hits) == 1 and np.all(np.delete(cov_g, hits[0]) <= other):
            labels.append("usable")
            cand.append((float(iou[hits[0]]), i, int(hits[0])))
        elif (cov_g >= cover).sum() >= 2:
            labels.append("merged")
        elif len(share_p) and share_p.max() > 0.5 and iou.max() < iou_thr:
            labels.append("fragment")
        elif share_p.sum() < 0.5:
            labels.append("spurious")
        else:
            labels.append("other")
    # one-to-one: an annotated phrase makes at most one clip usable (best IoU first)
    used_g, n_usable = set(), 0
    for _, i, g in sorted(cand, reverse=True):
        if g in used_g:
            labels[i] = "other"
            continue
        used_g.add(g)
        n_usable += 1
    return labels, n_usable


# ----------------------------------------------------------------------------- decoding
def decode_run(run, split, pred_root, o_min, end_incl):
    from count_oracle_decoding import peaks_with_heights, starts_to_segments, tune_peaks_on_val
    from phrase_stats import load_predictions
    from src.metrics import B_TAG, extract_segments
    val = load_predictions(pred_root, run, "val")
    probs, _, gold = load_predictions(pred_root, run, split)
    tuned = tune_peaks_on_val(val)
    out = {}
    for v, p in probs.items():
        idx, h = peaks_with_heights(p[B_TAG], tuned["min_dist"], tuned["sigma"])
        starts = np.sort(idx[h >= tuned["thr"]])
        pred = to_half_open(starts_to_segments(starts, p, o_min), end_incl)
        gseg = to_half_open(extract_segments(gold[v]), end_incl)
        out[v] = (pred, gseg)
    return out, tuned


def video_counts(decoded, iou_thr, cover, other):
    """{vid: (pred, gold)} -> {vid: {"usable": (u, n_pred, n_gold), class: (count, n_pred, n_gold), ...}}"""
    res = {}
    for v, (pred, gold) in decoded.items():
        labels, n_usable = classify(pred, gold, iou_thr, cover, other)
        n_pred, n_gold = len(pred), len(gold)
        res[v] = {c: ((n_usable if c == "usable" else labels.count(c)), n_pred, n_gold) for c in CLASSES}
    return res


# ----------------------------------------------------------------------------- metrics on pooled counts
def precision(k, n_pred, n_gold):
    return k / n_pred if n_pred else 0.0


def recovery(k, n_pred, n_gold):
    return k / n_gold if n_gold else 0.0


def f1(k, n_pred, n_gold):
    p, r = precision(k, n_pred, n_gold), recovery(k, n_pred, n_gold)
    return 2 * p * r / (p + r) if p + r else 0.0


def pooled(per_run, key, metric):
    """mean over runs of the metric on counts pooled over videos."""
    vals = []
    for c in per_run:
        tot = np.sum([c[v][key] for v in c], axis=0)
        vals.append(metric(*tot))
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def boot(per_run_a, per_run_b, key, metric, n_boot):
    from phrase_stats import paired_bootstrap
    ca = [{v: c[v][key] for v in c} for c in per_run_a]
    cb = [{v: c[v][key] for v in c} for c in per_run_b]
    return paired_bootstrap(ca, cb, n_boot=n_boot, metric=metric)


# ----------------------------------------------------------------------------- main
def run(arms, split, pred_root, out_dir, iou_list, cover, other, o_min, n_boot):
    end_incl = end_is_inclusive()
    decoded, tuned = {}, {}
    for arm, runs in arms.items():
        for r in runs:
            print(f"decoding {r} ...")
            decoded[r], tuned[r] = decode_run(r, split, pred_root, o_min, end_incl)

    L = [f"Usable-clip analysis ({split}; peak picking + threshold tuned on val, o_min={o_min})",
         f"usable = IoU >= thr with exactly one annotated phrase and no other phrase > {other:.0%} inside; "
         f"merged = covers >= {cover:.0%} of 2+ phrases; one-to-one matching.",
         "clip precision = usable / predicted clips; phrase recovery = usable / annotated phrases. "
         "Mean +- SD over runs; differences: paired bootstrap over videos, runs averaged.", ""]
    rows = []
    for thr in iou_list:
        counts = {arm: [video_counts(decoded[r], thr, cover, other) for r in runs] for arm, runs in arms.items()}
        n_gold = sum(len(g) for _, g in next(iter(decoded.values())).values())
        L.append(f"=== IoU >= {thr}  ({n_gold} annotated phrases) ===")
        L.append(f"{'model':<42s}{'clips':>7s}{'usable-clip F1':>18s}{'clip precision':>18s}{'phrase recovery':>18s}"
                 + "".join(f"{c:>10s}" for c in CLASSES[1:]))
        for arm, runs in arms.items():
            c = counts[arm]
            n_clips = np.mean([sum(x[v]["usable"][1] for v in x) for x in c])
            f, fs = pooled(c, "usable", f1)
            p, ps = pooled(c, "usable", precision)
            r_, rs = pooled(c, "usable", recovery)
            shares = [pooled(c, k, precision)[0] for k in CLASSES[1:]]
            L.append(f"{NAMES.get(arm, arm):<42s}{n_clips:>7.0f}{f:>10.3f} +- {fs:.3f}{p:>10.3f} +- {ps:.3f}"
                     f"{r_:>10.3f} +- {rs:.3f}" + "".join(f"{s:>10.1%}" for s in shares))
            for rr, cc in zip(runs, c):
                rows.append({"iou": thr, "arm": arm, "run": rr, **{f"{k}_count": sum(cc[v][k][0] for v in cc)
                                                                   for k in CLASSES},
                             "n_pred": sum(cc[v]["usable"][1] for v in cc), "n_gold": sum(cc[v]["usable"][2] for v in cc),
                             "usable_f1": f1(*np.sum([cc[v]["usable"] for v in cc], axis=0))})
        L.append("  merged / fragment / spurious / other = share of predicted clips")
        if "a" in arms:
            for other_arm in [a for a in arms if a != "a"]:
                L.append(f"  difference, {NAMES['a']} - {NAMES.get(other_arm, other_arm)}:")
                for label, key, metric in (("usable-clip F1", "usable", f1), ("clip precision", "usable", precision),
                                           ("phrase recovery", "usable", recovery),
                                           ("merged share", "merged", precision),
                                           ("fragment share", "fragment", precision)):
                    b = boot(counts["a"], counts[other_arm], key, metric, n_boot)
                    lo, hi = b["ci95"]
                    L.append(f"    {label:<18s} {b['diff']:+.3f} (95% CI: {lo:+.3f} to {hi:+.3f})")
        L.append("")
    L.append("Decoder settings tuned on val (per run): d = min peak distance, sigma = smoothing, t = peak threshold")
    for r, t in tuned.items():
        L.append(f"  {r}: d={t['min_dist']} sigma={t['sigma']} t={t['thr']}")
    text = "\n".join(L)
    print("\n" + text)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir}/summary.txt, per_run.csv")


def selftest():
    G = np.array([[0, 100], [100, 200], [300, 400]])          # three phrases, the first two back to back
    cases = [
        (np.array([[2, 98]]), ["usable"], 1, "matches phrase 1"),
        (np.array([[0, 200]]), ["merged"], 0, "covers phrases 1 and 2"),
        (np.array([[10, 40]]), ["fragment"], 0, "a third of phrase 1"),
        (np.array([[220, 290]]), ["spurious"], 0, "between phrases"),
        (np.array([[5, 95], [10, 90]]), ["usable", "other"], 1, "two clips on one phrase: one-to-one"),
        (np.array([[0, 130]]), ["other"], 0, "phrase 1 + 30% of phrase 2: not usable"),
    ]
    ok_all = True
    for pred, want, want_u, name in cases:
        labels, u = classify(pred, G)
        ok = labels == want and u == want_u
        ok_all &= ok
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {labels}, usable {u}")
    a = to_half_open([(1, 3)], True)
    ok = a.tolist() == [[1, 4]]
    ok_all &= ok
    print(f"  [{'ok' if ok else 'FAIL'}] inclusive end -> half-open")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", default=ARM_A, help="proposed model runs")
    ap.add_argument("--b", nargs="+", default=ARM_B, help="baseline runs (compared with --a)")
    ap.add_argument("--extra", nargs="+", default=None, help="optional third arm (e.g. bilstm_2023-302 ...)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou", nargs="+", type=float, default=[0.5, 0.7])
    ap.add_argument("--cover", type=float, default=0.5, help="share of a phrase that counts as covered (merged)")
    ap.add_argument("--other", type=float, default=0.2, help="max share of another phrase inside a usable clip")
    ap.add_argument("--o-min", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "usable_clips"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return 0 if selftest() else 1
    from phrase_stats import PRED_ROOT
    arms = {"a": args.a, "b": args.b}
    if args.extra:
        arms["extra"] = args.extra
    run(arms, args.split, args.pred_root or PRED_ROOT, args.out_dir, args.iou, args.cover, args.other,
        args.o_min, args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())