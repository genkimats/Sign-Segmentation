"""
hierarchy_phrase/data_audit.py -- step 1 of the brief. Numpy only; reads labels from disk, no features needed.

    python data_audit.py                     # all three splits, working rate 25 fps (stride 2)
    python data_audit.py --stride 1          # native 50 fps

Reports, per split:
  * B:I:O frame ratios for signs and phrases (the brief expects about 1:5:18 and 1:58:77 on train),
  * signs / phrases per video, signs per phrase,
  * how often a gold phrase START (END) falls EXACTLY on a gold sign start (end), and within +-1/+-2/+-5 frames,
  * gap length (frames between consecutive signs) at phrase boundaries vs elsewhere: mean, median, percentiles,
    the AUC of 'gap length predicts phrase boundary', and a histogram figure if matplotlib is available.
A phrase-boundary gap that is clearly longer than a within-phrase gap is the quantitative form of the hypothesis
that prosodic cues at sign gaps carry the signal.
"""
import argparse
import json
import os

import numpy as np

from common import (HERE, DEFAULT_STRIDE, fps_of, have_all_files, load_gold_segments, split_ids)
from metrics import evaluate_videos
from segments import TAG_RULES, phrase_tags_over_signs, phrases_from_tags, segments_to_bio


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), rank-based."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                                   # average ranks over ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def nearest_dist(x, ref):
    ref = np.asarray(ref)
    if len(ref) == 0:
        return np.full(len(x), np.inf)
    return np.abs(np.asarray(x)[:, None] - ref[None, :]).min(1)


def where_edges_fall(sign_segs, points, kind):
    """Classify phrase starts (kind='start') / ends (kind='end') against the sign segmentation:
    on_edge (exactly a sign start/end), inside a sign, in a gap between signs, or outside the sign range."""
    starts = np.array([s for s, _ in sign_segs]); ends = np.array([e for _, e in sign_segs])
    out = {"on_edge": 0, "inside_sign": 0, "in_gap": 0, "outside": 0}
    for p in points:
        if kind == "start":
            j = int(np.searchsorted(starts, p, side="right")) - 1
            if j < 0:
                out["outside"] += 1
            elif starts[j] == p:
                out["on_edge"] += 1
            elif p < ends[j]:
                out["inside_sign"] += 1
            else:
                out["in_gap"] += 1
        else:
            k = int(np.searchsorted(ends, p, side="left"))
            if k >= len(ends):
                out["outside"] += 1
            elif ends[k] == p:
                out["on_edge"] += 1
            elif starts[k] < p:
                out["inside_sign"] += 1
            else:
                out["in_gap"] += 1
    return out


def audit_split(split, stride, tol_list=(0, 1, 2, 5)):
    ids = [v for v in split_ids(split) if have_all_files(v)]
    cnt_s, cnt_p = np.zeros(3), np.zeros(3)
    n_sign, n_phr, n_vid = 0, 0, 0
    starts, ends, tot_starts, tot_ends = {t: 0 for t in tol_list}, {t: 0 for t in tol_list}, 0, 0
    gap_phr, gap_in = [], []
    where_s, where_e = {}, {}
    ceil = {r: {"P": [], "G": [], "L": [], "gp": [], "gi": []} for r in TAG_RULES}
    for vid in ids:
        g = load_gold_segments(vid, stride)
        if g is None or not g["sign"] or not g["phrase"]:
            continue
        n_vid += 1
        cnt_s += np.bincount(segments_to_bio(g["sign"], g["T"]), minlength=3)
        cnt_p += np.bincount(segments_to_bio(g["phrase"], g["T"]), minlength=3)
        n_sign += len(g["sign"]); n_phr += len(g["phrase"])
        ss = np.array([s for s, _ in g["sign"]]); se = np.array([e for _, e in g["sign"]])
        ps = np.array([s for s, _ in g["phrase"]]); pe = np.array([e for _, e in g["phrase"]])
        ds, de = nearest_dist(ps, ss), nearest_dist(pe, se)
        for t in tol_list:
            starts[t] += int((ds <= t).sum()); ends[t] += int((de <= t).sum())
        tot_starts += len(ps); tot_ends += len(pe)
        tags = phrase_tags_over_signs(g["sign"], g["phrase"])
        gaps = np.array([g["sign"][k + 1][0] - g["sign"][k][1] for k in range(len(g["sign"]) - 1)])
        gap_phr.extend(gaps[tags[1:] == 1].tolist()); gap_in.extend(gaps[tags[1:] == 0].tolist())
        for key, (kind, pts) in {"start": ("start", ps), "end": ("end", pe)}.items():
            acc = where_s if key == "start" else where_e
            for k2, v2 in where_edges_fall(g["sign"], pts, kind).items():
                acc[k2] = acc.get(k2, 0) + v2
        for r in TAG_RULES:
            tg = phrase_tags_over_signs(g["sign"], g["phrase"], r)
            ceil[r]["P"].append(phrases_from_tags(g["sign"], tg)); ceil[r]["G"].append(g["phrase"]); ceil[r]["L"].append(g["T"])
            ceil[r]["gp"].extend(gaps[tg[1:] == 1].tolist()); ceil[r]["gi"].extend(gaps[tg[1:] == 0].tolist())
    gp, gi = np.array(gap_phr), np.array(gap_in)
    pct = lambda a: {q: float(np.percentile(a, q)) for q in (10, 50, 90)} if len(a) else {}      # noqa: E731
    return {
        "split": split, "videos": n_vid, "signs": n_sign, "phrases": n_phr,
        "signs_per_video": n_sign / max(n_vid, 1), "phrases_per_video": n_phr / max(n_vid, 1),
        "signs_per_phrase": n_sign / max(n_phr, 1),
        "sign_BIO_ratio(B:I:O)": [round(float(x), 2) for x in cnt_s[[2, 1, 0]] / max(cnt_s[2], 1)],
        "phrase_BIO_ratio(B:I:O)": [round(float(x), 2) for x in cnt_p[[2, 1, 0]] / max(cnt_p[2], 1)],
        "phrase_start_on_sign_start": {f"<={t}": starts[t] / max(tot_starts, 1) for t in tol_list},
        "phrase_end_on_sign_end": {f"<={t}": ends[t] / max(tot_ends, 1) for t in tol_list},
        "gap_frames_at_phrase_boundary": {"n": len(gp), "mean": float(gp.mean()) if len(gp) else None, **pct(gp)},
        "gap_frames_within_phrase": {"n": len(gi), "mean": float(gi.mean()) if len(gi) else None, **pct(gi)},
        "share_of_zero_gaps": {"phrase_boundary": float((gp == 0).mean()) if len(gp) else None,
                               "within_phrase": float((gi == 0).mean()) if len(gi) else None},
        "AUC_gap_predicts_boundary": auc(gp, gi),
        "phrase_starts_vs_signs": {k: v / max(sum(where_s.values()), 1) for k, v in where_s.items()},
        "phrase_ends_vs_signs": {k: v / max(sum(where_e.values()), 1) for k, v in where_e.items()},
        "ORACLE_CEILING_gold_signs_to_gold_phrases": {
            r: {**{k: round(v, 3) for k, v in evaluate_videos(ceil[r]["P"], ceil[r]["G"], ceil[r]["L"], tols=(2, 5), iou_thrs=(0.5,)).items()
                   if k in ("frame_f1", "frame_f1_B", "mask_iou", "ratio", "start_f1@2", "start_f1@5", "end_f1@2", "end_f1@5", "seg_f1@0.5")},
                "AUC_gap_boundary": round(auc(np.array(ceil[r]["gp"]), np.array(ceil[r]["gi"])), 3)} for r in TAG_RULES},
        "_gaps": (gp, gi),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    out, fig_data = {}, None
    print(f"working frame rate: {fps_of(a.stride):.1f} fps (stride {a.stride})")
    for sp in a.splits:
        r = audit_split(sp, a.stride)
        gaps = r.pop("_gaps")
        if sp == "train":
            fig_data = gaps
        out[sp] = r
        print(f"\n=== {sp} ===")
        for k, v in r.items():
            print(f"  {k}: {v}")
    path = os.path.join(HERE, "data_audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if fig_data is not None and len(fig_data[0]) and len(fig_data[1]):
            gp, gi = fig_data
            hi = int(np.percentile(np.concatenate([gp, gi]), 99))
            bins = np.arange(0, hi + 2)
            plt.figure(figsize=(7, 4))
            plt.hist(gi, bins=bins, alpha=0.6, density=True, label=f"within phrase (n={len(gi)})")
            plt.hist(gp, bins=bins, alpha=0.6, density=True, label=f"at phrase boundary (n={len(gp)})")
            plt.xlabel(f"gap between consecutive signs (frames @ {fps_of(a.stride):.0f} fps)")
            plt.ylabel("density"); plt.legend(); plt.title("train: gap length vs phrase boundary")
            plt.savefig(os.path.join(HERE, "data_audit_gaps.png"), bbox_inches="tight", dpi=130)
            print("saved data_audit_gaps.png")
    except Exception as e:                                          # figure is optional
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()