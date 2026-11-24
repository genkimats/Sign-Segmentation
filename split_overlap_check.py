"""
split_overlap_check.py -- do conversations and signers stay within one split? And if not, does it change
the comparison?

Identity sources:
  document (conversation)  the video name: <document>_<A|B>.npy; train also has time-cut documents
                           <document>-<start>-<end>_<A|B>.npy, which are the same recording.
  session                  the "Transcript" column of the Public DGS Corpus table (data_download/index.html),
                           e.g. dgskorpus_ber_01 = one recording session = one pair of informants, who sign
                           several documents (tasks) in that session.
  signer                   (session, camera side A/B). This assumes an informant keeps the same side in all
                           documents of a session. With --cmdi-dir (the corpus .cmdi files, one per document)
                           the script prints the participant fields of the CMDI files so this can be checked;
                           it does not rely on them.

Report (stdout + evaluation_phrase/split_overlap/):
  1. exact duplicates within and between splits
  2. document overlap: val/test videos whose conversation partner (or the same video) is in train
  3. session overlap and signer overlap: val/test videos whose signer has other documents in train
  4. optional, with --a / --b runs: test boundary F1 (peak picking + threshold tuned on val) on videos with
     a seen signer vs an unseen signer, per arm and A - B, with paired bootstrap over videos -- if the gain is
     similar in both groups, the overlap does not explain the comparison. Also the validation comparison
     without any val video that is itself in train.

    python split_overlap_check.py
    python split_overlap_check.py --a multistream_bilstm-115 multistream_bilstm-116 multistream_bilstm-117 \
                                  --b bilstm_2023-244 bilstm_2023-245 bilstm_2023-246
    python split_overlap_check.py --cmdi-dir raw_data/annotations      # if the .cmdi files are there
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

K = 5


# ----------------------------------------------------------------------------- identities
def doc_of(f):
    return os.path.splitext(f)[0].rsplit("_", 1)[0]          # keeps time-cut suffixes: the index uses them too


def base_doc(f):
    return doc_of(f).split("-")[0]


def side_of(f):
    return os.path.splitext(f)[0].rsplit("_", 1)[1][0].upper()


def load_sessions(index_html):
    from bs4 import BeautifulSoup
    with open(index_html, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "html.parser")
    table = soup.find("table")
    heads = [th.get_text(" ", strip=True) for th in table.find_all("th")]
    i_age = heads.index("Age Group") if "Age Group" in heads else None
    i_fmt = heads.index("Format") if "Format" in heads else None
    sess, info = {}, {}
    for tr in table.find_all("tr")[1:]:
        td = tr.find_all("td")
        a = td[0].find("a") if td else None
        if a is None:
            continue
        m = re.search(r"html/(.+?)_en\.html", a.get("href", ""))
        if not m:
            continue
        doc = m.group(1)
        sess[doc] = td[0].get_text(strip=True)
        info[doc] = {"age_group": td[i_age].get_text(strip=True) if i_age is not None else "",
                     "format": td[i_fmt].get_text(" ", strip=True) if i_fmt is not None else ""}
    if not sess:
        raise RuntimeError(f"no transcript rows found in {index_html}")
    return sess, info


def cmdi_fields(cmdi_dir, docs, n_show=2):
    """Print participant-like fields of a few CMDI files (to check how A/B map to informants)."""
    files = [p for d in docs for p in glob.glob(os.path.join(cmdi_dir, "**", f"{d}*.cmdi"), recursive=True)]
    if not files:
        return [f"  no .cmdi files for these documents under {cmdi_dir}"]
    L = []
    for p in files[:n_show]:
        L.append(f"  {os.path.basename(p)}:")
        root = ET.parse(p).getroot()
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if re.search(r"actor|participant|informant|speaker|signer", tag, re.I):
                kids = {c.tag.split("}")[-1]: (c.text or "").strip() for c in el if (c.text or "").strip()}
                if kids:
                    L.append(f"    <{tag}> " + "; ".join(f"{k}={v}" for k, v in list(kids.items())[:10]))
    return L


# ----------------------------------------------------------------------------- overlap table
def overlap_table(splits, sess):
    train = splits["train"]
    rows = []
    for sp in ("val", "test"):
        for f in splits[sp]:
            s, sd = sess.get(doc_of(f)), side_of(f)
            same_video = f in train
            partner = [t for t in train if base_doc(t) == base_doc(f) and side_of(t) != sd]
            same_doc_same_side = [t for t in train if base_doc(t) == base_doc(f) and side_of(t) == sd]
            ss = [t for t in train if s is not None and sess.get(doc_of(t)) == s and side_of(t) == sd]
            so = [t for t in train if s is not None and sess.get(doc_of(t)) == s and side_of(t) != sd]
            rows.append({"split": sp, "video": f, "document": base_doc(f), "session": s or "?", "side": sd,
                         "video_in_train": int(same_video), "partner_in_train": int(bool(partner)),
                         "same_doc_same_side_in_train": len(same_doc_same_side),
                         "session_same_side_train_videos": len(ss), "session_other_side_train_videos": len(so),
                         "signer_seen": int(bool(ss)), "session_seen": int(bool(ss or so))})
    return rows


def summary_lines(splits, rows, sess):
    L = ["Split overlap check (dataset_splits.json)", ""]
    L.append("1. Exact duplicates")
    for k, v in splits.items():
        d = [x for x, n in collections.Counter(v).items() if n > 1]
        L.append(f"   {k}: {len(v)} entries, duplicated: {d or 'none'}")
    for a in splits:
        for b in splits:
            if a < b:
                both = sorted(set(splits[a]) & set(splits[b]))
                L.append(f"   in both {a} and {b}: {both or 'none'}")
    unmapped = sorted({doc_of(f) for v in splits.values() for f in v if doc_of(f) not in sess})
    if unmapped:
        L.append(f"   documents not in the corpus table (no session): {unmapped[:10]}{' ...' if len(unmapped) > 10 else ''}")
    for sp in ("val", "test"):
        R = [r for r in rows if r["split"] == sp]
        n = len(R)
        sess_sp = {r["session"] for r in R}
        sess_tr = {sess.get(doc_of(t)) for t in splits["train"]}
        L += ["", f"{sp.upper()} ({n} videos, {len({r['document'] for r in R})} documents, {len(sess_sp)} sessions)"]
        L.append(f"2. conversation: partner video in train      {sum(r['partner_in_train'] for r in R):3d} / {n}")
        L.append(f"               same video in train          {sum(r['video_in_train'] for r in R):3d} / {n}")
        L.append(f"3. session: session has train videos         {sum(r['session_seen'] for r in R):3d} / {n} videos; "
                 f"{len(sess_sp & sess_tr)} / {len(sess_sp)} sessions")
        L.append(f"   signer (same session + side) in train     {sum(r['signer_seen'] for r in R):3d} / {n}")
    L += ["", f"val and test share sessions: {sorted({r['session'] for r in rows if r['split'] == 'val'} & {r['session'] for r in rows if r['split'] == 'test'}) or 'none'}"]
    return L


# ----------------------------------------------------------------------------- effect on the comparison
def counts_per_video(run, split, pred_root):
    from count_oracle_decoding import peaks_with_heights, tune_peaks_on_val
    from phrase_stats import boundary_counts, gold_starts_of, load_predictions
    from src.metrics import B_TAG
    tuned = tune_peaks_on_val(load_predictions(pred_root, run, "val"))
    probs, _, gold = load_predictions(pred_root, run, split)
    out = {}
    for v, p in probs.items():
        idx, h = peaks_with_heights(p[B_TAG], tuned["min_dist"], tuned["sigma"])
        out[v] = boundary_counts(gold_starts_of(gold[v]), np.sort(idx[h >= tuned["thr"]]), K)
    return out


def group_effect(runs_a, runs_b, split, keep, pred_root, n_boot, label):
    from phrase_stats import f1_from_counts, paired_bootstrap
    ca = [counts_per_video(r, split, pred_root) for r in runs_a]
    cb = [counts_per_video(r, split, pred_root) for r in runs_b]
    ca = [{v: c[v] for v in c if os.path.splitext(v)[0] in keep} for c in ca]
    cb = [{v: c[v] for v in c if os.path.splitext(v)[0] in keep} for c in cb]
    if not ca[0]:
        return [f"   {label}: no videos"]
    fa = np.mean([f1_from_counts(*np.sum(list(c.values()), axis=0)) for c in ca])
    fb = np.mean([f1_from_counts(*np.sum(list(c.values()), axis=0)) for c in cb])
    L = [f"   {label} ({len(ca[0])} videos): A {fa:.3f}  B {fb:.3f}"]
    if len(ca[0]) >= 3:
        r = paired_bootstrap(ca, cb, n_boot=n_boot, metric=f1_from_counts)
        lo, hi = r["ci95"]
        L[-1] += f"  A - B {r['diff']:+.3f} (95% CI: {lo:+.3f} to {hi:+.3f})"
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="dataset_splits.json")
    ap.add_argument("--index-html", default=os.path.join("data_download", "index.html"))
    ap.add_argument("--cmdi-dir", default=None)
    ap.add_argument("--a", nargs="+", default=None, help="arm A runs (e.g. proposed)")
    ap.add_argument("--b", nargs="+", default=None, help="arm B runs (e.g. Moryossef et al.)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "split_overlap"))
    args = ap.parse_args()

    with open(args.splits) as fh:
        splits = json.load(fh)
    sess, _ = load_sessions(args.index_html)
    rows = overlap_table(splits, sess)
    L = summary_lines(splits, rows, sess)

    if args.cmdi_dir:
        docs = [r["document"] for r in rows[:2]]
        L += ["", "CMDI participant fields (check that side A/B is the same informant across a session):"]
        L += cmdi_fields(args.cmdi_dir, docs)

    if args.a and args.b:
        from phrase_stats import PRED_ROOT
        pred_root = args.pred_root or PRED_ROOT
        test = [r for r in rows if r["split"] == "test"]
        seen = {os.path.splitext(r["video"])[0] for r in test if r["signer_seen"]}
        unseen = {os.path.splitext(r["video"])[0] for r in test if not r["signer_seen"]}
        L += ["", "4. Does the overlap explain the comparison? (boundary F1 +-5 frames, peak picking + threshold tuned",
              "   on val, mean of seeds; A = " + " ".join(args.a) + "; B = " + " ".join(args.b) + ")", "   TEST"]
        L += group_effect(args.a, args.b, "test", seen | unseen, pred_root, args.n_boot, "all test videos")
        L += group_effect(args.a, args.b, "test", seen, pred_root, args.n_boot, "signer seen in train")
        L += group_effect(args.a, args.b, "test", unseen, pred_root, args.n_boot, "signer not seen in train")
        val = [r for r in rows if r["split"] == "val"]
        allv = {os.path.splitext(r["video"])[0] for r in val}
        clean = {os.path.splitext(r["video"])[0] for r in val if not r["video_in_train"]}
        L += ["   VALIDATION (decoder tuned on all val videos, as in the main pipeline)"]
        L += group_effect(args.a, args.b, "val", allv, pred_root, args.n_boot, "all val videos")
        if clean != allv:
            L += group_effect(args.a, args.b, "val", clean, pred_root, args.n_boot,
                              "without val videos that are also in train")

    os.makedirs(args.out_dir, exist_ok=True)
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(args.out_dir, "per_video.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out_dir}/summary.txt, per_video.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())