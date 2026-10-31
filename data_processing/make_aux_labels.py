"""
make_aux_labels.py -- frame-level auxiliary targets for Idea B (stream-specific supervision).

  mouth    for the FACE stream: mouthing (1) / mouth gesture (2) / none (0) from the participant's
           Mundbild tier; the whole video is -100 (ignored) when that tier does not exist.
  signcat  for the HaMeR stream: one of 7 boundary-relevant sign classes per sign (palm-up,
           pointing, depicting, fingerspelling/number, other gesture, connective, other lexical),
           -100 outside signs. Classes and the connective list: src/aux_labels.py.

Everything is parsed with the functions of phrase_oracle_analysis.py (read_eaf, find_tier,
to_frames, gloss_base, gloss_category, mouth_kind, sign_segments, overlap_matrix) and the gloss
of a sign is picked with the oracle's rule (right / left Lexem_Geb tier with the larger overlap),
so the labels match the oracle exactly.

Output:
  processed_data/aux_labels/<video>.npz     mouth (T,) int8, signcat (T,) int8, T
                                            (T = length of the v2 phrase labels)
  processed_data/aux_labels/_meta.json      class names, connective list (candidates, train counts,
                                            kept), per-class TRAIN frame counts (class weights),
                                            per-split distributions, skipped videos

Run from the repo root:
  python data_processing/make_aux_labels.py
  python data_processing/make_aux_labels.py --annotations raw_data/annotations --min-connective-count 20
Re-run it whenever CONNECTIVE_CANDIDATES changes (the number of classes is fixed per task).
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.aux_labels import (CLASS_NAMES, CONNECTIVE_CANDIDATES, IGNORE, NUM_CLASSES,  # noqa: E402
                            mouth_id, signcat_id)


def strip_ext(v):
    return re.sub(r"\.(npy|pt)$", "", v)


def video_items(vid, args, eaf_cache, poa):
    """Parses one video: (T, signs, glosses, mouth annotations or None, reason)."""
    transcript, participant = vid.rsplit("_", 1)
    pp = os.path.join(args.phrase_labels, f"{vid}.npy")
    if not os.path.exists(pp):
        return None, "missing_phrase_labels"
    T = int(len(np.load(pp, mmap_mode="r")))
    item = {"T": T, "signs": np.zeros((0, 2), dtype=np.int64), "glosses": [], "mouth": None, "flags": []}
    ep = os.path.join(args.annotations, f"{transcript}.eaf")
    if not os.path.exists(ep):
        item["flags"].append("missing_eaf")
        return item, None
    if transcript not in eaf_cache:
        eaf_cache.clear()
        eaf_cache[transcript] = poa.read_eaf(ep)
    tiers = eaf_cache[transcript]

    m_tier = poa.find_tier(tiers, "Mundbild", f"_{participant}")
    has_m_tier = any(tid.startswith("Mundbild") and tid.endswith(f"_{participant}") for tid in tiers)
    if has_m_tier:
        m_fr, m_val = poa.to_frames(m_tier, args.fps)
        item["mouth"] = (m_fr, m_val)
    else:
        item["flags"].append("no_mundbild_tier")

    gp = os.path.join(args.gloss_labels, f"{vid}.npy")
    if not os.path.exists(gp):
        item["flags"].append("missing_gloss_labels")
        return item, None
    gloss = np.load(gp)
    if len(gloss) != T:
        item["flags"].append("gloss_length_mismatch")
    signs = poa.sign_segments(gloss)
    signs = signs[signs[:, 0] < T]
    signs[:, 1] = np.minimum(signs[:, 1], T - 1)
    r_fr, r_val = poa.to_frames(poa.find_tier(tiers, "Lexem_Geb", f"_r_{participant}"), args.fps)
    l_fr, l_val = poa.to_frames(poa.find_tier(tiers, "Lexem_Geb", f"_l_{participant}"), args.fps)
    if len(r_fr) + len(l_fr) == 0:
        item["flags"].append("no_gloss_tier")
        return item, None
    ov_r, ov_l = poa.overlap_matrix(signs, r_fr), poa.overlap_matrix(signs, l_fr)
    best_r = ov_r.max(1) if ov_r.shape[1] else np.zeros(len(signs))
    best_l = ov_l.max(1) if ov_l.shape[1] else np.zeros(len(signs))
    glosses = []
    for i in range(len(signs)):                                      # oracle rule (video_transitions)
        if best_r[i] >= best_l[i] and best_r[i] > 0:
            glosses.append(r_val[int(ov_r[i].argmax())])
        elif best_l[i] > 0:
            glosses.append(l_val[int(ov_l[i].argmax())])
        else:
            glosses.append("?")
    item["signs"], item["glosses"] = signs, glosses
    return item, None


def build_labels(item, connectives, poa):
    T = item["T"]
    mouth = np.full(T, IGNORE, dtype=np.int8)
    if item["mouth"] is not None:
        mouth[:] = 0
        m_fr, m_val = item["mouth"]
        for (a, b), v in zip(m_fr, m_val):
            a, b = max(int(a), 0), min(int(b), T - 1)
            if b >= a:
                mouth[a:b + 1] = mouth_id(poa.mouth_kind(v))
    signcat = np.full(T, IGNORE, dtype=np.int8)
    for (s, e), g in zip(item["signs"], item["glosses"]):
        signcat[s:e + 1] = signcat_id(poa.gloss_category(g), poa.gloss_base(g), connectives)
    return mouth, signcat


def main():
    ap = argparse.ArgumentParser(description="Auxiliary mouth / sign-class labels for Idea B.")
    ap.add_argument("--annotations", default=os.path.join("raw_data", "annotations"))
    ap.add_argument("--gloss-labels", default=os.path.join("processed_data", "BIO_tags"))
    ap.add_argument("--phrase-labels", default=os.path.join("processed_data", "BIO_tags_phrase_v2"))
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--out-dir", default=os.path.join("processed_data", "aux_labels"))
    ap.add_argument("--min-connective-count", type=int, default=20,
                    help="drop connective candidates with fewer TRAIN occurrences (signs)")
    ap.add_argument("--max-videos", type=int, default=None, help="per split, for a quick test")
    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    import phrase_oracle_analysis as poa       # sklearn / scipy / tqdm: imported only here

    with open(args.split_file) as f:
        sp = json.load(f)
    splits = {}
    for split in ("train", "val", "test"):
        vids = sorted(strip_ext(v) for v in sp.get(split, []))      # sorted -> same transcript adjacent
        splits[split] = vids[: args.max_videos] if args.max_videos else vids

    items, skips, flags, eaf_cache = {}, Counter(), Counter(), {}
    for split, vids in splits.items():
        for v in vids:
            it, why = video_items(v, args, eaf_cache, poa)
            if it is None:
                skips[(split, why)] += 1
                continue
            items[v] = it
            for fl in it["flags"]:
                flags[(split, fl)] += 1

    # ---- connective list: train occurrences of each candidate (lexical signs only)
    cand = set(CONNECTIVE_CANDIDATES)
    train_counts = Counter()
    for v in splits["train"]:
        for g in items.get(v, {"glosses": []})["glosses"]:
            b = poa.gloss_base(g)
            if b in cand and poa.gloss_category(g) == "lexical":
                train_counts[b] += 1
    kept = [c for c in CONNECTIVE_CANDIDATES if train_counts[c] >= args.min_connective_count]
    dropped = [c for c in CONNECTIVE_CANDIDATES if c not in kept]
    print("Connective candidates (train occurrences): " +
          ", ".join(f"{c} {train_counts[c]}" for c in CONNECTIVE_CANDIDATES))
    print(f"  kept {len(kept)}: {kept}")
    if dropped:
        print(f"  dropped (< {args.min_connective_count} train occurrences): {dropped}")

    # ---- labels
    os.makedirs(args.out_dir, exist_ok=True)
    dist = {s: {t: np.zeros(NUM_CLASSES[t], dtype=np.int64) for t in CLASS_NAMES} for s in splits}
    ignored_mouth = Counter()
    n_written = Counter()
    for split, vids in splits.items():
        for v in vids:
            if v not in items:
                continue
            mouth, signcat = build_labels(items[v], set(kept), poa)
            np.savez_compressed(os.path.join(args.out_dir, f"{v}.npz"), mouth=mouth, signcat=signcat,
                                T=np.int64(items[v]["T"]))
            n_written[split] += 1
            if (mouth == IGNORE).all():
                ignored_mouth[split] += 1
            for t, arr in (("mouth", mouth), ("signcat", signcat)):
                ok = arr[arr != IGNORE].astype(np.int64)
                dist[split][t] += np.bincount(ok, minlength=NUM_CLASSES[t])[:NUM_CLASSES[t]]

    print()
    for split in splits:
        n = n_written[split]
        print(f"[{split}] {n} videos; mouth ignored (no Mundbild tier / no EAF) in "
              f"{ignored_mouth[split]} ({ignored_mouth[split] / max(n, 1):.1%})")
        for t in CLASS_NAMES:
            tot = max(int(dist[split][t].sum()), 1)
            print(f"   {t:8s} " + "  ".join(f"{name} {c / tot:.3f}" for name, c in zip(CLASS_NAMES[t], dist[split][t])))
    if skips or flags:
        print("skipped videos:", dict(skips), " flags:", dict(flags))

    meta = {
        "tasks": list(CLASS_NAMES), "class_names": CLASS_NAMES, "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE,
        "connective_source": "positive gloss_cur= weights >= 0.9 in evaluation_phrase/oracle/"
                             "oracle_lexical_weights.csv (lexical oracle fitted on TRAIN only)",
        "connective_candidates": CONNECTIVE_CANDIDATES,
        "connective_train_counts": {c: int(train_counts[c]) for c in CONNECTIVE_CANDIDATES},
        "min_connective_count": args.min_connective_count,
        "connectives": kept,
        "train_class_frames": {t: dist["train"][t].tolist() for t in CLASS_NAMES},
        "class_frames_by_split": {s: {t: d[t].tolist() for t in CLASS_NAMES} for s, d in dist.items()},
        "videos_written": dict(n_written),
        "videos_mouth_ignored": dict(ignored_mouth),
        "skipped": {f"{s}:{w}": n for (s, w), n in skips.items()},
        "flags": {f"{s}:{w}": n for (s, w), n in flags.items()},
        "settings": {"annotations": args.annotations, "gloss_labels": args.gloss_labels,
                     "phrase_labels": args.phrase_labels, "fps": args.fps, "split_file": args.split_file},
    }
    with open(os.path.join(args.out_dir, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {sum(n_written.values())} label files + _meta.json to {args.out_dir}")


if __name__ == "__main__":
    main()
