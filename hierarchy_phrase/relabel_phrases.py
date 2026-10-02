"""
hierarchy_phrase/relabel_phrases.py -- OPT-IN: rebuild phrase labels so every phrase edge is a SIGN edge.

WHY. processed_data/BIO_tags_phrase was built by create_phrase_bio_tags.py, whose snap_to_gloss_boundaries takes the first / last
non-Outside FRAME inside [raw_start-15, raw_end+15]. Translation units are contiguous, so each boundary is cut at a window edge
15 frames to either side -- usually in the middle of a sign, and pulling in the neighbouring phrase's sign tail. Measured on the real
labels (data_audit.py): only 57% of phrase starts and 30% of ends are on a sign edge, 42% of starts and 42% of ends are strictly
inside a sign. The paper's rule is "from the start of the first sign to the end of the last sign", which this script implements:
every sign goes to the translation span it overlaps most (or the nearest span within the tolerance), and a phrase runs from its first
sign's start to its last sign's end. Phrases therefore cannot overlap and always start/end on sign edges.

USAGE (from the project root or anywhere; needs the same environment as create_phrase_bio_tags.py: sign_language_datasets, cv2):
    python relabel_phrases.py                      # writes processed_data/BIO_tags_phrase_signaligned/
    HP_PHRASE_DIR=BIO_tags_phrase_signaligned python data_audit.py          # re-run the audit on the new labels
    HP_PHRASE_DIR=BIO_tags_phrase_signaligned python stage_a.py train ...   # train / export / evaluate on them
The existing labels are NOT touched. Report results on both label sets: the old one is the project's historical definition, the new
one is the paper's wording; they are not interchangeable gold standards.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from segments import bio_to_segments  # noqa: E402

OUT_NAME = "BIO_tags_phrase_signaligned"


def sign_aligned_spans(raw_spans, sign_segs, tolerance=15):
    """raw_spans [(start, end)] of translation units (any order, may overlap), sign_segs sorted [(start, end)].
    Returns (spans, n_unassigned_signs): one (first sign start, last sign end) per translation span that received a sign,
    sorted and non-overlapping."""
    spans = list(raw_spans)
    assigned = [[] for _ in spans]
    dropped = 0
    for k, (s, e) in enumerate(sign_segs):
        best, best_ov = -1, 0
        for i, (rs, re_) in enumerate(spans):
            ov = min(e, re_) - max(s, rs)
            if ov > best_ov:
                best, best_ov = i, ov
        if best < 0:                                       # no overlap: attach to the nearest span if close enough
            dist = [max(rs - e, s - re_, 0) for rs, re_ in spans]
            if dist and min(dist) <= tolerance:
                best = int(np.argmin(dist))
        if best >= 0:
            assigned[best].append(k)
        else:
            dropped += 1
    out = sorted((sign_segs[ks[0]][0], sign_segs[ks[-1]][1]) for ks in assigned if ks)
    fixed, prev_end = [], -1
    for s, e in out:                                       # interleaved assignments are rare; clip rather than overlap
        s = max(s, prev_end)
        if e > s:
            fixed.append((s, e)); prev_end = e
    return fixed, dropped


def main():
    os.chdir(PROJECT_ROOT)                                 # the builder uses paths relative to the project root
    sys.path.insert(0, PROJECT_ROOT)
    import create_phrase_bio_tags as B                     # reuse its EAF parsing and fps lookup exactly
    out_dir = os.path.join("processed_data", OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    eafs = sorted(f for f in os.listdir(B.ANNOTATIONS_DIR) if f.endswith(".eaf"))
    done, skipped = 0, {}
    for fname in eafs:
        vid = fname[:-4]
        for part in ("A", "B"):
            out_path = os.path.join(out_dir, f"{vid}_{part}.npy")
            if os.path.exists(out_path):
                continue
            gpath = os.path.join(B.GLOSS_LABELS_DIR, f"{vid}_{part}.npy")
            if not os.path.exists(gpath):
                skipped["missing_gloss_labels"] = skipped.get("missing_gloss_labels", 0) + 1
                continue
            gloss = np.load(gpath)
            fps = B.get_video_fps(vid)
            if fps is None:
                skipped["missing_video_or_fps"] = skipped.get("missing_video_or_fps", 0) + 1
                continue
            try:
                sentences = list(B.get_elan_sentences(os.path.join(B.ANNOTATIONS_DIR, fname)))
            except Exception:
                skipped["elan_parse_error"] = skipped.get("elan_parse_error", 0) + 1
                continue
            raw = [(int(round(s["start"] / 1000.0 * fps)), int(round(s["end"] / 1000.0 * fps)))
                   for s in sentences if s.get("participant") == part]
            spans, _ = sign_aligned_spans(raw, bio_to_segments(gloss), tolerance=B.SNAP_TOLERANCE_FRAMES)
            if not spans:
                skipped["no_valid_phrases"] = skipped.get("no_valid_phrases", 0) + 1
                continue
            np.save(out_path, B.build_phrase_bio_array(spans, len(gloss)))
            done += 1
    print(f"wrote {done} label files to {out_dir}; skipped: {skipped}")
    print(f"next: HP_PHRASE_DIR={OUT_NAME} python data_audit.py")


if __name__ == "__main__":
    main()