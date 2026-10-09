"""
create_phrase_bio_tags_v2.py -- phrase-level BIO tags with CORRECTED boundaries.

Why a v2:
  The translation tiers (Deutsche_Übersetzung / Translation_into_English, identical
  timings) are annotated roughly: a span usually starts some frames BEFORE its first
  sign and ends AFTER its last sign, and consecutive spans are made to touch. v1
  (create_phrase_bio_tags.py) searched for gloss activity in [start - 15, end + 15],
  i.e. it widened the rough span further. When two rough spans touch, that window
  reaches into the neighbouring phrase and grabs its first/last sign, so phrases
  became directly adjacent ("continuous") far more often than in the actual signing.

v2 rule (as in the 2023 paper: "from the start of its first sign to the end of its
last sign"): every SIGN (from the verified gloss labels, processed_data/BIO_tags) is
assigned to at most one translation span, and a phrase runs from the first frame of
its first assigned sign to the last frame of its last assigned sign. Nothing outside
the rough span is ever searched, so a phrase can never take a sign from its neighbour.

  --method overlap (default): a sign belongs to the span it overlaps most (ties: the
      span containing the sign's midpoint). Signs fully inside a span are assigned to
      it, exactly as with "inward"; a sign that straddles a rough boundary goes to the
      span holding most of it instead of being dropped.
  --method inward: only signs lying COMPLETELY inside a span are assigned to it
      (start: move forward from the rough start to the first sign that begins inside;
      end: move backward from the rough end to the last sign that ends inside).
      Straddling signs are left unassigned (Outside).

Output: processed_data/BIO_tags_phrase_v2/<video>_<participant>.npy, same convention
as every other label file here: 0 = Outside, 1 = Inside, 2 = Begin. v1 labels are not
touched. Run from the repo root:
    python data_processing/create_phrase_bio_tags_v2.py
    python data_processing/create_phrase_bio_tags_v2.py --method inward --output-dir processed_data/BIO_tags_phrase_v2_inward
"""
import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

try:
    from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
except ImportError as e:
    raise ImportError(
        "Could not import get_elan_sentences from sign_language_datasets (same import as "
        "create_phrase_bio_tags.py -- update the path there and here if your version differs)."
    ) from e

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(PARENT_DIR, "raw_data", "annotations")
VIDEOS_DIR = os.path.join(PARENT_DIR, "raw_data", "videos")
GLOSS_LABELS_DIR = "processed_data/BIO_tags"

O_TAG, I_TAG, B_TAG = 0, 1, 2


# ==============================================================================
# Helpers
# ==============================================================================
def get_video_fps(video_id):
    """Actual fps of the raw video (same lookup as create_phrase_bio_tags.py)."""
    for suffix in ("_1a1", "_1b1"):
        for ext in (".mp4", ".avi", ".mov"):
            path = os.path.join(VIDEOS_DIR, f"{video_id}{suffix}{ext}")
            if os.path.exists(path):
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0:
                    return fps
    return None


def sign_segments(gloss_labels):
    """Signs from the gloss BIO array as [(start, end_inclusive), ...]. B starts a sign, O ends it."""
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
        elif start == -1:  # I without an open sign: start one (robust to odd label files)
            start = i
    if start != -1:
        segs.append((start, len(gloss_labels) - 1))
    return segs


def assign_signs(signs, spans, method):
    """
    signs: [(s, e)] inclusive; spans: [(rs, re)] with re EXCLUSIVE (rough translation spans).
    Returns (assignment list: span index or -1 per sign, number of straddling signs).
    """
    assignment = [-1] * len(signs)
    straddling = 0
    for si, (s, e) in enumerate(signs):
        overlaps = []
        for pi, (rs, re_) in enumerate(spans):
            ov = min(e + 1, re_) - max(s, rs)
            if ov > 0:
                overlaps.append((ov, pi))
        if not overlaps:
            continue
        fully_inside = [pi for _, pi in overlaps if spans[pi][0] <= s and e + 1 <= spans[pi][1]]
        if not fully_inside:
            straddling += 1
        if method == "inward":
            if fully_inside:
                assignment[si] = fully_inside[0]
            continue
        # overlap: largest overlap; tie -> span containing the sign's midpoint
        best = max(ov for ov, _ in overlaps)
        tied = [pi for ov, pi in overlaps if ov == best]
        if len(tied) > 1:
            mid = (s + e) / 2.0
            containing = [pi for pi in tied if spans[pi][0] <= mid < spans[pi][1]]
            tied = containing or tied
        assignment[si] = tied[0]
    return assignment, straddling


def build_labels(phrases, num_frames):
    labels = np.zeros(num_frames, dtype=np.int64)
    for s, e in phrases:  # inclusive end
        labels[s] = B_TAG
        if e > s:
            labels[s + 1:e + 1] = I_TAG
    return labels


# ==============================================================================
# One participant-video
# ==============================================================================
def process_one(video_id, participant, eaf_path, method):
    gloss_path = os.path.join(GLOSS_LABELS_DIR, f"{video_id}_{participant}.npy")
    if not os.path.exists(gloss_path):
        return None, "missing_gloss_labels", None
    gloss = np.load(gloss_path)
    T = len(gloss)

    fps = get_video_fps(video_id)
    if fps is None:
        return None, "missing_video_or_fps", None
    try:
        sentences = list(get_elan_sentences(eaf_path))
    except Exception as e:
        return None, f"elan_parse_error: {str(e)[:100]}", None

    spans = []
    for sent in sentences:
        if sent.get("participant") != participant:
            continue
        rs = int(round(sent["start"] / 1000.0 * fps))
        re_ = int(round(sent["end"] / 1000.0 * fps))
        rs, re_ = max(0, rs), min(T, re_)
        if re_ > rs:
            spans.append((rs, re_))
    spans.sort()
    if not spans:
        return None, "no_valid_phrases", None

    signs = sign_segments(gloss)
    assignment, straddling = assign_signs(signs, spans, method)

    per_span = {}
    for (s, e), pi in zip(signs, assignment):
        if pi >= 0:
            lo, hi = per_span.get(pi, (s, e))
            per_span[pi] = (min(lo, s), max(hi, e))
    phrases = sorted(per_span.values())
    if not phrases:
        return None, "no_valid_phrases", None

    # Safety: phrases are disjoint by construction (signs are disjoint and each goes to
    # one span) unless spans themselves interleave; clip any overlap to be sure.
    clipped = 0
    fixed = [phrases[0]]
    for s, e in phrases[1:]:
        ps, pe = fixed[-1]
        if s <= pe:
            clipped += 1
            s = pe + 1
            if s > e:
                continue
        fixed.append((s, e))
    phrases = fixed

    gaps = [phrases[i + 1][0] - phrases[i][1] - 1 for i in range(len(phrases) - 1)]
    start_shift = [per_span[pi][0] - spans[pi][0] for pi in per_span]
    stats = {
        "spans": len(spans),
        "phrases": len(phrases),
        "spans_without_signs": len(spans) - len(per_span),
        "signs": len(signs),
        "signs_assigned": sum(1 for a in assignment if a >= 0),
        "signs_outside_all_spans": sum(1 for (s, e) in signs
                                       if not any(min(e + 1, re_) > max(s, rs) for rs, re_ in spans)),
        "signs_straddling": straddling,
        "overlaps_clipped": clipped,
        "gaps": gaps,
        "start_shift": start_shift,
    }
    return build_labels(phrases, T), None, stats


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phrase BIO tags v2: snap each translation span to the "
                                                 "signs it contains (no outward search).")
    parser.add_argument("--method", choices=["overlap", "inward"], default="overlap",
                        help="overlap (default): straddling signs go to the span covering most of them; "
                             "inward: only signs fully inside a span count.")
    parser.add_argument("--output-dir", default="processed_data/BIO_tags_phrase_v2")
    parser.add_argument("--short-gap", type=int, default=12,
                        help="Gap (frames) up to which a pause between phrases counts as short in the report.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild files that already exist.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    eaf_files = sorted(f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".eaf"))
    print(f"Method: {args.method} | output: {args.output_dir} | {len(eaf_files)} annotation files")

    skip_counts = {}
    totals = {"spans": 0, "phrases": 0, "spans_without_signs": 0, "signs": 0, "signs_assigned": 0,
              "signs_outside_all_spans": 0, "signs_straddling": 0, "overlaps_clipped": 0}
    all_gaps, all_shift, written = [], [], 0

    for fname in tqdm(eaf_files, desc="Building phrase BIO tags v2"):
        video_id = fname[:-4]
        for participant in ("A", "B"):
            out_path = os.path.join(args.output_dir, f"{video_id}_{participant}.npy")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            labels, reason, stats = process_one(video_id, participant,
                                                os.path.join(ANNOTATIONS_DIR, fname), args.method)
            if labels is None:
                key = "elan_parse_error" if reason.startswith("elan_parse_error") else reason
                skip_counts[key] = skip_counts.get(key, 0) + 1
                continue
            np.save(out_path, labels)
            written += 1
            for k in totals:
                totals[k] += stats[k]
            all_gaps += stats["gaps"]
            all_shift += stats["start_shift"]

    gaps = np.array(all_gaps)
    report = {"method": args.method, "files_written": written, "skips": skip_counts, **totals}
    if len(gaps):
        report["gap_between_phrases"] = {
            "continuous_0": int((gaps == 0).sum()),
            f"short_1_{args.short_gap}": int(((gaps >= 1) & (gaps <= args.short_gap)).sum()),
            f"long_over_{args.short_gap}": int((gaps > args.short_gap).sum()),
            "median_gap": float(np.median(gaps)),
        }
    if all_shift:
        report["phrase_start_minus_rough_start_frames"] = {
            "median": float(np.median(all_shift)), "p10": float(np.percentile(all_shift, 10)),
            "p90": float(np.percentile(all_shift, 90)),
        }

    print(f"\nDone. {written} files written to {args.output_dir}. Skips: {skip_counts}")
    if written:
        print(f"Spans: {totals['spans']} -> phrases: {totals['phrases']} "
              f"({totals['spans_without_signs']} spans contained no sign and were dropped)")
        print(f"Signs: {totals['signs']} | assigned: {totals['signs_assigned']} | outside every span: "
              f"{totals['signs_outside_all_spans']} | straddling a rough boundary: {totals['signs_straddling']}"
              + (" (left Outside with --method inward)" if args.method == "inward" else ""))
        if len(gaps):
            g = report["gap_between_phrases"]
            n = len(gaps)
            print(f"Gaps between consecutive phrases: continuous {g['continuous_0']} ({g['continuous_0']/n:.1%}) | "
                  f"short {g[f'short_1_{args.short_gap}']} ({g[f'short_1_{args.short_gap}']/n:.1%}) | "
                  f"long {g[f'long_over_{args.short_gap}']} ({g[f'long_over_{args.short_gap}']/n:.1%}) | "
                  f"median {g['median_gap']:.0f} frames")
        if all_shift:
            sh = report["phrase_start_minus_rough_start_frames"]
            print(f"Phrase start minus rough translation start: median {sh['median']:+.0f} frames "
                  f"(10%: {sh['p10']:+.0f}, 90%: {sh['p90']:+.0f})")
        if totals["overlaps_clipped"]:
            print(f"⚠️  {totals['overlaps_clipped']} overlapping phrases were clipped (interleaved translation spans).")
    report_path = os.path.join(args.output_dir, "_build_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()