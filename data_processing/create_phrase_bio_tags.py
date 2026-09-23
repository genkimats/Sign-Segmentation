"""
Builds PHRASE-level BIO tags (as opposed to the existing SIGN/gloss-level
processed_data/BIO_tags/), following the exact methodology described in the
2023 paper this whole pipeline is built on: phrase segments come from the
corpus's German/English TRANSLATION spans, corrected to snap to the actual
first/last gloss boundary they contain -- "the segment assumed to span from
the start of its first sign to the end of its last sign, correcting
imprecise annotation" (the translation tier's own timestamps are looser than
the frame-precise gloss annotations).

Uses get_elan_sentences() from sign_language_datasets (the same package
create_splits.py already depends on) to extract raw translation spans
directly from the same .eaf files clean_data.py processes -- no new
low-level ELAN-tier parsing needed.

Output format matches processed_data/BIO_tags/ EXACTLY (per-frame int array,
0=Outside, 1=Begin, 2=Inside), just in a separate directory
(processed_data/BIO_tags_phrase/) -- the EXISTING SignSegmentationDataset
class works completely unchanged against phrase labels, just point
labels_dir at the new folder. No dataset.py changes needed; all feature
extraction (keypoints, kinematic, HaMeR, DINOv2) is fully reused as-is,
since none of that depends on which label granularity you're training
against.

IMPORTANT -- verify before trusting a full run (this could not be tested
against your actual corpus/installed package in the environment this was
written in):
  1. The import path below -- get_elan_sentences's exact module location can
     differ between sign_language_datasets versions. If the import fails,
     run `pip show sign_language_datasets` and check its actual dgs_corpus/
     folder layout, then update the import.
  2. The "participant" field format printed in the first sanity check --
     this script assumes it matches "A"/"B" (same as clean_data.py's
     PARTICIPANT_TIERS convention). If the printed raw values look
     different (e.g. lowercase, or a different scheme entirely), fix the
     comparison in process_one_video() before trusting anything downstream.
  3. The per-video sanity-check printouts -- phrase counts should be much
     LOWER and spans much LONGER than the existing gloss-level labels for
     the same video (phrases contain multiple signs). Near-zero or
     near-100%-inside-a-phrase values indicate a frame-alignment or
     unit-conversion bug, not a real result.
"""
import os
import numpy as np
import cv2
from tqdm import tqdm

try:
    from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
except ImportError as e:
    raise ImportError(
        "Could not import get_elan_sentences from sign_language_datasets. This function's "
        "exact module path can differ between package versions -- run "
        "`pip show sign_language_datasets`, check its actual dgs_corpus/ folder layout, "
        "and update the import above if the path has changed."
    ) from e

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(PARENT_DIR, "raw_data", "annotations")
VIDEOS_DIR = os.path.join(PARENT_DIR, "raw_data", "videos")
GLOSS_LABELS_DIR = "processed_data/BIO_tags"          # existing, already-verified sign-level labels
OUTPUT_DIR = "processed_data/BIO_tags_phrase"          # new phrase-level labels go here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How much slack (in frames) to allow when matching a raw translation span
# against gloss frames for the "snap to first/last gloss" correction -- the
# translation tier's timing can be looser than the gloss tier's, so a gloss
# starting just barely outside the raw span is still very likely the genuine
# first sign of that phrase, not noise.
SNAP_TOLERANCE_FRAMES = 15

# Print a detailed breakdown for this many videos, so you can eyeball
# correctness before trusting a full corpus run.
SANITY_CHECK_COUNT = 5


def get_video_fps(video_id):
    """Reads the ACTUAL fps from the raw video file, matching how the rest of
    this pipeline (extract_poses.py etc.) handles frame alignment -- not a
    fixed assumed corpus-wide fps."""
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


def snap_to_gloss_boundaries(raw_start_frame, raw_end_frame, gloss_labels, tolerance):
    """
    Implements the paper's exact correction: "segment assumed to span from
    the start of its first sign to the end of its last sign." Finds the
    first and last NON-OUTSIDE frame (gloss B or I) within
    [raw_start-tolerance, raw_end+tolerance] of the existing, already-
    verified gloss BIO array, and uses THOSE as the corrected phrase bounds.
    Returns None if no gloss activity is found in that window at all (a
    translation span with no corresponding signs -- skip it, don't guess).
    """
    num_frames = len(gloss_labels)
    search_start = max(0, raw_start_frame - tolerance)
    search_end = min(num_frames, raw_end_frame + tolerance)
    if search_start >= search_end:
        return None

    window = gloss_labels[search_start:search_end]
    nonzero_positions = np.where(window != 0)[0]  # 0 = Outside
    if len(nonzero_positions) == 0:
        return None

    corrected_start = search_start + int(nonzero_positions[0])
    corrected_end = search_start + int(nonzero_positions[-1]) + 1  # +1: end is exclusive
    return corrected_start, corrected_end


def build_phrase_bio_array(phrase_spans, num_frames):
    """Builds a per-frame int array (0=Outside, 1=Begin, 2=Inside), matching
    processed_data/BIO_tags/'s exact existing format, from a list of
    (start_frame, end_frame) phrase spans (end EXCLUSIVE)."""
    labels = np.zeros(num_frames, dtype=np.int64)
    for start, end in phrase_spans:
        if start >= end or start < 0 or end > num_frames:
            continue
        labels[start] = 1              # Begin
        if end - start > 1:
            labels[start + 1:end] = 2  # Inside
    return labels


def process_one_video(video_id, participant, eaf_path, print_raw_participants=False):
    gloss_path = os.path.join(GLOSS_LABELS_DIR, f"{video_id}_{participant}.npy")
    if not os.path.exists(gloss_path):
        return None, "missing_gloss_labels"

    gloss_labels = np.load(gloss_path)
    num_frames = len(gloss_labels)

    fps = get_video_fps(video_id)
    if fps is None:
        return None, "missing_video_or_fps"

    try:
        sentences = list(get_elan_sentences(eaf_path))
    except Exception as e:
        return None, f"elan_parse_error: {str(e)[:100]}"

    if print_raw_participants and sentences:
        raw_values = sorted(set(s.get("participant") for s in sentences))
        print(f"  [DEBUG] raw 'participant' values seen in this file: {raw_values} "
              f"-- confirm this matches the 'A'/'B' comparison below before trusting output")

    phrase_spans = []
    for sentence in sentences:
        if sentence.get("participant") != participant:
            continue
        start_ms, end_ms = sentence["start"], sentence["end"]
        raw_start_frame = int(round(start_ms / 1000.0 * fps))
        raw_end_frame = int(round(end_ms / 1000.0 * fps))

        corrected = snap_to_gloss_boundaries(raw_start_frame, raw_end_frame, gloss_labels, SNAP_TOLERANCE_FRAMES)
        if corrected is None:
            continue  # translation span with no corresponding gloss activity -- skip, don't guess
        phrase_spans.append(corrected)

    if not phrase_spans:
        return None, "no_valid_phrases"

    phrase_labels = build_phrase_bio_array(phrase_spans, num_frames)
    return phrase_labels, None


def main():
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Directory not found: {ANNOTATIONS_DIR}")
        return

    eaf_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".eaf")]
    existing_ids = {f[:-4] for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")}

    all_out_ids = [f"{f[:-4]}_{p}" for f in eaf_files for p in ("A", "B")]
    skipped_already_done = sum(1 for oid in all_out_ids if oid in existing_ids)
    print(f"Found {len(eaf_files)} annotation files ({len(all_out_ids)} participant-videos), "
          f"{skipped_already_done} already extracted -- "
          f"processing the remaining {len(all_out_ids) - skipped_already_done}.")

    skip_counts = {"missing_gloss_labels": 0, "missing_video_or_fps": 0,
                   "no_valid_phrases": 0, "elan_parse_error": 0}
    processed_count = 0
    sanity_checked = 0
    debug_printed_for_file = False

    for fname in tqdm(eaf_files, desc="Building phrase BIO tags"):
        video_id = fname[:-4]
        eaf_path = os.path.join(ANNOTATIONS_DIR, fname)

        for participant in ("A", "B"):
            out_id = f"{video_id}_{participant}"
            if out_id in existing_ids:
                continue

            result, skip_reason = process_one_video(
                video_id, participant, eaf_path,
                print_raw_participants=not debug_printed_for_file
            )
            debug_printed_for_file = True

            if result is None:
                bucket = "elan_parse_error" if skip_reason.startswith("elan_parse_error") else skip_reason
                skip_counts[bucket] = skip_counts.get(bucket, 0) + 1
                continue

            np.save(os.path.join(OUTPUT_DIR, f"{out_id}.npy"), result)
            processed_count += 1

            if sanity_checked < SANITY_CHECK_COUNT:
                num_phrases = int((result == 1).sum())
                total_frames = len(result)
                inside_frac = float((result == 2).sum()) / total_frames if total_frames else 0
                print(f"\n[SANITY CHECK] {out_id}: {num_phrases} phrases detected, "
                      f"{total_frames} total frames, {inside_frac * 100:.1f}% of frames inside a phrase")
                sanity_checked += 1

    print(f"\nDone. {processed_count} phrase-label files written to {OUTPUT_DIR}")
    print(f"Skip reasons: {skip_counts}")
    print(f"\nBefore running/trusting a full training run: check the {SANITY_CHECK_COUNT} "
          f"sanity-check videos above. Phrase counts should be much LOWER and spans much "
          f"LONGER than the equivalent gloss-level labels for the same video (a phrase "
          f"contains multiple signs). Near-zero or near-100%-inside values indicate a "
          f"frame-alignment or unit-conversion bug, not a real result.")


if __name__ == "__main__":
    main()