"""
Extracts per-frame GLOSS IDENTITY labels (which specific sign is active at
each frame), aligned with the EXISTING processed_data/BIO_tags/ segmentation
labels. This is the data multi-task training needs: an auxiliary
classification target (which gloss) alongside the existing BIO tags (where
are the boundaries).

Reads the SAME "Sign_r_A"/"Sign_r_B" tiers that (per this project's existing
pipeline, confirmed directly against clean_data.py's own pympi usage) produced
processed_data/BIO_tags/ in the first place -- so gloss identity and BIO
boundaries come from the same underlying annotation, and should align closely
by construction. A sanity check against the EXISTING BIO_tags is built in
specifically to catch any mismatch before you trust this for training.

Vocabulary is built from the TRAIN split ONLY (standard practice -- avoids
leaking val/test gloss identity into label encoding). Val/test glosses not
seen in training map to the UNK id and are excluded from the gloss loss
during multi-task training (nothing meaningful to predict them as), while
still contributing normally to the BIO segmentation loss.

Output: processed_data/gloss_ids/{id}.npy -- one integer per frame:
  -1                     = Outside (background, no gloss active)
  0 .. vocab_size-2      = the specific gloss active at that frame
  vocab_size-1 (UNK id)  = a gloss that exists in the corpus but wasn't
                           frequent enough in train, or wasn't seen in train
                           at all, to get its own class
Also writes gloss_vocab.json (gloss string -> id mapping) alongside it.

This file lives in Sign-Segmentation/multitask_learning/. All paths are
computed relative to THIS FILE's own location, not the terminal's current
directory -- run it from wherever you like.

VERIFY BEFORE TRUSTING A FULL RUN (could not be tested against your actual
corpus/pympi installation in the environment this was written in):
  1. The sanity-check agreement percentages printed for the first few
     videos -- gloss-active frames should closely match BIO-tags-active
     frames (both come from the same Sign_r_A/B tiers, so high agreement is
     expected; if it's notably low, something about the tier reading or
     frame alignment doesn't match what produced the existing BIO_tags).
  2. The vocabulary size and MIN_GLOSS_FREQUENCY tradeoff printed in Pass 1
     -- adjust MIN_GLOSS_FREQUENCY if the kept-vocabulary size looks too
     small (losing too many real signs to UNK) or too large (many classes
     with only a handful of examples, prone to noisy gradients).
"""
import os
import json
import pympi
import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

ANNOTATIONS_DIR = os.path.join(_PROJECT_ROOT, "raw_data", "annotations")
VIDEOS_DIR = os.path.join(_PROJECT_ROOT, "raw_data", "videos")
GLOSS_LABELS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "BIO_tags")  # existing, for alignment
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "gloss_ids")
SPLIT_FILE = os.path.join(_PROJECT_ROOT, "dataset_splits.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTICIPANT_TIERS = {"A": "Sign_r_A", "B": "Sign_r_B"}

# Glosses seen fewer than this many times in the TRAIN split collapse into a
# shared UNK class -- a large sign-language corpus has a long tail of rare
# signs; giving each its own class with only a handful of examples is more
# likely to inject noisy gradients than useful signal.
MIN_GLOSS_FREQUENCY = 10

SANITY_CHECK_COUNT = 5


def get_video_fps(video_id):
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


def extract_gloss_spans(eaf, participant, fps):
    """Returns a list of (start_frame, end_frame, gloss_string) tuples for
    one participant's Sign tier, end EXCLUSIVE. Unpacks defensively in case
    this pympi version returns a 4th element (e.g. an svg ref) alongside
    (start_ms, end_ms, value)."""
    tier_name = PARTICIPANT_TIERS[participant]
    if tier_name not in eaf.get_tier_names():
        return []

    spans = []
    for ann in eaf.get_annotation_data_for_tier(tier_name):
        start_ms, end_ms, value = ann[0], ann[1], ann[2]
        if not value or not str(value).strip():
            continue
        start_frame = int(round(start_ms / 1000.0 * fps))
        end_frame = int(round(end_ms / 1000.0 * fps))
        if end_frame > start_frame:
            spans.append((start_frame, end_frame, str(value).strip()))
    return spans


def main():
    with open(SPLIT_FILE) as f:
        splits = json.load(f)
    train_ids = {v.replace(".npy", "").replace(".pt", "") for v in splits.get("train", [])}

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Directory not found: {ANNOTATIONS_DIR}")
        return

    eaf_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".eaf")]
    fps_cache = {}

    def cached_fps(video_id):
        if video_id not in fps_cache:
            fps_cache[video_id] = get_video_fps(video_id)
        return fps_cache[video_id]

    # --- Pass 1: build the vocabulary from TRAIN split only ---
    gloss_counts = Counter()
    print("Pass 1/2: building gloss vocabulary from the train split...")
    for fname in tqdm(eaf_files, desc="Scanning for vocabulary"):
        video_id = fname[:-4]
        eaf_path = os.path.join(ANNOTATIONS_DIR, fname)

        relevant_participants = [p for p in ("A", "B") if f"{video_id}_{p}" in train_ids]
        if not relevant_participants:
            continue

        fps = cached_fps(video_id)
        if fps is None:
            continue

        try:
            eaf = pympi.Elan.Eaf(eaf_path)
        except Exception:
            continue

        for participant in relevant_participants:
            for _, _, gloss in extract_gloss_spans(eaf, participant, fps):
                gloss_counts[gloss] += 1

    kept_glosses = sorted(g for g, c in gloss_counts.items() if c >= MIN_GLOSS_FREQUENCY)
    gloss_to_id = {g: i for i, g in enumerate(kept_glosses)}
    unk_id = len(kept_glosses)
    vocab_size = len(kept_glosses) + 1  # +1 for UNK

    print(f"Vocabulary: {len(gloss_counts)} distinct glosses seen in train, "
          f"{len(kept_glosses)} kept (>= {MIN_GLOSS_FREQUENCY} occurrences), "
          f"rest collapse to UNK (id={unk_id})")

    with open(os.path.join(OUTPUT_DIR, "gloss_vocab.json"), "w") as f:
        json.dump({"gloss_to_id": gloss_to_id, "unk_id": unk_id, "vocab_size": vocab_size,
                    "min_gloss_frequency": MIN_GLOSS_FREQUENCY}, f, indent=2)

    # --- Pass 2: build per-frame gloss-id arrays for ALL splits ---
    print("\nPass 2/2: building per-frame gloss-id arrays...")
    sanity_checked = 0
    skip_counts = {"missing_bio_tags": 0, "missing_video_or_fps": 0, "eaf_parse_error": 0}
    processed_count = 0

    for fname in tqdm(eaf_files, desc="Building gloss-id arrays"):
        video_id = fname[:-4]
        eaf_path = os.path.join(ANNOTATIONS_DIR, fname)

        fps = cached_fps(video_id)
        try:
            eaf = pympi.Elan.Eaf(eaf_path) if fps is not None else None
        except Exception:
            eaf = None

        for participant in ("A", "B"):
            out_id = f"{video_id}_{participant}"
            bio_path = os.path.join(GLOSS_LABELS_DIR, f"{out_id}.npy")
            if not os.path.exists(bio_path):
                skip_counts["missing_bio_tags"] += 1
                continue

            bio_labels = np.load(bio_path)
            num_frames = len(bio_labels)

            if fps is None:
                skip_counts["missing_video_or_fps"] += 1
                continue
            if eaf is None:
                skip_counts["eaf_parse_error"] += 1
                continue

            spans = extract_gloss_spans(eaf, participant, fps)

            gloss_ids = np.full(num_frames, -1, dtype=np.int64)  # -1 = Outside/no gloss
            for start, end, gloss in spans:
                start = max(0, start)
                end = min(num_frames, end)
                if start >= end:
                    continue
                gid = gloss_to_id.get(gloss, unk_id)
                gloss_ids[start:end] = gid

            np.save(os.path.join(OUTPUT_DIR, f"{out_id}.npy"), gloss_ids)
            processed_count += 1

            if sanity_checked < SANITY_CHECK_COUNT:
                # Alignment check: gloss_ids should be "not -1" almost exactly
                # where bio_labels is "not 0" (0 = Outside in the existing,
                # confirmed BIO_tags convention).
                gloss_active = gloss_ids != -1
                bio_active = bio_labels != 0
                agreement = (gloss_active == bio_active).mean()
                print(f"\n[SANITY CHECK] {out_id}: {len(spans)} gloss spans, "
                      f"frame-level agreement with existing BIO_tags active/inactive: "
                      f"{agreement * 100:.1f}% (should be very high, e.g. >95%)")
                sanity_checked += 1

    print(f"\nDone. {processed_count} gloss-id files written to {OUTPUT_DIR}")
    print(f"Skip reasons: {skip_counts}")
    print(f"\nBefore trusting this for training: check the sanity-check agreement percentages "
          f"above. If any are notably low, the gloss tier extraction and the existing BIO_tags "
          f"may be using different conventions or sources -- investigate before proceeding.")


if __name__ == "__main__":
    main()