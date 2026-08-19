import os

# --- Directory Resolution (same convention as clean_data.py: relative to repo root,
# regardless of the CWD the script is invoked from) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

VIDEOS_DIR = os.path.join(PARENT_DIR, "raw_data", "videos")
ANNOTATIONS_DIR = os.path.join(PARENT_DIR, "raw_data", "annotations")
KEYPOINTS_DIR = os.path.join(PARENT_DIR, "processed_data", "keypoints")

# Set to False to actually delete. Defaults to a dry run (prints what WOULD be
# removed, deletes nothing) given how many surprises this pipeline has produced
# recently -- review the printed list once before flipping this.
DRY_RUN = False


def get_keypoint_ids():
    """
    Parses processed_data/keypoints/{id}_A.npy / {id}_B.npy filenames into two
    sets of ids: ids that have an 'A' keypoints file, and ids that have a 'B'
    keypoints file. This is the ground truth for "was this participant actually
    successfully extracted".
    """
    ids_a, ids_b = set(), set()

    if not os.path.exists(KEYPOINTS_DIR):
        print(f"Directory not found: {KEYPOINTS_DIR}")
        return ids_a, ids_b

    for fname in os.listdir(KEYPOINTS_DIR):
        if not fname.endswith(".npy"):
            continue
        if fname.endswith("_A.npy"):
            ids_a.add(fname[:-len("_A.npy")])
        elif fname.endswith("_B.npy"):
            ids_b.add(fname[:-len("_B.npy")])
        else:
            print(f"⚠️ Unexpected keypoints filename (doesn't end in _A.npy/_B.npy): {fname}")

    return ids_a, ids_b


def clean_videos_by_keypoints(ids_a, ids_b):
    print("--- Reconciling raw_data/videos against processed_data/keypoints ---")

    if not os.path.exists(VIDEOS_DIR):
        print(f"Directory not found: {VIDEOS_DIR}")
        return

    removed, kept, skipped = 0, 0, 0

    for fname in sorted(os.listdir(VIDEOS_DIR)):
        if not fname.endswith(".mp4"):
            continue

        if fname.endswith("_1a1.mp4"):
            video_id = fname[:-len("_1a1.mp4")]
            has_keypoints = video_id in ids_a
        elif fname.endswith("_1b1.mp4"):
            video_id = fname[:-len("_1b1.mp4")]
            has_keypoints = video_id in ids_b
        else:
            # Not a recognized camera-angle suffix -- that's clean_data.py's
            # "Wrong Angle" check's job, not this script's. Leave it alone.
            skipped += 1
            continue

        if has_keypoints:
            kept += 1
            continue

        removed += 1
        if DRY_RUN:
            print(f"[DRY RUN] Would remove (no matching keypoints): {fname}")
        else:
            os.remove(os.path.join(VIDEOS_DIR, fname))
            print(f"🗑️ Removed (no matching keypoints): {fname}")

    verb = "Would keep" if DRY_RUN else "Kept"
    verb2 = "would remove" if DRY_RUN else "removed"
    print(f"\n✅ {verb} {kept} videos with matching keypoints; {verb2} {removed} without "
          f"({skipped} files skipped -- not a recognized _1a1/_1b1 filename).")


def clean_annotations_by_keypoints(ids_a, ids_b):
    """
    OPTIONAL: also reconciles raw_data/annotations -- drops a .eaf only if NEITHER
    participant (A nor B) has a matching keypoints file, i.e. nothing was ever
    successfully extracted for this document at all. Comment out the call in
    __main__ below if you'd rather leave annotations alone for now.
    """
    print("\n--- Reconciling raw_data/annotations against processed_data/keypoints ---")

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Directory not found: {ANNOTATIONS_DIR}")
        return

    removed, kept = 0, 0

    for fname in sorted(os.listdir(ANNOTATIONS_DIR)):
        if not fname.endswith(".eaf"):
            continue

        doc_id = fname[:-len(".eaf")]
        if doc_id in ids_a or doc_id in ids_b:
            kept += 1
            continue

        removed += 1
        if DRY_RUN:
            print(f"[DRY RUN] Would remove (no matching keypoints for either participant): {fname}")
        else:
            os.remove(os.path.join(ANNOTATIONS_DIR, fname))
            print(f"🗑️ Removed (no matching keypoints for either participant): {fname}")

    verb = "Would keep" if DRY_RUN else "Kept"
    verb2 = "would remove" if DRY_RUN else "removed"
    print(f"\n✅ {verb} {kept} ELAN files with at least one matching participant; {verb2} {removed} without.")


if __name__ == "__main__":
    if DRY_RUN:
        print("=== DRY RUN MODE -- nothing will actually be deleted. Set DRY_RUN = False to apply. ===\n")

    ids_a, ids_b = get_keypoint_ids()
    print(f"Found keypoints for {len(ids_a)} 'A' participants and {len(ids_b)} 'B' participants.\n")

    clean_videos_by_keypoints(ids_a, ids_b)
    clean_annotations_by_keypoints(ids_a, ids_b)