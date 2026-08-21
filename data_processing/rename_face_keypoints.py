import os

# Directories that might contain face keypoint files using the raw DGS video
# naming convention ("_1a1"/"_1b1") instead of the participant-letter convention
# ("_A"/"_B") used everywhere else in this pipeline: processed_data/keypoints,
# processed_data/BIO_tags, processed_data/kinematic_features, dataset_splits.json.
# Renaming is pure metadata -- file CONTENT is untouched, so this is safe to run
# on already-normalized files without recomputing anything.
TARGET_DIRS = [
    "processed_data/face_keypoints",
    "processed_data/face_keypoints_normalized",
]

RENAME_SUFFIXES = [
    ("_1a1.npy", "_A.npy"),
    ("_1b1.npy", "_B.npy"),
]

# Defaults to a dry run (prints what WOULD be renamed, renames nothing).
# Review the printed plan, then flip to False to actually apply it.
DRY_RUN = False


def main():
    if DRY_RUN:
        print("=== DRY RUN MODE -- nothing will actually be renamed. Set DRY_RUN = False to apply. ===\n")

    for target_dir in TARGET_DIRS:
        if not os.path.exists(target_dir):
            print(f"Directory not found, skipping: {target_dir}")
            continue

        print(f"--- {target_dir} ---")
        renamed, already_correct, unrecognized = 0, 0, 0

        for fname in sorted(os.listdir(target_dir)):
            if not fname.endswith(".npy"):
                continue

            matched = False
            for old_suffix, new_suffix in RENAME_SUFFIXES:
                if fname.endswith(old_suffix):
                    new_fname = fname[: -len(old_suffix)] + new_suffix
                    old_path = os.path.join(target_dir, fname)
                    new_path = os.path.join(target_dir, new_fname)

                    if os.path.exists(new_path):
                        print(f"  SKIP (target already exists): {fname} -> {new_fname}")
                    elif DRY_RUN:
                        print(f"  [DRY RUN] Would rename: {fname} -> {new_fname}")
                        renamed += 1
                    else:
                        os.rename(old_path, new_path)
                        print(f"  Renamed: {fname} -> {new_fname}")
                        renamed += 1
                    matched = True
                    break

            if matched:
                continue

            if fname.endswith("_A.npy") or fname.endswith("_B.npy"):
                already_correct += 1
            else:
                unrecognized += 1
                print(f"  ⚠️ Unrecognized naming pattern (left as-is): {fname}")

        verb = "would rename" if DRY_RUN else "renamed"
        print(f"  Summary: {renamed} {verb}, {already_correct} already correct, "
              f"{unrecognized} unrecognized.\n")


if __name__ == "__main__":
    main()