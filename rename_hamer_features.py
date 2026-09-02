import os

# Videos were originally sourced using the raw DGS camera-angle naming
# (_1a1.mp4/_1b1.mp4), so extract_hand_boxes.py / extract_hamer_features.py
# inherited that naming for their outputs too -- same root cause as the
# face-keypoints naming mismatch fixed earlier. Everything else in this
# pipeline (processed_data/keypoints, BIO_tags, dataset_splits.json) uses the
# participant-letter convention ("_A"/"_B") instead. Renaming is pure
# metadata -- file CONTENT is untouched, nothing needs re-extracting.
TARGET_DIR = "processed_data/hamer_features"

RENAME_SUFFIXES = [
    ("_1a1_hamer.pt", "_A_hamer.pt"),
    ("_1b1_hamer.pt", "_B_hamer.pt"),
]

# Defaults to a dry run (prints what WOULD be renamed, renames nothing).
# Review the printed plan, then flip to False to actually apply it.
DRY_RUN = False


def main():
    if DRY_RUN:
        print("=== DRY RUN MODE -- nothing will actually be renamed. Set DRY_RUN = False to apply. ===\n")

    if not os.path.exists(TARGET_DIR):
        print(f"Directory not found: {TARGET_DIR}")
        return

    print(f"--- {TARGET_DIR} ---")
    renamed, already_correct, unrecognized = 0, 0, 0

    for fname in sorted(os.listdir(TARGET_DIR)):
        if not fname.endswith(".pt"):
            continue

        matched = False
        for old_suffix, new_suffix in RENAME_SUFFIXES:
            if fname.endswith(old_suffix):
                new_fname = fname[: -len(old_suffix)] + new_suffix
                old_path = os.path.join(TARGET_DIR, fname)
                new_path = os.path.join(TARGET_DIR, new_fname)

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

        if fname.endswith("_A_hamer.pt") or fname.endswith("_B_hamer.pt"):
            already_correct += 1
        else:
            unrecognized += 1
            print(f"  ⚠️ Unrecognized naming pattern (left as-is): {fname}")

    verb = "would rename" if DRY_RUN else "renamed"
    print(f"  Summary: {renamed} {verb}, {already_correct} already correct, "
          f"{unrecognized} unrecognized.")


if __name__ == "__main__":
    main()