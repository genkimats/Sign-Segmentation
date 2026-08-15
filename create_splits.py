import os
import glob
import json
import numpy as np
from tqdm import tqdm

# Requires: pip install sign-language-datasets
# (same TFDS loader library used by Moryossef et al. 2023 / Zhang et al. 2023,
#  ships the canonical "3.0.0-uzh-document" split as a plain JSON file, plus
#  the DGS Corpus document index needed to map document IDs -> video files)
from sign_language_datasets.datasets.dgs_corpus.dgs_corpus import load_split, INDEX_PATH

LABELS_DIR = "processed_data/BIO_tags"
SPLIT_FILE = "dataset_splits.json"
OFFICIAL_SPLIT_NAME = "3.0.0-uzh-document"


# ==============================================================================
# STEP 1: FETCH THE OFFICIAL SPLIT + DOCUMENT INDEX
# ==============================================================================
def load_dgs_index():
    """The DGS Corpus document index (dgs.json) ships inside sign_language_datasets
    and maps each document_id -> its metadata, including video_a/video_b file paths."""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def stem(path_or_name):
    """Basename without extension."""
    base = os.path.basename(str(path_or_name))
    return os.path.splitext(base)[0]


def resolve_document_video_stems(document_id, datum):
    """
    Candidate local-file stems for a single DGS document.

    Your local .npy files are named "{document_id}_A.npy" / "{document_id}_B.npy"
    (one per participant camera) -- the document_id itself is already the exact
    prefix, so we don't need to parse the official video_a/video_b path suffixes
    (e.g. "..._1a1.mp4") at all. Only offer a letter if that camera actually
    exists for this document (some documents have a single signer only).
    """
    candidates = []
    if datum.get("video_a"):
        candidates.append(f"{document_id}_A")
    if datum.get("video_b"):
        candidates.append(f"{document_id}_B")
    return candidates


# ==============================================================================
# STEP 2: MATCH OFFICIAL DOCUMENTS AGAINST YOUR LOCAL LABEL FILES
# ==============================================================================
def build_local_file_index(labels_dir):
    """normalized stem (lowercase) -> actual filename on disk."""
    index = {}
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".npy"):
            continue
        index[stem(fname).lower()] = fname
    return index


def build_official_splits(labels_dir):
    print(f"🔎 Fetching official split '{OFFICIAL_SPLIT_NAME}' via sign_language_datasets...")
    split = load_split(OFFICIAL_SPLIT_NAME)  # {"train": [...], "dev": [...], "test": [...]} of document IDs

    print("📖 Loading DGS Corpus document index (dgs.json)...")
    index_data = load_dgs_index()

    print(f"📁 Scanning local label files in '{labels_dir}'...")
    local_index = build_local_file_index(labels_dir)

    final_splits = {"train": [], "val": [], "test": []}
    official_to_ours = {"train": "train", "dev": "val", "test": "test"}

    unmatched_documents = []
    claimed_local_keys = set()

    for official_name, our_name in official_to_ours.items():
        for document_id in split.get(official_name, []):
            datum = index_data.get(document_id)
            if datum is None:
                unmatched_documents.append((document_id, "not found in dgs.json index"))
                continue

            candidates = resolve_document_video_stems(document_id, datum)
            matched_any = False
            for cand in candidates:
                key = cand.lower()
                if key in local_index:
                    final_splits[our_name].append(local_index[key])
                    claimed_local_keys.add(key)
                    matched_any = True

            if not matched_any:
                unmatched_documents.append((document_id, f"no local file matched candidates {candidates}"))

    unmatched_local = set(local_index.keys()) - claimed_local_keys

    print("\n" + "=" * 60)
    print(f"MATCHED   -> train={len(final_splits['train'])}  val={len(final_splits['val'])}  test={len(final_splits['test'])}")
    print(f"UNMATCHED OFFICIAL DOCUMENTS: {len(unmatched_documents)}")
    for doc_id, reason in unmatched_documents[:20]:
        print(f"   - {doc_id}: {reason}")
    if len(unmatched_documents) > 20:
        print(f"   ... and {len(unmatched_documents) - 20} more")
    print(f"LOCAL FILES NEVER CLAIMED BY THE OFFICIAL SPLIT: {len(unmatched_local)}")
    for key in list(unmatched_local)[:20]:
        print(f"   - {local_index[key]}")
    print("=" * 60)

    if unmatched_documents or unmatched_local:
        print(
            "\n⚠️  Matching was not fully clean -- inspect the lists above BEFORE trusting "
            f"'{SPLIT_FILE}'. This almost always means your local .npy filenames don't share "
            "a naming convention with the official 'video_a' / 'video_b' basenames. Adjust "
            "`resolve_document_video_stems()` (e.g. strip a prefix/suffix, or match on a "
            "substring) until UNMATCHED counts are ~0, rather than shipping a partial split."
        )

    return final_splits


# ==============================================================================
# STEP 3 (OPTIONAL DIAGNOSTIC): REPORT GLOSS / BIO STATS FOR THE RESULTING SPLIT
# ==============================================================================
def report_split_stats(final_splits, labels_dir):
    print("\n📊 SPLIT DIAGNOSTICS (BIO distribution per split, informational only --")
    print("   the split membership itself comes from the official protocol, not these stats):")
    print("-" * 60)

    for name in ["train", "val", "test"]:
        b = i = o = glosses = 0
        for fname in tqdm(final_splits[name], desc=f"Analyzing {name}", leave=False):
            labels = np.load(os.path.join(labels_dir, fname))
            if labels.ndim > 1:
                hard = np.argmax(labels, axis=0 if labels.shape[0] == 3 else -1)
            else:
                hard = labels
            b += int(np.sum(hard == 2))
            i += int(np.sum(hard == 1))
            o += int(np.sum(hard == 0))
            glosses += int(np.sum(hard == 2))

        total = b + i + o
        print(f"[{name.upper()}] - {len(final_splits[name])} videos, {glosses} glosses")
        if total > 0:
            print(f"   BIO Distribution: B: {100*b/total:.2f}% | I: {100*i/total:.2f}% | O: {100*o/total:.2f}%")
        print("-" * 60)


if __name__ == "__main__":
    if not glob.glob(os.path.join(LABELS_DIR, "*.npy")):
        print(f"❌ Error: No .npy files found in {LABELS_DIR}")
    else:
        final_splits = build_official_splits(LABELS_DIR)
        report_split_stats(final_splits, LABELS_DIR)

        with open(SPLIT_FILE, "w") as f:
            json.dump(final_splits, f, indent=4)

        print(f"\n✅ Wrote '{SPLIT_FILE}' using the official '{OFFICIAL_SPLIT_NAME}' split.")