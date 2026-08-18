import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import pympi

try:
    import cv2
except ImportError:
    cv2 = None

# --- Directory Resolution ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

VIDEOS_DIR = os.path.join(PARENT_DIR, "raw_data", "videos")
ANNOTATIONS_DIR = os.path.join(PARENT_DIR, "raw_data", "annotations")

PARTICIPANT_TIERS = {"a": "Sign_r_A", "b": "Sign_r_B"}


def get_video_duration_seconds(file_path):
    if cv2 is None:
        return 0.0

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()

    if fps <= 0 or frame_count <= 0:
        return 0.0

    return frame_count / fps


def record_deletion(deletion_stats, reason, duration_seconds=0.0):
    entry = deletion_stats[reason]
    entry["count"] += 1
    entry["seconds"] += duration_seconds


def analyze_eaf(file_path):
    """
    X-rays the ELAN XML structure PER PARTICIPANT TIER, instead of as one
    whole-document pass/fail. A dangling reference or a pympi read failure
    confined to Sign_r_B's own tier no longer disqualifies Sign_r_A (and vice
    versa) -- only a genuinely whole-document problem (unparseable XML, or
    both tiers unusable) does.

    Returns a dict:
      {
        'fatal': bool, 'fatal_reason': str or None,   # can't be salvaged AT ALL
        'a_ok': bool,  'a_reason': str,
        'b_ok': bool,  'b_reason': str,
      }
    """
    result = {
        'fatal': False, 'fatal_reason': None,
        'a_ok': False, 'a_reason': f"{PARTICIPANT_TIERS['a']} tier not present",
        'b_ok': False, 'b_reason': f"{PARTICIPANT_TIERS['b']} tier not present",
    }

    try:
        # 1. XML Referential Integrity Check -- scoped PER TIER, not whole-document.
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Map every annotation ID to the TIER it belongs to (each <TIER> owns its
        # own <ANNOTATION> children directly in the EAF schema), so a dangling
        # reference can be attributed to the specific tier/participant it affects.
        ann_to_tier = {}
        for tier_el in root.findall('.//TIER'):
            tier_id = tier_el.get('TIER_ID')
            for ann in tier_el.findall('.//ALIGNABLE_ANNOTATION') + tier_el.findall('.//REF_ANNOTATION'):
                ann_to_tier[ann.get('ANNOTATION_ID')] = tier_id

        valid_ann_ids = set(ann_to_tier.keys())

        broken_tiers = set()
        for ref_ann in root.findall('.//REF_ANNOTATION'):
            ref_id = ref_ann.get('ANNOTATION_REF')
            if ref_id not in valid_ann_ids:
                owning_tier = ann_to_tier.get(ref_ann.get('ANNOTATION_ID'))
                broken_tiers.add(owning_tier)

        # 2. Pympi Dry-Run Check -- also scoped per tier.
        eaf = pympi.Elan.Eaf(file_path)
        tiers = eaf.get_tier_names()

        for participant, tier_name in PARTICIPANT_TIERS.items():
            if tier_name not in tiers:
                continue  # leave the default "tier not present" reason

            if tier_name in broken_tiers:
                result[f'{participant}_reason'] = f"Dangling reference within {tier_name}"
                continue

            try:
                _ = eaf.get_annotation_data_for_tier(tier_name)
                result[f'{participant}_ok'] = True
                result[f'{participant}_reason'] = "Healthy"
            except Exception as e:
                result[f'{participant}_reason'] = f"pympi error reading {tier_name}: {e}"

        return result

    except Exception as e:
        result['fatal'] = True
        result['fatal_reason'] = f"Fatal XML/Parsing Error: {str(e)}"
        return result


def clean_annotations(deletion_stats):
    print("--- Phase 1: Cleaning Annotations (ELAN) Directory ---")
    valid_ids_a = set()
    valid_ids_b = set()

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Directory not found: {ANNOTATIONS_DIR}")
        return valid_ids_a, valid_ids_b

    for filename in os.listdir(ANNOTATIONS_DIR):
        if filename.endswith(".eaf"):
            file_path = os.path.join(ANNOTATIONS_DIR, filename)
            base_id = filename.replace('.eaf', '')

            analysis = analyze_eaf(file_path)

            # --- WHOLE-DOCUMENT FAILURE: genuinely nothing salvageable ---
            if analysis['fatal']:
                # os.remove(file_path)
                record_deletion(deletion_stats, "Corrupted ELAN (fatal, whole document)")
                print(f"🗑️ Removed (Corrupted, fatal): {filename} -> {analysis['fatal_reason']}")
                continue

            try:
                eaf = pympi.Elan.Eaf(file_path)
                tiers = eaf.get_tier_names()
            except Exception:
                # os.remove(file_path)
                record_deletion(deletion_stats, "Unexpected ELAN Error")
                print(f"🗑️ Removed (Unexpected Error): {filename}")
                continue

            if len(tiers) == 0:
                # os.remove(file_path)
                record_deletion(deletion_stats, "Empty/No Tiers")
                print(f"🗑️ Removed (Empty/No Tiers): {filename}")
                continue

            # --- Neither participant has usable sign data: nothing to keep. ---
            # This is what should catch the "Joke"/unglossed category in the DGS
            # Corpus (present + technically valid EAF, but never translated/glossed).
            if not analysis['a_ok'] and not analysis['b_ok']:
                # os.remove(file_path)
                record_deletion(deletion_stats, "No usable Sign_r_A/Sign_r_B data (likely Joke/unglossed)")
                print(f"🗑️ Removed (No usable sign data): {filename} "
                      f"-> A: {analysis['a_reason']} | B: {analysis['b_reason']}")
                continue

            # --- KEEP the file. At least one participant has valid, usable data. ---
            # Each participant's validity is tracked SEPARATELY -- a problem with
            # B no longer takes A down with it, and vice versa.
            if analysis['a_ok']:
                valid_ids_a.add(base_id)
            else:
                print(f"⚠️ {filename}: keeping file, but participant A is unusable -> {analysis['a_reason']}")

            if analysis['b_ok']:
                valid_ids_b.add(base_id)
            else:
                print(f"⚠️ {filename}: keeping file, but participant B is unusable -> {analysis['b_reason']}")

    print(f"🟢 Valid ELAN participants -- A: {len(valid_ids_a)} documents, B: {len(valid_ids_b)} documents\n")
    return valid_ids_a, valid_ids_b


def clean_videos(valid_ids_a, valid_ids_b, deletion_stats):
    print("--- Phase 2: Cleaning Video Directory ---")

    if not os.path.exists(VIDEOS_DIR):
        print(f"Directory not found: {VIDEOS_DIR}")
        return

    for filename in os.listdir(VIDEOS_DIR):
        if filename.endswith(".mp4"):
            file_path = os.path.join(VIDEOS_DIR, filename)

            # 1. Delete wrong camera angles, and figure out which participant's
            #    validity set this specific video should be checked against.
            if filename.endswith("_1a1.mp4"):
                base_id = filename.replace('_1a1.mp4', '')
                participant, valid_ids = "A", valid_ids_a
            elif filename.endswith("_1b1.mp4"):
                base_id = filename.replace('_1b1.mp4', '')
                participant, valid_ids = "B", valid_ids_b
            else:
                duration_seconds = get_video_duration_seconds(file_path)
                # os.remove(file_path)
                record_deletion(deletion_stats, "Wrong Angle", duration_seconds)
                print(f"🗑️ Removed (Wrong Angle): {filename}")
                continue

            # 2. Delete Orphaned Videos -- checked against THIS PARTICIPANT's
            #    validity specifically now, not a shared document-level set.
            #    A's video is no longer collateral damage when B's data is the
            #    problem (and B's video is no longer kept alive by A's data).
            if base_id not in valid_ids:
                duration_seconds = get_video_duration_seconds(file_path)
                # os.remove(file_path)
                record_deletion(deletion_stats, f"Orphaned - No usable ELAN data for participant {participant}", duration_seconds)
                print(f"🗑️ Removed (Orphaned - participant {participant} unusable): {filename}")


def print_deletion_summary(deletion_stats):
    print("--- Phase 3: Deletion Summary ---")
    total_files = sum(entry["count"] for entry in deletion_stats.values())
    total_hours = sum(entry["seconds"] for entry in deletion_stats.values()) / 3600.0

    for reason, entry in deletion_stats.items():
        hours = entry["seconds"] / 3600.0
        print(f"{reason}: {entry['count']} files deleted, {hours:.2f} hours of video data")

    print(f"\n✅ Total deleted files: {total_files}")
    print(f"✅ Total deleted video hours: {total_hours:.2f}")
    print("Note: annotation deletions contribute file count only, not video duration.\n")

if __name__ == "__main__":
    print("Starting Raw Data Deep Cleanup...\n")
    deletion_stats = defaultdict(lambda: {"count": 0, "seconds": 0.0})
    valid_ids_a, valid_ids_b = clean_annotations(deletion_stats)
    clean_videos(valid_ids_a, valid_ids_b, deletion_stats)
    print_deletion_summary(deletion_stats)
    print("Cleanup Complete!")