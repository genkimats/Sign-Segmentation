import os
import pympi
import xml.etree.ElementTree as ET

# --- Directory Resolution ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

VIDEOS_DIR = os.path.join(PARENT_DIR, "raw_data", "videos")
ANNOTATIONS_DIR = os.path.join(PARENT_DIR, "raw_data", "annotations")

def is_eaf_corrupted(file_path):
    """
    X-rays the ELAN XML structure to check for dangling references
    and attempts a dry-run extraction of the target tiers.
    Returns: (bool: is_corrupted, str: reason)
    """
    try:
        # 1. XML Referential Integrity Check
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Collect every valid Annotation ID in the document
        valid_ids = set()
        for ann in root.findall('.//ALIGNABLE_ANNOTATION'):
            valid_ids.add(ann.get('ANNOTATION_ID'))
        for ann in root.findall('.//REF_ANNOTATION'):
            valid_ids.add(ann.get('ANNOTATION_ID'))

        # Check every child annotation to ensure its parent actually exists
        for ref_ann in root.findall('.//REF_ANNOTATION'):
            ref_id = ref_ann.get('ANNOTATION_REF')
            if ref_id not in valid_ids:
                return True, f"Dangling reference (Missing Parent: {ref_id})"

        # 2. Pympi Dry-Run Check
        # If the XML is structurally sound, ensure pympi can successfully map the tiers
        eaf = pympi.Elan.Eaf(file_path)
        tiers = eaf.get_tier_names()
        
        if "Sign_r_A" in tiers:
            _ = eaf.get_annotation_data_for_tier("Sign_r_A")
        if "Sign_r_B" in tiers:
            _ = eaf.get_annotation_data_for_tier("Sign_r_B")

        return False, "Healthy"

    except Exception as e:
        return True, f"Fatal XML/Parsing Error: {str(e)}"

def clean_annotations():
    print("--- Phase 1: Cleaning Annotations (ELAN) Directory ---")
    removed_count = 0
    warning_count = 0
    valid_ids = set() 
    
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Directory not found: {ANNOTATIONS_DIR}")
        return valid_ids

    for filename in os.listdir(ANNOTATIONS_DIR):
        if filename.endswith(".eaf"):
            file_path = os.path.join(ANNOTATIONS_DIR, filename)
            base_id = filename.replace('.eaf', '')
            
            # --- RUN THE DEEP XML INTEGRITY CHECK ---
            is_corrupt, reason = is_eaf_corrupted(file_path)
            
            if is_corrupt:
                os.remove(file_path)
                print(f"🗑️ Removed (Corrupted): {filename} -> Reason: {reason}")
                removed_count += 1
                continue
            
            # --- RUN STANDARD TIER CHECKS ---
            try:
                eaf = pympi.Elan.Eaf(file_path)
                tiers = eaf.get_tier_names()
                
                if len(tiers) == 0:
                    os.remove(file_path)
                    print(f"🗑️ Removed (Empty/No Tiers): {filename}")
                    removed_count += 1
                    continue
                
                if "Sign_r_A" not in tiers and "Sign_r_B" not in tiers:
                    print(f"⚠️ Warning: {filename} does not contain 'Sign_r_A' or 'Sign_r_B'.")
                    warning_count += 1
                    
                # If it survives all checks, add to whitelist!
                valid_ids.add(base_id)

            except Exception as e:
                os.remove(file_path)
                print(f"🗑️ Removed (Unexpected Error): {filename}")
                removed_count += 1

    print(f"✅ Total annotations removed: {removed_count}")
    print(f"🟢 Total valid ELAN files remaining: {len(valid_ids)}\n")
    return valid_ids

def clean_videos(valid_ids):
    print("--- Phase 2: Cleaning Video Directory ---")
    removed_count = 0
    
    if not os.path.exists(VIDEOS_DIR):
        print(f"Directory not found: {VIDEOS_DIR}")
        return

    for filename in os.listdir(VIDEOS_DIR):
        if filename.endswith(".mp4"):
            file_path = os.path.join(VIDEOS_DIR, filename)
            
            # 1. Delete wrong camera angles
            if not (filename.endswith("_1a1.mp4") or filename.endswith("_1b1.mp4")):
                os.remove(file_path)
                print(f"🗑️ Removed (Wrong Angle): {filename}")
                removed_count += 1
                continue
            
            # 2. Delete Orphaned Videos (Where the ELAN file was deleted in Phase 1)
            base_id = filename.replace('_1a1.mp4', '').replace('_1b1.mp4', '')
            
            if base_id not in valid_ids:
                os.remove(file_path)
                print(f"🗑️ Removed (Orphaned - No matching ELAN): {filename}")
                removed_count += 1

    print(f"✅ Total videos removed: {removed_count}\n")

if __name__ == "__main__":
    print("Starting Raw Data Deep Cleanup...\n")
    valid_elan_ids = clean_annotations()
    clean_videos(valid_elan_ids)
    print("Cleanup Complete!")