import pympi
import numpy as np
from config import FPS

def create_bio_labels(eaf_path, total_frames, target_tier):
    """
    Parses ELAN file and creates a frame-level BIO tag array 
    exclusively for the specified target tier (e.g., 'Sign_r_A').
    """
    eaf = pympi.Elan.Eaf(eaf_path)
    
    # Initialize array with 0s (Out)
    bio_array = np.zeros(total_frames, dtype=np.int8)
    
    # Check if the required tier exists in this specific file
    if target_tier not in eaf.get_tier_names():
        print(f"  [!] Warning: Tier '{target_tier}' not found in {eaf_path}. Returning empty labels (all 0s).")
        return bio_array
        
    # Extract only the annotations for the matching person
    annotations = eaf.get_annotation_data_for_tier(target_tier)
        
    for ann in annotations:
        start_ms = ann[0]
        end_ms = ann[1]
        
        # Convert ms to exact frame index (50 FPS)
        start_frame = int((start_ms / 1000.0) * FPS)
        end_frame = int((end_ms / 1000.0) * FPS)
        
        # Prevent out-of-bounds mapping
        start_frame = min(start_frame, total_frames - 1)
        end_frame = min(end_frame, total_frames)
        
        if start_frame >= end_frame:
            continue
            
        # Apply BIO tags: 2 = Begin, 1 = In
        bio_array[start_frame] = 2                  
        if end_frame > start_frame + 1:
            bio_array[start_frame + 1:end_frame] = 1 

    return bio_array