import os
import numpy as np
from config import *
from extract_poses import extract_video_features
from parse_labels import create_bio_labels

def setup_directories():
    # Make sure we are saving to the right places
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    print("✅ Verified processed data directories.")

def get_already_processed_jobs():
    """
    Scans the keypoints directory and returns a set of job names 
    (e.g., '12345_A') that have already been successfully processed.
    """
    processed_set = set()
    if os.path.exists(KEYPOINTS_DIR):
        for filename in os.listdir(KEYPOINTS_DIR):
            if filename.endswith(".npy"):
                # Filenames now look exactly like: 12345_A.npy
                base_name = filename.replace('.npy', '')
                processed_set.add(base_name)
                
    return processed_set

def match_files(processed_set):
    """Matches videos to .eaf files and skips completed jobs."""
    videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]
    annotations = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.eaf')]
    
    processing_jobs = []
    skipped_count = 0
    
    for vid in videos:
        # Route Person A
        if vid.endswith('_1a1.mp4'):
            vid_base = vid.replace('_1a1.mp4', '')
            target_tier = "Sign_r_A"
            save_name = f"{vid_base}_A" 
            
        # Route Person B
        elif vid.endswith('_1b1.mp4'):
            vid_base = vid.replace('_1b1.mp4', '')
            target_tier = "Sign_r_B"
            save_name = f"{vid_base}_B" 
            
        else:
            continue 
            
        # --- RESUME LOGIC ---
        if save_name in processed_set:
            skipped_count += 1
            continue
            
        # Match with the core ELAN file
        eaf_filename = f"{vid_base}.eaf"
        if eaf_filename in annotations:
            processing_jobs.append({
                'save_name': save_name,
                'video_path': os.path.join(VIDEOS_DIR, vid),
                'eaf_path': os.path.join(ANNOTATIONS_DIR, eaf_filename),
                'target_tier': target_tier
            })
            
    print(f"⏩ Skipped {skipped_count} jobs that were already processed.")
    return processing_jobs

def main():
    print("Starting Continuous Data Extraction Pipeline...")
    setup_directories()
    
    # 1. Build the list of already finished files
    processed_set = get_already_processed_jobs()
    
    # 2. Get the remaining jobs
    jobs = match_files(processed_set)
    print(f"Found {len(jobs)} remaining jobs to process.")
    
    if len(jobs) == 0:
        print("🎉 All data has been processed!")
        return
    
    for idx, job in enumerate(jobs):
        print(f"\n--- Processing Job {idx+1}/{len(jobs)}: {job['save_name']} ---")
        
        # Pre-check ELAN integrity
        test_parse = create_bio_labels(job['eaf_path'], 10, job['target_tier'])
        if test_parse is None:
            print(f"⚠️ Skipping {job['save_name']} due to corrupted ELAN annotations.")
            continue
        
        # Extract and Normalize Poses (Returns full continuous array)
        keypoints = extract_video_features(job['video_path'])
        total_frames = keypoints.shape[0]
        
        # Parse ELAN annotations for real (Returns full continuous array)
        bio_array = create_bio_labels(job['eaf_path'], total_frames, job['target_tier'])
        
        if bio_array is None:
            print(f"⚠️ Skipping {job['save_name']} due to label creation failure.")
            continue
        
        # --- NEW SAVING LOGIC: Save the full, unsliced arrays ---
        kp_filename = f"{job['save_name']}.npy"
        bio_filename = f"{job['save_name']}.npy"
        
        np.save(os.path.join(KEYPOINTS_DIR, kp_filename), keypoints)
        np.save(os.path.join(LABELS_DIR, bio_filename), bio_array)
        
        print(f"✅ Saved full sequence {job['save_name']} successfully.")

if __name__ == "__main__":
    main()