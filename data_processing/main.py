import os
import numpy as np
from config import *
from extract_poses import extract_video_features
from parse_labels import create_bio_labels
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

def process_job(job, position):
    """Worker function to process a single video and its ELAN annotations."""
    try:
        # Pre-check ELAN integrity
        test_parse = create_bio_labels(job['eaf_path'], 10, job['target_tier'])
        if test_parse is None:
            return job, False, "Corrupted ELAN annotations"
        
        # Extract and Normalize Poses (Returns full continuous array)
        # We pass the thread position down to extract_poses to stack the progress bars
        keypoints = extract_video_features(job['video_path'], position=position)
        total_frames = keypoints.shape[0]
        
        # Parse ELAN annotations for real
        bio_array = create_bio_labels(job['eaf_path'], total_frames, job['target_tier'])
        
        if bio_array is None:
            return job, False, "Label creation failure"
        
        # --- NEW SAVING LOGIC: Save the full, unsliced arrays ---
        kp_filename = f"{job['save_name']}.npy"
        bio_filename = f"{job['save_name']}.npy"
        
        np.save(os.path.join(KEYPOINTS_DIR, kp_filename), keypoints)
        np.save(os.path.join(LABELS_DIR, bio_filename), bio_array)
        
        return job, True, None
    except Exception as e:
        return job, False, str(e)

def main():
    setup_directories()
    processed_set = get_already_processed_jobs()
    jobs = match_files(processed_set)
    print(f"Found {len(jobs)} remaining jobs to process.")
    
    if len(jobs) == 0:
        print("🎉 All data has been processed!")
        return
        
    max_workers = 3
    print(f"🚀 Starting multi-threaded pipeline with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, job in enumerate(jobs):
            # Assign a position to the worker's progress bar (1 through max_workers)
            pos = (idx % max_workers) + 1 
            future = executor.submit(process_job, job, pos)
            futures[future] = job
            
        # Top-level progress bar sits at position 0
        for future in tqdm(as_completed(futures), total=len(jobs), desc="Total Pipeline", position=0, leave=True):
            job = futures[future]
            try:
                _, success, error_msg = future.result()
                if not success:
                    tqdm.write(f"⚠️ Skipping {job['save_name']}: {error_msg}")
            except Exception as exc:
                tqdm.write(f"❌ Fatal error on {job['save_name']}: {exc}")

if __name__ == "__main__":
    main()