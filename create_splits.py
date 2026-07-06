import os
import glob
import numpy as np
import json
from tqdm import tqdm

LABELS_DIR = "processed_data/BIO_tags"
SPLIT_FILE = "dataset_splits.json"

def analyze_dataset():
    """Scans all label files to count glosses and BIO frames per video."""
    print("🔍 Scanning dataset...")
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.npy")))
    
    if not label_files:
        print(f"❌ Error: No .npy files found in {LABELS_DIR}")
        return []

    video_stats = []
    
    for file_path in tqdm(label_files, desc="Analyzing labels"):
        filename = os.path.basename(file_path)
        
        # Load the (3, T) Gaussian soft-label array
        soft_labels = np.load(file_path)
        
        # Convert soft labels back to hard indices (0=O, 1=I, 2=B) for counting
        hard_labels = np.argmax(soft_labels, axis=0)
        
        # Count the frames for each class
        o_count = np.sum(hard_labels == 0)
        i_count = np.sum(hard_labels == 1)
        b_count = np.sum(hard_labels == 2)
        
        # In BIO tagging, the number of 'B' tags equals the number of glosses!
        total_glosses = b_count
        
        video_stats.append({
            "filename": filename,
            "gloss_count": int(total_glosses),
            "b_frames": int(b_count),
            "i_frames": int(i_count),
            "o_frames": int(o_count),
            "total_frames": len(hard_labels)
        })
        
    return video_stats

def create_balanced_split(video_stats, target_ratios=(0.8, 0.1, 0.1)):
    """Distributes videos into Train/Val/Test attempting to hit target gloss ratios."""
    print("\n⚖️ Calculating balanced splits...")
    
    # Sort videos by gloss count (descending) so we fit the biggest "rocks" in the buckets first
    video_stats.sort(key=lambda x: x['gloss_count'], reverse=True)
    
    total_glosses_dataset = sum(v['gloss_count'] for v in video_stats)
    
    # Target gloss counts for each bucket
    targets = {
        "train": total_glosses_dataset * target_ratios[0],
        "val": total_glosses_dataset * target_ratios[1],
        "test": total_glosses_dataset * target_ratios[2]
    }
    
    # Current state of each bucket
    buckets = {
        "train": {"videos": [], "glosses": 0, "b": 0, "i": 0, "o": 0},
        "val": {"videos": [], "glosses": 0, "b": 0, "i": 0, "o": 0},
        "test": {"videos": [], "glosses": 0, "b": 0, "i": 0, "o": 0}
    }
    
    for video in video_stats:
        # Calculate how "hungry" each bucket is relative to its target
        def hunger(bucket_name):
            if targets[bucket_name] == 0: return -float('inf')
            # The lower the current percentage of its target, the hungrier it is
            return 1.0 - (buckets[bucket_name]['glosses'] / targets[bucket_name])
            
        # Pick the hungriest bucket
        best_bucket = max(["train", "val", "test"], key=hunger)
        
        # Assign video to the best bucket
        b = buckets[best_bucket]
        b["videos"].append(video["filename"])
        b["glosses"] += video["gloss_count"]
        b["b"] += video["b_frames"]
        b["i"] += video["i_frames"]
        b["o"] += video["o_frames"]

    # --- Print Analytics ---
    print("\n📊 SPLIT RESULTS:")
    print("-" * 60)
    for name in ["train", "val", "test"]:
        b = buckets[name]
        total_frames = b["b"] + b["i"] + b["o"]
        actual_ratio = b["glosses"] / total_glosses_dataset
        
        b_pct = (b["b"] / total_frames) * 100 if total_frames > 0 else 0
        i_pct = (b["i"] / total_frames) * 100 if total_frames > 0 else 0
        o_pct = (b["o"] / total_frames) * 100 if total_frames > 0 else 0
        
        print(f"[{name.upper()}] - {len(b['videos'])} Videos")
        print(f"   Target Gloss Ratio:  {target_ratios[['train', 'val', 'test'].index(name)] * 100:.1f}%")
        print(f"   Actual Gloss Ratio:  {actual_ratio * 100:.1f}% ({b['glosses']} glosses)")
        print(f"   BIO Distribution:    B: {b_pct:.2f}% | I: {i_pct:.2f}% | O: {o_pct:.2f}%")
        print("-" * 60)

    # Prepare output dictionary (Just saving the filenames)
    final_split_dict = {
        "train": buckets["train"]["videos"],
        "val": buckets["val"]["videos"],
        "test": buckets["test"]["videos"]
    }
    
    return final_split_dict

if __name__ == "__main__":
    stats = analyze_dataset()
    if stats:
        final_splits = create_balanced_split(stats)
        
        with open(SPLIT_FILE, "w") as f:
            json.dump(final_splits, f, indent=4)
            
        print(f"✅ Splits successfully saved to '{SPLIT_FILE}'!")
        print("You can now update dataset.py to load this file instead of using random_split.")