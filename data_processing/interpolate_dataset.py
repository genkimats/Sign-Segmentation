import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

KEYPOINTS_DIR = "processed_data/keypoints"

def fix_and_interpolate_dataset():
    print(f"🔍 Scanning {KEYPOINTS_DIR} for 0.0 values or collapsed coordinates...")
    files = glob.glob(os.path.join(KEYPOINTS_DIR, "*.npy"))
    
    if not files:
        print("❌ No .npy files found!")
        return

    files_modified = 0
    total_missing_frames_fixed = 0

    for file_path in tqdm(files, desc="Interpolating gaps"):
        try:
            data = np.load(file_path)
            
            # Handle potential shape issues if data was flattened previously
            if data.ndim == 2:
                T = data.shape[0]
                data = data.reshape(T, -1, 3)
                
            T, V, C = data.shape
        except Exception as e:
            print(f"\n⚠️ Failed to read {file_path}. Skipping. Error: {e}")
            continue
        
        # 1. Identify literal near-zeros or NaNs (Using isclose prevents floating point misses)
        bad_mask = np.isclose(data, 0.0, atol=1e-5) | np.isnan(data)
        
        # 2. Identify fake normalized zeros (Collapsed Hands)
        # If 0s were passed through 3D normalization, they turned into identical non-zero values.
        # We detect this by checking if an entire hand has ~0 variance (all 21 points overlapping).
        if V >= 65:
            # Left hand: indices 23 to 44
            lh_var = np.var(data[:, 23:44, :2], axis=1) # Variance of X and Y
            lh_bad = lh_var < 1e-5
            bad_mask[lh_bad, 23:44, :] = True
            
            # Right hand: indices 44 to 65
            rh_var = np.var(data[:, 44:65, :2], axis=1)
            rh_bad = rh_var < 1e-5
            bad_mask[rh_bad, 44:65, :] = True
        else:
            # Fallback for dynamic shapes: check adjacent vertices
            for v in range(V - 1):
                collapsed = np.all(np.isclose(data[:, v, :2], data[:, v+1, :2], atol=1e-5), axis=1)
                bad_mask[collapsed, v, :] = True
                bad_mask[collapsed, v+1, :] = True
            
        if not bad_mask.any():
            continue  # File is perfectly clean, move to the next one
            
        files_modified += 1
        total_missing_frames_fixed += int(np.sum(bad_mask) // C)
        
        # Set all bad data to explicitly be NaN so Pandas knows it represents a gap
        data[bad_mask] = np.nan
        
        # 3. Apply the Linear Interpolation
        # Flatten to 2D (Time, Features) so pandas can interpolate every tracking point over Time
        flattened = data.reshape(T, -1)
        df = pd.DataFrame(flattened)
        
        # method='linear' smoothly and evenly connects the last valid point to the next valid point.
        # limit_direction='both' ensures if the video starts/ends with missing data, it holds the nearest frame steady.
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        
        # If a joint was missing for the ENTIRE video, it will still be NaN. 
        # We fill with 0.0 purely to prevent a crash (its Velocity will be exactly 0, which is safe).
        df.fillna(0.0, inplace=True)
        
        # Reshape back to 3D and overwrite the file
        data_fixed = df.values.reshape(T, V, C)
        np.save(file_path, data_fixed)

    print("\n✅ Dataset Interpolation Complete!")
    print(f"📊 Fixed approximately {total_missing_frames_fixed} missing coordinates across {files_modified} files.")
    print("🚀 Your kinematic features (Velocity, Acceleration, Jerk) are now smooth and spike-free!")

if __name__ == "__main__":
    fix_and_interpolate_dataset()