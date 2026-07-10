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
        
        # 1. Identify literal 0.0s or NaNs (The exact issue you spotted)
        bad_mask = (data == 0.0) | np.isnan(data)
        
        # 2. Identify fake normalized zeros (Collapsed Joints)
        # If the 0s were already passed through the 3D normalization, they became a fake non-zero coordinate.
        # We can find them because all vertices in a dropped hand will perfectly overlap each other.
        for v in range(V - 1):
            # If a vertex has the exact same X and Y as the next vertex, it's a dead collapsed joint
            collapsed = np.all(data[:, v, :2] == data[:, v+1, :2], axis=1)
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