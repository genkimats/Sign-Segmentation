import os
import glob
import numpy as np
from tqdm import tqdm

KEYPOINTS_DIR = "processed_data/keypoints"

def interpolate_missing_keypoints():
    print(f"🔍 Scanning dataset in {KEYPOINTS_DIR} for 0.0 values...")
    files = glob.glob(os.path.join(KEYPOINTS_DIR, "*.npy"))
    
    if not files:
        print("❌ No .npy files found!")
        return

    files_modified = 0
    total_missing_frames_fixed = 0

    for file_path in tqdm(files, desc="Interpolating"):
        # Load array of shape (Frames, Vertices, Channels)
        data = np.load(file_path)
        T, V, C = data.shape
        
        modified = False
        
        for v in range(V):
            # A vertex is dropped if X and Y are exactly 0.0 simultaneously, or if they are NaN
            missing_mask = ((data[:, v, 0] == 0.0) & (data[:, v, 1] == 0.0)) | np.isnan(data[:, v, 0])
            
            if not missing_mask.any():
                continue # This vertex was tracked perfectly for the whole video
                
            modified = True
            total_missing_frames_fixed += missing_mask.sum()
            
            # Interpolate channel by channel (X, Y, Z, Confidence)
            for c in range(C):
                series = data[:, v, c].copy()
                
                # Temporarily replace missing values with NaN for the interpolator
                series[missing_mask] = np.nan
                valid_mask = ~np.isnan(series)
                
                if valid_mask.any():
                    # np.interp performs linear interpolation between valid points.
                    # If the missing values are at the very beginning or end of the video,
                    # it performs 'flat extrapolation' (copies the nearest valid frame),
                    # which perfectly prevents velocity spikes at the edges!
                    x_all = np.arange(T)
                    x_valid = x_all[valid_mask]
                    y_valid = series[valid_mask]
                    
                    data[:, v, c] = np.interp(x_all, x_valid, y_valid)
                else:
                    # Extreme edge case: Vertex was never tracked once in the entire video.
                    # Keep it as 0.0 so we don't crash, the network will learn to ignore it.
                    data[:, v, c] = 0.0 

        if modified:
            # Overwrite the file with the newly smoothed, continuous physics
            np.save(file_path, data)
            files_modified += 1

    print("\n✅ Dataset Interpolation Complete!")
    print(f"📊 Fixed {total_missing_frames_fixed} missing vertex frames across {files_modified} files.")
    print("🚀 Your kinematic features (Velocity, Acceleration, Jerk) will now be smooth and stable!")

if __name__ == "__main__":
    interpolate_missing_keypoints()