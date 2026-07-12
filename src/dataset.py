import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import medfilt

def apply_label_smoothing(labels_array, window_size=5):
    """
    Converts 1D hard labels into 2D soft probability distributions.
    Applies a Gaussian tolerance window around 'Begin' (Class 2) tags.
    """
    T = labels_array.shape[0]
    num_classes = 3
    soft_labels = np.zeros((T, num_classes), dtype=np.float32)

    # Convert to strict one-hot vectors first
    soft_labels[np.arange(T), labels_array] = 1.0

    if window_size <= 1:
        return soft_labels

    spread = window_size // 2
    # Standard deviation scales with the window spread
    sigma = spread / 2.0 if spread > 0 else 1.0

    # Find all exact 'Begin' frames
    begin_indices = np.where(labels_array == 2)[0]

    for i in begin_indices:
        for d in range(-spread, spread + 1):
            idx = i + d
            if 0 <= idx < T:
                # Calculate Gaussian weight
                weight = np.exp(-(d**2) / (2 * sigma**2))
                
                # If this curve provides a higher 'Begin' probability, apply it
                if weight > soft_labels[idx, 2]:
                    soft_labels[idx, 2] = weight
                    
                    # Proportionally scale down whatever the original class was
                    orig_class = labels_array[idx]
                    if orig_class != 2:
                        soft_labels[idx, orig_class] = 1.0 - weight
                        
    return soft_labels

class SignSegmentationDataset(Dataset):
    def __init__(self, keypoints_dir, labels_dir, split_file="dataset_splits.json", split="train", window_size=1000, overlap=200, tolerance_window=5, use_full_length=False, base_features=None, kinematic_features=None, temporal_downsample_factor=1):
        self.keypoints_dir = keypoints_dir
        self.labels_dir = labels_dir
        self.split_file = split_file
        self.split = split
        self.window_size = window_size
        self.step_size = window_size - overlap
        self.tolerance_window = tolerance_window
        self.use_full_length = use_full_length
        self.temporal_downsample_factor = temporal_downsample_factor
        
        # Track parameters or fall back to defaults
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "confidence"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []
        self.slice_index = []
        
        # Raw index map to read from the raw source array (0:X, 1:Y, 2:Z/Conf)
        self.feature_map = {"x-cord": 0, "y-cord": 1, "z-cord": 2, "confidence": 2}
        
        self._build_index()

    def _build_index(self):
        # Load the split blueprint
        with open(self.split_file, "r") as f:
            splits = json.load(f)
            
        # Get only the files assigned to the current split (train, val, or test)
        allowed_files = splits.get(self.split, [])
        
        if not allowed_files:
            print(f"⚠️ Warning: No files found for split '{self.split}' in {self.split_file}")
            
        for file_name in allowed_files:
            base_name = file_name.replace('.npy', '')
            if self.use_full_length:
                self.slice_index.append({'base_name': base_name, 'start': None, 'end': None})
            else:
                kp_path = os.path.join(self.keypoints_dir, file_name)
                # Ensure the file exists before attempting to load
                if not os.path.exists(kp_path):
                    continue
                total_frames = np.load(kp_path, mmap_mode='r').shape[0]
                for start in range(0, total_frames - self.window_size + 1, self.step_size):
                    self.slice_index.append({'base_name': base_name, 'start': start, 'end': start + self.window_size})

    def _compute_full_kinematics(self, full_raw_pos):
        """
        Always calculates physics derivatives based on full 3D spatial points
        to guarantee mathematical precision, regardless of what user wants to keep.
        full_raw_pos shape: (3, T, V)
        """
        # 1. Full 3D Velocity
        v = torch.zeros_like(full_raw_pos)
        v[:, 1:, :] = full_raw_pos[:, 1:, :] - full_raw_pos[:, :-1, :]
        
        # 2. Full 3D Acceleration
        a = torch.zeros_like(v)
        a[:, 1:, :] = v[:, 1:, :] - v[:, :-1, :]
        
        # 3. Full 3D Jerk
        j = torch.zeros_like(a)
        j[:, 1:, :] = a[:, 1:, :] - a[:, :-1, :]
        
        # 4. Scalar Speed Magnitude (1 channel per vertex)
        v_mag = torch.norm(v, p=2, dim=0, keepdim=True)
        
        # 5. Trajectory Angular Velocity (1 channel per vertex)
        theta = torch.atan2(v[1, ...], v[0, ...]).unsqueeze(0) 
        omega = torch.zeros_like(theta)
        diff_theta = theta[:, 1:, :] - theta[:, :-1, :]
        omega[:, 1:, :] = torch.atan2(torch.sin(diff_theta), torch.cos(diff_theta))
        
        return v, a, j, v_mag, omega

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        slice_info = self.slice_index[idx]
        kp_path = os.path.join(self.keypoints_dir, f"{slice_info['base_name']}.npy")
        label_path = os.path.join(self.labels_dir, f"{slice_info['base_name']}.npy")
        
        if self.use_full_length:
            kp_array = np.load(kp_path)        
            label_array = np.load(label_path)  
            max_allowed_frames = 4000
            if kp_array.shape[0] > max_allowed_frames:
                kp_array = kp_array[:max_allowed_frames]
                label_array = label_array[:max_allowed_frames]
        else:
            kp_array = np.load(kp_path, mmap_mode='c')[slice_info['start']:slice_info['end']]
            label_array = np.load(label_path, mmap_mode='c')[slice_info['start']:slice_info['end']]
        
        # Scrub NaNs safely
        kp_array = np.nan_to_num(kp_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Median Filter
        clean_kp_array = medfilt(kp_array, kernel_size=(5, 1, 1))
        
        # Tensor conversion (C, T, V)
        full_raw_tensor = torch.tensor(clean_kp_array, dtype=torch.float32).permute(2, 0, 1) 
        
        # Physics
        v_full, a_full, j_full, v_mag, omega = self._compute_full_kinematics(full_raw_tensor)
        
        final_channels = []
        
        if self.base_features:
            base_indices = [self.feature_map[f] for f in self.base_features if f in self.feature_map]
            final_channels.append(full_raw_tensor[base_indices, :, :])
            
        deriv_indices = [self.feature_map[f] for f in self.base_features if f in self.feature_map] if self.base_features else [0, 1]
        
        if "velocity" in self.kinematic_features:
            final_channels.append(v_full[deriv_indices, :, :])
        if "acceleration" in self.kinematic_features:
            final_channels.append(a_full[deriv_indices, :, :])
        if "jerk" in self.kinematic_features:
            final_channels.append(j_full[deriv_indices, :, :])
        if "velocity-mag" in self.kinematic_features:
            final_channels.append(v_mag)
        if "angular-vel" in self.kinematic_features:
            final_channels.append(omega)
            
        # Cat channels
        final_input_tensor = torch.cat(final_channels, dim=0)
        
        soft_labels = apply_label_smoothing(label_array, self.tolerance_window)
        labels_tensor = torch.tensor(soft_labels, dtype=torch.float32).permute(1, 0)
        
        # --- NOVELTY: Temporal Downsampling ---
        # Slices the tensors to take every Nth frame, halving the sequence length but preserving accuracy!
        if self.temporal_downsample_factor > 1:
            final_input_tensor = final_input_tensor[:, ::self.temporal_downsample_factor, :]
            labels_tensor = labels_tensor[:, ::self.temporal_downsample_factor]
            
        return final_input_tensor, labels_tensor