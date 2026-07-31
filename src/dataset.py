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

def get_spatial_angles(features):
    """
    Converts (x,y,z) coordinates into 3D Spherical Angles (Pitch and Yaw)
    relative to the body center (node 0).
    
    Expected input shape: (Frames, Vertices, Channels)
    Returns shape: (Frames, Vertices, 2)
    """
    # Use Node 0 as the origin point (0,0,0)
    anchor = features[:, 0:1, :3] 
    
    # Calculate the vector from the anchor to every other joint
    rel_vec = features[:, :, :3] - anchor
    
    x = rel_vec[:, :, 0]
    y = rel_vec[:, :, 1]
    z = rel_vec[:, :, 2]
    
    # Convert to Spherical Angles
    r = torch.sqrt(x**2 + y**2 + z**2) + 1e-6
    theta = torch.acos(z / r)  # Elevation / Pitch (0 to pi)
    phi = torch.atan2(y, x)    # Azimuth / Yaw (-pi to pi)
    
    # Normalize between roughly -1.0 and 1.0 for the neural network
    theta = theta / np.pi
    phi = phi / np.pi
    
    # Stack along the channel dimension
    return torch.stack([theta, phi], dim=-1)


def get_temporal_angles(features):
    """
    Calculates the 3D angle of the trajectory of movement between frame t and t-1.
    Unlike velocity (which includes speed and causes noise), this ONLY looks at the direction.
    
    Expected input shape: (Frames, Vertices, Channels)
    Returns shape: (Frames, Vertices, 2)
    """
    # Calculate difference between frames
    motion_vec = torch.zeros_like(features[:, :, :3])
    motion_vec[1:] = features[1:, :, :3] - features[:-1, :, :3]
    
    x = motion_vec[:, :, 0]
    y = motion_vec[:, :, 1]
    z = motion_vec[:, :, 2]
    
    r = torch.sqrt(x**2 + y**2 + z**2) + 1e-6
    theta = torch.acos(z / r) / np.pi
    phi = torch.atan2(y, x) / np.pi
    
    # Stack along the channel dimension
    return torch.stack([theta, phi], dim=-1)

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

        # Make sure this is in your __init__
        self.feature_map = {
            "x-cord": 0,
            "y-cord": 1,
            "z-cord": 2
            # Add any confidence or other channels here if they exist in your .npy files
        }
        
        # Track parameters or fall back to defaults
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "confidence"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []
        self.slice_index = []
        
        self.samples = []  # Stores full video info
        self.windows = []  # Stores sliced window info (CRITICAL FOR TRAINING)

        for vid in sorted(self.video_ids):
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            if not os.path.exists(label_path):
                continue
                
            labels = np.load(label_path)
            num_frames = len(labels)
            
            # Store the full video for validation/full_length mode
            self.samples.append({
                'video_id': vid,
                'start_idx': 0,
                'end_idx': num_frames
            })
            
            # Generate the sliding windows for training
            if num_frames > self.window_size:
                step = self.window_size - self.overlap
                for start in range(0, num_frames - self.window_size + 1, step):
                    self.windows.append({
                        'video_id': vid,
                        'start_idx': start,
                        'end_idx': start + self.window_size
                    })
                # Add the final window if it doesn't align perfectly
                if start + self.window_size < num_frames:
                    self.windows.append({
                        'video_id': vid,
                        'start_idx': num_frames - self.window_size,
                        'end_idx': num_frames
                    })
            else:
                # If the video is shorter than the window, just use the whole video
                self.windows.append({
                    'video_id': vid,
                    'start_idx': 0,
                    'end_idx': num_frames
                })
        
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
        if self.use_full_length:
            window_info = self.samples[idx]
        else:
            window_info = self.windows[idx]
            
        vid = window_info['video_id']
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']
        
        feature_path = os.path.join(self.keypoints_dir, f"{vid}.npy")
        label_path = os.path.join(self.labels_dir, f"{vid}.npy")
        
        # 1. Load full raw data
        full_raw_array = np.load(feature_path)
        label_array = np.load(label_path)
        full_raw_tensor = torch.tensor(full_raw_array, dtype=torch.float32)
        
        # Apply median filter for noise smoothing
        full_raw_np = full_raw_tensor.numpy()
        full_raw_np = medfilt(full_raw_np, kernel_size=(5, 1, 1))
        full_raw_tensor = torch.tensor(full_raw_np, dtype=torch.float32)
        
        # 2. Calculate full-sequence kinematics (Velocity, Accel, Jerk)
        v_full = torch.zeros_like(full_raw_tensor)
        v_full[1:] = full_raw_tensor[1:] - full_raw_tensor[:-1]
        
        a_full = torch.zeros_like(v_full)
        a_full[1:] = v_full[1:] - v_full[:-1]
        
        j_full = torch.zeros_like(a_full)
        j_full[1:] = a_full[1:] - a_full[:-1]
        
        v_mag = torch.sqrt(torch.sum(v_full**2, dim=-1, keepdim=True))
        cross_prod = v_full[:, :, 0] * a_full[:, :, 1] - v_full[:, :, 1] * a_full[:, :, 0]
        v_mag_sq = v_mag.squeeze(-1)**2 + 1e-6
        omega = (cross_prod / v_mag_sq).unsqueeze(-1)
        
        # 3. Stack requested features
        final_channels = []
        
        # --- FIX: Dynamically add any requested base features ---
        base_indices = [self.feature_map[f] for f in self.base_features if f in self.feature_map]
        if base_indices:
            final_channels.append(full_raw_tensor[:, :, base_indices])
            
        # Determine the indices for derivative calculation (Default to x,y,z if none provided)
        deriv_indices = base_indices if base_indices else [0, 1, 2]
        
        if "velocity" in self.kinematic_features:
            final_channels.append(v_full[:, :, deriv_indices])
        if "acceleration" in self.kinematic_features:
            final_channels.append(a_full[:, :, deriv_indices])
        if "jerk" in self.kinematic_features:
            final_channels.append(j_full[:, :, deriv_indices])
        if "velocity-mag" in self.kinematic_features:
            final_channels.append(v_mag)
        if "angular-vel" in self.kinematic_features:
            final_channels.append(omega)
            
        # --- Skeletal Angles ---
        if "spatial_angles" in self.kinematic_features:
            s_angles = get_spatial_angles(full_raw_tensor)
            final_channels.append(s_angles)
            
        if "temporal_angles" in self.kinematic_features:
            t_angles = get_temporal_angles(full_raw_tensor)
            final_channels.append(t_angles)
            
        # 4. Construct the final tensor: Shape (Frames, Vertices, Total_Channels)
        full_feature_tensor = torch.cat(final_channels, dim=-1)
        
        # 5. Extract the required window slice
        window_features = full_feature_tensor[start_idx:end_idx]
        window_labels = label_array[start_idx:end_idx]
        
        # 6. Apply Label Smoothing
        soft_labels = apply_label_smoothing(window_labels, self.tolerance_window)
        
        # Permute to match PyTorch / ST-GCN expectations
        # Features: (Channels, Time, Vertices)
        final_input_tensor = window_features.permute(2, 0, 1)
        # Labels: (Classes, Time)
        labels_tensor = torch.tensor(soft_labels, dtype=torch.float32).permute(1, 0)
        
        # 7. Apply Temporal Downsampling (If requested)
        if self.temporal_downsample_factor > 1:
            final_input_tensor = final_input_tensor[:, ::self.temporal_downsample_factor, :]
            labels_tensor = labels_tensor[:, ::self.temporal_downsample_factor]
            
        # 8. Padding (Only needed if NOT using full length and window is short)
        if not self.use_full_length:
            C, T, V = final_input_tensor.shape
            target_T = self.window_size // self.temporal_downsample_factor
            
            if T < target_T:
                pad_T = target_T - T
                # Pad Features
                feat_pad = torch.zeros(C, pad_T, V, dtype=torch.float32)
                final_input_tensor = torch.cat([final_input_tensor, feat_pad], dim=1)
                
                # Pad Labels (Pad with Class 0: 'Outside')
                label_pad = torch.zeros(3, pad_T, dtype=torch.float32)
                label_pad[0, :] = 1.0 
                labels_tensor = torch.cat([labels_tensor, label_pad], dim=1)
                
        return final_input_tensor, labels_tensor