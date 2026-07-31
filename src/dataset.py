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

    soft_labels[np.arange(T), labels_array] = 1.0

    if window_size <= 1:
        return soft_labels

    spread = window_size // 2
    sigma = spread / 2.0 if spread > 0 else 1.0

    begin_indices = np.where(labels_array == 2)[0]

    for i in begin_indices:
        for d in range(-spread, spread + 1):
            idx = i + d
            if 0 <= idx < T:
                weight = np.exp(-(d**2) / (2 * sigma**2))
                if weight > soft_labels[idx, 2]:
                    soft_labels[idx, 2] = weight
                    remaining_prob = 1.0 - weight
                    original_class = labels_array[idx]
                    
                    if original_class != 2:
                        soft_labels[idx, original_class] = remaining_prob
                        other_class = 1 if original_class == 0 else 0
                        soft_labels[idx, other_class] = 0.0
                    else:
                        soft_labels[idx, 0] = remaining_prob / 2
                        soft_labels[idx, 1] = remaining_prob / 2

    sums = soft_labels.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    soft_labels = soft_labels / sums
    return soft_labels

def get_spatial_angles(features):
    anchor = features[:, 0:1, :3] 
    rel_vec = features[:, :, :3] - anchor
    x, y, z = rel_vec[:, :, 0], rel_vec[:, :, 1], rel_vec[:, :, 2]
    
    r = torch.sqrt(x**2 + y**2 + z**2) + 1e-6
    theta = torch.acos(z / r) / np.pi
    phi = torch.atan2(y, x) / np.pi
    return torch.stack([theta, phi], dim=-1)

def get_temporal_angles(features):
    motion_vec = torch.zeros_like(features[:, :, :3])
    motion_vec[1:] = features[1:, :, :3] - features[:-1, :, :3]
    x, y, z = motion_vec[:, :, 0], motion_vec[:, :, 1], motion_vec[:, :, 2]
    
    r = torch.sqrt(x**2 + y**2 + z**2) + 1e-6
    theta = torch.acos(z / r) / np.pi
    phi = torch.atan2(y, x) / np.pi
    return torch.stack([theta, phi], dim=-1)

class SignSegmentationDataset(Dataset):
    def __init__(self, keypoints_dir, labels_dir, split_file="dataset_splits.json", split="train", window_size=1000, overlap=200, tolerance_window=5, use_full_length=False, base_features=None, kinematic_features=None, temporal_downsample_factor=1):
        self.keypoints_dir = keypoints_dir
        self.labels_dir = labels_dir
        self.split_file = split_file
        self.split = split
        self.window_size = window_size
        self.overlap = overlap 
        self.step_size = window_size - overlap
        self.tolerance_window = tolerance_window
        self.use_full_length = use_full_length
        self.temporal_downsample_factor = temporal_downsample_factor

        self.feature_map = {
            "x-cord": 0,
            "y-cord": 1,
            "z-cord": 2
        }
        
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "z-cord"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
            
        self.video_ids = [vid.replace('.npy', '').replace('.pt', '') for vid in splits[split]]
        
        self.samples = []  
        self.windows = []  
        
        # --- NEW: CACHE DICTIONARIES ---
        self.video_cache = {}

        print(f"[{split.upper()}] Pre-computing physics and caching {len(self.video_ids)} videos to RAM...")
        for vid in self.video_ids: # Removed sorted() to speed up slightly, not strictly needed
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            feature_path = os.path.join(self.keypoints_dir, f"{vid}.npy")
            
            if not os.path.exists(label_path) or not os.path.exists(feature_path):
                continue
                
            labels = np.load(label_path)
            num_frames = len(labels)
            
            # --- 🚀 LOAD AND COMPUTE PHYSICS ONCE PER VIDEO ---
            full_raw_array = np.load(feature_path)
            full_raw_tensor = torch.tensor(full_raw_array, dtype=torch.float32)
            
            full_raw_np = full_raw_tensor.numpy()
            full_raw_np = medfilt(full_raw_np, kernel_size=(5, 1, 1))
            full_raw_tensor = torch.tensor(full_raw_np, dtype=torch.float32)
            
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
            
            final_channels = []
            base_indices = [self.feature_map[f] for f in self.base_features if f in self.feature_map]
            if base_indices:
                final_channels.append(full_raw_tensor[:, :, base_indices])
                
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
                
            if "spatial_angles" in self.kinematic_features:
                s_angles = get_spatial_angles(full_raw_tensor)
                final_channels.append(s_angles)
                
            if "temporal_angles" in self.kinematic_features:
                t_angles = get_temporal_angles(full_raw_tensor)
                final_channels.append(t_angles)
                
            full_feature_tensor = torch.cat(final_channels, dim=-1)
            
            # Store the heavily computed tensor and labels in RAM
            self.video_cache[vid] = {
                'features': full_feature_tensor,
                'labels': labels
            }
            # ---------------------------------------------------
            
            self.samples.append({
                'video_id': vid,
                'start_idx': 0,
                'end_idx': num_frames
            })
            
            if num_frames > self.window_size:
                step = self.window_size - self.overlap
                for start in range(0, num_frames - self.window_size + 1, step):
                    self.windows.append({
                        'video_id': vid,
                        'start_idx': start,
                        'end_idx': start + self.window_size
                    })
                if start + self.window_size < num_frames:
                    self.windows.append({
                        'video_id': vid,
                        'start_idx': num_frames - self.window_size,
                        'end_idx': num_frames
                    })
            else:
                self.windows.append({
                    'video_id': vid,
                    'start_idx': 0,
                    'end_idx': num_frames
                })

    def __len__(self):
        if self.use_full_length:
            return len(self.samples)
        return len(self.windows)

    def __getitem__(self, idx):
        if self.use_full_length:
            window_info = self.samples[idx]
        else:
            window_info = self.windows[idx]
            
        vid = window_info['video_id']
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']
        
        # --- 🚀 LIGHTNING FAST RETRIEVAL ---
        cached_data = self.video_cache[vid]
        window_features = cached_data['features'][start_idx:end_idx]
        window_labels = cached_data['labels'][start_idx:end_idx]
        
        soft_labels = apply_label_smoothing(window_labels, self.tolerance_window)
        
        final_input_tensor = window_features.permute(2, 0, 1)
        labels_tensor = torch.tensor(soft_labels, dtype=torch.float32).permute(1, 0)
        
        if self.temporal_downsample_factor > 1:
            final_input_tensor = final_input_tensor[:, ::self.temporal_downsample_factor, :]
            labels_tensor = labels_tensor[:, ::self.temporal_downsample_factor]
            
        if not self.use_full_length:
            C, T, V = final_input_tensor.shape
            target_T = self.window_size // self.temporal_downsample_factor
            
            if T < target_T:
                pad_T = target_T - T
                feat_pad = torch.zeros(C, pad_T, V, dtype=torch.float32)
                final_input_tensor = torch.cat([final_input_tensor, feat_pad], dim=1)
                
                label_pad = torch.zeros(3, pad_T, dtype=torch.float32)
                label_pad[0, :] = 1.0 
                labels_tensor = torch.cat([labels_tensor, label_pad], dim=1)
                
        return final_input_tensor, labels_tensor