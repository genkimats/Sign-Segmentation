import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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

class SignSegmentationDataset(Dataset):
    def __init__(self, keypoints_dir, labels_dir, split_file="dataset_splits.json", split="train", window_size=1000, overlap=200, tolerance_window=5, use_full_length=False, base_features=None, kinematic_features=None, temporal_downsample_factor=1):
        self.labels_dir = labels_dir
        self.kinetic_dir = "processed_data/kinematic_features" 
        self.split_file = split_file
        self.split = split
        self.window_size = window_size
        self.overlap = overlap 
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
            
        self.video_ids = [vid.replace('.npy', '') for vid in splits[split]]
        
        self.samples = []  
        self.windows = []  

        print(f"[{split.upper()}] Indexing {len(self.video_ids)} videos (Streaming from Disk)...")
        for vid in self.video_ids:
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")
            
            # Skip if the preprocessed feature file doesn't exist yet
            if not os.path.exists(label_path) or not os.path.exists(kinetic_path):
                continue
                
            labels = np.load(label_path)
            num_frames = len(labels)
            
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
        
        # 1. Load Labels
        window_labels = np.load(os.path.join(self.labels_dir, f"{vid}.npy"))[start_idx:end_idx]
        soft_labels = apply_label_smoothing(window_labels, self.tolerance_window)
        labels_tensor = torch.tensor(soft_labels, dtype=torch.float32).permute(1, 0)
        
        # 2. Stream Precomputed Kinematics from SSD
        kin_data = torch.load(os.path.join(self.kinetic_dir, f"{vid}.pt"), weights_only=False)

        # 3. Assemble Custom Tensor dynamically
        channels = []
        
        base_indices = [self.feature_map[f] for f in self.base_features if f in self.feature_map]
        if base_indices:
            channels.append(kin_data["base"][:, :, base_indices])
            
        deriv_indices = base_indices if base_indices else [0, 1, 2]
        
        if "velocity" in self.kinematic_features: channels.append(kin_data["velocity"][:, :, deriv_indices])
        if "acceleration" in self.kinematic_features: channels.append(kin_data["acceleration"][:, :, deriv_indices])
        if "jerk" in self.kinematic_features: channels.append(kin_data["jerk"][:, :, deriv_indices])
        if "velocity-mag" in self.kinematic_features: channels.append(kin_data["velocity-mag"])
        if "angular-vel" in self.kinematic_features: channels.append(kin_data["angular-vel"])
        if "spatial_angles" in self.kinematic_features: channels.append(kin_data["spatial_angles"])
        if "temporal_angles" in self.kinematic_features: channels.append(kin_data["temporal_angles"])
            
        # Convert back from FP16 to FP32, slice window, format for PyTorch
        final_input_tensor = torch.cat(channels, dim=-1).to(torch.float32)[start_idx:end_idx].permute(2, 0, 1)

        # 4. Handle Downsampling & Padding
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
                
        start_scaled = start_idx // self.temporal_downsample_factor
        end_scaled = end_idx // self.temporal_downsample_factor
        
        return final_input_tensor, labels_tensor, vid, start_scaled, end_scaled