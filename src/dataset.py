import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

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
    def __init__(self, keypoints_dir, labels_dir, split_file="dataset_splits.json", split="train", window_size=16, overlap=0, tolerance_window=5, use_full_length=False, base_features=None, kinematic_features=None, temporal_downsample_factor=1, use_face_keypoints=False, face_dir="processed_data/face_keypoints_normalized"):
        self.labels_dir = labels_dir
        self.kinetic_dir = "processed_data/kinematic_features" 
        self.split_file = split_file
        self.split = split
        self.window_size = window_size
        self.overlap = overlap 
        self.tolerance_window = tolerance_window
        self.use_full_length = use_full_length
        self.temporal_downsample_factor = temporal_downsample_factor
        self.use_face_keypoints = use_face_keypoints
        self.face_dir = face_dir

        self.feature_map = {
            "x-cord": 0,
            "y-cord": 1,
            "z-cord": 2
        }
        
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "z-cord"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []

        if self.use_face_keypoints:
            # IMPORTANT: this changes the vertex axis (65 -> 65 + NUM_FACE_VERTICES), not the
            # channel axis. Graph-based models (STGCN_Mamba, Decoupled_STGCN_Mamba, etc.) read
            # their adjacency matrix from src/graph.py's SkeletonGraph, which is still hardcoded
            # to 65 vertices -- using face keypoints with those models WILL fail with a shape
            # mismatch in SpatialGraphConv until SkeletonGraph is extended to include face
            # vertices/edges. Non-graph models (PureMambaBaseline, BiMambaBaseline) work as-is,
            # since they don't depend on a fixed adjacency structure.
            print(f"[{split.upper()}] use_face_keypoints=True -- vertex axis will be "
                  f"65 + face-vertex-count (see processed_data/face_keypoints/*.npy shapes). "
                  f"Make sure num_vertices in your config/queue matches, and that you're using "
                  f"a non-graph model unless src/graph.py has been extended for face vertices.")
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
            
        self.video_ids = [vid.replace('.npy', '').replace('.pt', '') for vid in splits[split]]
        
        self.samples = []  
        self.windows = []  
        
        # --- THE RAM CACHE ---
        self.video_cache = {}

        print(f"[{split.upper()}] Loading {len(self.video_ids)} videos into RAM Cache (Bypassing Disk I/O)...")
        
        # Load everything into RAM exactly ONCE
        for vid in tqdm(self.video_ids, desc=f"Caching {split}"):
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")
            
            if not os.path.exists(label_path) or not os.path.exists(kinetic_path):
                continue
                
            labels = np.load(label_path)
            num_frames = len(labels)
            
            # CPU Bottleneck Fix: Pre-calculate smoothing once per video, not once per window!
            soft_labels = apply_label_smoothing(labels, self.tolerance_window)
            
            try:
                kin_data = torch.load(kinetic_path, weights_only=False)
                # Robustly handle dictionary extraction
                if "mediapipe" in kin_data:
                    kin_data = kin_data["mediapipe"]
            except Exception:
                continue

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
                
            final_tensor = torch.cat(channels, dim=-1)  # (T, 65, K) -- defer FP16 cast until after optional face concat

            # --- OPTIONAL: FACE KEYPOINTS (from extract_face_keypoints.py) ---
            if self.use_face_keypoints:
                face_path = os.path.join(self.face_dir, f"{vid}.npy")
                if not os.path.exists(face_path):
                    continue  # keep vertex layout consistent across the whole split; don't silently zero-fill a missing video

                face_raw = torch.from_numpy(np.load(face_path)).float()  # (T_face, NUM_FACE_VERTICES, 3)
                if face_raw.shape[0] != num_frames:
                    continue  # frame count doesn't match labels/body -- skip rather than risk misaligning them

                face_selected = face_raw[:, :, base_indices] if base_indices else face_raw

                # extract_face_keypoints.py only outputs static x/y/z -- there's no face
                # velocity/acceleration/angle equivalent yet, so any *derivative* channels
                # requested via kinematic_features are zero-padded for the face vertices.
                K_total = final_tensor.shape[-1]
                face_padded = torch.zeros(num_frames, face_selected.shape[1], K_total, dtype=final_tensor.dtype)
                face_padded[:, :, :face_selected.shape[-1]] = face_selected

                # These are shoulder-normalized coordinates (see normalize_face_keypoints.py --
                # same (coords - shoulder_midpoint) / shoulder_xy_distance transform as body/
                # hands use), so this vertex block lives in the same coordinate frame as the
                # rest of final_tensor. If you point face_dir back at the raw, unnormalized
                # processed_data/face_keypoints/ directory instead, that guarantee no longer
                # holds -- only do that deliberately (e.g. for debugging).
                final_tensor = torch.cat([final_tensor, face_padded], dim=1)  # concat along the VERTEX axis

            # Compress to FP16 to keep RAM super low, and save it to the dictionary!
            final_tensor = final_tensor.to(torch.float16)
            
            self.video_cache[vid] = {
                'features': final_tensor,
                'labels': soft_labels
            }
            
            # Map out the windows only for videos successfully loaded
            self.samples.append({'video_id': vid, 'start_idx': 0, 'end_idx': num_frames})
            
            if num_frames > self.window_size:
                step = self.window_size - self.overlap
                for start in range(0, num_frames - self.window_size + 1, step):
                    self.windows.append({'video_id': vid, 'start_idx': start, 'end_idx': start + self.window_size})
                if start + self.window_size < num_frames:
                    self.windows.append({'video_id': vid, 'start_idx': num_frames - self.window_size, 'end_idx': num_frames})
            else:
                self.windows.append({'video_id': vid, 'start_idx': 0, 'end_idx': num_frames})

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
        
        # --- ⚡ 0-LATENCY MEMORY SLICE ⚡ ---
        cached_data = self.video_cache[vid]
        
        # Instantly slice the data and restore to FP32 for PyTorch's mathematical operations
        window_features = cached_data['features'][start_idx:end_idx].to(torch.float32)
        window_labels = cached_data['labels'][start_idx:end_idx]
        
        final_input_tensor = window_features.permute(2, 0, 1)
        labels_tensor = torch.tensor(window_labels, dtype=torch.float32).permute(1, 0)
        
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