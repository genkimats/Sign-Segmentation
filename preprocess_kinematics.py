import os
import json
import torch
import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm

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

def compute_kinematics(tensor):
    v = torch.zeros_like(tensor)
    v[1:] = tensor[1:] - tensor[:-1]
    a = torch.zeros_like(v)
    a[1:] = v[1:] - v[:-1]
    j = torch.zeros_like(a)
    j[1:] = a[1:] - a[:-1]

    v_mag = torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True))
    cross_prod = v[:, :, 0] * a[:, :, 1] - v[:, :, 1] * a[:, :, 0]
    v_mag_sq = v_mag.squeeze(-1)**2 + 1e-6
    omega = (cross_prod / v_mag_sq).unsqueeze(-1)
    
    return {
        "base": tensor,
        "velocity": v,
        "acceleration": a,
        "jerk": j,
        "velocity-mag": v_mag,
        "angular-vel": omega,
        "spatial_angles": get_spatial_angles(tensor),
        "temporal_angles": get_temporal_angles(tensor)
    }

if __name__ == "__main__":
    split_file = "dataset_splits.json"
    keypoints_dir = "processed_data/keypoints"
    out_dir = "processed_data/kinematic_features"
    
    os.makedirs(out_dir, exist_ok=True)
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
        
    all_vids = set(splits['train'] + splits['val'] + splits.get('test', []))
    all_vids = [v.replace('.npy', '') for v in all_vids]
    
    print(f"🚀 Found {len(all_vids)} total videos. Generating Offline Kinematic Database...")
    
    for vid in tqdm(all_vids):
        out_path = os.path.join(out_dir, f"{vid}.pt")
        if os.path.exists(out_path): 
            continue # Skip if already built!
            
        mp_path = os.path.join(keypoints_dir, f"{vid}.npy")
        if not os.path.exists(mp_path):
            continue
        
        # 1. Load and Clean the MediaPipe raw data
        mp_raw = np.load(mp_path)
        mp_raw = medfilt(mp_raw, kernel_size=(5, 1, 1))
        mp_tensor = torch.tensor(mp_raw, dtype=torch.float32)
        
        # 2. Compute physics
        vid_data = compute_kinematics(mp_tensor)
        
        # 3. FP16 Compression (Saves 50% Disk Space instantly)
        for feat_key, tensor in vid_data.items():
            vid_data[feat_key] = tensor.to(torch.float16)

        torch.save(vid_data, out_path)