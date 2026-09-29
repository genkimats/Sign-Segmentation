"""
Dataset for Stage 1 reconstruction pretraining: identical windowing/caching
pattern to src/dataset.py, but needs NO BIO labels at all -- reconstruction
is self-supervised (the target is the input itself). This means it can, in
principle, use every video with extracted keypoints, not just the ones with
gloss annotations -- though this version still reads from the same split
file for simplicity and reproducibility with Stage 2's labeled split.

This file lives in Sign-Segmentation/latent_transformer/. Paths are resolved
relative to THIS FILE's own location, not the terminal's current directory.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

DEFAULT_KINETIC_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "kinematic_features")
DEFAULT_SPLIT_FILE = os.path.join(_PROJECT_ROOT, "dataset_splits.json")


class ReconstructionDataset(Dataset):
    def __init__(self, split_file=None, split="train", window_size=64, overlap=0,
                 base_features=None, kinematic_features=None):
        self.window_size = window_size
        self.overlap = overlap
        self.kinetic_dir = DEFAULT_KINETIC_DIR
        self.feature_map = {"x-cord": 0, "y-cord": 1, "z-cord": 2}
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "z-cord"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []

        if split_file is None:
            split_file = DEFAULT_SPLIT_FILE
        with open(split_file, 'r') as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
        self.video_ids = [vid.replace('.npy', '').replace('.pt', '') for vid in splits[split]]

        self.windows = []
        self.video_cache = {}
        skip_counts = {"missing_kinetic": 0, "kinetic_load_error": 0}

        print(f"[{split.upper()}] (reconstruction pretraining, no labels needed) "
              f"Loading {len(self.video_ids)} videos into RAM cache...")
        for vid in tqdm(self.video_ids, desc=f"Caching {split}"):
            kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")
            if not os.path.exists(kinetic_path):
                skip_counts["missing_kinetic"] += 1
                continue
            try:
                kin_data = torch.load(kinetic_path, weights_only=False)
                if "mediapipe" in kin_data:
                    kin_data = kin_data["mediapipe"]
            except Exception:
                skip_counts["kinetic_load_error"] += 1
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

            final_tensor = torch.cat(channels, dim=-1).to(torch.float16)
            num_frames = final_tensor.shape[0]
            self.video_cache[vid] = final_tensor

            if num_frames > self.window_size:
                step = self.window_size - self.overlap
                start = 0
                for start in range(0, num_frames - self.window_size + 1, step):
                    self.windows.append((vid, start, start + self.window_size))
                if start + self.window_size < num_frames:
                    self.windows.append((vid, num_frames - self.window_size, num_frames))
            else:
                self.windows.append((vid, 0, num_frames))

        total_cached = len(self.video_cache)
        if total_cached < len(self.video_ids):
            print(f"[{split.upper()}] Cached {total_cached}/{len(self.video_ids)} videos. "
                  f"Skip reasons: {skip_counts}")

        if len(self.windows) == 0:
            raise RuntimeError(f"[{split.upper()}] ReconstructionDataset ended up with 0 windows -- "
                               f"EMPTY. Skip reasons: {skip_counts}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        vid, start, end = self.windows[idx]
        window = self.video_cache[vid][start:end].to(torch.float32)  # (T_win, V, C)
        T_win = window.shape[0]

        final_input_tensor = window.permute(2, 0, 1)  # (C, T_win, V)

        if T_win < self.window_size:
            C, _, V = final_input_tensor.shape
            pad = torch.zeros(C, self.window_size - T_win, V, dtype=torch.float32)
            final_input_tensor = torch.cat([final_input_tensor, pad], dim=1)

        # valid_length lets the training script mask padded frames OUT of the
        # reconstruction loss -- padding the input with zeros and then asking
        # the model to reconstruct those same zeros is a trivial, meaningless
        # objective for those frames, unlike BIO tagging where padding has a
        # genuine "Outside" label to predict.
        return final_input_tensor, T_win, vid