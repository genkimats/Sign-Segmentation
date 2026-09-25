"""
Dataset for DETR-style set-prediction segmentation: returns FULL-LENGTH
videos (no windowing/chunking at all) along with ground-truth SEGMENTS as a
list of (start, end) frame-index tuples, instead of per-frame BIO labels --
this is the data format set prediction needs, since it directly predicts
spans rather than classifying every frame.

Reuses the exact same feature-loading logic as src/dataset.py (kinematic
features, HaMeR, DINOv2) -- only the label representation differs.

DESIGN CHOICE: batch_size=1 (one full video per forward pass). Since
sequences are full, variable length (not fixed windows), proper batching
would need padding + attention masks. This mirrors how use_full_length=True
already forces batch_size=1 elsewhere in this codebase, and prioritizes a
correct, working first version of a genuinely new paradigm over a training-
speed optimization that can be added later if it becomes a bottleneck.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

# This file lives in Sign-Segmentation/detr/. processed_data/ and
# dataset_splits.json live in Sign-Segmentation/ itself (one level up) --
# computed from THIS FILE's own location, not the terminal's current
# directory, so this resolves correctly no matter where the importing
# script is actually run from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bio_to_segments(bio_array):
    """
    Converts a per-frame BIO array (0=Outside, 1=Begin, 2=Inside) into a list
    of (start, end) tuples, end EXCLUSIVE -- one per B...I* run. A lone Begin
    frame with no following Inside frames still produces a valid 1-frame
    segment (start, start+1).
    """
    segments = []
    begin_positions = np.where(bio_array == 1)[0]
    for start in begin_positions:
        end = int(start) + 1
        while end < len(bio_array) and bio_array[end] == 2:
            end += 1
        segments.append((int(start), end))
    return segments


class SignSegmentationDatasetDETR(Dataset):
    def __init__(self, keypoints_dir, labels_dir, split_file=None, split="train",
                 base_features=None, kinematic_features=None,
                 use_hamer_features=False, hamer_dir=None,
                 use_dinov2_features=False, dinov2_dir=None):
        self.labels_dir = labels_dir
        self.kinetic_dir = os.path.join(_PROJECT_ROOT, "processed_data", "kinematic_features")
        self.use_hamer_features = use_hamer_features
        self.hamer_dir = hamer_dir if hamer_dir is not None else os.path.join(_PROJECT_ROOT, "processed_data", "hamer_features")
        self.use_dinov2_features = use_dinov2_features
        self.dinov2_dir = dinov2_dir if dinov2_dir is not None else os.path.join(_PROJECT_ROOT, "processed_data", "dinov2_features")

        if split_file is None:
            split_file = os.path.join(_PROJECT_ROOT, "dataset_splits.json")

        self.feature_map = {"x-cord": 0, "y-cord": 1, "z-cord": 2}
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "z-cord"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []

        with open(split_file, 'r') as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
        self.video_ids = [vid.replace('.npy', '').replace('.pt', '') for vid in splits[split]]

        self.detected_hamer_dim = None
        self.detected_dinov2_dim = None

        skip_counts = {
            "missing_label_or_kinetic": 0, "kinetic_load_error": 0,
            "missing_hamer": 0, "hamer_load_error": 0,
            "missing_dinov2": 0, "dinov2_load_error": 0, "dinov2_frame_mismatch": 0,
            "no_segments": 0,
        }

        self.valid_ids = []
        print(f"[{split.upper()}] (DETR mode) Validating {len(self.video_ids)} videos...")

        for vid in self.video_ids:
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")
            if not os.path.exists(label_path) or not os.path.exists(kinetic_path):
                skip_counts["missing_label_or_kinetic"] += 1
                continue

            labels = np.load(label_path)
            segments = bio_to_segments(labels)
            if len(segments) == 0:
                skip_counts["no_segments"] += 1
                continue

            if self.use_hamer_features and not os.path.exists(os.path.join(self.hamer_dir, f"{vid}_hamer.pt")):
                skip_counts["missing_hamer"] += 1
                continue
            if self.use_dinov2_features and not os.path.exists(os.path.join(self.dinov2_dir, f"{vid}_dinov2.pt")):
                skip_counts["missing_dinov2"] += 1
                continue

            self.valid_ids.append(vid)

        if len(self.valid_ids) < len(self.video_ids):
            print(f"[{split.upper()}] {len(self.valid_ids)}/{len(self.video_ids)} videos passed validation. "
                  f"Skip reasons: {skip_counts}")

        if len(self.valid_ids) == 0:
            raise RuntimeError(
                f"[{split.upper()}] SignSegmentationDatasetDETR ended up with 0 valid videos -- EMPTY. "
                f"Skip reasons: {skip_counts}. Same naming-convention-mismatch causes as the other "
                f"dataset classes in this codebase apply here too."
            )

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        vid = self.valid_ids[idx]

        labels = np.load(os.path.join(self.labels_dir, f"{vid}.npy"))
        num_frames = len(labels)
        segments = bio_to_segments(labels)

        kin_data = torch.load(os.path.join(self.kinetic_dir, f"{vid}.pt"), weights_only=False)
        if "mediapipe" in kin_data:
            kin_data = kin_data["mediapipe"]

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

        final_tensor = torch.cat(channels, dim=-1)  # (T, V, C)
        final_input_tensor = final_tensor.permute(2, 0, 1).to(torch.float32)  # (C, T, V)

        hamer_tensor = None
        if self.use_hamer_features:
            hamer_data = torch.load(os.path.join(self.hamer_dir, f"{vid}_hamer.pt"), weights_only=False)
            hand_pose = hamer_data["hand_pose"]
            global_orient = hamer_data["global_orient"]
            ham_downsample = hamer_data.get("temporal_downsample_factor", 1)
            T_ham = hand_pose.shape[0]
            hand_pose_flat = torch.as_tensor(hand_pose, dtype=torch.float32).reshape(T_ham, 270)
            global_orient_flat = torch.as_tensor(global_orient, dtype=torch.float32).reshape(T_ham, 18)
            hamer_flat = torch.cat([hand_pose_flat, global_orient_flat], dim=-1)
            frame_to_hamer_idx = (torch.arange(num_frames) // ham_downsample).clamp(max=T_ham - 1)
            hamer_full = hamer_flat[frame_to_hamer_idx]
            hamer_tensor = hamer_full.permute(1, 0)  # (288, T)
            if self.detected_hamer_dim is None:
                self.detected_hamer_dim = hamer_tensor.shape[0]

        dinov2_tensor = None
        if self.use_dinov2_features:
            dinov2_data = torch.load(os.path.join(self.dinov2_dir, f"{vid}_dinov2.pt"), weights_only=False)
            dinov2_feats = dinov2_data["features"]
            D = dinov2_feats.shape[-1]
            dinov2_full = torch.as_tensor(dinov2_feats, dtype=torch.float32).reshape(num_frames, 2 * D)
            dinov2_tensor = dinov2_full.permute(1, 0)  # (2*D, T)
            if self.detected_dinov2_dim is None:
                self.detected_dinov2_dim = dinov2_tensor.shape[0]

        # Ground-truth segments, NORMALIZED to [0,1] fractions of num_frames --
        # matching DETR's own convention of predicting scale-invariant, well-
        # conditioned normalized box/span coordinates rather than raw frame
        # indices (which could be in the thousands for a long video).
        segments_frac = torch.tensor(
            [[s / num_frames, e / num_frames] for s, e in segments], dtype=torch.float32
        )

        return final_input_tensor, hamer_tensor, dinov2_tensor, segments_frac, vid, num_frames