"""
Multi-task version of src/dataset.py's SignSegmentationDataset: identical
window-based caching/padding pipeline (same window_size/overlap semantics,
same RAM-cache-everything-once design, same feature streams), with gloss-ID
labels loaded and windowed IN SYNC with everything else, for the auxiliary
gloss-classification head (see models_multitask.py / train_multitask.py).

Deliberately reuses the EXISTING, PROVEN windowing paradigm rather than
introducing a new one -- this is the direct lesson from the DETR experiment,
where a genuinely new paradigm (full-sequence set prediction) surfaced
memory-scaling problems that took many rounds to fully resolve. Multi-task
learning is a much smaller, safer change: same architecture family, same
training loop shape, one additional label stream and one additional loss
term.

This file lives in Sign-Segmentation/multitask_learning/. Paths are resolved
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

# Needed to reuse apply_label_smoothing exactly as the existing pipeline does,
# so the BIO-tag half of this dataset behaves identically to every other
# training script in this project.
import sys
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.dataset import apply_label_smoothing

DEFAULT_GLOSS_IDS_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "gloss_ids")
DEFAULT_KINETIC_DIR = os.path.join(_PROJECT_ROOT, "processed_data", "kinematic_features")


class SignSegmentationDatasetMultitask(Dataset):
    def __init__(self, keypoints_dir, labels_dir, split_file=None, split="train",
                 window_size=64, overlap=0, tolerance_window=5, use_full_length=False,
                 base_features=None, kinematic_features=None, temporal_downsample_factor=1,
                 use_hamer_features=False, hamer_dir=None,
                 use_dinov2_features=False, dinov2_dir=None,
                 gloss_ids_dir=None):
        self.labels_dir = labels_dir
        self.kinetic_dir = DEFAULT_KINETIC_DIR
        self.split = split
        self.window_size = window_size
        self.overlap = overlap
        self.tolerance_window = tolerance_window
        self.use_full_length = use_full_length
        self.temporal_downsample_factor = temporal_downsample_factor
        self.use_hamer_features = use_hamer_features
        self.hamer_dir = hamer_dir if hamer_dir is not None else os.path.join(_PROJECT_ROOT, "processed_data", "hamer_features")
        self.use_dinov2_features = use_dinov2_features
        self.dinov2_dir = dinov2_dir if dinov2_dir is not None else os.path.join(_PROJECT_ROOT, "processed_data", "dinov2_features")
        self.gloss_ids_dir = gloss_ids_dir if gloss_ids_dir is not None else DEFAULT_GLOSS_IDS_DIR

        self.feature_map = {"x-cord": 0, "y-cord": 1, "z-cord": 2}
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "z-cord"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []

        if split_file is None:
            split_file = os.path.join(_PROJECT_ROOT, "dataset_splits.json")
        with open(split_file, 'r') as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
        self.video_ids = [vid.replace('.npy', '').replace('.pt', '') for vid in splits[split]]

        self.samples = []
        self.windows = []
        self.video_cache = {}
        self.detected_hamer_dim = None
        self.detected_dinov2_dim = None

        # Loaded once, here, so train_multitask.py can read vocab_size without
        # re-parsing the file itself.
        vocab_path = os.path.join(self.gloss_ids_dir, "gloss_vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"{vocab_path} not found -- run extract_gloss_ids.py first (it writes this "
                f"alongside the per-video gloss-id .npy files)."
            )
        with open(vocab_path) as f:
            vocab_info = json.load(f)
        self.gloss_vocab_size = vocab_info["vocab_size"]
        self.gloss_unk_id = vocab_info["unk_id"]

        skip_counts = {
            "missing_label_or_kinetic": 0, "kinetic_load_error": 0,
            "missing_hamer": 0, "hamer_load_error": 0, "hamer_empty": 0,
            "missing_dinov2": 0, "dinov2_load_error": 0, "dinov2_frame_mismatch": 0,
            "missing_gloss_ids": 0, "gloss_frame_mismatch": 0,
        }

        print(f"[{split.upper()}] (multitask) Loading {len(self.video_ids)} videos into RAM cache...")
        for vid in tqdm(self.video_ids, desc=f"Caching {split}"):
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")
            gloss_path = os.path.join(self.gloss_ids_dir, f"{vid}.npy")

            if not os.path.exists(label_path) or not os.path.exists(kinetic_path):
                skip_counts["missing_label_or_kinetic"] += 1
                continue
            if not os.path.exists(gloss_path):
                skip_counts["missing_gloss_ids"] += 1
                continue

            labels = np.load(label_path)
            num_frames = len(labels)
            soft_labels = apply_label_smoothing(labels, self.tolerance_window)

            gloss_ids = np.load(gloss_path)
            if len(gloss_ids) != num_frames:
                skip_counts["gloss_frame_mismatch"] += 1
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

            final_tensor = torch.cat(channels, dim=-1)

            hamer_full = None
            if self.use_hamer_features:
                hamer_path = os.path.join(self.hamer_dir, f"{vid}_hamer.pt")
                if not os.path.exists(hamer_path):
                    skip_counts["missing_hamer"] += 1
                    continue
                try:
                    hamer_data = torch.load(hamer_path, weights_only=False)
                    hand_pose = hamer_data["hand_pose"]
                    global_orient = hamer_data["global_orient"]
                    ham_downsample = hamer_data.get("temporal_downsample_factor", 1)
                except Exception:
                    skip_counts["hamer_load_error"] += 1
                    continue
                T_ham = hand_pose.shape[0]
                if T_ham == 0:
                    skip_counts["hamer_empty"] += 1
                    continue
                hand_pose_flat = torch.as_tensor(hand_pose, dtype=torch.float32).reshape(T_ham, 270)
                global_orient_flat = torch.as_tensor(global_orient, dtype=torch.float32).reshape(T_ham, 18)
                hamer_flat = torch.cat([hand_pose_flat, global_orient_flat], dim=-1)
                frame_to_hamer_idx = (torch.arange(num_frames) // ham_downsample).clamp(max=T_ham - 1)
                hamer_full = hamer_flat[frame_to_hamer_idx].to(torch.float16)
                if self.detected_hamer_dim is None:
                    self.detected_hamer_dim = hamer_full.shape[-1]

            dinov2_full = None
            if self.use_dinov2_features:
                dinov2_path = os.path.join(self.dinov2_dir, f"{vid}_dinov2.pt")
                if not os.path.exists(dinov2_path):
                    skip_counts["missing_dinov2"] += 1
                    continue
                try:
                    dinov2_data = torch.load(dinov2_path, weights_only=False)
                    dinov2_feats = dinov2_data["features"]
                except Exception:
                    skip_counts["dinov2_load_error"] += 1
                    continue
                if dinov2_feats.shape[0] != num_frames:
                    skip_counts["dinov2_frame_mismatch"] += 1
                    continue
                D = dinov2_feats.shape[-1]
                dinov2_full = torch.as_tensor(dinov2_feats, dtype=torch.float32).reshape(num_frames, 2 * D).to(torch.float16)
                if self.detected_dinov2_dim is None:
                    self.detected_dinov2_dim = 2 * D
                elif self.detected_dinov2_dim != 2 * D:
                    raise RuntimeError(
                        f"{vid}: dinov2 feature dim ({2 * D}) doesn't match the dimension "
                        f"detected from an earlier video in this split ({self.detected_dinov2_dim})."
                    )

            final_tensor = final_tensor.to(torch.float16)

            self.video_cache[vid] = {
                'features': final_tensor,
                'labels': soft_labels,
                'gloss_ids': gloss_ids,
            }
            if self.use_hamer_features:
                self.video_cache[vid]['hamer_features'] = hamer_full
            if self.use_dinov2_features:
                self.video_cache[vid]['dinov2_features'] = dinov2_full

            self.samples.append({'video_id': vid, 'start_idx': 0, 'end_idx': num_frames})

            if num_frames > self.window_size:
                step = self.window_size - self.overlap
                for start in range(0, num_frames - self.window_size + 1, step):
                    self.windows.append({'video_id': vid, 'start_idx': start, 'end_idx': start + self.window_size})
                if start + self.window_size < num_frames:
                    self.windows.append({'video_id': vid, 'start_idx': num_frames - self.window_size, 'end_idx': num_frames})
            else:
                self.windows.append({'video_id': vid, 'start_idx': 0, 'end_idx': num_frames})

        total_attempted = len(self.video_ids)
        total_cached = len(self.video_cache)
        if total_cached < total_attempted:
            print(f"[{split.upper()}] Cached {total_cached}/{total_attempted} videos "
                  f"({total_attempted - total_cached} skipped). Skip reasons: {skip_counts}")

        if len(self.windows) == 0:
            raise RuntimeError(
                f"[{split.upper()}] SignSegmentationDatasetMultitask ended up with 0 windows -- "
                f"EMPTY. Skip reasons: {skip_counts}. If missing_gloss_ids is nonzero, run "
                f"extract_gloss_ids.py first; if it's already been run, check that its output "
                f"directory matches gloss_ids_dir."
            )

    def __len__(self):
        return len(self.samples) if self.use_full_length else len(self.windows)

    def __getitem__(self, idx):
        window_info = self.samples[idx] if self.use_full_length else self.windows[idx]
        vid = window_info['video_id']
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']

        cached_data = self.video_cache[vid]

        window_features = cached_data['features'][start_idx:end_idx].to(torch.float32)
        window_labels = cached_data['labels'][start_idx:end_idx]
        window_gloss_ids = cached_data['gloss_ids'][start_idx:end_idx]

        final_input_tensor = window_features.permute(2, 0, 1)
        labels_tensor = torch.tensor(window_labels, dtype=torch.float32).permute(1, 0)
        gloss_ids_tensor = torch.tensor(window_gloss_ids, dtype=torch.long)  # (T_win,)

        if self.use_hamer_features:
            window_hamer = cached_data['hamer_features'][start_idx:end_idx].to(torch.float32)
            hamer_tensor = window_hamer.permute(1, 0)

        if self.use_dinov2_features:
            window_dinov2 = cached_data['dinov2_features'][start_idx:end_idx].to(torch.float32)
            dinov2_tensor = window_dinov2.permute(1, 0)

        if self.temporal_downsample_factor > 1:
            final_input_tensor = final_input_tensor[:, ::self.temporal_downsample_factor, :]
            labels_tensor = labels_tensor[:, ::self.temporal_downsample_factor]
            gloss_ids_tensor = gloss_ids_tensor[::self.temporal_downsample_factor]
            if self.use_hamer_features:
                hamer_tensor = hamer_tensor[:, ::self.temporal_downsample_factor]
            if self.use_dinov2_features:
                dinov2_tensor = dinov2_tensor[:, ::self.temporal_downsample_factor]

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

                # -1 (no gloss / ignore) for padded frames -- matches the "Outside"
                # semantics of label_pad above, and is masked out of the gloss loss
                # in train_multitask.py exactly like real Outside frames are.
                gloss_pad = torch.full((pad_T,), -1, dtype=torch.long)
                gloss_ids_tensor = torch.cat([gloss_ids_tensor, gloss_pad], dim=0)

                if self.use_hamer_features:
                    hamer_pad = torch.zeros(hamer_tensor.shape[0], pad_T, dtype=torch.float32)
                    hamer_tensor = torch.cat([hamer_tensor, hamer_pad], dim=1)

                if self.use_dinov2_features:
                    dinov2_pad = torch.zeros(dinov2_tensor.shape[0], pad_T, dtype=torch.float32)
                    dinov2_tensor = torch.cat([dinov2_tensor, dinov2_pad], dim=1)

        start_scaled = start_idx // self.temporal_downsample_factor
        end_scaled = end_idx // self.temporal_downsample_factor

        # Fixed order: features, labels, gloss_ids, [hamer], [dinov2], vid, start, end.
        # gloss_ids sits right after labels (not with the optional extras) since it's
        # core to this dataset's purpose, not an optional feature stream.
        extras = []
        if self.use_hamer_features:
            extras.append(hamer_tensor)
        if self.use_dinov2_features:
            extras.append(dinov2_tensor)

        return (final_input_tensor, labels_tensor, gloss_ids_tensor, *extras, vid, start_scaled, end_scaled)