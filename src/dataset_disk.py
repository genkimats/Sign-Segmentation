"""
Disk-based variant of SignSegmentationDataset -- reads each video's features
from disk on demand in __getitem__, instead of pre-loading the ENTIRE dataset
into a RAM cache at __init__ (which is what src/dataset.py does, and which is
the right choice as long as everything fits in memory).

Use this when your feature combination is too large for RAM -- e.g. DINOv2
features at ~100GB, against 28GB of available memory. HaMeR alone (~26GB)
still fits (barely) with the original RAM-cached dataset; this exists for the
cases that don't.

TRADEOFF, stated plainly: every window read is a disk read (plus a full
per-video feature reconstruction) instead of an instant memory slice. A small
per-worker LRU cache (default: 2 videos) softens this for windows that happen
to land on an already-loaded video, but this will NOT be as fast as the
RAM-cached version, especially with shuffling (which scatters window access
order across videos, giving the small cache little to work with). Expect
slower epochs -- this is about making a run FEASIBLE within memory limits,
not fast. See train.py's docstring / the accompanying chat message for
concrete numbers on how much slower to expect, and for lower-effort
alternatives (smaller DINOv2 variant, fp16 storage, PCA) that might let you
use the RAM-cached dataset instead.

VALIDATION TRADEOFF, also stated plainly: __init__ does a full read-and-check
pass for labels/kinetic/face/HaMeR (cheap enough in aggregate to validate
upfront, matching src/dataset.py's robustness), but only an EXISTENCE check
for DINOv2 (reading the full ~100GB just to validate frame counts, then
discarding it, would take a very long time for no lasting benefit). This
means a DINOv2 frame-count mismatch -- if one exists -- won't be caught until
the first time that specific video is actually requested during training,
where it will raise a clear, actionable error (not silently corrupt data),
just later than ideal. Everything else fails exactly as loudly and as early
as src/dataset.py does.
"""
import os
import json
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset import apply_label_smoothing  # identical label-smoothing logic, reused as-is


class SignSegmentationDatasetDisk(Dataset):
    def __init__(self, keypoints_dir, labels_dir, split_file="dataset_splits.json", split="train",
                 window_size=16, overlap=0, tolerance_window=5, use_full_length=False,
                 base_features=None, kinematic_features=None, temporal_downsample_factor=1,
                 use_face_keypoints=False, face_dir="processed_data/face_keypoints_normalized",
                 use_hamer_features=False, hamer_dir="processed_data/hamer_features",
                 use_dinov2_features=False, dinov2_dir="processed_data/dinov2_features",
                 cache_size=2):
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
        self.use_hamer_features = use_hamer_features
        self.hamer_dir = hamer_dir
        self.use_dinov2_features = use_dinov2_features
        self.dinov2_dir = dinov2_dir

        self.feature_map = {"x-cord": 0, "y-cord": 1, "z-cord": 2}
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "z-cord"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []

        # Small, BOUNDED per-worker cache -- at most `cache_size` videos' full feature
        # sets held in memory at once (evicting least-recently-used). With num_workers=4
        # and cache_size=2, worst case is 4 x 2 = 8 videos resident at once (each
        # worker keeps its own separate cache) -- comfortably small regardless of how
        # large the underlying feature set is, since it's bounded by video COUNT, not
        # dataset size. Raise cache_size if you have RAM headroom to spare and want
        # fewer redundant re-reads; keep it low if memory is genuinely tight.
        self.cache_size = cache_size
        self._video_lru = OrderedDict()

        with open(split_file, 'r') as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
        self.video_ids = [vid.replace('.npy', '').replace('.pt', '') for vid in splits[split]]

        self.samples = []
        self.windows = []

        skip_counts = {
            "missing_label_or_kinetic": 0,
            "missing_face": 0,
            "missing_hamer": 0,
            "missing_dinov2": 0,
        }

        print(f"[{split.upper()}] (disk mode) Validating {len(self.video_ids)} videos -- "
              f"checking existence + reading labels only, NOT loading full feature sets "
              f"into memory (that's the whole point of this dataset variant).")

        for vid in tqdm(self.video_ids, desc=f"Validating {split} (disk mode)"):
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")
            if not os.path.exists(label_path) or not os.path.exists(kinetic_path):
                skip_counts["missing_label_or_kinetic"] += 1
                continue

            if self.use_face_keypoints and not os.path.exists(os.path.join(self.face_dir, f"{vid}.npy")):
                skip_counts["missing_face"] += 1
                continue
            if self.use_hamer_features and not os.path.exists(os.path.join(self.hamer_dir, f"{vid}_hamer.pt")):
                skip_counts["missing_hamer"] += 1
                continue
            if self.use_dinov2_features and not os.path.exists(os.path.join(self.dinov2_dir, f"{vid}_dinov2.pt")):
                skip_counts["missing_dinov2"] += 1
                continue

            # Only the label file gets read here -- tiny (a few KB), unlike kinetic/
            # face/hamer/dinov2, so this is cheap even across the whole split.
            num_frames = len(np.load(label_path))

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
        total_valid = len(self.samples)
        if total_valid < total_attempted:
            print(f"[{split.upper()}] {total_valid}/{total_attempted} videos passed validation "
                  f"({total_attempted - total_valid} skipped). Skip reasons: {skip_counts}")

        if len(self.windows) == 0:
            raise RuntimeError(
                f"[{split.upper()}] SignSegmentationDatasetDisk ended up with 0 windows/samples "
                f"(0 of {total_attempted} videos passed validation) -- this dataset is EMPTY. "
                f"Skip reasons: {skip_counts}. Same naming-convention-mismatch causes as the "
                f"RAM-cached dataset -- check {self.face_dir!r} / {self.hamer_dir!r} / "
                f"{self.dinov2_dir!r} contain files matching your split's video ids exactly."
            )

    def __len__(self):
        if self.use_full_length:
            return len(self.samples)
        return len(self.windows)

    def _load_video(self, vid):
        """
        Loads and constructs the full feature set for ONE video from disk, exactly
        matching src/dataset.py's per-video construction logic. Checked against the
        small LRU cache first; on a miss, loads fresh and inserts (evicting the
        least-recently-used entry if the cache is full).
        """
        if vid in self._video_lru:
            self._video_lru.move_to_end(vid)
            return self._video_lru[vid]

        label_path = os.path.join(self.labels_dir, f"{vid}.npy")
        kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")

        labels = np.load(label_path)
        num_frames = len(labels)
        soft_labels = apply_label_smoothing(labels, self.tolerance_window)

        kin_data = torch.load(kinetic_path, weights_only=False)
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

        final_tensor = torch.cat(channels, dim=-1)

        if self.use_face_keypoints:
            face_path = os.path.join(self.face_dir, f"{vid}.npy")
            face_raw = torch.from_numpy(np.load(face_path)).float()
            if face_raw.shape[0] != num_frames:
                raise RuntimeError(f"{vid}: face frame count ({face_raw.shape[0]}) != "
                                    f"label frame count ({num_frames}). This should have been "
                                    f"caught upfront -- investigate before trusting this video.")
            face_selected = face_raw[:, :, base_indices] if base_indices else face_raw
            K_total = final_tensor.shape[-1]
            face_padded = torch.zeros(num_frames, face_selected.shape[1], K_total, dtype=final_tensor.dtype)
            face_padded[:, :, :face_selected.shape[-1]] = face_selected
            final_tensor = torch.cat([final_tensor, face_padded], dim=1)

        hamer_full = None
        if self.use_hamer_features:
            hamer_path = os.path.join(self.hamer_dir, f"{vid}_hamer.pt")
            hamer_data = torch.load(hamer_path, weights_only=False)
            hand_pose = hamer_data["hand_pose"]
            global_orient = hamer_data["global_orient"]
            ham_downsample = hamer_data.get("temporal_downsample_factor", 1)
            T_ham = hand_pose.shape[0]
            if T_ham == 0:
                raise RuntimeError(f"{vid}: hamer feature file has 0 frames. This should have "
                                    f"been caught upfront -- investigate before trusting this video.")
            hand_pose_flat = torch.as_tensor(hand_pose, dtype=torch.float32).reshape(T_ham, 270)
            global_orient_flat = torch.as_tensor(global_orient, dtype=torch.float32).reshape(T_ham, 18)
            hamer_flat = torch.cat([hand_pose_flat, global_orient_flat], dim=-1)
            frame_to_hamer_idx = (torch.arange(num_frames) // ham_downsample).clamp(max=T_ham - 1)
            hamer_full = hamer_flat[frame_to_hamer_idx].to(torch.float16)

        dinov2_full = None
        if self.use_dinov2_features:
            dinov2_path = os.path.join(self.dinov2_dir, f"{vid}_dinov2.pt")
            dinov2_data = torch.load(dinov2_path, weights_only=False)
            dinov2_feats = dinov2_data["features"]
            if dinov2_feats.shape[0] != num_frames:
                # NOT caught upfront (see module docstring) -- this is the one place a
                # bad video can surface partway through training instead of at startup.
                raise RuntimeError(f"{vid}: dinov2 frame count ({dinov2_feats.shape[0]}) != "
                                    f"label frame count ({num_frames}). Investigate this video's "
                                    f"extraction before trusting results that included it.")
            D = dinov2_feats.shape[-1]
            dinov2_full = torch.as_tensor(dinov2_feats, dtype=torch.float32).reshape(num_frames, 2 * D).to(torch.float16)

        final_tensor = final_tensor.to(torch.float16)

        result = {'features': final_tensor, 'labels': soft_labels}
        if self.use_hamer_features:
            result['hamer_features'] = hamer_full
        if self.use_dinov2_features:
            result['dinov2_features'] = dinov2_full

        self._video_lru[vid] = result
        if len(self._video_lru) > self.cache_size:
            self._video_lru.popitem(last=False)  # evict least-recently-used

        return result

    def __getitem__(self, idx):
        # From here down, this is IDENTICAL to src/dataset.py's __getitem__ -- same
        # slicing, downsampling, padding, and return-tuple construction. Only the line
        # that fetches `cached_data` differs (disk-backed lazy load vs instant RAM slice).
        if self.use_full_length:
            window_info = self.samples[idx]
        else:
            window_info = self.windows[idx]

        vid = window_info['video_id']
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']

        cached_data = self._load_video(vid)

        window_features = cached_data['features'][start_idx:end_idx].to(torch.float32)
        window_labels = cached_data['labels'][start_idx:end_idx]

        final_input_tensor = window_features.permute(2, 0, 1)
        labels_tensor = torch.tensor(window_labels, dtype=torch.float32).permute(1, 0)

        if self.use_hamer_features:
            window_hamer = cached_data['hamer_features'][start_idx:end_idx].to(torch.float32)
            hamer_tensor = window_hamer.permute(1, 0)

        if self.use_dinov2_features:
            window_dinov2 = cached_data['dinov2_features'][start_idx:end_idx].to(torch.float32)
            dinov2_tensor = window_dinov2.permute(1, 0)

        if self.temporal_downsample_factor > 1:
            final_input_tensor = final_input_tensor[:, ::self.temporal_downsample_factor, :]
            labels_tensor = labels_tensor[:, ::self.temporal_downsample_factor]
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

                if self.use_hamer_features:
                    hamer_pad = torch.zeros(hamer_tensor.shape[0], pad_T, dtype=torch.float32)
                    hamer_tensor = torch.cat([hamer_tensor, hamer_pad], dim=1)

                if self.use_dinov2_features:
                    dinov2_pad = torch.zeros(dinov2_tensor.shape[0], pad_T, dtype=torch.float32)
                    dinov2_tensor = torch.cat([dinov2_tensor, dinov2_pad], dim=1)

        start_scaled = start_idx // self.temporal_downsample_factor
        end_scaled = end_idx // self.temporal_downsample_factor

        extras = []
        if self.use_hamer_features:
            extras.append(hamer_tensor)
        if self.use_dinov2_features:
            extras.append(dinov2_tensor)

        return (final_input_tensor, labels_tensor, *extras, vid, start_scaled, end_scaled)