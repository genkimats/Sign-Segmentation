import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from src.face_subsets import face_subset_positions, SAVED_FACE_INDICES, FACE_LAYOUTS
from src.cue_sets import resolve_cue_set, cue_names_path
from src.aux_labels import IGNORE as AUX_IGNORE, apply_label_map
from src.frontend import dominant_wrist, mirror_graph_tv, mirror_hamer, face_mirror_perm, L_WRIST, R_WRIST

CUE_DEFAULT_DIR = "processed_data/cue_features"


def extra_dataset_kwargs(config):
    """
    Dataset arguments beyond the original ones, read from a run config (so training,
    evaluation and the viewers always build the SAME data): the interpretable-cue stream
    and the handedness options. mirror_augment_p only acts on the train split.
    The auxiliary labels of Idea B (aux_label_dir / aux_tasks) are only LOADED for the train
    split (or with force_aux=True, aux_head_eval.py), so validation / evaluation data is unchanged.
    """
    return {"use_cue_features": bool(config.get("use_cue_features", False)),
            "cue_dir": config.get("cue_dir", CUE_DEFAULT_DIR),
            "cue_set": config.get("cue_set", "selected"),
            "hand_canonicalize": bool(config.get("hand_canonicalize", False)),
            "face_layout": config.get("face_layout", "saved83"),
            "mirror_augment_p": float(config.get("mirror_augment_p", 0.0)),
            "train_fraction": float(config.get("train_fraction", 1.0)),
            "aux_label_dir": config.get("aux_label_dir") if config.get("aux_heads") else None,
            "aux_tasks": tuple(config.get("aux_tasks") or (config.get("aux_heads") or {}).keys()),
            "aux_label_map": config.get("aux_label_map")}


cue_dataset_kwargs = extra_dataset_kwargs  # old name

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

def downsample_labels(labels, k):
    """
    Soft labels (3, T) -> (3, ceil(T / k)), keeping phrase STARTS: plain [::k] subsampling
    drops every B that falls on a frame not divisible by k (B is a single frame). Here
    each output frame is the first frame of its group of k, but if the group contains a
    B, the output frame becomes B (with that B's probability; O/I rescaled).
    """
    if k <= 1:
        return labels
    C, T = labels.shape
    sub = labels[:, ::k].clone()
    Tp = sub.shape[1]
    pad = Tp * k - T
    b = labels[2]
    if pad:
        b = torch.cat([b, torch.zeros(pad, dtype=b.dtype)])
    bmax = b.view(Tp, k).max(dim=1).values
    upd = bmax > sub[2]
    if upd.any():
        rest = sub[:2, upd]
        tot = rest.sum(dim=0).clamp(min=1e-8)
        sub[:2, upd] = rest / tot * (1 - bmax[upd])
        sub[2, upd] = bmax[upd]
    return sub


class SignSegmentationDataset(Dataset):
    def __init__(self, keypoints_dir, labels_dir, split_file="dataset_splits.json", split="train", window_size=16, overlap=0, tolerance_window=1, use_full_length=False, base_features=None, kinematic_features=None, temporal_downsample_factor=1, use_face_keypoints=False, face_dir="processed_data/face_keypoints_normalized_v2", face_subset="full", face_only=False, face_dims=2, use_hamer_features=False, hamer_dir="processed_data/hamer_features", use_dinov2_features=False, dinov2_dir="processed_data/dinov2_features", use_cue_features=False, cue_dir=CUE_DEFAULT_DIR, cue_set="selected", hand_canonicalize=False, mirror_augment_p=0.0, face_layout="saved83", train_fraction=1.0, soft_label_dir=None, aux_label_dir=None, aux_tasks=(), aux_label_map=None, force_aux=False):
        self.labels_dir = labels_dir
        # Idea B auxiliary targets (data_processing/make_aux_labels.py): per-video <dir>/<vid>.npz
        # with one (T,) int8 array per task. Loaded only for the TRAIN split (like soft_label_dir)
        # unless force_aux=True (aux_head_eval.py); returned as the LAST extra, (n_tasks, T) int64,
        # padded with -100. A missing file / track becomes -100 (ignored), never a skipped video.
        self.aux_tasks = tuple(aux_tasks or ())
        self.aux_label_dir = aux_label_dir
        self.aux_label_map = aux_label_map or {}
        self.use_aux = bool(aux_label_dir and self.aux_tasks and (split == "train" or force_aux))
        self.aux_missing = []
        # soft_label_dir (train split only, set by train_phrase.py): per-video soft TRAINING
        # targets (T, 3) from make_soft_labels.py replace the one-hot / tolerance-smoothed
        # targets. 'hard_labels' (evaluation) always stay the gold labels from labels_dir.
        self.soft_label_dir = soft_label_dir
        self.soft_label_missing = []
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
        # Which of the saved face points to load (src/face_subsets.py): "full" (83),
        # "compact" (31), "eyes_brows" (22) or "minimal" (18). Selected at load time;
        # the saved face files are not changed.
        self.face_subset = face_subset
        # FACE-ONLY MODE: keep ONLY the face vertices (drop body + hands), e.g. to measure
        # how much phrase information the face carries on its own. Needs
        # use_face_keypoints=True; graph models then build a face-only graph (face_only=True).
        self.face_only = face_only
        if face_only:
            if not use_face_keypoints:
                raise ValueError("face_only=True requires use_face_keypoints=True.")
            if kinematic_features:
                raise ValueError("face_only=True with kinematic_features: the face has no kinematic "
                                 "channels (they would be all zeros). Use kinematic_features=[].")
        # face_layout: "saved83" = the 83-point files (extract_face_keypoints.py), "full478" =
        # all landmarks (extract_face_keypoints_full.py; needed for the 2023 contour subset,
        # the face oval and the irises).
        if face_layout not in FACE_LAYOUTS:
            raise ValueError(f"face_layout must be one of {sorted(FACE_LAYOUTS)}, got '{face_layout}'.")
        self.face_layout = face_layout
        self.face_positions = face_subset_positions(face_subset, face_layout) if use_face_keypoints else None
        # Face coordinates used: 2 = x, y only (z dropped; the default, matching
        # face_keypoints_normalized_v2), 3 = legacy x, y, z (only so runs trained on the old
        # 3D face files can still be evaluated).
        if face_dims not in (2, 3):
            raise ValueError(f"face_dims must be 2 or 3, got {face_dims}.")
        self.face_dims = face_dims
        self.use_hamer_features = use_hamer_features
        self.hamer_dir = hamer_dir
        self.use_dinov2_features = use_dinov2_features
        self.dinov2_dir = dinov2_dir
        # INTERPRETABLE CUE STREAM (data_processing/extract_cue_features.py): a (T, C) matrix
        # of per-frame cues (eye openness, brow height, head nod, wrist height, finger
        # flexion ...). cue_set picks the channels at load time (src/cue_sets.py).
        self.use_cue_features = use_cue_features
        self.cue_dir = cue_dir
        self.cue_set = cue_set
        self.cue_names, self.cue_indices = None, None
        if use_cue_features:
            names_path = cue_names_path(cue_dir)
            if not os.path.exists(names_path):
                raise FileNotFoundError(f"{names_path} not found -- run data_processing/extract_cue_features.py first.")
            with open(names_path) as f:
                file_names = json.load(f)
            self.cue_names, self.cue_indices = resolve_cue_set(cue_set, file_names)
            print(f"[{split.upper()}] cue features: set '{cue_set if isinstance(cue_set, str) else 'custom'}' "
                  f"({len(self.cue_names)} of {len(file_names)} channels) from {cue_dir}")

        self.feature_map = {
            "x-cord": 0,
            "y-cord": 1,
            "z-cord": 2
        }
        
        self.base_features = base_features if base_features is not None else ["x-cord", "y-cord", "z-cord"]
        self.kinematic_features = kinematic_features if kinematic_features is not None else []

        # Where the face's x and y go in the per-vertex channel layout (the body's x-cord /
        # y-cord channels). All other channels of face vertices are 0. Models that process
        # the face separately (multistream face stream, face_only) use only these channels.
        coord_order = [f for f in self.base_features if f in ("x-cord", "y-cord", "z-cord")]
        self.face_channel_indices = [coord_order.index(f) for f in ("x-cord", "y-cord") if f in coord_order]
        if use_face_keypoints and face_dims == 2 and not self.face_channel_indices:
            raise ValueError("2D face keypoints need 'x-cord' and/or 'y-cord' in base_features.")

        # PURE HAMER MODE: base_features=["pure_hamer"] makes the 288-dim HaMeR vector
        # the MAIN feature tensor, shaped (T, 1, 288) -> (288, T, 1) per window, so the
        # graph-free baselines (bilstm_baseline / transformer_baseline) can train on it
        # directly. No MediaPipe keypoints, kinematics or face are loaded in this mode.
        # Keep use_hamer_features=False here: HaMeR is the main input, not a side branch.
        # DINOv2 (use_dinov2_features=True) still works as a separate side stream.
        self.pure_hamer = "pure_hamer" in self.base_features
        if self.pure_hamer:
            if use_hamer_features:
                raise ValueError(
                    "base_features contains 'pure_hamer' AND use_hamer_features=True. In pure "
                    "HaMeR mode HaMeR is already the main input -- set use_hamer_features=False."
                )
            if use_face_keypoints:
                raise ValueError("use_face_keypoints=True is not supported in pure_hamer mode.")
            if self.kinematic_features:
                print(f"[{split.upper()}] pure_hamer mode: ignoring kinematic_features "
                      f"{self.kinematic_features} (they are MediaPipe-based).")

        # HANDEDNESS (src/frontend.py):
        #   hand_canonicalize=True : every video whose dominant hand (the pose wrist that
        #       moves more) is MediaPipe's "left" (15) is mirrored, so the dominant hand is
        #       always the "right" one (pose 16, hand block 44-64, HaMeR slot 1). Uses no
        #       labels; applied identically to every split.
        #   mirror_augment_p > 0   : TRAIN split only -- each window is mirrored with this
        #       probability (the model sees both handednesses instead).
        self.hand_canonicalize = bool(hand_canonicalize)
        self.mirror_augment_p = float(mirror_augment_p) if split == "train" else 0.0
        self.mirrored_videos = set()
        if self.hand_canonicalize or float(mirror_augment_p) > 0:
            if self.pure_hamer or face_only:
                raise ValueError("hand_canonicalize / mirror_augment_p need the body + hands keypoints "
                                 "(not pure_hamer / face_only).")
            if use_dinov2_features:
                raise ValueError("hand_canonicalize / mirror_augment_p cannot mirror DINOv2 appearance "
                                 "features -- turn DINOv2 off for these runs.")
            unsupported = [f for f in self.kinematic_features
                           if f not in ("velocity", "acceleration", "jerk", "velocity-mag")]
            if unsupported:
                raise ValueError(f"Mirroring is only defined for velocity / acceleration / jerk / "
                                 f"velocity-mag, not {unsupported}.")
            if "x-cord" not in self.base_features:
                raise ValueError("Mirroring needs 'x-cord' in base_features.")
        # channels carrying x (positions + their derivatives), in the cached tensor's layout
        _coords = [f for f in self.base_features if f in ("x-cord", "y-cord", "z-cord")]
        self._mirror_x_channels = []
        if "x-cord" in _coords:
            self._mirror_x_channels.append(_coords.index("x-cord"))
            _off = len(_coords)
            for _f in ("velocity", "acceleration", "jerk"):
                if _f in self.kinematic_features:
                    self._mirror_x_channels.append(_off + _coords.index("x-cord"))
                    _off += len(_coords)
        mirroring = self.hand_canonicalize or float(mirror_augment_p) > 0
        self._face_mirror_perm = (face_mirror_perm(face_subset)
                                  if mirroring and use_face_keypoints and not face_only else None)

        if self.use_face_keypoints:
            # Face points are appended along the VERTEX axis: 65 + subset size. Graph models
            # build the matching face graph from num_vertices (src/graph.py).
            n_vertices = len(self.face_positions) + (0 if face_only else 65)
            print(f"[{split.upper()}] face keypoints: subset '{face_subset}' "
                  f"({len(self.face_positions)} of {FACE_LAYOUTS[face_layout]} saved points, layout {face_layout}) -> "
                  f"{n_vertices} vertices, {face_dims}D from {face_dir}"
                  + (" (FACE ONLY: body + hands dropped)" if face_only else ""))

        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
            
        self.video_ids = [vid.replace('.npy', '').replace('.pt', '') for vid in splits[split]]
        # train_fraction < 1 (data-scaling study): keep a fixed, NESTED subset of the training
        # videos -- one permutation (seed 0, independent of the run seed), so the 25% subset is
        # inside the 50% subset and every seed of a fraction trains on the same videos.
        if split == "train" and float(train_fraction) < 1.0:
            ids = sorted(self.video_ids)
            order = np.random.RandomState(0).permutation(len(ids))
            n_keep = max(1, int(np.ceil(float(train_fraction) * len(ids))))
            self.video_ids = [ids[i] for i in sorted(order[:n_keep])]
            print(f"[TRAIN] train_fraction={train_fraction}: {n_keep}/{len(ids)} videos")
        
        self.samples = []  
        self.windows = []  
        
        # --- THE RAM CACHE ---
        self.video_cache = {}

        # Populated from the actual data during caching below (see the hamer/dinov2
        # loading blocks) -- train.py should read these to build the model, instead
        # of guessing a hardcoded dimension that can silently drift out of sync with
        # whatever's actually been extracted.
        self.detected_hamer_dim = None
        self.detected_dinov2_dim = None
        self.detected_cue_dim = len(self.cue_indices) if use_cue_features else None

        # Tracks WHY a video got skipped, so an empty/near-empty dataset fails loudly
        # with an actionable reason instead of surfacing as a cryptic
        # "num_samples=0" error three layers deep in the DataLoader/RandomSampler.
        skip_counts = {
            "missing_label_or_kinetic": 0,
            "kinetic_load_error": 0,
            "missing_face": 0,
            "face_frame_mismatch": 0,
            "missing_hamer": 0,
            "hamer_load_error": 0,
            "hamer_empty": 0,
            "missing_dinov2": 0,
            "dinov2_load_error": 0,
            "dinov2_frame_mismatch": 0,
            "missing_cue": 0,
            "cue_frame_mismatch": 0,
        }

        # reason -> [(video_id, detail_or_None), ...] -- WHICH videos were skipped, not just how many.
        skipped_videos = {}

        print(f"[{split.upper()}] Loading {len(self.video_ids)} videos into RAM Cache (Bypassing Disk I/O)...")
        
        # Load everything into RAM exactly ONCE
        for vid in tqdm(self.video_ids, desc=f"Caching {split}"):
            label_path = os.path.join(self.labels_dir, f"{vid}.npy")
            kinetic_path = os.path.join(self.kinetic_dir, f"{vid}.pt")
            
            if not os.path.exists(label_path) or not os.path.exists(kinetic_path):
                skip_counts["missing_label_or_kinetic"] += 1
                skipped_videos.setdefault("missing_label_or_kinetic", []).append((vid, "kinematic file missing" if os.path.exists(label_path) else "label file missing"))
                continue
                
            labels = np.load(label_path)
            num_frames = len(labels)
            
            # CPU Bottleneck Fix: Pre-calculate smoothing once per video, not once per window!
            soft_labels = apply_label_smoothing(labels, self.tolerance_window)
            if self.soft_label_dir:
                sp = os.path.join(self.soft_label_dir, f"{vid}.npy")
                if os.path.exists(sp):
                    soft = np.load(sp).astype(np.float32)
                    if soft.shape == (num_frames, 3):
                        soft_labels = soft
                    else:
                        self.soft_label_missing.append((vid, f"shape {soft.shape} != ({num_frames}, 3)"))
                else:
                    self.soft_label_missing.append((vid, "file missing"))

            aux_arr = None
            if self.use_aux:
                aux_arr = np.full((len(self.aux_tasks), num_frames), AUX_IGNORE, dtype=np.int8)
                ap_ = os.path.join(self.aux_label_dir, f"{vid}.npz")
                if os.path.exists(ap_):
                    with np.load(ap_) as z:
                        for j, task in enumerate(self.aux_tasks):
                            if task not in z.files:
                                self.aux_missing.append((vid, f"no '{task}' track"))
                            elif len(z[task]) != num_frames:
                                self.aux_missing.append((vid, f"{task}: {len(z[task])} != {num_frames} frames"))
                            else:
                                aux_arr[j] = apply_label_map(z[task].astype(np.int8),
                                                             self.aux_label_map.get(task))
                else:
                    self.aux_missing.append((vid, "file missing"))
            
            try:
                kin_data = torch.load(kinetic_path, weights_only=False)
                # Robustly handle dictionary extraction
                if "mediapipe" in kin_data:
                    kin_data = kin_data["mediapipe"]
            except Exception:
                skip_counts["kinetic_load_error"] += 1
                skipped_videos.setdefault("kinetic_load_error", []).append((vid, None))
                continue

            if self.pure_hamer:
                final_tensor = None  # filled from the HaMeR stream below
                base_indices = []
            else:
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
            if self.use_face_keypoints and not self.pure_hamer:
                face_path = os.path.join(self.face_dir, f"{vid}.npy")
                if not os.path.exists(face_path):
                    skip_counts["missing_face"] += 1
                    skipped_videos.setdefault("missing_face", []).append((vid, None))
                    continue  # keep vertex layout consistent across the whole split; don't silently zero-fill a missing video

                face_raw = torch.from_numpy(np.load(face_path)).float()  # (T_face, NUM_FACE_VERTICES, 3)
                if face_raw.shape[0] != num_frames:
                    skip_counts["face_frame_mismatch"] += 1
                    skipped_videos.setdefault("face_frame_mismatch", []).append((vid, f"face frames={face_raw.shape[0]}, label frames={num_frames}"))
                    continue  # frame count doesn't match labels/body -- skip rather than risk misaligning them

                if face_raw.shape[1] != FACE_LAYOUTS[self.face_layout]:
                    raise ValueError(
                        f"{face_path} has {face_raw.shape[1]} face points, expected "
                        f"{FACE_LAYOUTS[self.face_layout]} for face_layout='{self.face_layout}' "
                        f"(83-point files -> 'saved83', face_keypoints_full* files -> 'full478')."
                    )
                face_raw = face_raw[:, self.face_positions, :]  # keep only the chosen subset

                K_total = final_tensor.shape[-1]
                face_padded = torch.zeros(num_frames, face_raw.shape[1], K_total, dtype=final_tensor.dtype)
                if self.face_dims == 2:
                    # 2D face: x -> the x-cord channel, y -> the y-cord channel, everything
                    # else (z, kinematic channels) stays 0. Works with 2-channel v2 files
                    # and, by dropping z, with old 3-channel files.
                    if face_raw.shape[-1] < 2:
                        raise ValueError(f"{face_path}: expected at least x, y per face point, got {face_raw.shape}.")
                    coord_src = {"x-cord": 0, "y-cord": 1}
                    coord_order = [f for f in self.base_features if f in ("x-cord", "y-cord", "z-cord")]
                    for feat, src_ch in coord_src.items():
                        if feat in coord_order:
                            face_padded[:, :, coord_order.index(feat)] = face_raw[:, :, src_ch]
                else:
                    # Legacy 3D face (old face_keypoints_normalized files only).
                    if face_raw.shape[-1] < 3:
                        raise ValueError(f"{face_path} has {face_raw.shape[-1]} coordinates; face_dims=3 "
                                         f"needs the old 3D files (use face_dims=2 for the v2 files).")
                    face_selected = face_raw[:, :, base_indices] if base_indices else face_raw
                    face_padded[:, :, :face_selected.shape[-1]] = face_selected

                # v2 files are FACE-LOCAL (origin between the inner eye corners, units of
                # inter-ocular distance, roll-aligned -- see normalize_face_keypoints_v2.py),
                # i.e. they describe facial movement, not head position (the body stream
                # has the head). Old v1 files were in the body's shoulder frame.
                final_tensor = torch.cat([final_tensor, face_padded], dim=1)  # concat along the VERTEX axis
                if self.face_only:
                    final_tensor = final_tensor[:, 65:, :]  # drop body (23) + hands (42)
                    if self.face_dims == 2:
                        final_tensor = final_tensor[:, :, self.face_channel_indices]  # x, y only
                    final_tensor = final_tensor.contiguous()

            # --- OPTIONAL: HAMER HAND-POSE FEATURES (from extract_hamer_features.py) ---
            # Unlike face keypoints, HaMeR features are NOT folded into the per-vertex
            # graph tensor -- they're MANO rotation parameters, not 3D point coordinates,
            # so they're kept as a SEPARATE (num_frames, 288) stream and fused inside the
            # model itself via a dedicated MLP branch (see models.py's hamer_dim support),
            # matching the 2025 Hands-On paper's own architecture for this feature.
            hamer_full = None
            if self.use_hamer_features or self.pure_hamer:
                hamer_path = os.path.join(self.hamer_dir, f"{vid}_hamer.pt")
                if not os.path.exists(hamer_path):
                    skip_counts["missing_hamer"] += 1
                    skipped_videos.setdefault("missing_hamer", []).append((vid, None))
                    continue

                try:
                    hamer_data = torch.load(hamer_path, weights_only=False)
                    hand_pose = hamer_data["hand_pose"]          # (T_ham, 2, 15, 3, 3)
                    global_orient = hamer_data["global_orient"]  # (T_ham, 2, 3, 3)
                    ham_downsample = hamer_data.get("temporal_downsample_factor", 1)
                except Exception:
                    skip_counts["hamer_load_error"] += 1
                    skipped_videos.setdefault("hamer_load_error", []).append((vid, None))
                    continue

                T_ham = hand_pose.shape[0]
                if T_ham == 0:
                    skip_counts["hamer_empty"] += 1
                    skipped_videos.setdefault("hamer_empty", []).append((vid, None))
                    continue

                # Flatten each hand's (15,3,3) pose + (3,3) orientation into 144 values,
                # concatenate left+right -> 288 per frame (matches F in R^288 from the
                # 2025 paper exactly: 2 hands x (15x3x3 + 1x3x3) = 2 x 144 = 288).
                # Flatten to match the 2025 paper's construction EXACTLY: flatten ALL of H
                # (both hands, all 15 joints) into H' in R^270 FIRST, then ALL of G (both
                # hands) into G' in R^18, then concatenate [H' | G'] = F in R^288. NOT
                # interleaved per-hand (that would give the same 288 numbers in a different
                # order -- mathematically equivalent once fed into a Linear layer, but not
                # a literal match to the paper's stated construction).
                hand_pose_flat = torch.as_tensor(hand_pose, dtype=torch.float32).reshape(T_ham, 270)    # H' in R^270
                global_orient_flat = torch.as_tensor(global_orient, dtype=torch.float32).reshape(T_ham, 18)  # G' in R^18
                hamer_flat = torch.cat([hand_pose_flat, global_orient_flat], dim=-1)  # F in R^288, [H' | G']

                # extract_hamer_features.py runs at ham_downsample x the native frame
                # rate (every Nth frame). Upsample back to num_frames via nearest-
                # neighbor/forward-fill from the last extracted HaMeR frame, so this
                # stream aligns 1:1 with labels/body and windowing downstream doesn't
                # need to know HaMeR was extracted at a different rate.
                frame_to_hamer_idx = (torch.arange(num_frames) // ham_downsample).clamp(max=T_ham - 1)
                hamer_full = hamer_flat[frame_to_hamer_idx].to(torch.float16)  # (num_frames, 288)
                if self.detected_hamer_dim is None:
                    self.detected_hamer_dim = hamer_full.shape[-1]

            # --- PURE HAMER: HaMeR becomes the main feature tensor ---
            if self.pure_hamer:
                final_tensor = hamer_full.to(torch.float32).unsqueeze(1)  # (num_frames, 1, 288)

            # --- OPTIONAL: DINOV2 VISUAL HAND-CROP FEATURES (from extract_dinov2_features.py) ---
            # Same treatment as HaMeR -- kept as a SEPARATE (num_frames, 2*D) stream, fused
            # inside the model via its own MLP branch, not folded into the per-vertex graph
            # tensor. This is an APPEARANCE feature (comes straight from pixels, no geometric
            # derivation at all), so it's a genuinely different information source than every
            # other stream here -- combinable independently with use_hamer_features.
            dinov2_full = None
            if self.use_dinov2_features:
                dinov2_path = os.path.join(self.dinov2_dir, f"{vid}_dinov2.pt")
                if not os.path.exists(dinov2_path):
                    skip_counts["missing_dinov2"] += 1
                    skipped_videos.setdefault("missing_dinov2", []).append((vid, None))
                    continue

                try:
                    dinov2_data = torch.load(dinov2_path, weights_only=False)
                    dinov2_feats = dinov2_data["features"]  # (T_dino, 2, D)
                except Exception:
                    skip_counts["dinov2_load_error"] += 1
                    skipped_videos.setdefault("dinov2_load_error", []).append((vid, None))
                    continue

                if dinov2_feats.shape[0] != num_frames:
                    skip_counts["dinov2_frame_mismatch"] += 1
                    skipped_videos.setdefault("dinov2_frame_mismatch", []).append((vid, f"dinov2 frames={dinov2_feats.shape[0]}, label frames={num_frames}"))
                    continue  # extracted at full frame rate by default -- a mismatch here means
                              # something upstream (frame count, video decode) doesn't line up;
                              # skip rather than risk silently misaligning it with labels/body

                D = dinov2_feats.shape[-1]
                dinov2_full = torch.as_tensor(dinov2_feats, dtype=torch.float32).reshape(num_frames, 2 * D).to(torch.float16)
                # Record the ACTUAL dimension found in the data, the first time we see it --
                # this is what train.py should build the model with, not a hardcoded guess.
                # A hardcoded default is exactly what broke when the extraction script's
                # model variant changed (768-dim -> 384-dim) and this number had to be
                # remembered and updated in a second, unrelated file -- self-detecting from
                # the data makes that whole bug class impossible going forward, regardless
                # of what extraction settings or PCA reduction you use.
                if self.detected_dinov2_dim is None:
                    self.detected_dinov2_dim = 2 * D
                elif self.detected_dinov2_dim != 2 * D:
                    raise RuntimeError(
                        f"{vid}: dinov2 feature dim ({2 * D}) doesn't match the dimension "
                        f"detected from an earlier video in this split ({self.detected_dinov2_dim}). "
                        f"This means {self.dinov2_dir!r} contains files extracted with DIFFERENT "
                        f"settings (e.g. some with the old model variant, some with the new one, "
                        f"or a mix of raw and PCA-reduced files) -- re-extract consistently or "
                        f"separate them into different directories before training."
                    )

            # --- OPTIONAL: INTERPRETABLE CUES (from extract_cue_features.py) ---
            cue_full = None
            if self.use_cue_features:
                cue_path = os.path.join(self.cue_dir, f"{vid}.npy")
                if not os.path.exists(cue_path):
                    skip_counts["missing_cue"] += 1
                    skipped_videos.setdefault("missing_cue", []).append((vid, None))
                    continue
                cue_arr = np.load(cue_path)
                if cue_arr.shape[0] != num_frames:
                    skip_counts["cue_frame_mismatch"] += 1
                    skipped_videos.setdefault("cue_frame_mismatch", []).append(
                        (vid, f"cue frames={cue_arr.shape[0]}, label frames={num_frames}"))
                    continue
                cue_full = torch.from_numpy(np.ascontiguousarray(cue_arr[:, self.cue_indices])).to(torch.float16)

            # --- OPTIONAL: HANDEDNESS CANONICALIZATION (dominant hand -> MediaPipe "right") ---
            if self.hand_canonicalize:
                base_xy = kin_data["base"][:, :, :2]
                base_xy = base_xy.numpy() if hasattr(base_xy, "numpy") else np.asarray(base_xy)
                if dominant_wrist(base_xy) == L_WRIST:
                    final_tensor = mirror_graph_tv(final_tensor.to(torch.float32), self._mirror_x_channels,
                                                   self._face_mirror_perm)
                    if hamer_full is not None:
                        hamer_full = mirror_hamer(hamer_full.to(torch.float32)).to(torch.float16)
                    self.mirrored_videos.add(vid)

            # Compress to FP16 to keep RAM super low, and save it to the dictionary!
            final_tensor = final_tensor.to(torch.float16)
            
            self.video_cache[vid] = {
                'features': final_tensor,
                'labels': soft_labels,
                # Raw, HARD per-frame labels at the RAW frame rate. Used only for
                # evaluation (src/evaluation.py): ground truth never depends on
                # tolerance_window smoothing or temporal_downsample_factor.
                'hard_labels': np.asarray(labels).astype(np.int64)
            }
            if self.use_hamer_features:
                self.video_cache[vid]['hamer_features'] = hamer_full
            if self.use_dinov2_features:
                self.video_cache[vid]['dinov2_features'] = dinov2_full
            if self.use_cue_features:
                self.video_cache[vid]['cue_features'] = cue_full
            if self.use_aux:
                self.video_cache[vid]['aux_labels'] = torch.from_numpy(aux_arr)
            
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

        if self.soft_label_dir:
            n_bad = len(self.soft_label_missing)
            print(f"[{split.upper()}] soft training targets from {self.soft_label_dir}: "
                  f"{len(self.video_cache) - n_bad}/{len(self.video_cache)} videos"
                  + (f" -- {n_bad} fall back to hard labels, e.g. {self.soft_label_missing[:3]}" if n_bad else ""))
            if n_bad > 0.05 * max(1, len(self.video_cache)):
                raise RuntimeError(f"{n_bad} of {len(self.video_cache)} videos have no usable soft labels in "
                                   f"{self.soft_label_dir} -- run make_soft_labels.py first.")
        if self.use_aux:
            n_bad = len({v for v, _ in self.aux_missing if v in self.video_cache})
            print(f"[{split.upper()}] auxiliary labels {list(self.aux_tasks)} from {self.aux_label_dir}: "
                  f"{len(self.video_cache) - n_bad}/{len(self.video_cache)} videos complete"
                  + (f" -- {n_bad} partly ignored (-100), e.g. {self.aux_missing[:3]}" if n_bad else ""))
            if n_bad > 0.05 * max(1, len(self.video_cache)):
                raise RuntimeError(f"{n_bad} of {len(self.video_cache)} videos have no usable auxiliary labels in "
                                   f"{self.aux_label_dir} -- run data_processing/make_aux_labels.py first.")
        if self.hand_canonicalize:
            print(f"[{split.upper()}] hand canonicalization: {len(self.mirrored_videos)}/{len(self.video_cache)} "
                  f"videos mirrored (dominant hand moved to MediaPipe 'right')")
        if self.mirror_augment_p > 0:
            print(f"[{split.upper()}] mirror augmentation: p = {self.mirror_augment_p}")
        total_attempted = len(self.video_ids)
        total_cached = len(self.video_cache)
        if total_cached < total_attempted:
            print(f"[{split.upper()}] Cached {total_cached}/{total_attempted} videos "
                  f"({total_attempted - total_cached} skipped). Skip reasons: {skip_counts}")

        self.skip_counts = skip_counts
        self.skipped_videos = skipped_videos

        if len(self.windows) == 0:
            raise RuntimeError(
                f"[{split.upper()}] SignSegmentationDataset ended up with 0 windows/samples "
                f"(0 of {total_attempted} videos were successfully cached) -- this dataset is "
                f"EMPTY. Skip reasons: {skip_counts}. This fails here, loudly, instead of three "
                f"layers deep in DataLoader/RandomSampler as a confusing 'num_samples=0' error. "
                f"If missing_face/missing_hamer/missing_dinov2 is nonzero, check that "
                f"{self.face_dir!r} / {self.hamer_dir!r} / {self.dinov2_dir!r} actually contain "
                f"files named EXACTLY like your split's video ids (e.g. '{self.video_ids[0]}.npy', "
                f"'{self.video_ids[0]}_hamer.pt', or '{self.video_ids[0]}_dinov2.pt' if that helps) "
                f"-- a naming-convention mismatch between how a feature stream was extracted and "
                f"how body/hand keypoints are named is the most common cause."
            )

    def inverse_class_ratio(self):
        """
        Class weights as in Moryossef et al. (2023) (data.py inverse_classes_ratio):
        weight_c = total frames / frames of class c, counted on the TRAINING targets
        (hard labels, after this dataset's temporal downsampling). Returns [O, I, B].
        """
        counts = np.zeros(3, dtype=np.float64)
        k = max(1, int(self.temporal_downsample_factor))
        for cached in self.video_cache.values():
            lab = torch.as_tensor(np.asarray(cached["labels"]), dtype=torch.float32).permute(1, 0)  # (3, T)
            hard = downsample_labels(lab, k).argmax(dim=0).numpy()
            counts += np.bincount(hard, minlength=3)[:3]
        if (counts == 0).any():
            raise RuntimeError(f"A class never occurs in the training labels: counts {counts}.")
        return (counts.sum() / counts).tolist()

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
        # keep_half (set by train_phrase.py for chunked full-video batches): stay float16 until the GPU
        feat_dtype = torch.float16 if getattr(self, 'keep_half', False) else torch.float32
        window_features = cached_data['features'][start_idx:end_idx].to(feat_dtype)
        window_labels = cached_data['labels'][start_idx:end_idx]
        
        final_input_tensor = window_features.permute(2, 0, 1)
        labels_tensor = torch.tensor(window_labels, dtype=torch.float32).permute(1, 0)

        if self.use_hamer_features:
            window_hamer = cached_data['hamer_features'][start_idx:end_idx].to(feat_dtype)  # (T_win, 288)
            hamer_tensor = window_hamer.permute(1, 0)  # (288, T_win) -- channel-first, matches final_input_tensor's convention

        if self.use_dinov2_features:
            window_dinov2 = cached_data['dinov2_features'][start_idx:end_idx].to(torch.float32)  # (T_win, 2*D)
            dinov2_tensor = window_dinov2.permute(1, 0)  # (2*D, T_win) -- channel-first

        if self.use_cue_features:
            cue_tensor = cached_data['cue_features'][start_idx:end_idx].to(torch.float32).permute(1, 0)  # (C, T_win)

        if self.use_aux:
            aux_tensor = cached_data['aux_labels'][:, start_idx:end_idx].to(torch.int64)  # (n_tasks, T_win)

        # Mirror augmentation (train split only). The cue stream is dominant/non-dominant
        # based and left/right symmetric, so it needs no change.
        if self.mirror_augment_p > 0 and torch.rand(1).item() < self.mirror_augment_p:
            final_input_tensor = mirror_graph_tv(final_input_tensor.permute(1, 2, 0), self._mirror_x_channels,
                                                 self._face_mirror_perm).permute(2, 0, 1)
            if self.use_hamer_features:
                hamer_tensor = mirror_hamer(hamer_tensor.permute(1, 0)).permute(1, 0)
        
        # 4. Handle Downsampling & Padding
        if self.temporal_downsample_factor > 1:
            final_input_tensor = final_input_tensor[:, ::self.temporal_downsample_factor, :]
            labels_tensor = downsample_labels(labels_tensor, self.temporal_downsample_factor)
            if self.use_hamer_features:
                hamer_tensor = hamer_tensor[:, ::self.temporal_downsample_factor]
            if self.use_dinov2_features:
                dinov2_tensor = dinov2_tensor[:, ::self.temporal_downsample_factor]
            if self.use_cue_features:
                cue_tensor = cue_tensor[:, ::self.temporal_downsample_factor]
            if self.use_aux:
                aux_tensor = aux_tensor[:, ::self.temporal_downsample_factor]
            
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

                if self.use_cue_features:
                    cue_pad = torch.zeros(cue_tensor.shape[0], pad_T, dtype=torch.float32)
                    cue_tensor = torch.cat([cue_tensor, cue_pad], dim=1)

                if self.use_aux:
                    aux_pad = torch.full((aux_tensor.shape[0], pad_T), AUX_IGNORE, dtype=torch.int64)
                    aux_tensor = torch.cat([aux_tensor, aux_pad], dim=1)
                
        start_scaled = start_idx // self.temporal_downsample_factor
        end_scaled = end_idx // self.temporal_downsample_factor
        
        # Consistent ordering across all 4 combinations: features, labels, [hamer],
        # [dinov2], vid, start, end -- optional streams appear in this fixed order
        # whenever enabled, never swapped, so callers can unpack unambiguously based
        # on which two flags are set.
        extras = []
        if self.use_hamer_features:
            extras.append(hamer_tensor)
        if self.use_dinov2_features:
            extras.append(dinov2_tensor)
        if self.use_cue_features:
            extras.append(cue_tensor)   # always LAST among the optional streams
        if self.use_aux:
            extras.append(aux_tensor)   # auxiliary TARGETS (train split only): after every input stream

        return (final_input_tensor, labels_tensor, *extras, vid, start_scaled, end_scaled)