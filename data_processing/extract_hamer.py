import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# HaMeR imports
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset   # <-- reuse HaMeR's own preprocessing
from hamer.utils import recursive_to

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_VIDEO_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/data/raw_videos")
OUTPUT_FEATURE_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/processed_data/hamer_features")
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU batch size for the HaMeR forward pass (unrelated to video frame stride below)
BATCH_SIZE = 32

# HaMeR's own demo/dataset code uses rescale_factor=2.0-2.5 (padding around the tight
# hand bbox before cropping). The previous version of this script used 1.5, which under-pads
# the crop relative to how HaMeR was actually trained/demonstrated. Match their convention.
RESCALE_FACTOR = 2.0

# --- Temporal downsampling ---
# The 2025 paper downsamples HaMeR features by a factor of 2 *after* extraction, inside the
# model (see Fig. 2: "T x 288 HaMeR features" -> downsample -> "T/2 x 512"), purely to shrink
# the transformer's sequence length. We instead downsample AT EXTRACTION time (i.e. only run
# detection/HaMeR on every 2nd frame) because storage, not compute, is the bottleneck here.
# This produces the same effective input resolution but is not literally what the paper did.
#
# IMPORTANT: because of this, the saved tensors here run at HALF the frame rate of your
# keypoints/BIO-tag/kinematic-feature files. You must account for this when combining streams
# in SignSegmentationDataset (e.g. by using the *same* temporal_downsample_factor=2 for every
# stream so time axes line up, or by upsampling this stream 2x with repeat-interpolation
# before concatenation). Silently concatenating this with full-rate kinematic features will
# misalign every frame after the first.
TEMPORAL_DOWNSAMPLE_FACTOR = 2

# MANO hand_pose is (15, 3, 3) rotation matrices per hand, global_orient is (3, 3) per hand.
# HaMeR always regresses in a canonical RIGHT-hand frame: left-hand crops are horizontally
# mirrored before being fed to the network (ViTDetDataset does this via the `right` flag), and
# the output must be mirrored back. For 3D points this is a coordinate negation; for rotation
# matrices it is a conjugation by the same reflection. Mirroring about the vertical (x) axis:
MIRROR_AXIS = 0


def unmirror_rotation_batch(R: np.ndarray, mirror_axis: int = MIRROR_AXIS) -> np.ndarray:
    """
    Corrects one or more rotation matrices that were predicted on a horizontally-mirrored
    crop (i.e. a left hand shown to HaMeR as if it were a right hand), converting them back
    to the true rotation of the original, unmirrored hand.

    For a reflection M (M @ M = I), if a rotation R was estimated in the mirrored frame,
    the equivalent rotation in the true frame is M @ R @ M.

    R: array of shape (..., 3, 3)
    """
    M = np.eye(3, dtype=R.dtype)
    M[mirror_axis, mirror_axis] = -1.0
    return np.einsum('ij,...jk,kl->...il', M, R, M)


def get_pixel_bbox(landmarks, img_w, img_h):
    """
    Tight pixel-space bbox [x1, y1, x2, y2] around MediaPipe hand landmarks.
    Do NOT add extra padding/scale here -- ViTDetDataset applies its own context
    padding via RESCALE_FACTOR, matching how HaMeR derives boxes from keypoints
    in its own demo (min/max of detected keypoints, unpadded).
    """
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    return [min(xs), min(ys), max(xs), max(ys)]


# ==============================================================================
# MAIN EXTRACTION LOOP
# ==============================================================================
def main():
    print(f"Loading HaMeR Model on {DEVICE} with Batch Size {BATCH_SIZE}...")
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(DEVICE)
    model.eval()

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_name in tqdm(video_files, desc="Processing Videos with HaMeR"):
        video_path = os.path.join(INPUT_VIDEO_DIR, video_name)

        save_name = video_name.rsplit('.', 1)[0] + "_hamer.pt"
        save_path = os.path.join(OUTPUT_FEATURE_DIR, save_name)
        if os.path.exists(save_path):
            continue

        cap = cv2.VideoCapture(video_path)

        # Per kept-frame outputs. Index 0 = left hand, index 1 = right hand.
        video_hand_pose = []       # each entry: (2, 15, 3, 3) float32
        video_global_orient = []   # each entry: (2, 3, 3) float32
        kept_source_frame_idx = [] # original (pre-downsample) frame index for each kept slot

        # Batch queues, spanning potentially many frames
        hand_queue = []   # list of per-hand item dicts, as returned by ViTDetDataset.__getitem__
        meta_queue = []   # list of (slot_idx, is_right) matching hand_queue order

        def flush_queue():
            """Pushes the accumulated queue through the GPU."""
            if not hand_queue:
                return

            batch = torch.utils.data.dataloader.default_collate(hand_queue)
            batch = recursive_to(batch, DEVICE)

            with torch.no_grad():
                out = model(batch)

            # NOTE: first time you run this, sanity-check these keys/shapes:
            #   print(out['pred_mano_params'].keys())
            #   print(out['pred_mano_params']['hand_pose'].shape)      # expect (B, 15, 3, 3)
            #   print(out['pred_mano_params']['global_orient'].shape)  # expect (B, 1, 3, 3) or (B, 3, 3)
            mano = out['pred_mano_params']
            hand_pose = mano['hand_pose'].detach().cpu().numpy()        # (B, 15, 3, 3)
            global_orient = mano['global_orient'].detach().cpu().numpy()
            if global_orient.ndim == 4:
                global_orient = global_orient[:, 0]                     # (B, 1, 3, 3) -> (B, 3, 3)

            for i, (slot_idx, is_right) in enumerate(meta_queue):
                hp = hand_pose[i].copy()
                go = global_orient[i].copy()

                if not is_right:
                    # Un-mirror: HaMeR always predicts in right-hand canonical space.
                    hp = unmirror_rotation_batch(hp)
                    go = unmirror_rotation_batch(go)

                slot = 1 if is_right else 0
                video_hand_pose[slot_idx][slot] = hp
                video_global_orient[slot_idx][slot] = go

            hand_queue.clear()
            meta_queue.clear()

        raw_frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if raw_frame_idx % TEMPORAL_DOWNSAMPLE_FACTOR != 0:
                raw_frame_idx += 1
                continue

            img_h, img_w = frame.shape[:2]

            slot_idx = len(video_hand_pose)
            video_hand_pose.append(np.zeros((2, 15, 3, 3), dtype=np.float32))
            video_global_orient.append(np.zeros((2, 3, 3), dtype=np.float32))
            kept_source_frame_idx.append(raw_frame_idx)

            results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                boxes, rights = [], []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
                    is_right = (label == 'Right')
                    boxes.append(get_pixel_bbox(hand_landmarks, img_w, img_h))
                    rights.append(1.0 if is_right else 0.0)

                if boxes:
                    boxes_arr = np.array(boxes, dtype=np.float32)
                    rights_arr = np.array(rights, dtype=np.float32)

                    # ViTDetDataset takes ONE full frame + all detected boxes on it, and
                    # replicates HaMeR's exact crop/scale/flip/normalize pipeline per box.
                    frame_ds = ViTDetDataset(model_cfg, frame, boxes_arr, rights_arr,
                                              rescale_factor=RESCALE_FACTOR)
                    for i in range(len(frame_ds)):
                        hand_queue.append(frame_ds[i])
                        meta_queue.append((slot_idx, bool(rights_arr[i])))

                    if len(hand_queue) >= BATCH_SIZE:
                        flush_queue()

            raw_frame_idx += 1

        cap.release()
        flush_queue()

        video_features = {
            "hand_pose": np.array(video_hand_pose, dtype=np.float32),         # (T', 2, 15, 3, 3)
            "global_orient": np.array(video_global_orient, dtype=np.float32), # (T', 2, 3, 3)
            "hand_order": "index 0 = left, index 1 = right",
            "temporal_downsample_factor": TEMPORAL_DOWNSAMPLE_FACTOR,
            "source_frame_indices": np.array(kept_source_frame_idx, dtype=np.int64),
        }

        torch.save(video_features, save_path)


if __name__ == "__main__":
    main()