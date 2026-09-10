"""
Extracts DINOv2 (self-supervised Vision Transformer) visual embeddings from
hand crops -- the representation used by SHuBERT (2024) and SignMusketeers
(2024) for sign language. Unlike everything else in this pipeline (raw
coordinates, spatial angles, kinematic derivatives, HaMeR's MANO parameters),
this is an APPEARANCE-based feature: it comes directly from pixels, not from
any geometric/kinematic derivation, so it can capture texture, shading, and
subtle handshape/contact cues that keypoint- or mesh-based representations
throw away.

Reuses the hand bounding boxes already extracted by extract_hand_boxes.py
(processed_data/hand_boxes/{id}_boxes.pkl) -- no new hand detection needed,
and no re-running of any MediaPipe code. DINOv2 is a plain PyTorch model with
no MediaPipe dependency, so (unlike extract_hamer_features.py) this runs as a
single script/process: no MediaPipe+PyTorch segfault concern, and no
multi-worker GPU-memory-multiplication concern either -- this batches crops
WITHIN one loaded model instead of loading multiple model copies.

Box/handedness format read here is IDENTICAL to what extract_hamer_features.py
reads from the same .pkl files: frame_records[i] = {"source_frame_idx": int,
"boxes": [[x1,y1,x2,y2], ...], "rights": [1.0 or 0.0, ...]} (parallel lists,
0-2 entries per frame). Same left/right slot convention too: slot 0 = left
hand, slot 1 = right hand (rights[i]==1.0 means MediaPipe's raw "Right" label
-- see extract_hand_boxes.py's own caveat about this being possibly mirrored
for non-selfie-style recordings; this script doesn't correct for it, matching
extract_hamer_features.py's convention, so both stay consistent with each
other even if that caveat turns out to matter later).

REQUIRES INTERNET ACCESS the first time it runs, to download the pretrained
DINOv2 checkpoint via torch.hub (from facebookresearch/dinov2 on GitHub). If
this machine has no internet access, download the checkpoint on one that does
and point TORCH_HOME at a local cache directory before running this.

Run AFTER extract_hand_boxes.py. Does not depend on extract_hamer_features.py
at all -- these two feature streams are independent and can be extracted in
either order (or skipped independently).
"""
import os
import pickle
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_VIDEO_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/data/raw_videos")
INPUT_BOX_DIR = "processed_data/hand_boxes"
OUTPUT_FEATURE_DIR = "processed_data/dinov2_features"
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)

# DINOv2 model variant. SHuBERT/SignMusketeers use the Small variant (384-dim)
# specifically for efficiency; since efficiency isn't a priority here, this
# defaults to the Base variant (768-dim) for richer features instead. The
# "_reg" suffix uses register tokens (Darcet et al. 2024), which both those
# papers also used -- generally cleaner attention maps, recommended over the
# non-reg variants. Change to 'dinov2_vits14_reg' (384-dim, matches the
# literature exactly) or 'dinov2_vitl14_reg' (1024-dim, heaviest) if you want
# a different point on that tradeoff -- just update DINOV2_FEATURE_DIM to match.
DINOV2_MODEL_NAME = "dinov2_vitb14_reg"
DINOV2_FEATURE_DIM = 768  # vits14=384, vitb14=768, vitl14=1024, vitg14=1536

CROP_SIZE = 224  # must be a multiple of 14 (DINOv2's patch size); 224 = 16x16 patches
BATCH_SIZE = 64  # crops per forward pass -- lower this if you hit GPU OOM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((CROP_SIZE, CROP_SIZE), antialias=True),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_dinov2_model():
    print(f"Loading {DINOV2_MODEL_NAME} (downloads on first run if not already cached)...")
    model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL_NAME)
    model.eval().to(DEVICE)
    return model


def crop_box(frame_rgb, box):
    """box: [x1, y1, x2, y2] in pixel coords (as produced by extract_hand_boxes.py's
    get_pixel_bbox). Returns an HxWx3 RGB uint8 crop, or None if degenerate."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_rgb[y1:y2, x1:x2]


@torch.no_grad()
def extract_batch(model, crops):
    """crops: list of HxWx3 uint8 RGB numpy arrays (no Nones). Returns (N, DINOV2_FEATURE_DIM)."""
    if not crops:
        return torch.zeros((0, DINOV2_FEATURE_DIM))
    batch = torch.stack([preprocess(c) for c in crops]).to(DEVICE)
    features = model(batch)  # (N, DINOV2_FEATURE_DIM) pooled/CLS-token output
    return features.cpu()


def process_one_video(model, video_path, box_path, save_path):
    with open(box_path, "rb") as f:
        cached = pickle.load(f)
    frame_records = cached["frames"]

    cap = cv2.VideoCapture(video_path)

    # Index 0 = left hand, index 1 = right hand (matches extract_hamer_features.py).
    features = torch.zeros((len(frame_records), 2, DINOV2_FEATURE_DIM), dtype=torch.float32)

    crop_queue = []      # pending crops awaiting a batched forward pass
    meta_queue = []      # (record_ptr, slot) for each queued crop, same order as crop_queue

    def flush_queue():
        if not crop_queue:
            return
        feats = extract_batch(model, crop_queue)
        for i, (record_ptr, slot) in enumerate(meta_queue):
            features[record_ptr, slot] = feats[i]
        crop_queue.clear()
        meta_queue.clear()

    raw_frame_idx = 0
    record_ptr = 0
    while cap.isOpened() and record_ptr < len(frame_records):
        ret, frame = cap.read()
        if not ret:
            break

        record = frame_records[record_ptr]
        if raw_frame_idx != record["source_frame_idx"]:
            # This frame was dropped by stage 1's downsample stride; skip it.
            raw_frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, rights = record["boxes"], record["rights"]

        for box, is_right in zip(boxes, rights):
            crop = crop_box(frame_rgb, box)
            if crop is None:
                continue  # degenerate box -- leave this slot zero, same as "not detected"
            slot = 1 if is_right else 0
            crop_queue.append(crop)
            meta_queue.append((record_ptr, slot))

            if len(crop_queue) >= BATCH_SIZE:
                flush_queue()

        record_ptr += 1
        raw_frame_idx += 1

    cap.release()
    flush_queue()

    torch.save({
        "features": features,  # (T, 2, DINOV2_FEATURE_DIM), zero where a hand wasn't detected
        "model_name": DINOV2_MODEL_NAME,
        "temporal_downsample_factor": cached["temporal_downsample_factor"],
    }, save_path)


def find_video_path(vid):
    for ext in (".mp4", ".avi", ".mov"):
        candidate = os.path.join(INPUT_VIDEO_DIR, vid + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    all_box_files = [f for f in os.listdir(INPUT_BOX_DIR) if f.endswith("_boxes.pkl")]
    all_ids = [f[:-len("_boxes.pkl")] for f in all_box_files]

    existing_ids = {
        f[:-len("_dinov2.pt")] for f in os.listdir(OUTPUT_FEATURE_DIR) if f.endswith("_dinov2.pt")
    }
    ids_to_process = [vid for vid in all_ids if vid not in existing_ids]

    skipped = len(all_ids) - len(ids_to_process)
    print(f"Found {len(all_ids)} videos with cached hand boxes, {skipped} already extracted -- "
          f"processing the remaining {len(ids_to_process)}.")

    if not ids_to_process:
        print("Nothing to do.")
        return

    model = load_dinov2_model()

    for vid in tqdm(ids_to_process, desc=f"Extracting DINOv2 features ({DINOV2_MODEL_NAME})"):
        video_path = find_video_path(vid)
        if video_path is None:
            tqdm.write(f"Skipping {vid}: no source video found in {INPUT_VIDEO_DIR}.")
            continue

        box_path = os.path.join(INPUT_BOX_DIR, f"{vid}_boxes.pkl")
        save_path = os.path.join(OUTPUT_FEATURE_DIR, f"{vid}_dinov2.pt")

        try:
            process_one_video(model, video_path, box_path, save_path)
        except Exception as e:
            tqdm.write(f"Error processing {vid}: {e}")
            continue


if __name__ == "__main__":
    main()