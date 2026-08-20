"""
Stage 1 of 2 for HaMeR feature extraction.

Runs ONLY MediaPipe Hands over each video and caches per-frame hand bounding
boxes + handedness to disk. Deliberately imports nothing from `torch` or
`hamer` -- keeping MediaPipe and PyTorch/HaMeR in separate processes avoids a
known class of native-library (ABI) segfault that happens when both get
loaded into the same interpreter.

Videos are distributed across NUM_WORKERS worker processes for speed.

Run this first, then run extract_hamer_features.py (stage 2) as a SEPARATE
`python` invocation.
"""
import os
import pickle
import multiprocessing as mproc
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

INPUT_VIDEO_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/data/raw_videos")
OUTPUT_BOX_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/processed_data/hand_boxes")
os.makedirs(OUTPUT_BOX_DIR, exist_ok=True)

# mp.solutions.hands was removed in MediaPipe >= 0.10.31; this now uses the
# Tasks API, which requires an explicit model bundle. Download it once with:
#   wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# (same pattern as the face_landmarker.task you already have at the repo root).
HAND_LANDMARKER_MODEL_PATH = os.path.expanduser(
    "~/Genki_GR/Sign-Segmentation/hand_landmarker.task"
)

# Must match TEMPORAL_DOWNSAMPLE_FACTOR in extract_hamer_features.py.
# (Stage 2 also reads this value back out of the cached file, so the two
# scripts can never silently disagree -- but keep them equal for clarity.)
TEMPORAL_DOWNSAMPLE_FACTOR = 2

# Number of videos processed in parallel. This is CPU-bound work (MediaPipe
# Tasks API on CPU), so this is a straightforward core-count tradeoff -- raise
# it if you have more cores to spare and want it faster, lower it (or drop to
# 1) if it's contending with something else running on the same machine.
NUM_WORKERS = 2


def get_pixel_bbox(landmarks, img_w, img_h):
    """
    Tight pixel-space bbox [x1, y1, x2, y2] around hand landmarks.
    `landmarks` here is a single hand's list of NormalizedLandmark (Tasks API
    returns hand_landmarks as list[list[NormalizedLandmark]] directly -- no
    `.landmark` wrapper attribute like the old mp.solutions API had).
    """
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    return [min(xs), min(ys), max(xs), max(ys)]


def process_one_video(video_name):
    """
    Runs in a worker process. Builds its own HandLandmarker (this is already
    cheap/required per-video regardless of parallelism -- see the note below),
    so there's no expensive per-worker setup to hoist into a Pool initializer
    here, unlike extract_hamer_features.py.

    Returns (video_name, message_or_None) so the parent can report skips/
    warnings without multiple processes' prints colliding in the terminal.
    """
    video_path = os.path.join(INPUT_VIDEO_DIR, video_name)
    save_name = video_name.rsplit('.', 1)[0] + "_boxes.pkl"
    save_path = os.path.join(OUTPUT_BOX_DIR, save_name)
    if os.path.exists(save_path):
        return video_name, None

    base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,  # replaces static_image_mode=False
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    # A fresh landmarker per video: VIDEO running mode tracks an internal
    # "last timestamp seen" and requires strictly increasing timestamps for
    # the LIFETIME OF THE INSTANCE. Reusing one landmarker across multiple
    # videos (each restarting its own timestamp count from 0) violates that
    # and throws "Input timestamp must be monotonically increasing."
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # VIDEO mode needs real, monotonic timestamps
        frames_out = []

        raw_frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if raw_frame_idx % TEMPORAL_DOWNSAMPLE_FACTOR != 0:
                raw_frame_idx += 1
                continue

            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(raw_frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            boxes, rights = [], []
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                # NOTE: MediaPipe's handedness assumes the input is a mirrored/selfie-style
                # image (front-facing camera). Corpus recordings are typically NOT mirrored,
                # so this label may be flipped relative to the signer's actual hand -- verify
                # against one known video before trusting it, and swap if needed.
                label = handedness[0].category_name  # "Left" or "Right"
                boxes.append(get_pixel_bbox(hand_landmarks, img_w, img_h))
                rights.append(1.0 if label == 'Right' else 0.0)

            frames_out.append({
                "source_frame_idx": raw_frame_idx,
                "boxes": boxes,
                "rights": rights,
            })

            raw_frame_idx += 1

        cap.release()

    with open(save_path, "wb") as f:
        pickle.dump({
            "temporal_downsample_factor": TEMPORAL_DOWNSAMPLE_FACTOR,
            "frames": frames_out,
        }, f)

    return video_name, None


def main():
    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    # 'spawn', not the Linux default 'fork': keeps each worker a genuinely fresh
    # interpreter, avoiding any ambiguity around forking a process that may have
    # touched native library state (safe habit to share with extract_hamer_features.py,
    # even though this script itself never touches CUDA).
    ctx = mproc.get_context('spawn')
    with ctx.Pool(processes=NUM_WORKERS) as pool:
        for video_name, message in tqdm(
            pool.imap_unordered(process_one_video, video_files),
            total=len(video_files),
            desc=f"Detecting hands (MediaPipe only, {NUM_WORKERS} workers)",
        ):
            if message:
                tqdm.write(message)


if __name__ == "__main__":
    main()