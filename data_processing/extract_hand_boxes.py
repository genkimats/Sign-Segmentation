"""
Stage 1 of 2 for HaMeR feature extraction.

Runs ONLY MediaPipe Hands over each video and caches per-frame hand bounding
boxes + handedness to disk. Deliberately imports nothing from `torch` or
`hamer` -- keeping MediaPipe and PyTorch/HaMeR in separate processes avoids a
known class of native-library (ABI) segfault that happens when both get
loaded into the same interpreter.

Run this first, then run extract_hamer_features.py (stage 2) as a SEPARATE
`python` invocation.
"""
import os
import pickle
import cv2
import mediapipe as mp
from tqdm import tqdm

INPUT_VIDEO_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/data/raw_videos")
OUTPUT_BOX_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/processed_data/hand_boxes")
os.makedirs(OUTPUT_BOX_DIR, exist_ok=True)

# Must match TEMPORAL_DOWNSAMPLE_FACTOR in extract_hamer_features.py.
# (Stage 2 also reads this value back out of the cached file, so the two
# scripts can never silently disagree -- but keep them equal for clarity.)
TEMPORAL_DOWNSAMPLE_FACTOR = 2


def get_pixel_bbox(landmarks, img_w, img_h):
    """Tight pixel-space bbox [x1, y1, x2, y2] around MediaPipe hand landmarks."""
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    return [min(xs), min(ys), max(xs), max(ys)]


def main():
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_name in tqdm(video_files, desc="Detecting hands (MediaPipe only)"):
        video_path = os.path.join(INPUT_VIDEO_DIR, video_name)
        save_name = video_name.rsplit('.', 1)[0] + "_boxes.pkl"
        save_path = os.path.join(OUTPUT_BOX_DIR, save_name)
        if os.path.exists(save_path):
            continue

        cap = cv2.VideoCapture(video_path)
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
            results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            boxes, rights = [], []
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
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


if __name__ == "__main__":
    main()