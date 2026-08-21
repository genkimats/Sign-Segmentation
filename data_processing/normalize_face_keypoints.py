import os
import multiprocessing as mproc
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RAW_VIDEO_DIR = "raw_data/videos"
RAW_FACE_DIR = "processed_data/face_keypoints"                    # untouched, read-only input
NORMALIZED_FACE_DIR = "processed_data/face_keypoints_normalized"  # new output directory

# Optional cache written by the (optionally) patched extract_poses.py -- see the
# accompanying patch. If present for a video, it's used instead of re-deriving
# shoulder positions from the video (much cheaper). Safe to leave this directory
# empty/nonexistent; the script just falls back to the slower path below.
POSE_NORM_CACHE_DIR = "processed_data/pose_normalization_params"

os.makedirs(NORMALIZED_FACE_DIR, exist_ok=True)

POSE_MODEL_PATH = "pose_landmarker_full.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# Number of files normalized in parallel. Every video either hits the cheap
# cache path (pure numpy, trivially parallel) or the fallback path (another
# CPU-bound MediaPipe Pose-only pass, same cost profile as extract_hand_boxes.py) --
# no GPU/big-model-per-worker concern here, but this is still CPU-core-bound,
# so more than your actual core count may not help. Check with `nproc`.
NUM_WORKERS = 6

# MediaPipe Pose Landmarker's own (raw, un-reduced, 33-point) landmark indices.
# Confirmed against extract_poses.py's normalize_skeleton() comment ("Shoulders are
# typically at index 11 and 12 in the 65-point array"), which matches MediaPipe's
# standard Pose topology directly -- strong evidence BODY_LANDMARKS_KEPT preserves
# raw MediaPipe indices in order (0..22), so kept-array position 11/12 = raw Pose
# index 11/12 = left/right shoulder. If your config.py's BODY_LANDMARKS_KEPT
# reorders or remaps indices, update these two constants to match.
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def download_pose_model_if_needed():
    if not os.path.exists(POSE_MODEL_PATH):
        import urllib.request
        print("Downloading MediaPipe Pose Landmarker model...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("Download complete.")


def compute_root_and_scale_from_video(video_path):
    """
    Re-derives the SAME per-frame (root, scale) that extract_poses.py's
    normalize_skeleton() would have used, by running a lightweight Pose-ONLY
    landmarker pass (not the full Holistic+hands pipeline extract_poses.py runs)
    over the video and reading just the two shoulder landmarks.

    root:  (T, 3) midpoint of left/right shoulder, raw MediaPipe image-normalized coords
    scale: (T, 1) XY-only distance between the shoulders (matches normalize_skeleton exactly)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 50.0

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    shoulders = []  # per frame: [[lx,ly,lz], [rx,ry,rz]], or NaNs if undetected

    # Fresh landmarker per video (VIDEO mode keeps internal timestamp state that
    # must not be reused across separate videos -- see extract_hand_boxes.py fix).
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                l = pose[LEFT_SHOULDER_IDX]
                r = pose[RIGHT_SHOULDER_IDX]
                shoulders.append([[l.x, l.y, l.z], [r.x, r.y, r.z]])
            else:
                shoulders.append([[np.nan] * 3, [np.nan] * 3])

            frame_idx += 1

    cap.release()

    shoulders = np.array(shoulders, dtype=np.float32)  # (T, 2, 3)

    # Interpolate missed-detection frames -- same spirit as extract_poses.py's own
    # NaN handling, so a dropped frame here doesn't introduce a discontinuity that
    # wasn't already present in the body/hand normalization.
    flat = shoulders.reshape(shoulders.shape[0], -1)
    df = pd.DataFrame(flat)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(0, inplace=True)
    shoulders = df.values.reshape(shoulders.shape)

    l_shoulder, r_shoulder = shoulders[:, 0, :], shoulders[:, 1, :]

    # EXACT same formula as extract_poses.py's normalize_skeleton().
    scale = np.linalg.norm(l_shoulder[:, :2] - r_shoulder[:, :2], axis=1, keepdims=True) + 1e-6
    root = (l_shoulder + r_shoulder) / 2.0

    return root.astype(np.float32), scale.astype(np.float32)


def normalize_face_array(face_raw, root, scale):
    """Applies the identical (coords - root) / scale transform used for body/hands."""
    root_expanded = root[:, np.newaxis, :]    # (T, 1, 3)
    scale_expanded = scale[:, np.newaxis, :]  # (T, 1, 1)
    return (face_raw - root_expanded) / scale_expanded


def find_video_path(vid):
    for ext in (".mp4", ".avi", ".mov"):
        candidate = os.path.join(RAW_VIDEO_DIR, vid + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_one_file(fname):
    """
    Runs in a worker process. Returns (fname, message_or_None) so the parent
    can report skips/warnings without multiple processes' prints colliding in
    the terminal.
    """
    vid = fname[:-4]
    out_path = os.path.join(NORMALIZED_FACE_DIR, fname)

    face_raw = np.load(os.path.join(RAW_FACE_DIR, fname))  # (T, NUM_FACE_VERTICES, 3)

    # 1. Cheap path: reuse cached root/scale if extract_poses.py already wrote one.
    cache_path = os.path.join(POSE_NORM_CACHE_DIR, f"{vid}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        root, scale = cached["root"], cached["scale"]
    else:
        # 2. Fallback: re-derive it from the raw video (one more decode + a
        #    lightweight Pose-only inference pass -- slower, but exact).
        video_path = find_video_path(vid)
        if video_path is None:
            return fname, (f"Skipping {vid}: no source video found to re-derive shoulder "
                            f"root/scale (and no cached {cache_path}).")
        root, scale = compute_root_and_scale_from_video(video_path)

    if root.shape[0] != face_raw.shape[0]:
        return fname, (f"Skipping {vid}: frame count mismatch (face={face_raw.shape[0]}, "
                        f"pose={root.shape[0]}). Investigate before trusting this video.")

    face_normalized = normalize_face_array(face_raw, root, scale).astype(np.float32)
    np.save(out_path, face_normalized)

    return fname, None


def main():
    all_face_files = sorted(f for f in os.listdir(RAW_FACE_DIR) if f.endswith('.npy'))

    # Only normalize videos whose output doesn't already exist. Scanning the
    # output directory ONCE up front (by id, i.e. video stem) rather than relying
    # solely on a per-file check keeps this consistent with extract_hand_boxes.py /
    # extract_hamer_features.py, and means a fully-resumed run skips the pose-model
    # download check and the worker pool entirely.
    existing_ids = {f[:-4] for f in os.listdir(NORMALIZED_FACE_DIR) if f.endswith('.npy')}
    face_files = [f for f in all_face_files if f[:-4] not in existing_ids]

    skipped = len(all_face_files) - len(face_files)
    print(f"Found {len(all_face_files)} raw face-keypoint files, {skipped} already "
          f"normalized -- processing the remaining {len(face_files)}.")

    if not face_files:
        print("Nothing to do.")
        return

    download_pose_model_if_needed()

    # 'spawn', not the Linux default 'fork': keeps each worker a genuinely fresh
    # interpreter, avoiding any ambiguity around forking a process that may have
    # touched native library state (same habit as extract_hand_boxes.py /
    # extract_hamer_features.py).
    ctx = mproc.get_context('spawn')
    with ctx.Pool(processes=NUM_WORKERS) as pool:
        for fname, message in tqdm(
            pool.imap_unordered(process_one_file, face_files),
            total=len(face_files),
            desc=f"Normalizing face keypoints ({NUM_WORKERS} workers)",
        ):
            if message:
                tqdm.write(message)

    print(f"Done. Normalized files written to {NORMALIZED_FACE_DIR}")
    print(f"Raw files in {RAW_FACE_DIR} were not modified.")


if __name__ == "__main__":
    main()