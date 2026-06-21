import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# We import the config variables assuming you have them defined
from config import TOTAL_LANDMARKS, BODY_LANDMARKS_KEPT

# Get the exact directory of this script to load the downloaded .task file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "holistic_landmarker.task")

def normalize_skeleton(keypoints):
    """Centers and scales the skeleton based on shoulder width for X, Y, and Z."""
    # keypoints shape: (T, 65, 3)
    # Shoulders are typically at index 11 and 12 in the 65-point array 
    l_shoulder = keypoints[:, 11, :]
    r_shoulder = keypoints[:, 12, :]
    
    # Use X and Y for scale to prevent depth-warping
    scale = np.linalg.norm(l_shoulder[:, :2] - r_shoulder[:, :2], axis=1, keepdims=True) + 1e-6
    
    # Root uses X, Y, and Z
    root = (l_shoulder + r_shoulder) / 2.0
    
    root_expanded = np.expand_dims(root, axis=1)
    scale_expanded = np.expand_dims(scale, axis=1)
    
    normalized = np.copy(keypoints)
    # Normalize X, Y, and Z
    normalized[:, :, :3] = (keypoints[:, :, :3] - root_expanded) / scale_expanded
    return normalized

def extract_video_features(video_path, position=1):
    """Extracts 3D pose and hand landmarks (X, Y, Z) using the Tasks API."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Please run the curl command to download it.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback just in case OpenCV fails to read DGS properties
    if fps == 0 or np.isnan(fps):
        fps = 50.0  
        
    all_keypoints = np.full((total_frames, TOTAL_LANDMARKS, 3), np.nan)
    
    # Fast VIDEO running mode
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False
    )
    
    # Safely handle BODY_LANDMARKS_KEPT whether it's an int or a list
    if isinstance(BODY_LANDMARKS_KEPT, int):
        body_indices = range(BODY_LANDMARKS_KEPT)
        num_body_lms = BODY_LANDMARKS_KEPT
    else:
        body_indices = BODY_LANDMARKS_KEPT
        num_body_lms = len(BODY_LANDMARKS_KEPT)
    
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        
        # Dynamic position for multi-threaded progress bars
        vid_name = os.path.basename(video_path)
        # Truncate long names so the progress bar fits on screen
        short_name = vid_name[:15] + ".." if len(vid_name) > 15 else vid_name
        
        for frame_idx in tqdm(range(total_frames), desc=f"Extr: {short_name}", position=position, leave=False):
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_data = []
            
            # A. Body
            if result.pose_landmarks:
                pose = result.pose_landmarks
                if len(pose) > 0 and not hasattr(pose[0], 'x'):
                    pose = pose[0]
                    
                for i in body_indices:
                    if i < len(pose):
                        lm = pose[i]
                        frame_data.append([lm.x, lm.y, lm.z]) 
                    else:
                        frame_data.append([np.nan, np.nan, np.nan])
            else:
                frame_data.extend([[np.nan, np.nan, np.nan]] * num_body_lms)
                
            # B. Left Hand
            if result.left_hand_landmarks:
                left_hand = result.left_hand_landmarks
                if len(left_hand) > 0 and not hasattr(left_hand[0], 'x'):
                    left_hand = left_hand[0]
                    
                for i in range(21):
                    if i < len(left_hand):
                        lm = left_hand[i]
                        frame_data.append([lm.x, lm.y, lm.z]) 
                    else:
                        frame_data.append([np.nan, np.nan, np.nan])
            else:
                frame_data.extend([[np.nan, np.nan, np.nan]] * 21)
                
            # C. Right Hand
            if result.right_hand_landmarks:
                right_hand = result.right_hand_landmarks
                if len(right_hand) > 0 and not hasattr(right_hand[0], 'x'):
                    right_hand = right_hand[0]
                    
                for i in range(21):
                    if i < len(right_hand):
                        lm = right_hand[i]
                        frame_data.append([lm.x, lm.y, lm.z]) 
                    else:
                        frame_data.append([np.nan, np.nan, np.nan])
            else:
                frame_data.extend([[np.nan, np.nan, np.nan]] * 21)
                
            all_keypoints[frame_idx] = np.array(frame_data)
            
    cap.release()
    
    # -- Interpolate missing frames (NaNs) --
    orig_shape = all_keypoints.shape
    flattened = all_keypoints.reshape(orig_shape[0], -1)
    df = pd.DataFrame(flattened)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(0, inplace=True) # just in case a column is entirely NaN
    all_keypoints = df.values.reshape(orig_shape)
    
    # -- Normalize using 3D Math --
    all_keypoints = normalize_skeleton(all_keypoints)
    
    return all_keypoints

def process_directory(video_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    videos = [v for v in os.listdir(video_dir) if v.endswith('.mp4')]
    
    for video_name in videos:
        video_path = os.path.join(video_dir, video_name)
        output_path = os.path.join(output_dir, video_name.replace('.mp4', '.npy'))
        
        if os.path.exists(output_path):
             continue
             
        # Just process sequentially if run directly as a script
        keypoints = extract_video_features(video_path, position=1)
        np.save(output_path, keypoints)

if __name__ == "__main__":
    RAW_VIDEOS_DIR = "raw_data/videos"
    PROCESSED_KP_DIR = "processed_data/keypoints"
    process_directory(RAW_VIDEOS_DIR, PROCESSED_KP_DIR)