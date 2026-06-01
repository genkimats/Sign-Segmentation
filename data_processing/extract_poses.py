import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from config import TOTAL_LANDMARKS, BODY_LANDMARKS_KEPT

# Get the exact directory of this script to load the downloaded .task file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "holistic_landmarker.task")

def normalize_skeleton(keypoints):
    """Centers and scales the skeleton based on shoulder width."""
    l_shoulder = keypoints[:, 11, :2]
    r_shoulder = keypoints[:, 12, :2]
    
    root = (l_shoulder + r_shoulder) / 2.0
    scale = np.linalg.norm(l_shoulder - r_shoulder, axis=1, keepdims=True) + 1e-6
    
    root_expanded = np.expand_dims(root, axis=1)
    scale_expanded = np.expand_dims(scale, axis=1)
    
    normalized = np.copy(keypoints)
    normalized[:, :, :2] = (keypoints[:, :, :2] - root_expanded) / scale_expanded
    return normalized

def extract_video_features(video_path):
    """Extracts 2D pose and hand landmarks using the Tasks API."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Please run the curl command to download it.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback just in case OpenCV fails to read DGS properties
    if fps == 0 or np.isnan(fps):
        fps = 50.0  
        
    all_keypoints = np.zeros((total_frames, TOTAL_LANDMARKS, 3), dtype=np.float32)
    
    # 1. Configure the Holistic Landmarker Task
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    
    # 2. Run the processing pipeline
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for frame_idx in tqdm(range(total_frames), desc=f"Extracting {os.path.basename(video_path)}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_data = []
            
            # A. Body Pose
            if result.pose_landmarks:
                pose = result.pose_landmarks
                # Dynamic check: If it's a nested list of people, grab the first person
                if not hasattr(pose[0], 'x'):
                    pose = pose[0]
                
                for i in range(BODY_LANDMARKS_KEPT):
                    if i < len(pose):
                        lm = pose[i]
                        frame_data.append([lm.x, lm.y, getattr(lm, 'visibility', 1.0)])
                    else:
                        frame_data.append([np.nan, np.nan, 0.0])
            else:
                frame_data.extend([[np.nan, np.nan, 0.0]] * BODY_LANDMARKS_KEPT)
                
            # B. Left Hand
            if result.left_hand_landmarks:
                left_hand = result.left_hand_landmarks
                if not hasattr(left_hand[0], 'x'):
                    left_hand = left_hand[0]
                
                for i in range(21):
                    if i < len(left_hand):
                        lm = left_hand[i]
                        frame_data.append([lm.x, lm.y, 1.0])
                    else:
                        frame_data.append([np.nan, np.nan, 0.0])
            else:
                frame_data.extend([[np.nan, np.nan, 0.0]] * 21)
                
            # C. Right Hand
            if result.right_hand_landmarks:
                right_hand = result.right_hand_landmarks
                if not hasattr(right_hand[0], 'x'):
                    right_hand = right_hand[0]
                
                for i in range(21):
                    if i < len(right_hand):
                        lm = right_hand[i]
                        frame_data.append([lm.x, lm.y, 1.0])
                    else:
                        frame_data.append([np.nan, np.nan, 0.0])
            else:
                frame_data.extend([[np.nan, np.nan, 0.0]] * 21)
                
            all_keypoints[frame_idx] = np.array(frame_data)
            
    cap.release()
    
    # -- Interpolate missing frames (NaNs) --
    orig_shape = all_keypoints.shape
    flattened = all_keypoints.reshape(orig_shape[0], -1)
    df = pd.DataFrame(flattened)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(0, inplace=True) 
    interpolated_keypoints = df.values.reshape(orig_shape)
    
    # -- Normalize --
    final_keypoints = normalize_skeleton(interpolated_keypoints)
    return final_keypoints