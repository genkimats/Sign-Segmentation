import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ==============================================================================
# 🧠 LINGUISTICALLY MOTIVATED SUBSET (To save SSD Space & RAM)
# ==============================================================================
# Instead of 468 points, we only extract the boundaries of the eyes, eyebrows, and lips.
# This captures all Non-Manual Markers (NMMs) needed for boundary detection.
LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

SELECTED_INDICES = sorted(list(set(
    LIPS_INDICES + LEFT_EYE_INDICES + RIGHT_EYE_INDICES + LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES
)))
NUM_FACE_VERTICES = len(SELECTED_INDICES)

# ==============================================================================
# ⚙️ EXTRACTION CONFIGURATION
# ==============================================================================
RAW_VIDEO_DIR = "raw_videos" # <-- UPDATE THIS to your actual video folder
OUTPUT_DIR = "processed_data/face_keypoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh

def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return None
        
    frames_data = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # MediaPipe requires RGB images
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            frame_keypoints = np.zeros((NUM_FACE_VERTICES, 3), dtype=np.float32)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                for i, idx in enumerate(SELECTED_INDICES):
                    landmark = face_landmarks.landmark[idx]
                    frame_keypoints[i, 0] = landmark.x
                    frame_keypoints[i, 1] = landmark.y
                    frame_keypoints[i, 2] = landmark.z
            
            frames_data.append(frame_keypoints)
            
    cap.release()
    
    if not frames_data:
        return None
        
    # Convert to Numpy Array: Shape (T, V, C) -> (Frames, ~80 Vertices, 3 Coordinates)
    video_tensor = np.array(frames_data, dtype=np.float32)
    
    # Forward-fill any frames where the face tracker temporarily failed (e.g. signer looked away)
    for i in range(1, len(video_tensor)):
        if np.sum(video_tensor[i]) == 0:
            video_tensor[i] = video_tensor[i-1]
            
    return video_tensor

if __name__ == "__main__":
    video_files = glob.glob(os.path.join(RAW_VIDEO_DIR, "*.mp4")) # Or .avi / .mkv
    
    print(f"🚀 Found {len(video_files)} videos. Starting Face Mesh Extraction...")
    print(f"🎯 Extracting {NUM_FACE_VERTICES} linguistically significant keypoints per frame.")
    
    for vid_path in tqdm(video_files):
        vid_name = os.path.basename(vid_path).split('.')[0]
        out_path = os.path.join(OUTPUT_DIR, f"{vid_name}.npy")
        
        if os.path.exists(out_path):
            continue # Skip if already processed
            
        face_data = extract_faces_from_video(vid_path)
        
        if face_data is not None:
            np.save(out_path, face_data)
        else:
            print(f"⚠️ Warning: No frames extracted for {vid_name}")
            
    print(f"✅ Extraction complete! Files saved to {OUTPUT_DIR}")