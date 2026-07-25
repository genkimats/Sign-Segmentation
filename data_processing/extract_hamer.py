import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import torchvision.transforms as transforms

# HaMeR Imports
from hamer.models import load_hamer, DEFAULT_CHECKPOINT

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_VIDEO_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/raw_data/videos")
OUTPUT_FEATURE_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/processed_data/hamer_features")
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standard ImageNet Normalization required by HaMeR's ViT Backbone
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_square_bbox(landmarks, img_w, img_h, scale=1.5):
    """Calculates a square bounding box around the hand with context padding."""
    x_coords = [lm.x * img_w for lm in landmarks.landmark]
    y_coords = [lm.y * img_h for lm in landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    width, height = x_max - x_min, y_max - y_min
    
    # Make it a square and apply scale context
    box_size = max(width, height) * scale
    
    x1 = int(cx - box_size / 2)
    y1 = int(cy - box_size / 2)
    x2 = int(cx + box_size / 2)
    y2 = int(cy + box_size / 2)
    
    # Clamp to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    
    return x1, y1, x2, y2

def process_hand(img, bbox, model, is_right_hand=True):
    """Crops, normalizes, and passes a single hand to HaMeR."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
        
    # 🚨 CRITICAL HaMeR LOGIC: Flip left hands so the model thinks it's a right hand
    if not is_right_hand:
        crop = cv2.flip(crop, 1)
        
    # Resize to HaMeR's expected input size
    crop_resized = cv2.resize(crop, (256, 256))
    
    # Convert to RGB (OpenCV uses BGR) and apply transforms
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    input_tensor = transform(crop_rgb).unsqueeze(0).to(DEVICE)
    
    # --- FIX: Wrap the tensor in a dictionary with the 'img' key ---
    batch = {'img': input_tensor}
    
    with torch.no_grad():
        out = model(batch)
        
    joints = out['pred_keypoints_3d'][0].cpu().numpy()  # (21, 3)
    vertices = out['pred_vertices'][0].cpu().numpy()    # (778, 3)
    
    # 🚨 CRITICAL HaMeR LOGIC: Flip the resulting 3D X-coordinates back for the left hand
    if not is_right_hand:
        joints[:, 0] = -joints[:, 0]
        vertices[:, 0] = -vertices[:, 0]
        
    return {
        "joints": joints,
        "vertices": vertices
    }

# ==============================================================================
# MAIN EXTRACTION LOOP
# ==============================================================================
def main():
    print(f"Loading HaMeR Model on {DEVICE}...")
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
        cap = cv2.VideoCapture(video_path)
        
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_features = {
            "left_hand_joints": [],
            "left_hand_vertices": [],
            "right_hand_joints": [],
            "right_hand_vertices": []
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Initialize empty arrays for this frame (Handles missing hands)
            lh_j, lh_v = np.zeros((21, 3)), np.zeros((778, 3))
            rh_j, rh_v = np.zeros((21, 3)), np.zeros((778, 3))
            
            # Use MediaPipe just to get the Bounding Boxes
            results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine if left or right hand
                    label = results.multi_handedness[idx].classification[0].label
                    is_right = (label == 'Right')
                    
                    # Get Bbox and Process through HaMeR
                    bbox = get_square_bbox(hand_landmarks, img_w, img_h)
                    hamer_out = process_hand(frame, bbox, model, is_right_hand=is_right)
                    
                    if hamer_out:
                        if is_right:
                            rh_j, rh_v = hamer_out["joints"], hamer_out["vertices"]
                        else:
                            lh_j, lh_v = hamer_out["joints"], hamer_out["vertices"]
            
            # Append frame data
            video_features["left_hand_joints"].append(lh_j)
            video_features["left_hand_vertices"].append(lh_v)
            video_features["right_hand_joints"].append(rh_j)
            video_features["right_hand_vertices"].append(rh_v)
            
        cap.release()
        
        # Convert lists to NumPy arrays (Frames, Nodes, 3)
        for k in video_features.keys():
            video_features[k] = np.array(video_features[k], dtype=np.float32)
            
        # Save the feature block
        save_name = video_name.rsplit('.', 1)[0] + "_hamer.pt"
        torch.save(video_features, os.path.join(OUTPUT_FEATURE_DIR, save_name))

if __name__ == "__main__":
    main()
