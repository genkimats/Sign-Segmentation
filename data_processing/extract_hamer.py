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
INPUT_VIDEO_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/data/raw_videos")
OUTPUT_FEATURE_DIR = os.path.expanduser("~/Genki_GR/Sign-Segmentation/processed_data/hamer_features")
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🚀 GPU BATCH SIZE
# 32 is highly optimized for an 8GB/16GB RTX 4060 Ti. 
# If it crashes with "CUDA Out of Memory", lower this to 16.
BATCH_SIZE = 32 

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
    
    box_size = max(width, height) * scale
    
    x1 = int(cx - box_size / 2)
    y1 = int(cy - box_size / 2)
    x2 = int(cx + box_size / 2)
    y2 = int(cy + box_size / 2)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    
    return x1, y1, x2, y2

def prepare_hand_tensor(img, bbox, is_right_hand=True):
    """Crops and normalizes the hand without passing it to the model yet."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
        
    if not is_right_hand:
        crop = cv2.flip(crop, 1)
        
    crop_resized = cv2.resize(crop, (256, 256))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    
    # Returns shape: (1, 3, 256, 256)
    return transform(crop_rgb).unsqueeze(0)

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
        
        # Skip if already processed
        save_name = video_name.rsplit('.', 1)[0] + "_hamer.pt"
        if os.path.exists(os.path.join(OUTPUT_FEATURE_DIR, save_name)):
            continue
            
        cap = cv2.VideoCapture(video_path)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Final output arrays for this video
        video_lh_j, video_lh_v = [], []
        video_rh_j, video_rh_v = [], []
        
        # Batch Queues
        hand_queue = []
        meta_queue = []
        
        def flush_queue():
            """Pushes the accumulated queue through the GPU."""
            if not hand_queue:
                return
                
            # Stack into shape: (Batch, 3, 256, 256)
            batch_tensor = torch.cat(hand_queue, dim=0).to(DEVICE)
            
            with torch.no_grad():
                out = model({'img': batch_tensor})
                
            joints = out['pred_keypoints_3d'].cpu().numpy()  # (Batch, 21, 3)
            vertices = out['pred_vertices'].cpu().numpy()    # (Batch, 778, 3)
            
            # Unpack the batch and assign to the correct frame & hand
            for i, (f_idx, is_right) in enumerate(meta_queue):
                j, v = joints[i].copy(), vertices[i].copy()
                
                if not is_right:
                    j[:, 0] = -j[:, 0]
                    v[:, 0] = -v[:, 0]
                    video_lh_j[f_idx] = j
                    video_lh_v[f_idx] = v
                else:
                    video_rh_j[f_idx] = j
                    video_rh_v[f_idx] = v
                    
            hand_queue.clear()
            meta_queue.clear()

        # Read frames sequentially
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Initialize empty defaults for this frame
            video_lh_j.append(np.zeros((21, 3), dtype=np.float32))
            video_lh_v.append(np.zeros((778, 3), dtype=np.float32))
            video_rh_j.append(np.zeros((21, 3), dtype=np.float32))
            video_rh_v.append(np.zeros((778, 3), dtype=np.float32))
            
            results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
                    is_right = (label == 'Right')
                    
                    bbox = get_square_bbox(hand_landmarks, img_w, img_h)
                    tensor = prepare_hand_tensor(frame, bbox, is_right_hand=is_right)
                    
                    if tensor is not None:
                        hand_queue.append(tensor)
                        meta_queue.append((frame_idx, is_right))
                        
            # If queue is full, push to GPU
            if len(hand_queue) >= BATCH_SIZE:
                flush_queue()
                
            frame_idx += 1
            
        cap.release()
        
        # Flush any remaining hands in the queue at the end of the video
        flush_queue()
        
        # Save the finished video
        video_features = {
            "left_hand_joints": np.array(video_lh_j, dtype=np.float32),
            "left_hand_vertices": np.array(video_lh_v, dtype=np.float32),
            "right_hand_joints": np.array(video_rh_j, dtype=np.float32),
            "right_hand_vertices": np.array(video_rh_v, dtype=np.float32)
        }
        
        torch.save(video_features, os.path.join(OUTPUT_FEATURE_DIR, save_name))

if __name__ == "__main__":
    main()