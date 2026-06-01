import os

# --- Directory Resolution ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Raw Data Paths
RAW_DIR = os.path.join(PARENT_DIR, "raw_data")
VIDEOS_DIR = os.path.join(RAW_DIR, "videos")
ANNOTATIONS_DIR = os.path.join(RAW_DIR, "annotations")

# Processed Data Paths
PROCESSED_DIR = os.path.join(PARENT_DIR, "processed_data")
KEYPOINTS_DIR = os.path.join(PROCESSED_DIR, "keypoints")
LABELS_DIR = os.path.join(PROCESSED_DIR, "BIO_tags")

# --- Hyperparameters ---
FPS = 50                     # Native DGS Corpus framerate (No downsampling)
WINDOW_SIZE = 1000           # 20 seconds per slice (Great for Mamba context)
WINDOW_OVERLAP = 200         # 4 seconds overlap to protect boundary transitions

# --- Landmark Filtering ---
# MediaPipe Pose: 0-24 (Upper body + Hips), 25-32 (Lower body - DROPPED)
BODY_LANDMARKS_KEPT = 23
HAND_LANDMARKS = 21
TOTAL_LANDMARKS = BODY_LANDMARKS_KEPT + (HAND_LANDMARKS * 2) # 67 vertices for ST-GCN