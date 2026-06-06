import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SignSegmentationDataset(Dataset):
    def __init__(self, keypoints_dir, labels_dir, window_size=1000, overlap=200, mode='train'):
        """
        Args:
            keypoints_dir: Path to processed_data/keypoints
            labels_dir: Path to processed_data/BIO_tags
            window_size: Number of frames per slice
            overlap: Number of frames to overlap between slices
            mode: 'train', 'val', or 'test' (Useful later for data splitting)
        """
        self.keypoints_dir = keypoints_dir
        self.labels_dir = labels_dir
        self.window_size = window_size
        self.step_size = window_size - overlap
        
        # We will build an index of all possible slices across all files
        self.slice_index = []
        
        self._build_index()

    def _build_index(self):
        """Scans all files and pre-calculates the start/end frames for every slice."""
        files = [f for f in os.listdir(self.keypoints_dir) if f.endswith('.npy')]
        
        for file_name in files:
            base_name = file_name.replace('.npy', '')
            kp_path = os.path.join(self.keypoints_dir, file_name)
            
            # Load just the shape of the array using mmap_mode to save RAM
            # This allows us to know the length of the video without loading the 1GB file
            kp_shape = np.load(kp_path, mmap_mode='r').shape
            total_frames = kp_shape[0]
            
            # Calculate all valid slice windows for this file
            for start in range(0, total_frames - self.window_size + 1, self.step_size):
                end = start + self.window_size
                self.slice_index.append({
                    'base_name': base_name,
                    'start': start,
                    'end': end
                })

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        # 1. Get the pre-calculated slice metadata
        slice_info = self.slice_index[idx]
        base_name = slice_info['base_name']
        start = slice_info['start']
        end = slice_info['end']
        
        # 2. Load only the specific slice from the disk arrays
        kp_path = os.path.join(self.keypoints_dir, f"{base_name}.npy")
        label_path = os.path.join(self.labels_dir, f"{base_name}.npy")
        
        # Use mmap_mode to efficiently slice directly from disk
        kp_array = np.load(kp_path, mmap_mode='c')[start:end]
        label_array = np.load(label_path, mmap_mode='c')[start:end]
        
        # 3. Format tensors for PyTorch and ST-GCN
        # ST-GCN typically expects inputs in the shape: (Channels, Frames, Vertices)
        # Your data is currently saved as: (Frames, Vertices, Channels) -> (1000, 67, 3)
        keypoints_tensor = torch.tensor(kp_array, dtype=torch.float32)
        keypoints_tensor = keypoints_tensor.permute(2, 0, 1) # Now (3, 1000, 67)
        
        labels_tensor = torch.tensor(label_array, dtype=torch.long)
        
        return keypoints_tensor, labels_tensor

# --- Quick Test to verify it works ---
if __name__ == "__main__":
    import time
    
    # Adjust these paths to point to your processed_data directories
    KEYPOINTS_DIR = "../processed_data/keypoints"
    LABELS_DIR = "../processed_data/BIO_tags"
    
    print("Building Dataset Index...")
    start_time = time.time()
    
    dataset = SignSegmentationDataset(KEYPOINTS_DIR, LABELS_DIR, window_size=1000, overlap=200)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    print(f"Index built in {time.time() - start_time:.2f} seconds.")
    print(f"Total overlapping slices available: {len(dataset)}")
    
    # Test one batch
    for batch_idx, (features, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"Feature Shape: {features.shape}") # Should be [4, 3, 1000, 67]
        print(f"Target Shape: {targets.shape}")   # Should be [4, 1000]
        break # Just run one batch to test