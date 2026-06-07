import os
import torch
import numpy as np
from torch.utils.data import Dataset

def apply_label_smoothing(labels_array, window_size=5):
    """
    Converts 1D hard labels into 2D soft probability distributions.
    Applies a Gaussian tolerance window around 'Begin' (Class 2) tags.
    """
    T = labels_array.shape[0]
    num_classes = 3
    soft_labels = np.zeros((T, num_classes), dtype=np.float32)

    # Convert to strict one-hot vectors first
    soft_labels[np.arange(T), labels_array] = 1.0

    if window_size <= 1:
        return soft_labels

    spread = window_size // 2
    # Standard deviation scales with the window spread
    sigma = spread / 2.0 if spread > 0 else 1.0

    # Find all exact 'Begin' frames
    begin_indices = np.where(labels_array == 2)[0]

    for i in begin_indices:
        for d in range(-spread, spread + 1):
            idx = i + d
            if 0 <= idx < T:
                # Calculate Gaussian weight
                weight = np.exp(-(d**2) / (2 * sigma**2))
                
                # If this curve provides a higher 'Begin' probability, apply it
                if weight > soft_labels[idx, 2]:
                    soft_labels[idx, 2] = weight
                    
                    # Proportionally scale down whatever the original class was
                    orig_class = labels_array[idx]
                    if orig_class != 2:
                        soft_labels[idx, orig_class] = 1.0 - weight
                        
    return soft_labels

class SignSegmentationDataset(Dataset):
    def __init__(self, keypoints_dir, labels_dir, window_size=1000, overlap=200, tolerance_window=5):
        self.keypoints_dir = keypoints_dir
        self.labels_dir = labels_dir
        self.window_size = window_size
        self.step_size = window_size - overlap
        self.tolerance_window = tolerance_window
        self.slice_index = []
        
        self._build_index()

    def _build_index(self):
        files = [f for f in os.listdir(self.keypoints_dir) if f.endswith('.npy')]
        for file_name in files:
            base_name = file_name.replace('.npy', '')
            kp_path = os.path.join(self.keypoints_dir, file_name)
            total_frames = np.load(kp_path, mmap_mode='r').shape[0]
            
            for start in range(0, total_frames - self.window_size + 1, self.step_size):
                self.slice_index.append({'base_name': base_name, 'start': start, 'end': start + self.window_size})

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        slice_info = self.slice_index[idx]
        kp_path = os.path.join(self.keypoints_dir, f"{slice_info['base_name']}.npy")
        label_path = os.path.join(self.labels_dir, f"{slice_info['base_name']}.npy")
        
        kp_array = np.load(kp_path, mmap_mode='c')[slice_info['start']:slice_info['end']]
        label_array = np.load(label_path, mmap_mode='c')[slice_info['start']:slice_info['end']]
        
        # Format Keypoints
        keypoints_tensor = torch.tensor(kp_array, dtype=torch.float32).permute(2, 0, 1) 
        
        # --- NEW: Apply Gaussian Tolerance and return Soft Labels ---
        soft_labels = apply_label_smoothing(label_array, self.tolerance_window)
        # Expected shape for PyTorch Cross Entropy is (Classes, SequenceLength)
        labels_tensor = torch.tensor(soft_labels, dtype=torch.float32).permute(1, 0)
        
        return keypoints_tensor, labels_tensor