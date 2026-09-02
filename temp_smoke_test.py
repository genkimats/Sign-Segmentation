from src.dataset import SignSegmentationDataset
from src.models import STGCN_Mamba
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds = SignSegmentationDataset(
    keypoints_dir="processed_data/keypoints", labels_dir="processed_data/BIO_tags",
    window_size=16, use_hamer_features=True
)
features, labels, hamer, vid, s, e = ds[0]
print(features.shape, hamer.shape)  # (C, 16, 65), (288, 16)

model = STGCN_Mamba(num_vertices=65, in_channels=features.shape[0], hamer_dim=288).to(device)
logits, emb = model(features.unsqueeze(0).to(device), hamer=hamer.unsqueeze(0).to(device))
print(logits.shape)  # expect (1, 3, 16)