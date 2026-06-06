import torch
import torch.nn as nn

class SpatialGraphConv(nn.Module):
    """Passes spatial messages along the physical bones of the skeleton."""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        # Store the adjacency matrix in the model (non-trainable parameter)
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        
        # 1x1 Convolution maps node channels to a higher dimension
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Channels, Frames, Vertices)
        x = self.conv(x) 
        
        # Matrix multiplication using Einstein Summation
        # Multiplies the spatial dimension (v) by the Adjacency Matrix
        # to pull data from neighboring joints (w).
        x = torch.einsum('n c t v, v w -> n c t w', x, self.A)
        return x.contiguous()

class STGCNBlock(nn.Module):
    """Combines Spatial Graph Conv with Temporal Convolution."""
    def __init__(self, in_channels, out_channels, A, temporal_kernel=9, residual=True):
        super().__init__()
        
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        
        # Temporal Convolution acts across the 'Frames' dimension
        padding = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection matches channel dimensions if they change
        if not residual or in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)