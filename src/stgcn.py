import torch
import torch.nn as nn
from src.graph import SkeletonGraph

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x) 
        x = torch.einsum('n c t v, v w -> n c t w', x, self.A)
        return x.contiguous()

class STGCNBlock(nn.Module):
    """Standard ST-GCN: Processes the full graph with shared filters."""
    def __init__(self, in_channels, out_channels, A, temporal_kernel=9, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        padding = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual or in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)

class DecoupledSTGCNBlock(nn.Module):
    """
    Splits the 65-vertex graph into Body (23), LH (21), and RH (21),
    processes them through independent GCN filters, and concatenates them back.
    """
    def __init__(self, in_channels, out_channels, temporal_kernel=9):
        super().__init__()
        
        # Get the sub-graphs
        graph = SkeletonGraph(num_vertices=65)
        self.body_indices = graph.body_indices
        self.lh_indices = graph.lh_indices
        self.rh_indices = graph.rh_indices
        
        # 1. Independent Spatial Graph Convolutions
        # Because we decoupled them, we can allocate different channel widths if we want, 
        # but we keep them equal here for simplicity.
        self.gcn_body = SpatialGraphConv(in_channels, out_channels, graph.A_body)
        self.gcn_lh = SpatialGraphConv(in_channels, out_channels, graph.A_lh)
        self.gcn_rh = SpatialGraphConv(in_channels, out_channels, graph.A_rh)
        
        # 2. Shared Temporal Convolution (Runs across the time dimension after fusion)
        padding = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        
        # Extract specific anatomical tensors: Shape -> (B, C, T, Num_Nodes)
        x_body = x[:, :, :, self.body_indices]
        x_lh = x[:, :, :, self.lh_indices]
        x_rh = x[:, :, :, self.rh_indices]
        
        # Process them with their specialized filters
        out_body = self.gcn_body(x_body)
        out_lh = self.gcn_lh(x_lh)
        out_rh = self.gcn_rh(x_rh)
        
        # Fuse them back together along the vertex dimension
        # Resulting shape goes back to (B, out_channels, T, 65)
        x_fused = torch.cat([out_body, out_lh, out_rh], dim=3)
        
        # Apply temporal convolution to the fused graph
        x_fused = self.tcn(x_fused)
        
        return self.relu(x_fused + res)