import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock
from src.stgcn import DecoupledSTGCNBlock
from mamba_ssm import Mamba


class Decoupled_STGCN_Mamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        
        # 1. Decoupled Spatial Encoder
        self.stgcn_blocks = nn.Sequential(
            DecoupledSTGCNBlock(in_channels, stgcn_channels),
            DecoupledSTGCNBlock(stgcn_channels, stgcn_channels)
        )
        
        # 2. Bridge (THE FIX: LayerNorm and GELU restored)
        self.bridge_dim = num_vertices * stgcn_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model), # Critical for Mamba stability
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 3. Standard Causal Mamba
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """ x shape expected: (B, 3, T, 65) """
        x = self.stgcn_blocks(x) 
        
        # Bridge to Sequence
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), x.size(1), -1) 
        x = self.feature_proj(x) 
        
        # Mamba Sequence Parsing
        for layer in self.mamba_layers:
            x = layer(x)
            
        embeddings = x.permute(0, 2, 1) # Extract for BCL
        logits = self.classifier(x).permute(0, 2, 1) # Standard Classification
        
        return logits, embeddings

class Decoupled_STGCN_BiMamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        
        # 1. Decoupled Spatial Encoder
        self.stgcn_blocks = nn.Sequential(
            DecoupledSTGCNBlock(in_channels, stgcn_channels),
            DecoupledSTGCNBlock(stgcn_channels, stgcn_channels)
        )
        
        # 2. Bridge
        self.bridge_dim = num_vertices * stgcn_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 3. Bi-Mamba Temporal Encoder
        self.fwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.bwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        
        # 4. Fusion and Classification
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """ x shape expected: (B, 3, T, 65) """
        B, C, T, V = x.shape
        
        # Pass through the Anatomically Decoupled GCN
        x = self.stgcn_blocks(x) # (B, 64, T, 65)
        
        # Bridge to Sequence
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1) 
        x = self.feature_proj(x) # (B, T, 256)
        
        # Bi-Mamba Sweeps
        fwd_out = x
        for layer in self.fwd_mamba:
            fwd_out = layer(fwd_out)
            
        bwd_out = torch.flip(x, dims=[1])
        for layer in self.bwd_mamba:
            bwd_out = layer(bwd_out)
        bwd_out = torch.flip(bwd_out, dims=[1])
        
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        x = self.fusion(combined)
        
        # Extract embeddings for Boundary Contrastive Loss
        embeddings = x.permute(0, 2, 1) # (B, 256, T)
        
        logits = self.classifier(x)
        logits = logits.permute(0, 2, 1) # (B, 3, T)
        
        return logits, embeddings

class STGCN_BiMamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        
        # 1. Get the physical skeleton graph
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        
        # 2. ST-GCN Front-End (Spatial Encoder)
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        
        # 3. Graph-to-Sequence Bridge
        self.bridge_dim = num_vertices * stgcn_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1) # Prevent overfitting on spatial features
        )
        
        # 4. Bi-Mamba Back-End (Temporal Encoder)
        self.fwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.bwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        
        # 5. Fusion and Classifier
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """ x shape expected: (Batch, Channels, Frames, Vertices) -> (B, 3, 1000, 65) """
        B, C, T, V = x.shape
        
        # Phase 1: Spatial-Temporal Graph Parsing
        x = self.stgcn_blocks(x) # Shape: (B, 64, 1000, 65)
        
        # Phase 2: Bridge Formatting
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1) 
        x = self.feature_proj(x) # Shape: (B, 1000, 256)
        
        # Phase 3: Bidirectional Mamba Sweeps
        # Forward sweep
        fwd_out = x
        for layer in self.fwd_mamba:
            fwd_out = layer(fwd_out)
            
        # Backward sweep (flip time dimension, process, flip back)
        bwd_out = torch.flip(x, dims=[1])
        for layer in self.bwd_mamba:
            bwd_out = layer(bwd_out)
        bwd_out = torch.flip(bwd_out, dims=[1])
        
        # Phase 4: Fusion & Classification
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        x = self.fusion(combined)
        
        # --- NEW: Extract Latent Embeddings before Classification ---
        embeddings = x.permute(0, 2, 1) # Shape: (B, 256, 1000)
        
        # Phase 4: Classification
        logits = self.classifier(x)
        logits = logits.permute(0, 2, 1) # Shape: (B, 3, 1000)
        
        # Return BOTH for the combined loss function
        return logits, embeddings

class STGCN_Mamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        
        # 1. Get the physical skeleton graph
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        
        # 2. ST-GCN Front-End (Spatial Encoder)
        # Maps 3 (x,y,z) channels to stgcn_channels (e.g., 64) through structural context
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        
        # 3. Graph-to-Sequence Bridge
        # We flatten the (Vertices * Channels) into a 1D vector per frame
        self.bridge_dim = num_vertices * stgcn_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1) # Added slight dropout to prevent ST-GCN overfitting
        )
        
        # 4. Mamba Back-End (Temporal Encoder)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """ x shape expected: (Batch, Channels, Frames, Vertices) -> (B, 3, 1000, 65) """
        B, C, T, V = x.shape
        
        # Phase 1: Spatial-Temporal Graph Parsing
        x = self.stgcn_blocks(x) # Shape becomes (B, 64, 1000, 65)
        
        # Phase 2: Bridge Formatting
        # Permute to (B, T, V, C_new) -> (B, 1000, 65, 64)
        x = x.permute(0, 2, 3, 1).contiguous()
        # Flatten spatial info -> (B, 1000, 65 * 64)
        x = x.view(B, T, -1) 
        
        # Project down to d_model -> (B, 1000, 256)
        x = self.feature_proj(x)
        
        # Phase 3: Mamba Sequence Parsing
        for layer in self.mamba_layers:
            x = layer(x)
            
        # --- NEW: Extract Latent Embeddings before Classification ---
        embeddings = x.permute(0, 2, 1) # Shape: (B, 256, 1000)
        
        # Phase 4: Classification
        logits = self.classifier(x)
        logits = logits.permute(0, 2, 1) # Shape: (B, 3, 1000)
        
        # Return BOTH for the combined loss function
        return logits, embeddings

class BiMambaBaseline(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        
        self.input_dim = num_vertices * in_channels
        self.d_model = d_model
        
        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # We need two Mamba streams
        self.fwd_mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.bwd_mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        
        # Fusion layer to combine forward and backward features
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, V * C)
        x = self.feature_proj(x)
        
        # Forward pass
        fwd_out = x
        for layer in self.fwd_mamba:
            fwd_out = layer(fwd_out)
            
        # Backward pass
        # Flip the temporal dimension (dim 1)
        bwd_out = torch.flip(x, dims=[1])
        for layer in self.bwd_mamba:
            bwd_out = layer(bwd_out)
        # Flip back to original order
        bwd_out = torch.flip(bwd_out, dims=[1])
        
        # Fusion: Concat and project
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        x = self.fusion(combined)
        
        logits = self.classifier(x)
        return logits.permute(0, 2, 1) # (B, 3, T)

class PureMambaBaseline(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        
        # 1. Feature Projection
        # Flatten the 65 vertices and 3 channels (65 * 3 = 195) into a single vector
        # Then project it up to a rich hidden dimension (d_model)
        self.input_dim = num_vertices * in_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 2. The Mamba Sequence Backbone
        # Stack multiple Mamba blocks to learn the temporal dynamics over the 1000 frames
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension
                d_state=16,      # SSM state expansion factor (Standard is 16)
                d_conv=4,        # Local convolution width
                expand=2,        # Block expansion factor
            ) for _ in range(n_layers)
        ])
        
        # 3. BIO Tag Classifier
        # Maps the d_model features back down to our 3 classes (0: Out, 1: In, 2: Begin)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: Input tensor from DataLoader of shape (Batch, Channels, Frames, Vertices)
           Example: (4, 3, 1000, 65)
        """
        B, C, T, V = x.shape
        
        # Reshape to (Batch, Frames, Vertices, Channels) -> (4, 1000, 65, 3)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Flatten the spatial dimensions: (Batch, Frames, V * C) -> (4, 1000, 195)
        x = x.view(B, T, V * C)
        
        # Project up to d_model: (4, 1000, 256)
        x = self.feature_proj(x)
        
        # Pass through Mamba layers
        for layer in self.mamba_layers:
            # Mamba natively processes (Batch, SequenceLength, d_model)
            x = layer(x) 
            
        # Classify each frame: (4, 1000, 3)
        logits = self.classifier(x)
        
        # PyTorch CrossEntropy/Focal Loss expects shape (Batch, Classes, SequenceLength)
        # So we permute before returning -> (4, 3, 1000)
        return logits.permute(0, 2, 1)

# --- Quick Test ---
if __name__ == "__main__":
    # Simulate a batch from your DataLoader (Batch=4, Channels=3, Frames=1000, Vertices=65)
    dummy_input = torch.randn(4, 3, 1000, 65).cuda() 
    
    # Initialize the model and move to GPU
    model = PureMambaBaseline(num_vertices=65, d_model=256, n_layers=4).cuda()
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") 
    # Output should be [4, 3, 1000] -> Ready for Loss Calculation!