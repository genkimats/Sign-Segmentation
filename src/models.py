import torch
import torch.nn as nn
from mamba_ssm import Mamba

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