import math
import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock
from src.stgcn import DecoupledSTGCNBlock


class STGCN_MLP_Mamba(nn.Module):
    """
    ST-GCN + MLP Bridge + Mamba Architecture.
    Expands the flattened spatial graph using a Multi-Layer Perceptron (MLP) 
    before compressing it down to d_model for the Mamba sequence parser.
    """
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, mlp_expansion_factor=4, dropout=0.2):
        super().__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        
        self.bridge_dim = num_vertices * stgcn_channels
        hidden_dim = d_model * mlp_expansion_factor
        
        # --- NEW: MLP Bridge ---
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1) 
        
        x = self.feature_proj(x + 1e-5) 
        
        for layer in self.mamba_layers:
            x = layer(x)
            
        embeddings = x.permute(0, 2, 1) 
        logits = self.classifier(x)
        logits = logits.permute(0, 2, 1) 
        return logits, embeddings
    

# ==============================================================================
# TRANSFORMER UTILITIES
# ==============================================================================
class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens 
    in the sequence. Required for pure Transformers since they have no inherent 
    sense of time/order.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x shape: (Batch, Sequence Length, d_model) """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# ARCHITECTURES
# ==============================================================================

class TransformerBaseline(nn.Module):
    """
    A pure Multi-Head Self-Attention model. 
    Flattens the spatial graph entirely and relies purely on standard Transformer 
    Encoders and Positional Encodings to map temporal dependencies.
    """
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, nhead=8, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        
        self.feature_dim = in_channels * num_vertices
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V) 
        features = self.projection(x + 1e-5) # Epsilon addition prevents NaN
        
        features = self.pos_encoder(features)
        
        # Raw, un-checkpointed Global Self-Attention
        embeddings = self.transformer_encoder(features) 
        
        logits = self.classifier(embeddings)            
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_Transformer(nn.Module):
    """
    Hybrid Architecture: 
    Extracts isolated spatial kinetics using a Graph Convolutional Network, 
    then applies global temporal attention using a Transformer Encoder.
    """
    def __init__(self, in_channels, num_vertices, num_classes=3, stgcn_channels=64, d_model=256, n_layers=4, nhead=8, dim_feedforward=1024, dropout=0.2, hamer_dim=None, hamer_proj_dim=64):
        super().__init__()
        
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        
        self.bridge_dim = num_vertices * stgcn_channels

        # Optional separate HaMeR branch -- see STGCN_Mamba's comment for why this is
        # fused here (before feature_proj) rather than folded into the graph.
        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None):
        B, C, T, V = x.shape
        
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)

        features = self.feature_proj(x + 1e-5) 
        
        features = self.pos_encoder(features)
        
        # Raw, un-checkpointed Global Self-Attention
        embeddings = self.transformer_encoder(features)
        
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_Mamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, hamer_dim=None, hamer_proj_dim=64):
        super().__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        self.bridge_dim = num_vertices * stgcn_channels

        # Optional separate HaMeR branch (2025 Hands-On paper's design: HaMeR gets its
        # own MLP, fused with the graph-based stream BEFORE the shared temporal
        # backbone -- not folded into the per-vertex graph itself, since HaMeR's MANO
        # rotation parameters aren't per-vertex 3D coordinates the graph conv expects).
        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.bridge_dim += hamer_proj_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1) 
        )
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None):
        B, C, T, V = x.shape
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1) 

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)

        x = self.feature_proj(x + 1e-5) 
        
        # Raw, sequential state-space memory calculation
        for layer in self.mamba_layers:
            x = layer(x)
            
        embeddings = x.permute(0, 2, 1) 
        logits = self.classifier(x)
        logits = logits.permute(0, 2, 1) 
        return logits, embeddings


class Decoupled_STGCN_Mamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        self.stgcn_blocks = nn.Sequential(
            DecoupledSTGCNBlock(in_channels, stgcn_channels, num_vertices=num_vertices),
            DecoupledSTGCNBlock(stgcn_channels, stgcn_channels, num_vertices=num_vertices)
        )
        self.bridge_dim = num_vertices * stgcn_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model), 
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), x.size(1), -1) 
        x = self.feature_proj(x + 1e-5)
        
        embeddings = x
        for layer in self.mamba_layers:
            embeddings = layer(embeddings)
            
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_BiMamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3):
        super().__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        self.bridge_dim = num_vertices * stgcn_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.mamba_fwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.mamba_bwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), x.size(1), -1) 
        x = self.feature_proj(x + 1e-5)
        
        fwd_emb = x
        bwd_emb = torch.flip(x, dims=[1]) 
        
        for fwd_layer, bwd_layer in zip(self.mamba_fwd, self.mamba_bwd):
            fwd_emb = fwd_layer(fwd_emb)
            bwd_emb = bwd_layer(bwd_emb)
            
        bwd_emb = torch.flip(bwd_emb, dims=[1])
        embeddings = torch.cat([fwd_emb, bwd_emb], dim=-1)
        
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_BiLSTM(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64):
        super(STGCN_BiLSTM, self).__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        self.bridge_dim = num_vertices * stgcn_channels

        # Optional separate HaMeR branch -- see STGCN_Mamba's comment for why this is
        # fused here (before the LSTM) rather than folded into the graph.
        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        # Matches the 2023 paper's exact spec: "flattened and projected into a
        # standard dimension (256), then fed through an LSTM encoder" -- project
        # to d_model, NOT d_model*2 (that was doubling the LSTM's actual input
        # size relative to what the paper describes and what their own
        # hyperparameter sweep found optimal for this hidden size).
        self.projection = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x, hamer=None):
        B, C, T, V = x.shape
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = x.view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)

        features = self.projection(x + 1e-5)      
        lstm_out, _ = self.lstm(features)      
        logits = self.classifier(lstm_out)     
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)


class BiLSTM_Baseline(nn.Module):
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2):
        super(BiLSTM_Baseline, self).__init__()
        self.feature_dim = in_channels * num_vertices
        # Matches the 2023 paper's spec: project to d_model (256), not d_model*2 --
        # see STGCN_BiLSTM's comment for the full reasoning.
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)
        features = self.projection(x + 1e-5) 
        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out) 
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)


class PureMambaBaseline(nn.Module):
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2):
        super().__init__()
        self.feature_dim = in_channels * num_vertices
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)
        features = self.projection(x + 1e-5)
        
        embeddings = features
        for layer in self.mamba_layers:
            embeddings = layer(embeddings)
            
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class BiMambaBaseline(nn.Module):
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2):
        super().__init__()
        self.feature_dim = in_channels * num_vertices
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.bwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)
        features = self.projection(x + 1e-5) 
        
        fwd_emb = features
        bwd_emb = torch.flip(features, dims=[1])
        
        for fwd_layer, bwd_layer in zip(self.fwd_mamba, self.bwd_mamba):
            fwd_emb = fwd_layer(fwd_emb)
            bwd_emb = bwd_layer(bwd_emb)
            
        bwd_emb = torch.flip(bwd_emb, dims=[1])
        combined = torch.cat([fwd_emb, bwd_emb], dim=-1)
        
        embeddings = self.fusion(combined)
        logits = self.classifier(embeddings)
        
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)

class Latent_STGCN_Mamba(nn.Module):
    """
    Discriminative adaptation of 'Sign-Mamba' Latent Space Extractor.
    Compresses the spatial graph into a dense continuous latent space, 
    uses a dedicated Mamba block to extract temporal latent dynamics, 
    then up-projects to the main sequence modeler.
    """
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64):
        super().__init__()
        # 1. Spatial Graph Encoder
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        
        flat_dim = num_vertices * stgcn_channels

        # Optional separate HaMeR branch -- see STGCN_Mamba's comment for why this is
        # fused here (before the latent bottleneck) rather than folded into the graph.
        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            flat_dim += hamer_proj_dim
        
        # 2. Continuous Latent Space Bottleneck (Encoder)
        self.latent_encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Latent Mamba Extractor (Smooths the latent space dynamically)
        self.latent_mamba = Mamba(d_model=latent_dim, d_state=16, d_conv=4, expand=2)
        
        # 4. Up-Projection to Main Sequence Dimension
        self.latent_to_main = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 5. Main Temporal Sequence Modeler (BiMamba Backend)
        self.fwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.bwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None):
        B, C, T, V = x.shape
        
        # Spatial Graph Processing
        x = self.stgcn_blocks(x)                   # (B, 64, T, 65)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1) # Flatten to (B, T, 4160)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)
        
        # Transform into Latent Space
        z = self.latent_encoder(x)                 # (B, T, 128)
        
        # Extract Temporal Latent Dynamics (with residual connection)
        z_smooth = self.latent_mamba(z) + z        # (B, T, 128)
        
        # Project to Deep Sequence Modeler
        features = self.latent_to_main(z_smooth)   # (B, T, 256)
        
        fwd_emb = features
        bwd_emb = torch.flip(features, dims=[1])
        
        for f_layer, b_layer in zip(self.fwd_mamba, self.bwd_mamba):
            fwd_emb = f_layer(fwd_emb)
            bwd_emb = b_layer(bwd_emb)
            
        bwd_emb = torch.flip(bwd_emb, dims=[1])
        merged = torch.cat([fwd_emb, bwd_emb], dim=-1)
        fused = self.fusion(merged)
        
        logits = self.classifier(fused)
        
        # Return Logits -> (B, Classes, T) and final embeddings -> (B, T, d_model)
        return logits.permute(0, 2, 1), fused

    # ==============================================================================
# 🧩 MODERN SPATIAL EXTRACTOR BLOCKS
# ==============================================================================

class CTRGCNBlock(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution.
    Learns a dynamic adjacency matrix specific to each feature channel.
    """
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.V = A.shape[-1]
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32) + 1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        B, C, T, V = x.shape
        
        # 1. Base Physical Graph
        x_base = torch.einsum('bctv,vw->bctw', x, self.A)
        x_base = self.conv1(x_base)

        # 2. Dynamic Channel-Wise Topology
        x1 = self.conv1(x).mean(dim=2)  # (B, Cout, V)
        x2 = self.conv2(x).mean(dim=2)  # (B, Cout, V)
        dynamic_A = torch.einsum('bcv,bcw->bcvw', x1, x2)
        dynamic_A = torch.softmax(dynamic_A, dim=-1)
        x_dyn = torch.einsum('bctv,bcvw->bctw', self.conv2(x), dynamic_A)

        # 3. Fusion
        out = self.bn(x_base + self.alpha * x_dyn)
        return self.relu(out)


class InfoGCNBlock(nn.Module):
    """
    InfoGCN: Multi-Scale Attention Topology.
    Learns an additive attention-based structural prior.
    """
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.V = A.shape[-1]
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.spatial_attention = nn.Parameter(torch.ones(self.V, self.V) / self.V)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        # Combines the fixed physical bones with data-driven spatial attention
        A_dynamic = self.A + self.spatial_attention
        x_gcn = torch.einsum('bctv,vw->bctw', x, A_dynamic)
        return self.relu(self.bn(self.conv(x_gcn)))


class ShiftGCNBlock(nn.Module):
    """
    ShiftGCN: Eliminates the adjacency matrix entirely.
    Uses receptive field shifting (via 1D Conv) across the spatial dimension.
    """
    def __init__(self, in_channels, out_channels, num_vertices):
        super().__init__()
        # Groups=in_channels mathematically forces a perfect spatial shift
        self.shift = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        B, C, T, V = x.shape
        x_shift = x.permute(0, 2, 1, 3).reshape(B*T, C, V)
        x_shift = self.shift(x_shift)
        x_shift = x_shift.reshape(B, T, C, V).permute(0, 2, 1, 3)
        return self.relu(self.bn(self.conv(x_shift)))


class SpatialTransformerBlock(nn.Module):
    """
    Treats vertices as tokens and computes global Self-Attention across the body.
    """
    def __init__(self, in_channels, out_channels, num_vertices, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.pos_emb = nn.Parameter(torch.randn(1, num_vertices, out_channels))
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=heads, dim_feedforward=out_channels*2, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x = self.proj(x) + self.pos_emb
        
        # Merge Batch and Time to process each frame's spatial graph independently
        x = x.reshape(B*T, V, -1)
        x = self.transformer(x)
        
        # Return to standard shape
        x = x.reshape(B, T, V, -1).permute(0, 3, 1, 2)  # (B, Cout, T, V)
        return x


# ==============================================================================
# 🚀 MAMBA MODEL WRAPPERS
# ==============================================================================
class Base_Latent_Mamba_Wrapper(nn.Module):
    """
    Base shell for all the models to compress the spatial topology into Mamba.
    """
    def __init__(self, num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim=None, hamer_proj_dim=64):
        super().__init__()
        flat_dim = stgcn_channels * num_vertices

        # Optional separate HaMeR branch -- see STGCN_Mamba's comment for why this is
        # fused here (before the latent bottleneck) rather than folded into the graph.
        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            flat_dim += hamer_proj_dim

        self.latent_encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.latent_mamba = Mamba(d_model=latent_dim, d_state=16, d_conv=4, expand=2)
        self.latent_to_main = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.fwd_mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.bwd_mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None):
        B, C, T, V = x.shape
        # Spatial Processing (To be defined by subclasses)
        x = self.spatial_blocks(x)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)
        
        z = self.latent_encoder(x)
        z_smooth = self.latent_mamba(z) + z
        features = self.latent_to_main(z_smooth)
        
        fwd_emb = features
        bwd_emb = torch.flip(features, dims=[1])
        
        for f_layer, b_layer in zip(self.fwd_mamba, self.bwd_mamba):
            fwd_emb = f_layer(fwd_emb)
            bwd_emb = b_layer(bwd_emb)
            
        bwd_emb = torch.flip(bwd_emb, dims=[1])
        fused = self.fusion(torch.cat([fwd_emb, bwd_emb], dim=-1))
        return self.classifier(fused).permute(0, 2, 1), fused


class CTRGCN_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim)
        A = SkeletonGraph(num_vertices=num_vertices).A
        self.spatial_blocks = nn.Sequential(
            CTRGCNBlock(in_channels, stgcn_channels, A),
            CTRGCNBlock(stgcn_channels, stgcn_channels, A)
        )

class InfoGCN_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim)
        A = SkeletonGraph(num_vertices=num_vertices).A
        self.spatial_blocks = nn.Sequential(
            InfoGCNBlock(in_channels, stgcn_channels, A),
            InfoGCNBlock(stgcn_channels, stgcn_channels, A)
        )

class ShiftGCN_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim)
        self.spatial_blocks = nn.Sequential(
            ShiftGCNBlock(in_channels, stgcn_channels, num_vertices),
            ShiftGCNBlock(stgcn_channels, stgcn_channels, num_vertices)
        )

class SpatialTransformer_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim)
        self.spatial_blocks = nn.Sequential(
            SpatialTransformerBlock(in_channels, stgcn_channels, num_vertices),
            SpatialTransformerBlock(stgcn_channels, stgcn_channels, num_vertices)
        )


# ==============================================================================
# 🆕 HD-GCN: hierarchical (multi-hop) graph decomposition + attention aggregation
# ==============================================================================
class HDGCNBlock(nn.Module):
    """
    Core idea from HD-GCN (Lee et al., ICCV 2023): decompose the graph into
    multiple hop-DISTANCE levels (1-hop = direct neighbors, 2-hop, 3-hop, ...),
    each with its own dedicated graph convolution, then combine the levels
    with a learned, per-sample ATTENTION weighting -- "highlight the dominant
    hierarchical edge sets" (the paper's Attention-Guided Hierarchy
    Aggregation / A-HA module).

    NOT implemented (simplified out of scope): the paper's S-EdgeConv
    sample-wise key-relationship extraction, RSAP center-of-mass pooling, and
    6-way joint/bone/motion ensemble. This captures the hierarchical-
    decomposition + attention-aggregation core, not the full benchmark
    pipeline -- describe as "HD-GCN-inspired" in any writeup, not a full
    reproduction.
    """
    def __init__(self, in_channels, out_channels, hop_adjacencies):
        super().__init__()
        self.num_levels = len(hop_adjacencies)
        self.A_levels = nn.ParameterList([
            nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
            for A in hop_adjacencies
        ])
        self.level_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(self.num_levels)
        ])
        # Simplified Attention-Guided Hierarchy Aggregation: score each level's
        # (globally-pooled) output, softmax across levels, weighted-sum combine.
        self.level_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        # x: (B, C, T, V)
        B = x.shape[0]
        level_outputs = []
        level_scores = []
        for A_h, conv_h in zip(self.A_levels, self.level_convs):
            x_h = conv_h(x)                                  # (B, Cout, T, V)
            x_h = torch.einsum('bctv,vw->bctw', x_h, A_h)     # aggregate over THIS hop level
            level_outputs.append(x_h)
            level_scores.append(self.level_score(x_h))        # (B, 1)

        scores = torch.softmax(torch.cat(level_scores, dim=1), dim=1)  # (B, num_levels)
        stacked = torch.stack(level_outputs, dim=1)            # (B, num_levels, Cout, T, V)
        weights = scores.view(B, self.num_levels, 1, 1, 1)
        fused = (stacked * weights).sum(dim=1)                 # (B, Cout, T, V)

        return self.relu(self.bn(fused))


class HDGCN_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hd_max_hop=3, hamer_dim=None, hamer_proj_dim=64):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim)
        graph = SkeletonGraph(num_vertices=num_vertices)
        self.spatial_blocks = nn.Sequential(
            HDGCNBlock(in_channels, stgcn_channels, graph.get_hop_adjacencies(max_hop=hd_max_hop)),
            HDGCNBlock(stgcn_channels, stgcn_channels, graph.get_hop_adjacencies(max_hop=hd_max_hop))
        )


# ==============================================================================
# 🆕 HyperSign: pairwise graph + fixed anatomical hyperedges + learned soft hyperedges
# ==============================================================================
class HyperSignBlock(nn.Module):
    """
    Core idea from HyperSign (hierarchical hypergraph co-occurrence modeling
    for sign language): fuse THREE complementary structures over the same
    vertex set, matching the paper's three pathways:

      1. Standard pairwise graph convolution over the existing skeleton graph
         -- "traditional graph convolutions for modeling physical joint
         connections."
      2. FIXED, hand-designed anatomical hyperedges (e.g. "all 5 left-hand
         fingertips", "the lips as a group", from SkeletonGraph.
         get_anatomical_hyperedges()) -- a simplification of the paper's
         k-NN-built "dynamic geometric hypergraphs encoding local spatial
         patterns": explicit, interpretable groups instead of a
         differentiable k-NN construction.
      3. A LEARNABLE "soft hypergraph": P learnable prototype hyperedges,
         each with a softmax-normalized membership weight over all V
         vertices -- a direct analog of the paper's "soft hypergraphs
         generated by learnable prototypes to reveal latent semantic
         associations."

    NOT implemented: the paper's full multi-scale hierarchy and its specific
    co-occurrence loss terms. This captures the three-pathway fusion idea,
    not the complete paper -- describe as "HyperSign-inspired" in any
    writeup, not a full reproduction.
    """
    def __init__(self, in_channels, out_channels, A, hyperedges, num_vertices, num_soft_hyperedges=8):
        super().__init__()
        self.V = num_vertices

        # 1. Standard pairwise graph path
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        self.pair_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 2. Fixed anatomical hyperedges -> incidence matrix (V, E_fixed)
        H_fixed = np.zeros((num_vertices, len(hyperedges)), dtype=np.float32)
        for e, members in enumerate(hyperedges):
            for v in members:
                H_fixed[v, e] = 1.0
        self.register_buffer("H_fixed", torch.tensor(H_fixed, dtype=torch.float32))
        self.fixed_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 3. Learnable soft hyperedges: (V, P) incidence, softmax-normalized per hyperedge
        self.soft_incidence_logits = nn.Parameter(torch.randn(num_vertices, num_soft_hyperedges) * 0.01)
        self.soft_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    @staticmethod
    def _hypergraph_propagate(x, H):
        """
        Standard hypergraph convolution propagation (Feng et al., HGNN 2019):
        vertex -> hyperedge (averaged over hyperedge members) -> vertex
        (averaged over the hyperedges each vertex belongs to).
        x: (B, C, T, V); H: (V, E) incidence matrix (fixed 0/1, or soft).
        """
        deg_e = H.sum(dim=0).clamp(min=1e-6)              # (E,) hyperedge size
        deg_v = H.sum(dim=1).clamp(min=1e-6)               # (V,) vertex's hyperedge count
        H_norm = H / deg_e.unsqueeze(0)                    # normalize by hyperedge size
        msg = torch.einsum('bctv,ve->bcte', x, H_norm)     # (B, C, T, E) vertex -> hyperedge
        out = torch.einsum('bcte,ve->bctv', msg, H)         # (B, C, T, V) hyperedge -> vertex
        out = out / deg_v.view(1, 1, 1, -1)
        return out

    def forward(self, x):
        # 1. Standard pairwise path
        x_pair = self.pair_conv(x)
        x_pair = torch.einsum('bctv,vw->bctw', x_pair, self.A)

        # 2. Fixed anatomical hyperedges
        x_fixed = self.fixed_conv(x)
        x_fixed = self._hypergraph_propagate(x_fixed, self.H_fixed)

        # 3. Learnable soft hyperedges
        H_soft = torch.softmax(self.soft_incidence_logits, dim=0)  # each hyperedge sums to 1 over vertices
        x_soft = self.soft_conv(x)
        x_soft = self._hypergraph_propagate(x_soft, H_soft)

        fused = self.fuse(torch.cat([x_pair, x_fixed, x_soft], dim=1))
        return self.relu(self.bn(fused))


class HyperSign_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, num_soft_hyperedges=8, hamer_dim=None, hamer_proj_dim=64):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim)
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        hyperedges = graph.get_anatomical_hyperedges()
        self.spatial_blocks = nn.Sequential(
            HyperSignBlock(in_channels, stgcn_channels, A, hyperedges, num_vertices, num_soft_hyperedges),
            HyperSignBlock(stgcn_channels, stgcn_channels, A, hyperedges, num_vertices, num_soft_hyperedges)
        )