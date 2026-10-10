import math
import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock
from src.stgcn import DecoupledSTGCNBlock
from src.skeleton_angles import skeleton_angle_features, ANGLE_FEATURE_DIM


class STGCN_MLP_Mamba(nn.Module):
    """
    ST-GCN + MLP Bridge + Mamba Architecture.
    Expands the flattened spatial graph using a Multi-Layer Perceptron (MLP) 
    before compressing it down to d_model for the Mamba sequence parser.
    """
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, mlp_expansion_factor=4, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        
        self.bridge_dim = num_vertices * stgcn_channels
        hidden_dim = d_model * mlp_expansion_factor

        # Optional separate HaMeR branch -- see STGCN_Mamba's comment for why this is
        # fused here (before the MLP bridge) rather than folded into the graph.
        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim
        
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
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
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

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)
        
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
    Also used for pure-HaMeR training (input (288, T, 1) -> 288 features/frame).

    Optional DINOv2 side branch (dinov2_dim): own Linear+LayerNorm projection,
    concatenated with the main per-frame features before the shared projection.
    """
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, nhead=8,
                 dim_feedforward=1024, dropout=0.2, dinov2_dim=None, dinov2_proj_dim=128):
        super().__init__()
        
        self.feature_dim = in_channels * num_vertices

        self.dinov2_dim = dinov2_dim
        proj_in_dim = self.feature_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            proj_in_dim += dinov2_proj_dim

        self.projection = nn.Sequential(
            nn.Linear(proj_in_dim, d_model),
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

    def forward(self, x, dinov2=None):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V) 

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                 "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)

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
    def __init__(self, in_channels, num_vertices, num_classes=3, stgcn_channels=64, d_model=256, n_layers=4, nhead=8, dim_feedforward=1024, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, stgcn_proj_dim=None, face_only=False):
        super().__init__()
        
        # face_only=True: input holds ONLY face vertices (dataset face_only mode)
        graph = SkeletonGraph(num_vertices=num_vertices, face_only=face_only)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )

        # Optional graph-feature projection ("stream balancing"): without it, the
        # flattened ST-GCN output (num_vertices * stgcn_channels, e.g. 4160) is
        # concatenated directly with the much smaller HaMeR/DINOv2 branches, so it
        # dominates the shared projection simply by having far more dimensions.
        # With stgcn_proj_dim set, the graph features get their own branch
        # (Linear -> LayerNorm -> GELU -> Dropout, same design as the HaMeR branch)
        # down to stgcn_proj_dim before the concatenation. None = original behaviour
        # (keeps old checkpoints loadable).
        self.stgcn_proj_dim = stgcn_proj_dim
        if stgcn_proj_dim is not None:
            self.stgcn_proj = nn.Sequential(
                nn.Linear(num_vertices * stgcn_channels, stgcn_proj_dim),
                nn.LayerNorm(stgcn_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim = stgcn_proj_dim
        else:
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

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

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

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1)
        if self.stgcn_proj_dim is not None:
            x = self.stgcn_proj(x)  # (B, T, stgcn_proj_dim)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)

        features = self.feature_proj(x + 1e-5) 
        
        features = self.pos_encoder(features)
        
        # Raw, un-checkpointed Global Self-Attention
        embeddings = self.transformer_encoder(features)
        
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_Mamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
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

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.bridge_dim += dinov2_proj_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1) 
        )
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
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

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)

        x = self.feature_proj(x + 1e-5) 
        
        # Raw, sequential state-space memory calculation
        for layer in self.mamba_layers:
            x = layer(x)
            
        embeddings = x.permute(0, 2, 1) 
        logits = self.classifier(x)
        logits = logits.permute(0, 2, 1) 
        return logits, embeddings


class Decoupled_STGCN_Mamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.stgcn_blocks = nn.Sequential(
            DecoupledSTGCNBlock(in_channels, stgcn_channels, num_vertices=num_vertices),
            DecoupledSTGCNBlock(stgcn_channels, stgcn_channels, num_vertices=num_vertices)
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
                nn.Dropout(0.1)
            )
            self.bridge_dim += hamer_proj_dim

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.bridge_dim += dinov2_proj_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model), 
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), x.size(1), -1) 

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)

        x = self.feature_proj(x + 1e-5)
        
        embeddings = x
        for layer in self.mamba_layers:
            embeddings = layer(embeddings)
            
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_BiMamba(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
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
                nn.Dropout(0.1)
            )
            self.bridge_dim += hamer_proj_dim

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.bridge_dim += dinov2_proj_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.mamba_fwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        self.mamba_bwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), x.size(1), -1) 

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)

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
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, stgcn_proj_dim=None, face_only=False):
        super(STGCN_BiLSTM, self).__init__()
        graph = SkeletonGraph(num_vertices=num_vertices, face_only=face_only)  # face_only: face vertices only
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        # Optional graph-feature projection ("stream balancing"): without it, the
        # flattened ST-GCN output (num_vertices * stgcn_channels, e.g. 4160) is
        # concatenated directly with the much smaller HaMeR/DINOv2 branches, so it
        # dominates the shared projection simply by having far more dimensions.
        # With stgcn_proj_dim set, the graph features get their own branch
        # (Linear -> LayerNorm -> GELU -> Dropout, same design as the HaMeR branch)
        # down to stgcn_proj_dim before the concatenation. None = original behaviour
        # (keeps old checkpoints loadable).
        self.stgcn_proj_dim = stgcn_proj_dim
        if stgcn_proj_dim is not None:
            self.stgcn_proj = nn.Sequential(
                nn.Linear(num_vertices * stgcn_channels, stgcn_proj_dim),
                nn.LayerNorm(stgcn_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim = stgcn_proj_dim
        else:
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

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

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

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = x.view(B, T, -1)
        if self.stgcn_proj_dim is not None:
            x = self.stgcn_proj(x)  # (B, T, stgcn_proj_dim)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))  # (B, T, hamer_proj_dim)
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)

        features = self.projection(x + 1e-5)      
        lstm_out, _ = self.lstm(features)      
        logits = self.classifier(lstm_out)     
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)


class BiLSTM_Baseline(nn.Module):
    """
    Graph-free BiLSTM baseline. Flattens (C, V) per frame and runs a BiLSTM.
    Also used for pure-HaMeR training (input (288, T, 1) -> 288 features/frame).

    Optional DINOv2 side branch (dinov2_dim): DINOv2 hand-crop embeddings get their
    own Linear+LayerNorm projection (to dinov2_proj_dim) and are concatenated with the
    main per-frame features BEFORE the shared projection -- same fusion design as the
    graph models' DINOv2 branch, so a large unnormalized appearance vector can't
    dominate the smaller main input.
    """
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2,
                 dinov2_dim=None, dinov2_proj_dim=128):
        super(BiLSTM_Baseline, self).__init__()
        self.feature_dim = in_channels * num_vertices

        self.dinov2_dim = dinov2_dim
        proj_in_dim = self.feature_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            proj_in_dim += dinov2_proj_dim

        # Matches the 2023 paper's spec: project to d_model (256), not d_model*2 --
        # see STGCN_BiLSTM's comment for the full reasoning.
        self.projection = nn.Sequential(
            nn.Linear(proj_in_dim, d_model),
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

    def forward(self, x, dinov2=None):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                 "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)

        features = self.projection(x + 1e-5)
        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out)
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)


class PureMambaBaseline(nn.Module):
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.feature_dim = in_channels * num_vertices
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
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
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.feature_dim = in_channels * num_vertices
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        self.bwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
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
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
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

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            flat_dim += dinov2_proj_dim
        
        # 2. Continuous Latent Space Bottleneck (Encoder)
        self.latent_encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Latent Mamba Extractor (Smooths the latent space dynamically)
        self.latent_mamba = Mamba(d_model=latent_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        
        # 4. Up-Projection to Main Sequence Dimension
        self.latent_to_main = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 5. Main Temporal Sequence Modeler (BiMamba Backend)
        self.fwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        self.bwd_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)
        ])
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
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

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)
        
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
    def __init__(self, num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
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

        # Optional separate DINOv2 branch: a self-supervised ViT embedding of the
        # hand crop itself (SHuBERT/SignMusketeers-style) -- an APPEARANCE feature,
        # not a geometric/kinematic one, so it's fused the same way as HaMeR (own
        # MLP branch, concatenated before the temporal backbone) but is a genuinely
        # different information source, combinable independently with HaMeR.
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            flat_dim += dinov2_proj_dim

        self.latent_encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.latent_mamba = Mamba(d_model=latent_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.latent_to_main = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.fwd_mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)])
        self.bwd_mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(n_layers)])
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
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

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))  # (B, T, dinov2_proj_dim)
            x = torch.cat([x, dinov2_feat], dim=-1)
        
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
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim, mamba_d_state, mamba_d_conv, mamba_expand)
        A = SkeletonGraph(num_vertices=num_vertices).A
        self.spatial_blocks = nn.Sequential(
            CTRGCNBlock(in_channels, stgcn_channels, A),
            CTRGCNBlock(stgcn_channels, stgcn_channels, A)
        )

class InfoGCN_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim, mamba_d_state, mamba_d_conv, mamba_expand)
        A = SkeletonGraph(num_vertices=num_vertices).A
        self.spatial_blocks = nn.Sequential(
            InfoGCNBlock(in_channels, stgcn_channels, A),
            InfoGCNBlock(stgcn_channels, stgcn_channels, A)
        )

class ShiftGCN_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim, mamba_d_state, mamba_d_conv, mamba_expand)
        self.spatial_blocks = nn.Sequential(
            ShiftGCNBlock(in_channels, stgcn_channels, num_vertices),
            ShiftGCNBlock(stgcn_channels, stgcn_channels, num_vertices)
        )

class SpatialTransformer_Mamba(Base_Latent_Mamba_Wrapper):
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim, mamba_d_state, mamba_d_conv, mamba_expand)
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
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, hd_max_hop=3, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim, mamba_d_state, mamba_d_conv, mamba_expand)
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
    def __init__(self, num_vertices=65, in_channels=5, stgcn_channels=64, latent_dim=128, d_model=256, n_layers=4, num_classes=3, dropout=0.2, num_soft_hyperedges=8, hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__(num_vertices, stgcn_channels, latent_dim, d_model, n_layers, num_classes, dropout, hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim, mamba_d_state, mamba_d_conv, mamba_expand)
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        hyperedges = graph.get_anatomical_hyperedges()
        self.spatial_blocks = nn.Sequential(
            HyperSignBlock(in_channels, stgcn_channels, A, hyperedges, num_vertices, num_soft_hyperedges),
            HyperSignBlock(stgcn_channels, stgcn_channels, A, hyperedges, num_vertices, num_soft_hyperedges)
        )

# ==============================================================================
# 🆕 Mamba-Transformer hybrids
# ==============================================================================
class HybridInterleavedBlock(nn.Module):
    """
    One 'hybrid block': a BIDIRECTIONAL Mamba sub-layer (long-range context,
    linear cost) followed by a self-attention sub-layer (precise pairwise
    comparison), each with a pre-norm residual connection -- Jamba/Zamba-style
    interleaving, but Mamba-THEN-attention specifically: attention refines an
    ALREADY-CONTEXTUALIZED representation rather than raw input. Motivated by
    this task's structure: "is this frame a boundary" is fundamentally a
    pairwise question (attention-favorable), evaluated against "what sign is
    currently in progress" (a long-range question, recurrence-favorable) --
    so Mamba builds context first, attention sharpens boundaries using it.
    Stays bidirectional throughout, matching STGCN_BiMamba's proven edge over
    unidirectional Mamba on this task.
    """
    def __init__(self, d_model, mamba_d_state, mamba_d_conv, mamba_expand, nhead, dim_feedforward, dropout):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.mamba_fuse = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm residual bidirectional Mamba sub-layer
        residual = x
        x_norm = self.norm1(x)
        fwd = self.mamba_fwd(x_norm)
        bwd = torch.flip(self.mamba_bwd(torch.flip(x_norm, dims=[1])), dims=[1])
        x = residual + self.dropout(self.mamba_fuse(torch.cat([fwd, bwd], dim=-1)))

        # Pre-norm residual self-attention sub-layer
        residual = x
        x_norm = self.norm2(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.dropout(attn_out)

        # Pre-norm residual feedforward sub-layer
        residual = x
        x_norm = self.norm3(x)
        x = residual + self.dropout(self.ffn(x_norm))

        return x


class ParallelHybridBlock(nn.Module):
    """
    Runs bidirectional Mamba and self-attention as TWO PARALLEL branches over
    the SAME input, fused via a LEARNED, PER-TIMESTEP sigmoid gate -- not
    fixed concatenation. This lets the network decide, frame by frame, how
    much to rely on Mamba's compressed long-range context versus attention's
    precise pairwise comparison, rather than assuming a fixed sequential
    order is right for every moment. A bigger bet than the interleaved
    variant (more parameters, less production precedent), but it directly
    encodes "different frames need different mechanisms" instead of assuming
    Mamba-then-attention is universally the right order.
    """
    def __init__(self, d_model, mamba_d_state, mamba_d_conv, mamba_expand, nhead, dim_feedforward, dropout):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.mamba_fuse = nn.Linear(d_model * 2, d_model)

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_in = nn.LayerNorm(d_model)
        # Per-timestep, per-channel gate in [0,1]: 1 -> fully Mamba, 0 -> fully attention.
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm_out = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.norm_in(x)

        fwd = self.mamba_fwd(x_norm)
        bwd = torch.flip(self.mamba_bwd(torch.flip(x_norm, dims=[1])), dims=[1])
        mamba_out = self.mamba_fuse(torch.cat([fwd, bwd], dim=-1))

        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)

        g = self.gate(torch.cat([mamba_out, attn_out], dim=-1))
        fused = g * mamba_out + (1 - g) * attn_out

        x = residual + self.dropout(fused)
        x = self.norm_out(x)

        residual = x
        x = residual + self.dropout(self.ffn(x))
        x = self.norm_ffn(x)

        return x


class STGCN_HybridSequential(nn.Module):
    """
    Interleaved Mamba+Attention hybrid (Jamba/Zamba-style). See
    HybridInterleavedBlock for the full reasoning.
    """
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4,
                 num_classes=3, nhead=8, dim_feedforward=1024, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        self.bridge_dim = num_vertices * stgcn_channels

        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.hybrid_blocks = nn.ModuleList([
            HybridInterleavedBlock(d_model, mamba_d_state, mamba_d_conv, mamba_expand,
                                    nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        x = self.stgcn_blocks(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))
            x = torch.cat([x, dinov2_feat], dim=-1)

        x = self.feature_proj(x + 1e-5)

        for block in self.hybrid_blocks:
            x = block(x)

        embeddings = x
        logits = self.classifier(x)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_HybridParallel(nn.Module):
    """
    Parallel Mamba+Attention hybrid with a learned per-timestep gate. See
    ParallelHybridBlock for the full reasoning.
    """
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4,
                 num_classes=3, nhead=8, dim_feedforward=1024, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        self.bridge_dim = num_vertices * stgcn_channels

        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.hybrid_blocks = nn.ModuleList([
            ParallelHybridBlock(d_model, mamba_d_state, mamba_d_conv, mamba_expand,
                                 nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        x = self.stgcn_blocks(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))
            x = torch.cat([x, dinov2_feat], dim=-1)

        x = self.feature_proj(x + 1e-5)

        for block in self.hybrid_blocks:
            x = block(x)

        embeddings = x
        logits = self.classifier(x)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


# ==============================================================================
# 🆕 MLP auxiliary-module encoder (2025 Hands-On paper style, no graph)
# ==============================================================================
class AuxiliaryMLPEncoder(nn.Module):
    """
    Replaces the graph-convolution spatial encoder (STGCNBlock) with a plain
    3-layer trainable MLP applied to the FLATTENED per-frame input (all
    vertices x channels concatenated into one vector per frame) -- matching
    the 2025 Hands-On paper's "auxiliary module" design, which processes
    each feature stream through a dedicated MLP rather than an explicit
    skeleton graph.

    This is a genuine ablation of the STGCN-based approach used everywhere
    else in this codebase: does the explicit graph-topology inductive bias
    actually help, or does a topology-free MLP do just as well (or better)?
    Given this project's own earlier finding that the spatial graph mattered
    more than temporal-backbone choice, this is a real test of that finding,
    not just a style change.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, C, T, V) -- same input convention as STGCNBlock, for drop-in compatibility
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)  # flatten per-frame: (B, T, C*V)
        return self.net(x)  # (B, T, output_dim)


class MLPAux_Mamba(nn.Module):
    """
    Same overall pipeline as STGCN_Mamba, but the graph-convolution spatial
    encoder is replaced with AuxiliaryMLPEncoder -- see that class's
    docstring for the full motivation. Identical everything else (HaMeR/
    DINOv2 fusion, Mamba backbone, classifier); only the spatial-encoding
    mechanism differs, for a direct, controlled ablation.
    """
    def __init__(self, num_vertices=65, in_channels=3, mlp_hidden_dim=512, d_model=256, n_layers=4,
                 num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64,
                 dinov2_dim=None, dinov2_proj_dim=128,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.spatial_encoder = AuxiliaryMLPEncoder(
            input_dim=num_vertices * in_channels, hidden_dim=mlp_hidden_dim,
            output_dim=d_model, dropout=dropout
        )
        self.bridge_dim = d_model

        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

        # Only need a re-projection if hamer/dinov2 changed bridge_dim away from d_model
        if self.bridge_dim != d_model:
            self.feature_proj = nn.Sequential(
                nn.Linear(self.bridge_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_proj = nn.Identity()

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        x = self.spatial_encoder(x)  # (B, T, d_model)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))
            x = torch.cat([x, dinov2_feat], dim=-1)

        x = self.feature_proj(x + 1e-5)

        for layer in self.mamba_layers:
            x = layer(x)

        embeddings = x
        logits = self.classifier(x)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class MLPAux_BiMamba(nn.Module):
    """Same as MLPAux_Mamba, but with a bidirectional Mamba backbone (matching
    STGCN_BiMamba's proven edge over unidirectional Mamba on this task)."""
    def __init__(self, num_vertices=65, in_channels=3, mlp_hidden_dim=512, d_model=256, n_layers=4,
                 num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64,
                 dinov2_dim=None, dinov2_proj_dim=128,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.spatial_encoder = AuxiliaryMLPEncoder(
            input_dim=num_vertices * in_channels, hidden_dim=mlp_hidden_dim,
            output_dim=d_model, dropout=dropout
        )
        self.bridge_dim = d_model

        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

        if self.bridge_dim != d_model:
            self.feature_proj = nn.Sequential(
                nn.Linear(self.bridge_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_proj = nn.Identity()

        self.mamba_fwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(n_layers)
        ])
        self.mamba_bwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        x = self.spatial_encoder(x)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))
            x = torch.cat([x, dinov2_feat], dim=-1)

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


class MLPAux_BiLSTM(nn.Module):
    """Same overall pipeline as STGCN_BiLSTM, but with AuxiliaryMLPEncoder
    instead of the graph-convolution spatial encoder."""
    def __init__(self, num_vertices=65, in_channels=3, mlp_hidden_dim=512, d_model=256, n_layers=4,
                 num_classes=3, dropout=0.2, hamer_dim=None, hamer_proj_dim=64,
                 dinov2_dim=None, dinov2_proj_dim=128):
        super().__init__()
        self.spatial_encoder = AuxiliaryMLPEncoder(
            input_dim=num_vertices * in_channels, hidden_dim=mlp_hidden_dim,
            output_dim=d_model, dropout=dropout
        )
        self.bridge_dim = d_model

        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

        if self.bridge_dim != d_model:
            self.feature_proj = nn.Sequential(
                nn.Linear(self.bridge_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_proj = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=True
        )
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        x = self.spatial_encoder(x)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))
            x = torch.cat([x, dinov2_feat], dim=-1)

        features = self.feature_proj(x + 1e-5)
        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out)
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)


class MLPAux_Transformer(nn.Module):
    """Same overall pipeline as STGCN_Transformer, but with AuxiliaryMLPEncoder
    instead of the graph-convolution spatial encoder."""
    def __init__(self, num_vertices=65, in_channels=3, mlp_hidden_dim=512, d_model=256, n_layers=4,
                 num_classes=3, nhead=8, dim_feedforward=1024, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128):
        super().__init__()
        self.spatial_encoder = AuxiliaryMLPEncoder(
            input_dim=num_vertices * in_channels, hidden_dim=mlp_hidden_dim,
            output_dim=d_model, dropout=dropout
        )
        self.bridge_dim = d_model

        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim),
                nn.LayerNorm(hamer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += hamer_proj_dim

        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim),
                nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.bridge_dim += dinov2_proj_dim

        if self.bridge_dim != d_model:
            self.feature_proj = nn.Sequential(
                nn.Linear(self.bridge_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_proj = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, hamer=None, dinov2=None):
        x = self.spatial_encoder(x)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))
            x = torch.cat([x, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))
            x = torch.cat([x, dinov2_feat], dim=-1)

        features = self.feature_proj(x + 1e-5)
        features = self.pos_encoder(features)
        embeddings = self.transformer_encoder(features)
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


# ==============================================================================
# 🆕 2025 Hands-On paper architecture ("Hands-On: Segmenting Individual Signs from
#    Continuous Sequences"), trainable in train.py as basename "handson_2025"
# ==============================================================================
class SequenceMLP3(nn.Module):
    """Three-layer MLP applied per frame to a (B, T, D) sequence.
    Same Linear-LayerNorm-GELU-Dropout block as AuxiliaryMLPEncoder.net (the paper does not
    state norm/activation, so this follows the codebase's convention)."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class HandsOn2025(nn.Module):
    """
    Architecture of the 2025 Hands-On sign segmentation paper (Sec. III-C, Fig. 2):

        HaMeR (288)  --> 3-layer MLP adapter --> 512 --+
                                                         +-> temporal downsample x2 -> concat (1024)
        pose stream  --> 3-layer MLP adapter --> 512 --+      -> 3-layer MLP mixer -> d_model
        -> Transformer encoder -> per-frame BIO logits   (+ a CTC head for the sign-level CTC loss)

    HaMeR is a mandatory input (use_hamer_features=True): the model refuses to build or run without it.

    POSE STREAM (`pose_stream`). The paper's second stream is a 104-d 3D-skeleton-angle vector from a separate
    pose model, which this project does not have. Instead the angles are COMPUTED from the MediaPipe xyz
    coordinates already in x (src/skeleton_angles.py: finger/elbow flexion, finger spread, shoulder/wrist angles,
    limb directions, palm normals; ANGLE_FEATURE_DIM features). Options:
        "angles"      joint angles only                     (closest to the paper)
        "xyz"         flattened skeleton coordinates        (all in_channels x num_vertices)
        "xyz+angles"  both concatenated
    The angle features read channels 0-2 of x, which must be x/y/z (base_features = ["x-cord","y-cord","z-cord"]).
    `angle_y_scale` = video height / width corrects MediaPipe's normalised-image anisotropy (see skeleton_angles).

    CTC: a separate Linear head on the Transformer output (ctc_num_tokens + 1 classes, blank = 0, at the
    DOWNSAMPLED frame rate) is exposed after every forward() as `self.ctc_logits` (B, T', ctc_num_tokens + 1);
    forward() keeps returning (logits, embeddings) like every other model. Set ctc_num_tokens=0 to drop it.

    Not stated in the paper, so chosen here and exposed as arguments: MLP hidden widths, mixer output width
    (= d_model), Transformer size, norm/activation, dropout, pre-LN vs post-LN (`norm_first`).

    Downsampling is done inside the model (every `downsample`-th frame, parameter-free) and the BIO logits are
    repeated back to the input frame rate, so the dataset/labels/metrics keep their usual (B, 3, T) shape.
    Input convention (same as every other model here): x (B, C, T, V), hamer (B, hamer_dim, T).
    """
    POSE_STREAMS = ("angles", "xyz", "xyz+angles")

    def __init__(self, num_vertices=65, in_channels=3, num_classes=3, d_model=256, n_layers=4,
                 nhead=8, dim_feedforward=None, dropout=0.2, adapter_dim=512, adapter_hidden=None,
                 mixer_hidden=512, downsample=2, pose_stream="angles", angle_y_scale=1.0,
                 ctc_num_tokens=1, norm_first=True, hamer_dim=None, dinov2_dim=None):
        super().__init__()
        if hamer_dim is None:
            raise ValueError("HandsOn2025 needs HaMeR features: set use_hamer_features=True.")
        if dinov2_dim is not None:
            raise ValueError("HandsOn2025 has no DINOv2 stream; set use_dinov2_features=False.")
        if pose_stream not in self.POSE_STREAMS:
            raise ValueError(f"pose_stream must be one of {self.POSE_STREAMS}, got '{pose_stream}'.")
        if "angles" in pose_stream and (in_channels < 3 or num_vertices < 65):
            raise ValueError("angle features need x/y/z in channels 0-2 and the 65-vertex skeleton.")
        adapter_hidden = adapter_hidden or adapter_dim
        dim_feedforward = dim_feedforward or d_model * 4
        self.downsample = max(1, int(downsample))
        self.hamer_dim = hamer_dim
        self.pose_stream = pose_stream
        self.angle_y_scale = float(angle_y_scale)
        self.ctc_logits = None

        pose_in = {"angles": ANGLE_FEATURE_DIM,
                   "xyz": in_channels * num_vertices,
                   "xyz+angles": in_channels * num_vertices + ANGLE_FEATURE_DIM}[pose_stream]
        self.hamer_adapter = SequenceMLP3(hamer_dim, adapter_hidden, adapter_dim, dropout)
        self.pose_adapter = SequenceMLP3(pose_in, adapter_hidden, adapter_dim, dropout)
        self.mixer = SequenceMLP3(2 * adapter_dim, mixer_hidden, d_model, dropout)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=norm_first
        )
        # norm_first=True (pre-LN) trains far more stably than post-LN without a long warm-up; it needs a final LayerNorm.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=n_layers, norm=nn.LayerNorm(d_model) if norm_first else None,
            enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, num_classes)
        self.ctc_head = nn.Linear(d_model, ctc_num_tokens + 1) if ctc_num_tokens > 0 else None

    def pose_features(self, x):
        """x: (B, C, T', V) already downsampled -> (B, T', pose_dim)."""
        B, C, T, V = x.shape
        parts = []
        if "xyz" in self.pose_stream:
            parts.append(x.permute(0, 2, 1, 3).reshape(B, T, C * V))
        if "angles" in self.pose_stream:
            with torch.no_grad():                                  # fixed geometry, no learnable parameters
                xyz = x[:, :3].permute(0, 2, 3, 1).float()         # (B, T', V, 3)
                parts.append(skeleton_angle_features(xyz, y_scale=self.angle_y_scale).to(x.dtype))
        return torch.cat(parts, dim=-1)

    def forward(self, x, hamer=None, dinov2=None):
        if hamer is None:
            raise ValueError("HandsOn2025.forward() was called without a `hamer` tensor.")
        if dinov2 is not None:
            raise ValueError("HandsOn2025 has no DINOv2 stream.")
        T = x.shape[2]
        ham = hamer.permute(0, 2, 1)                             # (B, T, hamer_dim)
        if self.downsample > 1:                                  # downsample first: everything before the Transformer is per-frame
            x = x[:, :, ::self.downsample]
            ham = ham[:, ::self.downsample]

        fused = torch.cat([self.hamer_adapter(ham), self.pose_adapter(self.pose_features(x))], dim=-1)   # (B, T', 1024)
        feats = self.pos_encoder(self.mixer(fused))              # (B, T', d_model)
        emb = self.transformer_encoder(feats)                    # (B, T', d_model)
        logits = self.classifier(emb)                            # (B, T', 3)
        self.ctc_logits = self.ctc_head(emb) if self.ctc_head is not None else None   # (B, T', K), downsampled rate

        if self.downsample > 1:                                  # back to the input frame rate
            logits = logits.repeat_interleave(self.downsample, dim=1)[:, :T]
            emb = emb.repeat_interleave(self.downsample, dim=1)[:, :T]
        return logits.permute(0, 2, 1), emb.permute(0, 2, 1)


# ==============================================================================
# RELATIVE-POSITION TRANSFORMER ENCODER (RoPE / ALiBi / sinusoidal / none)
# ==============================================================================
def _alibi_slopes(n_heads):
    """Per-head ALiBi slopes (Press et al., 2022): geometric sequence 2^(-8/n), 2^(-16/n), ..."""
    def pow2_slopes(n):
        start = 2 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return pow2_slopes(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return pow2_slopes(closest) + pow2_slopes(2 * closest)[0::2][: n_heads - closest]


class RotaryEmbedding(nn.Module):
    """
    RoPE (Su et al., 2021): rotates each query/key feature pair by an angle proportional
    to the frame index, so q.k depends only on the DISTANCE between two frames, not on
    where they sit inside the (arbitrarily cut) window.
    """
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head dimension, got {head_dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, T, device, dtype):
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))      # (T, head_dim/2)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def apply(x, cos, sin):
        """x: (B, H, T, head_dim) -- rotate-half formulation."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, pos_encoding, rope_base=10000.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.pos_encoding = pos_encoding
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, rope_base) if pos_encoding == "rope" else None
        if pos_encoding == "alibi":
            self.register_buffer("alibi_slopes", torch.tensor(_alibi_slopes(nhead), dtype=torch.float32),
                                 persistent=False)
        self.store_attention = False  # set True to keep the last attention map (analysis only)
        self.last_attention = None

    def _alibi_bias(self, T, device, dtype):
        pos = torch.arange(T, device=device)
        dist = (pos[None, :] - pos[:, None]).abs().to(torch.float32)          # (T, T)
        return (-self.alibi_slopes.to(device)[:, None, None] * dist).to(dtype)  # (H, T, T)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            cos, sin = self.rope.cos_sin(T, x.device, q.dtype)
            q, k = RotaryEmbedding.apply(q, cos, sin), RotaryEmbedding.apply(k, cos, sin)
        bias = self._alibi_bias(T, x.device, q.dtype) if self.pos_encoding == "alibi" else None

        if self.store_attention:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if bias is not None:
                scores = scores + bias
            attn = scores.softmax(dim=-1)
            self.last_attention = attn.detach()                                 # (B, H, T, T)
            attn = nn.functional.dropout(attn, self.attn_dropout, self.training)
            y = attn @ v
        else:
            y = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=self.attn_dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class RelPosEncoderLayer(nn.Module):
    """Transformer encoder layer with pre- or post-norm and a ReLU feed-forward (as nn.TransformerEncoderLayer)."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, pos_encoding, norm="pre", rope_base=10000.0):
        super().__init__()
        self.norm_first = norm == "pre"
        self.attn = RelPosSelfAttention(d_model, nhead, dropout, pos_encoding, rope_base)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(dim_feedforward, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        if self.norm_first:
            x = x + self.drop1(self.attn(self.norm1(x)))
            x = x + self.drop2(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop1(self.attn(x)))
            x = self.norm2(x + self.drop2(self.ff(x)))
        return x


class RelPosTransformerEncoder(nn.Module):
    """
    pos_encoding:
      "sinusoidal" -- absolute sine/cosine added to the input (classic Transformer)
      "rope"       -- rotary embedding inside every attention layer (relative distance)
      "alibi"      -- per-head linear distance penalty on attention scores (relative, favours nearby frames)
      "none"       -- no position information (attention is order-blind)
    norm: "pre" (LayerNorm before each sublayer, + final LayerNorm) or "post".
    """
    POS_ENCODINGS = ("sinusoidal", "rope", "alibi", "none")

    def __init__(self, d_model, nhead, n_layers, dim_feedforward, dropout, pos_encoding="rope",
                 norm="pre", rope_base=10000.0):
        super().__init__()
        if pos_encoding not in self.POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {self.POS_ENCODINGS}, got '{pos_encoding}'.")
        if norm not in ("pre", "post"):
            raise ValueError(f"transformer_norm must be 'pre' or 'post', got '{norm}'.")
        self.pos_encoding = pos_encoding
        self.input_pos = PositionalEncoding(d_model, dropout) if pos_encoding == "sinusoidal" else nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            RelPosEncoderLayer(d_model, nhead, dim_feedforward, dropout, pos_encoding, norm, rope_base)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model) if norm == "pre" else nn.Identity()

    def set_attention_capture(self, on=True):
        for layer in self.layers:
            layer.attn.store_attention = on
            if not on:
                layer.attn.last_attention = None

    def attention_maps(self):
        """List (one per layer) of the last stored (B, H, T, T) attention maps."""
        return [layer.attn.last_attention for layer in self.layers]

    def forward(self, x):
        x = self.input_pos(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ==============================================================================
# MULTI-STREAM MODELS: (body + hands) / face / HaMeR as separately switchable streams
# ==============================================================================
class MultiStreamSegmenter(nn.Module):
    """
    Up to three input streams, each switchable on/off:
      body_hands : ST-GCN on the 65 body + hand vertices        -> own branch
      face       : ST-GCN on the face vertices (face-only graph) -> own branch
      hamer      : MLP on the 288-dim HaMeR vector               -> own branch
    Each branch is Linear -> LayerNorm -> GELU -> Dropout, so every stream enters the
    fusion normalised and at a size you choose.

    fusion="concat"     (option A): branches -> sizes *_proj_dim, concatenated, then ONE
                                    shared projection to d_model (encoder width stays
                                    d_model whichever streams are on).
    fusion="gated_sum"  (option B): every branch -> d_model, multiplied by a learnable
                                    scalar gate per stream, summed, LayerNorm. Encoder
                                    width is d_model whichever streams are on; the gates
                                    show how much each stream is used (stream_gates()).

    d_model = the size going into the encoder (BiLSTM hidden size / Transformer width).

    Input layout (from the dataset): x is (B, C, T, V) with the 65 body+hand vertices
    first and, when face keypoints are loaded, the face vertices after them
    (V = 65 + face subset size). The face stream uses x[..., 65:].

    For check_stream_balance.py the model can report each stream's contribution to the
    fused representation (set_capture) and replace one stream by a constant (set_ablation).
    """
    STREAM_ORDER = ("body_hands", "face", "hamer")

    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4,
                 dropout=0.2, stgcn_channels=64,
                 use_stream_body_hands=True, use_stream_face=False, use_stream_hamer=False,
                 fusion="concat", body_hands_proj_dim=256, face_proj_dim=256, hamer_proj_dim=256,
                 hamer_dim=None, encoder="bilstm", nhead=8, dim_feedforward=None,
                 pos_encoding="sinusoidal", transformer_norm="post", rope_base=10000.0,
                 face_channels=None):
        super().__init__()
        if fusion not in ("concat", "gated_sum"):
            raise ValueError(f"fusion must be 'concat' or 'gated_sum', got '{fusion}'.")
        if encoder not in ("bilstm", "transformer"):
            raise ValueError(f"encoder must be 'bilstm' or 'transformer', got '{encoder}'.")
        self.streams = [name for name, on in zip(self.STREAM_ORDER,
                                                 (use_stream_body_hands, use_stream_face, use_stream_hamer)) if on]
        if not self.streams:
            raise ValueError("At least one stream must be on (use_stream_body_hands / "
                             "use_stream_face / use_stream_hamer).")
        self.fusion = fusion
        self.encoder_type = encoder
        self.d_model = d_model

        def branch(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))

        out_dims = {
            "body_hands": body_hands_proj_dim, "face": face_proj_dim, "hamer": hamer_proj_dim,
        } if fusion == "concat" else {name: d_model for name in self.STREAM_ORDER}

        self.branches = nn.ModuleDict()

        if "body_hands" in self.streams:
            if num_vertices < 65:
                raise ValueError(f"body_hands stream needs the 65 body+hand vertices (num_vertices={num_vertices}).")
            A_bh = SkeletonGraph(num_vertices=65).A
            self.body_hands_stgcn = nn.Sequential(
                STGCNBlock(in_channels, stgcn_channels, A_bh),
                STGCNBlock(stgcn_channels, stgcn_channels, A_bh),
            )
            self.branches["body_hands"] = branch(65 * stgcn_channels, out_dims["body_hands"])

        if "face" in self.streams:
            self.num_face = num_vertices - 65
            if self.num_face <= 0:
                raise ValueError("face stream is on but the input has no face vertices -- set "
                                 "use_face_keypoints=True (the queue does this for multistream models).")
            A_face = SkeletonGraph(num_vertices=self.num_face, face_only=True).A
            # face_channels: which input channels carry face data (2D face: x and y only).
            # None = all channels (legacy 3D face runs).
            self.face_channels = list(face_channels) if face_channels is not None else None
            face_in = len(self.face_channels) if self.face_channels is not None else in_channels
            self.face_stgcn = nn.Sequential(
                STGCNBlock(face_in, stgcn_channels, A_face),
                STGCNBlock(stgcn_channels, stgcn_channels, A_face),
            )
            self.branches["face"] = branch(self.num_face * stgcn_channels, out_dims["face"])

        if "hamer" in self.streams:
            if hamer_dim is None:
                raise ValueError("hamer stream is on but hamer_dim is None -- set use_hamer_features=True.")
            self.hamer_dim = hamer_dim
            self.branches["hamer"] = branch(hamer_dim, out_dims["hamer"])

        if fusion == "concat":
            self.stream_dims = {name: out_dims[name] for name in self.streams}
            self.concat_dim = sum(self.stream_dims.values())
            self.fusion_proj = nn.Sequential(
                nn.Linear(self.concat_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            self.stream_dims = {name: d_model for name in self.streams}
            self.gates = nn.Parameter(torch.ones(len(self.streams)))
            self.fusion_norm = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout))

        if encoder == "bilstm":
            self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers,
                                batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=True)
            self.classifier = nn.Linear(d_model * 2, num_classes)
        else:
            if d_model % nhead != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
            self.pos_encoding = pos_encoding
            # sinusoidal + post-norm = the original PyTorch encoder (old checkpoints keep loading);
            # every other combination uses RelPosTransformerEncoder.
            self.legacy_encoder = (pos_encoding == "sinusoidal" and transformer_norm == "post")
            if self.legacy_encoder:
                self.pos_encoder = PositionalEncoding(d_model, dropout)
                layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward or d_model * 4,
                                                   dropout=dropout, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            else:
                self.rel_encoder = RelPosTransformerEncoder(
                    d_model, nhead, n_layers, dim_feedforward or d_model * 4, dropout,
                    pos_encoding=pos_encoding, norm=transformer_norm, rope_base=rope_base)
            self.classifier = nn.Linear(d_model, num_classes)

        self._capture = None   # callable(stream_inputs, stream_contributions) for diagnostics
        self._ablation = None  # (stream_name, constant vector) for diagnostics

    # ---- diagnostics hooks (used by check_stream_balance.py) ----------------
    def set_capture(self, fn):
        self._capture = fn

    def set_ablation(self, stream_name=None, value=None):
        self._ablation = None if stream_name is None else (stream_name, value)

    def stream_gates(self):
        """Gated-sum only: {stream: gate value}."""
        if self.fusion != "gated_sum":
            return {}
        return {name: float(g) for name, g in zip(self.streams, self.gates.detach().cpu())}

    # ---- forward ------------------------------------------------------------
    def _graph_stream(self, stgcn, x):
        B, C, T, V = x.shape
        h = stgcn(x)                                   # (B, stgcn_channels, T, V)
        return h.permute(0, 2, 3, 1).reshape(B, T, -1)  # (B, T, V * stgcn_channels)

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        z = {}
        if "body_hands" in self.streams:
            z["body_hands"] = self.branches["body_hands"](self._graph_stream(self.body_hands_stgcn, x[..., :65]))
        if "face" in self.streams:
            xf = x[..., 65:]
            if self.face_channels is not None:
                xf = xf[:, self.face_channels]
            z["face"] = self.branches["face"](self._graph_stream(self.face_stgcn, xf))
        if "hamer" in self.streams:
            if hamer is None:
                raise ValueError("hamer stream is on but forward() got no `hamer` tensor.")
            z["hamer"] = self.branches["hamer"](hamer.permute(0, 2, 1))   # (B, T, dim)

        if self._ablation is not None:
            name, value = self._ablation
            z[name] = value.to(z[name].dtype).expand_as(z[name])

        if self.fusion == "concat":
            cat = torch.cat([z[name] for name in self.streams], dim=-1)
            fused = self.fusion_proj(cat)
            if self._capture is not None:
                W = self.fusion_proj[0].weight
                contrib, start = {}, 0
                for name in self.streams:
                    d = self.stream_dims[name]
                    contrib[name] = z[name] @ W[:, start:start + d].T
                    start += d
                self._capture(z, contrib)
        else:
            contrib = {name: g * z[name] for name, g in zip(self.streams, self.gates)}
            fused = self.fusion_norm(sum(contrib.values()))
            if self._capture is not None:
                self._capture(z, contrib)

        if self.encoder_type == "bilstm":
            emb, _ = self.lstm(fused)
        elif self.legacy_encoder:
            emb = self.transformer_encoder(self.pos_encoder(fused))
        else:
            emb = self.rel_encoder(fused)
        logits = self.classifier(emb)
        return logits.permute(0, 2, 1), emb.permute(0, 2, 1)


class MultiStream_BiLSTM(MultiStreamSegmenter):
    def __init__(self, **kwargs):
        kwargs.pop("encoder", None)
        super().__init__(encoder="bilstm", **kwargs)


class MultiStream_Transformer(MultiStreamSegmenter):
    def __init__(self, **kwargs):
        kwargs.pop("encoder", None)
        super().__init__(encoder="transformer", **kwargs)