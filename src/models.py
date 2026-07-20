import math
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
    def __init__(self, in_channels, num_vertices, num_classes=3, stgcn_channels=64, d_model=256, n_layers=4, nhead=8, dim_feedforward=1024, dropout=0.2):
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
        
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, T, -1)
        features = self.feature_proj(x + 1e-5) 
        
        features = self.pos_encoder(features)
        
        # Raw, un-checkpointed Global Self-Attention
        embeddings = self.transformer_encoder(features)
        
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_Mamba(nn.Module):
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
            DecoupledSTGCNBlock(in_channels, stgcn_channels),
            DecoupledSTGCNBlock(stgcn_channels, stgcn_channels)
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
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, dropout=0.2):
        super(STGCN_BiLSTM, self).__init__()
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        self.bridge_dim = num_vertices * stgcn_channels
        self.projection = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=d_model * 2,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = self.stgcn_blocks(x) 
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = x.view(B, T, -1)                   
        features = self.projection(x + 1e-5)      
        lstm_out, _ = self.lstm(features)      
        logits = self.classifier(lstm_out)     
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)


class BiLSTM_Baseline(nn.Module):
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2):
        super(BiLSTM_Baseline, self).__init__()
        self.feature_dim = in_channels * num_vertices
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=d_model * 2,
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