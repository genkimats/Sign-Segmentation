import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock
from src.stgcn import DecoupledSTGCNBlock
from torch.utils.checkpoint import checkpoint


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
        
        # 1. Feature Projection (Flatten spatial nodes into 1D embedding)
        self.feature_dim = in_channels * num_vertices
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Sequence Encoder (Positional Encoding + Transformer)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # Ensures input is (Batch, Time, Features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # 3. Output Head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Current x shape: (B, C, T, V)
        B, C, T, V = x.shape
        
        # Phase 1: Flatten spatial dimension
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V) # (B, T, C*V)
        features = self.projection(x)                  # (B, T, d_model)
        
        # Phase 2: Add temporal position awareness
        features = self.pos_encoder(features)
        
        # Phase 3: Global Multi-Head Self Attention
        embeddings = self.transformer_encoder(features) # (B, T, d_model)
        
        # Phase 4: Classification
        logits = self.classifier(embeddings)            # (B, T, num_classes)
        
        # Match PyTorch CrossEntropy sequence signature: (B, C, T)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


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
        # Forward sweep with Checkpointing
        fwd_out = x
        for layer in self.fwd_mamba:
            # Replaces: fwd_out = layer(fwd_out)
            fwd_out = checkpoint(layer, fwd_out, use_reentrant=False)
            
        # Backward sweep with Checkpointing
        bwd_out = torch.flip(x, dims=[1])
        for layer in self.bwd_mamba:
            bwd_out = checkpoint(layer, bwd_out, use_reentrant=False)
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
        x = self.feature_proj(x + 1e-5) # Epsilon
        
        for layer in self.mamba_layers:
            x = layer(x)
            
        embeddings = x.permute(0, 2, 1) 
        logits = self.classifier(x)
        logits = logits.permute(0, 2, 1) 
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
    """
    Ablation Baseline: No Spatial GCN.
    Flattens spatial nodes directly and feeds to a Causal Mamba sequence model.
    """
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
        # Flatten spatial dimension
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)
        features = self.projection(x)
        
        embeddings = features
        for layer in self.mamba_layers:
            embeddings = layer(embeddings)
            
        logits = self.classifier(embeddings)
        return logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class BiLSTM_Baseline(nn.Module):
    """
    A direct implementation of the architecture from the "Linguistically Motivated 
    Sign Language Segmentation" paper for baseline ablation testing.
    
    Architecture:
    - Linear feature projection (No GCN, matching their "Feature Engineering" approach)
    - 4-Layer Bidirectional LSTM
    - Hidden Size: 512 (256 per direction)
    - Dropout: 0.2
    - Linear classification head for BIO tags
    """
    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=256, n_layers=4, dropout=0.2):
        super(BiLSTM_Baseline, self).__init__()
        
        # 1. Feature Projection (Flatten spatial nodes, mimicking their MLP setup)
        # Input shape from dataset: (Batch, Channels, Frames, Vertices)
        # Flattened size per frame = in_channels * num_vertices
        self.feature_dim = in_channels * num_vertices
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Sequence Encoder (4-Layer BiLSTM)
        # We use d_model as the hidden size per direction, so the total hidden state is d_model * 2
        self.lstm = nn.LSTM(
            input_size=d_model * 2,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # 3. Output Head
        # Output of BiLSTM is (Batch, Seq_Len, d_model * 2)
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        # Current x shape: (B, C, T, V)
        B, C, T, V = x.shape
        
        # Step 1: Flatten spatial dimension and move time to seq_len position
        # (B, C, T, V) -> (B, T, C, V) -> (B, T, C * V)
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)
        
        # Step 2: Linear Projection
        features = self.projection(x)  # Shape: (B, T, d_model * 2)
        
        # Step 3: BiLSTM Sequence Encoding
        # lstm_out shape: (B, T, hidden_size * 2)
        lstm_out, _ = self.lstm(features)
        
        # Step 4: Classification
        logits = self.classifier(lstm_out)  # Shape: (B, T, num_classes)
        
        # 🚨 THE FIX: Permute logits to (B, C, T) to match PyTorch's native sequence CrossEntropyLoss!
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)
    

class STGCN_BiLSTM(nn.Module):
    """
    Hybrid Architecture:
    Extracts spatial relationships using Graph Convolutions, then uses a BiLSTM 
    for sequence modeling. Directly bridges the gap between GCNs and baseline LSTMs.
    """
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4, num_classes=3, dropout=0.2):
        super(STGCN_BiLSTM, self).__init__()
        
        # 1. Graph Convolutional Front-End
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        
        # 2. Bridge (Match the BiLSTM Baseline's entry dimensions)
        self.bridge_dim = num_vertices * stgcn_channels
        self.projection = nn.Sequential(
            nn.Linear(self.bridge_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. BiLSTM Sequence Back-End
        self.lstm = nn.LSTM(
            input_size=d_model * 2,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # 4. Output Head
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        
        # Phase 1: GCN extracts spatial kinematics
        x = self.stgcn_blocks(x) # Shape: (B, 64, T, 65)
        
        # Phase 2: Flatten spatial graphs into 1D vectors per frame
        x = x.permute(0, 2, 3, 1).contiguous() # (B, T, V, C)
        x = x.view(B, T, -1)                   # (B, T, V*C)
        features = self.projection(x)          # (B, T, d_model * 2)
        
        # Phase 3: BiLSTM Sequence Modeling
        lstm_out, _ = self.lstm(features)      # (B, T, d_model * 2)
        
        # Phase 4: Classification
        logits = self.classifier(lstm_out)     # (B, T, num_classes)
        
        # Match PyTorch CrossEntropy sequence signature: (B, C, T)
        return logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)