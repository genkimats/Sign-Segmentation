"""
Multi-task variants of this project's most-tested architectures: identical
STGCN spatial encoder + temporal backbone as the existing STGCN_BiLSTM /
STGCN_Transformer / STGCN_BiMamba, with a SECOND classification head added
on top of the SAME shared embeddings -- predicting gloss identity alongside
the existing BIO (Begin/Inside/Outside) segmentation output.

The gloss head is training-time scaffolding, not a deployed recognizer: see
the design discussion in chat for why an auxiliary, imperfect gloss signal
can still be useful for shaping the shared representation even though a
dedicated downstream recognition model would do much better at gloss
classification itself. At inference, you can ignore the gloss head entirely
and use bio_logits exactly as with every other model in this project.

This file lives in Sign-Segmentation/multitask_learning/, and reaches back
into the existing Sign-Segmentation/src/ package (SkeletonGraph, STGCNBlock,
PositionalEncoding) via sys.path, the same pattern used for the DETR work.
"""
import os
import sys
import torch
import torch.nn as nn
from mamba_ssm import Mamba

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock
from src.models import PositionalEncoding


def _build_shared_encoder_prefix(num_vertices, in_channels, stgcn_channels, d_model, dropout,
                                  hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim):
    """Builds the STGCN spatial encoder + optional HaMeR/DINOv2 fusion +
    feature_proj stack shared by every multi-task model below -- exactly the
    same construction used throughout models.py, factored out here since all
    three backbones below share it verbatim."""
    graph = SkeletonGraph(num_vertices=num_vertices)
    A = graph.A
    stgcn_blocks = nn.Sequential(
        STGCNBlock(in_channels, stgcn_channels, A),
        STGCNBlock(stgcn_channels, stgcn_channels, A)
    )
    bridge_dim = num_vertices * stgcn_channels

    hamer_encoder = None
    if hamer_dim is not None:
        hamer_encoder = nn.Sequential(
            nn.Linear(hamer_dim, hamer_proj_dim), nn.LayerNorm(hamer_proj_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        bridge_dim += hamer_proj_dim

    dinov2_encoder = None
    if dinov2_dim is not None:
        dinov2_encoder = nn.Sequential(
            nn.Linear(dinov2_dim, dinov2_proj_dim), nn.LayerNorm(dinov2_proj_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        bridge_dim += dinov2_proj_dim

    feature_proj = nn.Sequential(
        nn.Linear(bridge_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
    )
    return stgcn_blocks, hamer_encoder, dinov2_encoder, feature_proj


class STGCN_BiLSTM_Multitask(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4,
                 num_classes=3, gloss_vocab_size=1000, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128):
        super().__init__()
        self.hamer_dim = hamer_dim
        self.dinov2_dim = dinov2_dim
        (self.stgcn_blocks, self.hamer_encoder, self.dinov2_encoder,
         self.feature_proj) = _build_shared_encoder_prefix(
            num_vertices, in_channels, stgcn_channels, d_model, dropout,
            hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim
        )

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=True
        )
        self.bio_classifier = nn.Linear(d_model * 2, num_classes)
        self.gloss_classifier = nn.Linear(d_model * 2, gloss_vocab_size)

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        feat = self.stgcn_blocks(x)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            feat = torch.cat([feat, self.hamer_encoder(hamer.permute(0, 2, 1))], dim=-1)
        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            feat = torch.cat([feat, self.dinov2_encoder(dinov2.permute(0, 2, 1))], dim=-1)

        features = self.feature_proj(feat + 1e-5)
        lstm_out, _ = self.lstm(features)

        bio_logits = self.bio_classifier(lstm_out)
        gloss_logits = self.gloss_classifier(lstm_out)
        return bio_logits.permute(0, 2, 1), gloss_logits.permute(0, 2, 1), lstm_out.permute(0, 2, 1)


class STGCN_Transformer_Multitask(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4,
                 num_classes=3, gloss_vocab_size=1000, nhead=8, dim_feedforward=1024, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128):
        super().__init__()
        self.hamer_dim = hamer_dim
        self.dinov2_dim = dinov2_dim
        (self.stgcn_blocks, self.hamer_encoder, self.dinov2_encoder,
         self.feature_proj) = _build_shared_encoder_prefix(
            num_vertices, in_channels, stgcn_channels, d_model, dropout,
            hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.bio_classifier = nn.Linear(d_model, num_classes)
        self.gloss_classifier = nn.Linear(d_model, gloss_vocab_size)

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        feat = self.stgcn_blocks(x)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            feat = torch.cat([feat, self.hamer_encoder(hamer.permute(0, 2, 1))], dim=-1)
        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            feat = torch.cat([feat, self.dinov2_encoder(dinov2.permute(0, 2, 1))], dim=-1)

        features = self.feature_proj(feat + 1e-5)
        features = self.pos_encoder(features)
        embeddings = self.transformer_encoder(features)

        bio_logits = self.bio_classifier(embeddings)
        gloss_logits = self.gloss_classifier(embeddings)
        return bio_logits.permute(0, 2, 1), gloss_logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)


class STGCN_BiMamba_Multitask(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4,
                 num_classes=3, gloss_vocab_size=1000, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.hamer_dim = hamer_dim
        self.dinov2_dim = dinov2_dim
        (self.stgcn_blocks, self.hamer_encoder, self.dinov2_encoder,
         self.feature_proj) = _build_shared_encoder_prefix(
            num_vertices, in_channels, stgcn_channels, d_model, dropout,
            hamer_dim, hamer_proj_dim, dinov2_dim, dinov2_proj_dim
        )

        self.mamba_fwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(n_layers)
        ])
        self.mamba_bwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(n_layers)
        ])
        self.bio_classifier = nn.Linear(d_model * 2, num_classes)
        self.gloss_classifier = nn.Linear(d_model * 2, gloss_vocab_size)

    def forward(self, x, hamer=None, dinov2=None):
        B, C, T, V = x.shape
        feat = self.stgcn_blocks(x)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            feat = torch.cat([feat, self.hamer_encoder(hamer.permute(0, 2, 1))], dim=-1)
        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            feat = torch.cat([feat, self.dinov2_encoder(dinov2.permute(0, 2, 1))], dim=-1)

        features = self.feature_proj(feat + 1e-5)

        fwd_emb = features
        bwd_emb = torch.flip(features, dims=[1])
        for fwd_layer, bwd_layer in zip(self.mamba_fwd, self.mamba_bwd):
            fwd_emb = fwd_layer(fwd_emb)
            bwd_emb = bwd_layer(bwd_emb)
        bwd_emb = torch.flip(bwd_emb, dims=[1])
        embeddings = torch.cat([fwd_emb, bwd_emb], dim=-1)

        bio_logits = self.bio_classifier(embeddings)
        gloss_logits = self.gloss_classifier(embeddings)
        return bio_logits.permute(0, 2, 1), gloss_logits.permute(0, 2, 1), embeddings.permute(0, 2, 1)