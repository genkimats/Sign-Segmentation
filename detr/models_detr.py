"""
DETR-style set-prediction model for sign/phrase segmentation. Instead of
classifying every frame (BIO tagging, everything else in this codebase), a
Transformer ENCODER contextualizes the full, unchunked sequence, and a small
FIXED set of learned "segment queries" cross-attend to that encoded sequence
via a Transformer DECODER -- each query directly predicts ONE (confidence,
start, end) segment. Trained via bipartite (Hungarian) matching against
ground-truth segments -- see detr_loss.py.

This is a genuinely different paradigm from every other model in this
codebase: no window size, no overlap, no per-frame BIO smoothing, no
chunked/streaming inference distinction. The whole point is to retire that
entire class of problem by operating on the full sequence directly.

Encoder: the same STGCNBlock spatial encoder used everywhere else in this
codebase, then a standard Transformer encoder stack (full self-attention --
this pairs naturally with a DETR-style decoder, matching how the actual
DETR/TadTR/ActionFormer lineage is built). Optional HaMeR/DINOv2 fusion,
same pattern as every other model here.

Designed for batch_size=1 -- see dataset_segments.py's docstring for why.
"""
import os
import sys
import torch
import torch.nn as nn

# This file lives in Sign-Segmentation/detr/, but needs the existing
# Sign-Segmentation/src/ package (SkeletonGraph, STGCNBlock,
# PositionalEncoding), which is NOT moving into detr/. Add the project root
# to sys.path explicitly, computed from THIS FILE's own location, so this
# resolves correctly no matter what directory the terminal is in when the
# importing script (train_detr.py) is actually run.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock
from src.models import PositionalEncoding


class STGCN_DETR(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256,
                 num_encoder_layers=4, num_decoder_layers=4, num_queries=100,
                 nhead=8, dim_feedforward=1024, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128,
                 pos_encoding_max_len=50000):
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

        # max_len set generously past your longest known video (51129 frames from
        # the inference-scaling results) -- bump pos_encoding_max_len if a longer
        # video ever shows up; this buffer caused a real crash once already
        # (see the DINOv2 streaming-inference bug from a few messages back).
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=pos_encoding_max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Learned query embeddings -- one per potential segment, matching DETR's
        # learned "object queries" convention (here: "segment queries"). Must
        # comfortably exceed the maximum number of signs/phrases in any single
        # video in your data -- see the training script's setup check for this.
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Prediction heads, applied per query.
        self.confidence_head = nn.Linear(d_model, 1)  # logit: is this query a real segment?
        self.span_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)  # (start_frac, end_frac), pre-sigmoid
        )

    def forward(self, x, hamer=None, dinov2=None):
        # x: (1, C, T, V) -- batch_size=1, see module docstring
        B, C, T, V = x.shape
        assert B == 1, "STGCN_DETR is designed for batch_size=1 (one full video per forward pass)"

        feat = self.stgcn_blocks(x)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, T, -1)

        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            hamer_feat = self.hamer_encoder(hamer.permute(0, 2, 1))
            feat = torch.cat([feat, hamer_feat], dim=-1)

        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            dinov2_feat = self.dinov2_encoder(dinov2.permute(0, 2, 1))
            feat = torch.cat([feat, dinov2_feat], dim=-1)

        feat = self.feature_proj(feat + 1e-5)
        feat = self.pos_encoder(feat)  # (1, T, d_model)

        memory = self.encoder(feat)  # (1, T, d_model)

        queries = self.query_embed.unsqueeze(0)  # (1, num_queries, d_model)
        decoded = self.decoder(queries, memory)  # (1, num_queries, d_model)

        confidence_logits = self.confidence_head(decoded).squeeze(-1)  # (1, num_queries)
        spans = torch.sigmoid(self.span_head(decoded))                 # (1, num_queries, 2), each in [0,1]

        return confidence_logits, spans