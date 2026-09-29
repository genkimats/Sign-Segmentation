"""
Two-stage latent-space Transformer, following Sign-Mamba's actual mechanism
(Feng et al., ICASSP 2025) rather than this project's existing
Latent_STGCN_Mamba -- which, on inspection, is a simple linear bottleneck
with no reconstruction objective, not a faithful implementation of what that
paper describes. See the chat discussion for the full comparison.

Sign-Mamba's latent space is the OUTPUT of a self-supervised RECONSTRUCTION
autoencoder (Eq. 4-5 in the paper): an encoder maps input to a latent
sequence, a symmetric decoder reconstructs the input from it, trained with
MSE, with no labels involved. Their own ablation (Table II) confirms this
mechanism helps a Transformer backbone too, not just Mamba (T w/o latent:
9.45 BLEU1, T latent: 14.07) -- this file builds that Transformer version
for the segmentation task instead of their generation task.

Two classes:
  TransformerReconstructionAutoencoder -- Stage 1. Encoder: STGCN spatial
    blocks (this project's existing spatial prior) -> Transformer encoder
    stack -> latent. Decoder: a symmetric Transformer stack (self-attention
    only, no cross-attention -- this is a pure autoencoder, not seq2seq) ->
    projects back to raw input coordinates. Trained with MSE reconstruction
    loss only; needs no BIO labels at all.
  TransformerLatentSegmenter -- Stage 2. Rebuilds the SAME encoder
    architecture, loads Stage 1's pretrained encoder weights, optionally
    freezes them (matching the paper's own "freeze the pretrained half,
    train the new half" pattern -- they freeze S-Decoder during their joint
    training; here it's the mirror image, freezing the encoder while
    training a new classification head), then attaches a BIO classifier.
    HaMeR/DINOv2 fusion happens AFTER the pretrained latent, not before --
    Stage 1 never saw those features, so fusing them into the frozen
    encoder's input would just be discarded by weights that never learned
    to use them.

This file lives in Sign-Segmentation/latent_transformer/, and reaches back
into the existing Sign-Segmentation/src/ package the same way the DETR and
multitask work did.
"""
import os
import sys
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock
from src.models import PositionalEncoding


class TransformerReconstructionAutoencoder(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256,
                 n_layers=4, nhead=8, dim_feedforward=1024, dropout=0.2, pos_encoding_max_len=5000):
        super().__init__()
        self.num_vertices = num_vertices
        self.in_channels = in_channels

        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        flat_dim = num_vertices * stgcn_channels

        self.encoder_proj = nn.Sequential(
            nn.Linear(flat_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=pos_encoding_max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder: a SEPARATE Transformer stack, self-attention only -- this
        # mirrors the paper's own symmetric-but-simple design (their decoder
        # is a reversed Mamba stack, no cross-attention either, since this is
        # reconstructing the SAME sequence, not translating between two).
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=pos_encoding_max_len)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.decoder_proj = nn.Linear(d_model, num_vertices * in_channels)

    def forward(self, x):
        # x: (B, C, T, V) raw input, same convention as every other model here.
        B, C, T, V = x.shape
        raw_target = x.permute(0, 2, 3, 1).reshape(B, T, V * C)  # what we're reconstructing

        feat = self.stgcn_blocks(x)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, T, -1)
        feat = self.encoder_proj(feat)
        feat = self.pos_encoder(feat)
        latent = self.transformer_encoder(feat)  # (B, T, d_model) -- THE latent

        dec = self.pos_decoder(latent)
        dec = self.transformer_decoder(dec)
        reconstruction = self.decoder_proj(dec)  # (B, T, V*C)

        return reconstruction, raw_target, latent


class TransformerLatentSegmenter(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256,
                 n_layers=4, nhead=8, dim_feedforward=1024, dropout=0.2, num_classes=3,
                 pos_encoding_max_len=5000, freeze_encoder=True,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128):
        super().__init__()
        self.freeze_encoder = freeze_encoder

        # Rebuilds EXACTLY the same encoder architecture as
        # TransformerReconstructionAutoencoder, so a Stage 1 state dict loads
        # cleanly with strict=True -- no decoder here, since fine-tuning
        # never needs to reconstruct anything.
        graph = SkeletonGraph(num_vertices=num_vertices)
        A = graph.A
        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, stgcn_channels, A),
            STGCNBlock(stgcn_channels, stgcn_channels, A)
        )
        flat_dim = num_vertices * stgcn_channels
        self.encoder_proj = nn.Sequential(
            nn.Linear(flat_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=pos_encoding_max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # HaMeR/DINOv2 fusion happens AFTER the pretrained latent -- see module
        # docstring for why fusing before it would be pointless.
        self.hamer_dim = hamer_dim
        if hamer_dim is not None:
            self.hamer_encoder = nn.Sequential(
                nn.Linear(hamer_dim, hamer_proj_dim), nn.LayerNorm(hamer_proj_dim),
                nn.GELU(), nn.Dropout(dropout)
            )
        self.dinov2_dim = dinov2_dim
        if dinov2_dim is not None:
            self.dinov2_encoder = nn.Sequential(
                nn.Linear(dinov2_dim, dinov2_proj_dim), nn.LayerNorm(dinov2_proj_dim),
                nn.GELU(), nn.Dropout(dropout)
            )

        post_latent_dim = d_model
        if hamer_dim is not None:
            post_latent_dim += hamer_proj_dim
        if dinov2_dim is not None:
            post_latent_dim += dinov2_proj_dim

        if post_latent_dim != d_model:
            self.post_fusion_proj = nn.Sequential(
                nn.Linear(post_latent_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
            )
        else:
            self.post_fusion_proj = nn.Identity()

        self.classifier = nn.Linear(d_model, num_classes)

        if freeze_encoder:
            for p in self.stgcn_blocks.parameters(): p.requires_grad = False
            for p in self.encoder_proj.parameters(): p.requires_grad = False
            for p in self.transformer_encoder.parameters(): p.requires_grad = False

    def load_pretrained_encoder(self, checkpoint_path, map_location=None, strict=True):
        """Loads a TransformerReconstructionAutoencoder checkpoint's encoder
        weights (stgcn_blocks, encoder_proj, transformer_encoder only -- the
        decoder/decoder_proj keys in that checkpoint are simply absent from
        this model and are ignored)."""
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        encoder_prefixes = ("stgcn_blocks.", "encoder_proj.", "transformer_encoder.")
        encoder_state = {k: v for k, v in state.items() if k.startswith(encoder_prefixes)}
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        # Only the encoder keys should be missing-from-the-filtered-state (none, since
        # we filtered to exactly those prefixes) -- classifier/hamer/dinov2/post_fusion_proj
        # are EXPECTED to be reported missing here, since Stage 1 never had them.
        unexpected_encoder_keys = [k for k in unexpected if k.startswith(encoder_prefixes)]
        if strict and unexpected_encoder_keys:
            raise RuntimeError(f"Unexpected encoder keys not consumed: {unexpected_encoder_keys}")
        return missing, unexpected

    def _encode(self, x):
        feat = self.stgcn_blocks(x)
        B, _, T, _ = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(feat.shape[0], feat.shape[2], -1)
        feat = self.encoder_proj(feat)
        feat = self.pos_encoder(feat)
        return self.transformer_encoder(feat)

    def forward(self, x, hamer=None, dinov2=None):
        if self.freeze_encoder:
            with torch.no_grad():
                latent = self._encode(x)
        else:
            latent = self._encode(x)

        combined = latent
        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("This model was built with hamer_dim set, but forward() "
                                  "was called without a `hamer` tensor.")
            combined = torch.cat([combined, self.hamer_encoder(hamer.permute(0, 2, 1))], dim=-1)
        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("This model was built with dinov2_dim set, but forward() "
                                  "was called without a `dinov2` tensor.")
            combined = torch.cat([combined, self.dinov2_encoder(dinov2.permute(0, 2, 1))], dim=-1)

        combined = self.post_fusion_proj(combined)
        logits = self.classifier(combined)
        return logits.permute(0, 2, 1), combined.permute(0, 2, 1)