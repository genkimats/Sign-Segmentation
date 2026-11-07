"""
Re-implementation of the 2026 architecture of Moryossef et al. ("CNN-medium-attn + RoPE"),
sign-language-processing/segmentation, commit 22ca3a6 (2026-04-29):
  sign_language_segmentation/model/pose_encoder.py   PoseEncoderUNetBlock (ported unchanged)
  sign_language_segmentation/model/model.py          RoPETransformerEncoderLayer, ClassifierHead,
                                                     PoseTaggingModel.frame_cnn / encode / heads
Architecture only: their training recipe (Dice loss, fps / frame / hand dropout augmentation,
velocity channels, 1024-frame windows, AdamW one-cycle at 1e-3) is NOT copied; every run trains
with this repository's pipeline (train_phrase.py), so the comparison is about the network.

  Moryossef2026           the whole model on our keypoints: per joint a temporal UNet over the
                          coordinates (stage 1, "spatial compression"), Linear(joints -> hidden),
                          a second UNet over time that treats each hidden unit as a 1-channel
                          signal (stage 2, "temporal context"), RMSNorm, N pre-norm RoPE
                          transformer layers, an MLP phrase head and (sign_head=True) an MLP sign
                          head -- as in the original, which trains both heads jointly.
                          Our additions (off by default): hamer_dim -> the HaMeR vector is
                          projected and fused with the stage-1 output before stage 2.
  UNetRoPETemporalEncoder stage 2 + RMSNorm + RoPE layers only: the temporal part of the 2026
                          model, used as the encoder of the multi-stream model
                          (MultiStreamSegmenter encoder="unet_rope") in place of the BiLSTM.

Defaults are the 2026 training command (README): hidden 384, depth 4, 8 heads, FFN x2, dropout 0.1.
"""
from collections import namedtuple
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

ConvDef = namedtuple("ConvDef", ["in_channels", "out_channels", "kernel_size", "stride"])

# their stage-1 (per joint, over the coordinate channels) and stage-2 (per hidden unit) convolutions
STAGE1_CONVS = lambda c_in: [ConvDef(c_in, 16, 5, 1), ConvDef(16, 32, 11, 1), ConvDef(32, 64, 21, 2)]   # noqa: E731
STAGE2_CONVS = [ConvDef(1, 16, 5, 1), ConvDef(16, 32, 11, 2), ConvDef(32, 64, 21, 2), ConvDef(64, 128, 21, 2)]


class _RMSNorm(nn.Module):
    """nn.RMSNorm for torch < 2.4 (same maths: x / rms(x) * weight, eps = finfo eps)."""

    def __init__(self, dim, eps=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight


RMSNorm = getattr(nn, "RMSNorm", _RMSNorm)


class PoseEncoderUNetBlock(nn.Module):
    """
    Ported unchanged. x (B, T, N, C): every one of the N inputs is a C-channel signal over time;
    a 1D conv encoder (stride s halves T) and a transposed-conv decoder with additive skips, applied
    to each input separately (shared weights); the output channels are averaged, then Linear(N -> out).
    Returns (B, T, out).
    """

    def __init__(self, input_size: int, output_size: int, convolutions: List[ConvDef]):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for conv in convolutions:
            assert conv.kernel_size % 2 == 1, "Temporal window (kernel size) must be odd"
            assert conv.stride & (conv.stride - 1) == 0, "Stride must be a power of 2"
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                          stride=conv.stride, padding=conv.kernel_size // 2),
                nn.BatchNorm1d(conv.out_channels),
                nn.SiLU()))
        self.decoder_layers = nn.ModuleList()
        for conv in reversed(convolutions):
            if conv.stride not in (1, 2, 4, 8):
                raise ValueError(f"Stride {conv.stride} not supported.")
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose1d(conv.out_channels, conv.in_channels, kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.kernel_size // 2),
                nn.BatchNorm1d(conv.in_channels),
                nn.SiLU()))
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)
        for layer in self.decoder_layers:
            skip = skips.pop()
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (diff // 2, diff - diff // 2))
            x = layer(x + skip)
        x = x.reshape(B, N, x.shape[1], x.shape[-1]).permute(0, 3, 1, 2)    # (B, T', N, C_out)
        return self.fc(x.mean(dim=-1))


class RoPETransformerEncoderLayer(nn.Module):
    """Ported unchanged: pre-norm (RMSNorm) self-attention + GELU FFN, RoPE on Q / K.
    Positions are frame indices (the original scales seconds by 50 fps -> the same at 50 fps)."""

    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout=0.1, rope_base=10000.0):
        super().__init__()
        assert hidden_dim % nhead == 0, f"hidden_dim {hidden_dim} must be divisible by nhead {nhead}"
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_p = float(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, hidden_dim), nn.Dropout(dropout))
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, positions=None):
        B, T, D = x.shape
        if positions is None:
            positions = torch.arange(T, device=x.device, dtype=torch.float32)
        freqs = positions.float()[..., None] * self.inv_freq                 # (T, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        h = self.norm1(x)
        q, k, v = self.qkv(h).reshape(B, T, 3, self.nhead, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)     # (B, H, T, hd)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p if self.training else 0.0)
        x = x + self.out_proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ffn(self.norm2(x))


class ClassifierHead(nn.Module):
    """Ported unchanged: Linear -> GELU -> Linear."""

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.net(x)


class UNetRoPETemporalEncoder(nn.Module):
    """The temporal part of the 2026 model: (B, T, D) -> stage-2 UNet -> RMSNorm -> RoPE layers -> (B, T, D)."""

    def __init__(self, dim, n_layers=4, nhead=8, ff_mult=2, dropout=0.1, rope_base=10000.0, unet=True):
        super().__init__()
        self.unet = PoseEncoderUNetBlock(dim, dim, STAGE2_CONVS) if unet else None
        self.norm = RMSNorm(dim)
        self.layers = nn.ModuleList([RoPETransformerEncoderLayer(dim, nhead, dim * ff_mult, dropout, rope_base)
                                     for _ in range(n_layers)])

    def forward(self, x):
        if self.unet is not None:
            x = self.unet(x[..., None])                                       # each unit = 1-channel signal
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Moryossef2026(nn.Module):
    """
    The 2026 model on this repository's inputs. x (B, C, T, V): V joints (65 body + hands, then the
    face subset), C coordinate channels (2D face: z = 0, as for every other model here).
    forward() returns (phrase logits (B, 3, T), encoder output (B, d_model, T)); with sign_head=True
    the sign logits are left in self._sign_logits (B, 3, T) for train_phrase.py's sign loss.
    d_model = their hidden_dim, n_layers = their encoder_depth.
    """

    def __init__(self, in_channels, num_vertices, num_classes=3, d_model=384, n_layers=4, nhead=8,
                 attn_ff_mult=2, attn_dropout=0.1, rope_base=10000.0, hamer_dim=None, sign_head=False,
                 use_pose_input=True, **unused):
        super().__init__()
        if not use_pose_input:
            raise ValueError("moryossef_2026 needs the keypoints (use_pose_input=True).")
        self.stage1 = PoseEncoderUNetBlock(num_vertices, d_model, STAGE1_CONVS(in_channels))
        self.hamer_dim = int(hamer_dim) if hamer_dim else 0
        if self.hamer_dim:   # our addition: HaMeR joins the per-frame vector before the temporal stage
            self.hamer_proj = nn.Linear(self.hamer_dim, d_model)
            self.hamer_fuse = nn.Linear(2 * d_model, d_model)
        self.encoder = UNetRoPETemporalEncoder(d_model, n_layers, nhead, attn_ff_mult, attn_dropout, rope_base)
        self.phrase_head = ClassifierHead(d_model, num_classes)
        self.sign_head = bool(sign_head)
        if self.sign_head:
            self.sign_classifier = ClassifierHead(d_model, 3)
        self._sign_logits = None
        self._aux_logits = {}
        self._extra_losses = {}

    def forward(self, x, hamer=None, **unused):
        B, C, T, V = x.shape
        h = self.stage1(x.permute(0, 2, 3, 1))                                # (B, T, d_model)
        if self.hamer_dim:
            if hamer is None:
                raise ValueError("moryossef_2026 was built with HaMeR input but got hamer=None.")
            hm = self.hamer_proj(hamer.permute(0, 2, 1)[:, :T])
            h = self.hamer_fuse(torch.cat([h, hm], dim=-1))
        emb = self.encoder(h)
        self._sign_logits = self.sign_classifier(emb).permute(0, 2, 1) if self.sign_head else None
        return self.phrase_head(emb).permute(0, 2, 1), emb.permute(0, 2, 1)
