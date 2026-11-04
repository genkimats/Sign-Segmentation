"""
fusion_layers.py -- the building blocks of the fusion study (torch).

  FrameGate          idea 1  per-frame softmax weights over the streams, computed from the streams
  LocalLagAttention  idea 2  each query frame attends to the face stream within +-K frames, with a
                             learned per-offset bias -> the model can learn that non-manual cues
                             lead or lag the manual signal
  FaceFiLM           idea 5  the face stream produces a per-frame, per-channel scale and shift that
                             modulates another stream (default: body+hands)
  StreamHead         idea 3  a per-stream BiLSTM + B/I/O classifier for late fusion

Every block that changes the original model's computation starts as (close to) a no-op:
FrameGate starts with uniform weights (= a plain sum, like gated_sum with all gates 1), FaceFiLM
starts as the identity (zero-initialised output layer), LocalLagAttention starts with a mild
preference for offset 0 (bias -slope*|offset|). Each block leaves what an analysis needs in
`self.trace` (detached tensors), which MultiStream_Fusion collects into model._trace.
"""
import math

import torch
import torch.nn as nn


def shift_time(x, d):
    """y[:, t] = x[:, t + d] (zeros where t + d is outside the sequence). x: (B, T, ...)."""
    if d == 0:
        return x
    T = x.shape[1]
    if abs(d) >= T:
        return torch.zeros_like(x)
    pad = torch.zeros_like(x[:, :abs(d)])
    if d > 0:
        return torch.cat([x[:, d:], pad], dim=1)
    return torch.cat([pad, x[:, :d]], dim=1)


class DepthwiseTemporal(nn.Module):
    """Residual depthwise temporal convolution on (B, T, C), identity at initialisation."""

    def __init__(self, channels, kernel):
        super().__init__()
        k = int(kernel) | 1
        self.conv = nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


class FrameGate(nn.Module):
    """
    Idea 1. Input: the stream features z_s (B, T, D) (all already projected to the same width D).
    w(t) = softmax_s( MLP([z_1(t), ..., z_S(t)]) / temperature ), with a residual depthwise temporal
    convolution inside the MLP so the gate sees a short context (default 9 frames, ~0.2 s).
    Output: sum_s S * w_s(t) * z_s(t)  -- at initialisation w = 1/S, i.e. the plain sum.
    trace["gate"]: (B, T, S).
    """

    def __init__(self, n_streams, dim, hidden=64, temporal_kernel=9, temperature=1.0):
        super().__init__()
        self.S, self.temperature = n_streams, float(temperature)
        self.inp = nn.Sequential(nn.Linear(n_streams * dim, hidden), nn.GELU())
        self.temporal = DepthwiseTemporal(hidden, temporal_kernel) if temporal_kernel and temporal_kernel > 1 else None
        self.out = nn.Linear(hidden, n_streams)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.trace = {}

    def forward(self, zs):
        h = self.inp(torch.cat(zs, dim=-1))
        if self.temporal is not None:
            h = self.temporal(h)
        w = torch.softmax(self.out(h) / self.temperature, dim=-1)              # (B, T, S)
        self.trace = {"gate": w.detach()}
        fused = sum(self.S * w[..., i:i + 1] * z for i, z in enumerate(zs))
        return fused, w


class LocalLagAttention(nn.Module):
    """
    Idea 2. For every frame t, a query from `q_src` (B, T, Dq) attends to the face features
    `kv_src` (B, T, Dkv) at frames t + d, d in [-K, K]:

        logit_h(t, d) = <q_h(t), k_h(t + d)> / sqrt(dh) + bias_h(d)     (attention="content")
        logit_h(t, d) = bias_h(d)                                         (attention="fixed")
        out(t)        = W_o [ sum_d softmax_d(logit) v_h(t + d) ]_h  -> LayerNorm   (B, T, out_dim)

    "fixed" is the content-free ablation: one learned temporal filter (a learned lag) per head.
    Offsets that fall outside the window are masked; d = 0 is always valid, so synchronous fusion
    is a special case. The loop over the 2K+1 offsets keeps memory at O(B T H (2K+1)) instead of
    materialising a (B, T, 2K+1, D) tensor.
    trace["lag_attn"]: (B, T, 2K+1) attention averaged over heads; trace["lag_mean"]: (B, T)
    expected offset in frames (+ = the face information used comes AFTER the query frame).
    """

    def __init__(self, q_dim, kv_dim, out_dim, max_lag=25, heads=4, head_dim=32, attention="content",
                 bias_slope=0.1, dropout=0.0):
        super().__init__()
        if attention not in ("content", "fixed"):
            raise ValueError(f"attention must be 'content' or 'fixed', got {attention!r}")
        self.K, self.H, self.dh, self.attention = int(max_lag), int(heads), int(head_dim), attention
        inner = self.H * self.dh
        self.q = nn.Linear(q_dim, inner) if attention == "content" else None
        self.k = nn.Linear(kv_dim, inner) if attention == "content" else None
        self.v = nn.Linear(kv_dim, inner)
        self.o = nn.Linear(inner, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        offsets = torch.arange(-self.K, self.K + 1, dtype=torch.float32)
        self.register_buffer("offsets", offsets, persistent=False)
        self.bias = nn.Parameter((-float(bias_slope) * offsets.abs()).repeat(self.H, 1))   # (H, 2K+1)
        self.trace = {}

    def forward(self, q_src, kv_src):
        B, T, _ = kv_src.shape
        H, dh = self.H, self.dh
        v = self.v(kv_src).view(B, T, H, dh)
        n_off = 2 * self.K + 1
        t_idx = torch.arange(T, device=kv_src.device)
        valid = torch.stack([((t_idx + d) >= 0) & ((t_idx + d) < T) for d in range(-self.K, self.K + 1)], dim=-1)
        if self.attention == "content":
            q = self.q(q_src).view(B, T, H, dh) / math.sqrt(dh)
            k = self.k(kv_src).view(B, T, H, dh)
            logits = torch.stack([(q * shift_time(k, d)).sum(-1) for d in range(-self.K, self.K + 1)], dim=-1)
            logits = logits + self.bias.view(1, 1, H, n_off)
        else:
            logits = self.bias.view(1, 1, H, n_off).expand(B, T, H, n_off)
        logits = logits.masked_fill(~valid.view(1, T, 1, n_off), float("-inf"))
        attn = torch.softmax(logits, dim=-1)                                     # (B, T, H, 2K+1)
        out = sum(attn[..., j:j + 1] * shift_time(v, d) for j, d in enumerate(range(-self.K, self.K + 1)))
        out = self.norm(self.o(self.drop(out.reshape(B, T, H * dh))))
        a = attn.mean(2).detach()
        self.trace = {"lag_attn": a, "lag_mean": (a * self.offsets).sum(-1)}
        return out


class FaceFiLM(nn.Module):
    """
    Idea 5. gamma(t), beta(t) = MLP(face(t)) (with a residual depthwise temporal conv), applied as
    target * (1 + gamma) + beta. The output layer is zero-initialised, so the block starts as the
    identity and the face can only ever act as a modulator of the target stream.
    trace["film_gamma"], trace["film_beta"]: (B, T) mean |gamma|, |beta| per frame.
    """

    def __init__(self, face_dim, target_dim, hidden=128, temporal_kernel=9):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(face_dim, hidden), nn.GELU())
        self.temporal = DepthwiseTemporal(hidden, temporal_kernel) if temporal_kernel and temporal_kernel > 1 else None
        self.out = nn.Linear(hidden, 2 * target_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.trace = {}

    def forward(self, face, target):
        h = self.inp(face)
        if self.temporal is not None:
            h = self.temporal(h)
        gamma, beta = self.out(h).chunk(2, dim=-1)
        self.trace = {"film_gamma": gamma.detach().abs().mean(-1), "film_beta": beta.detach().abs().mean(-1)}
        return target * (1 + gamma) + beta


class StreamHead(nn.Module):
    """Idea 3. One stream's own encoder for late fusion: BiLSTM -> Linear(3). (B, T, D) -> logits (B, 3, T)."""

    def __init__(self, in_dim, d_model=128, n_layers=4, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, d_model, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0, bidirectional=True)
        self.classifier = nn.Linear(2 * d_model, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.classifier(h).permute(0, 2, 1), h
