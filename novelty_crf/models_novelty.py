"""
models_novelty.py -- similarity-augmented, CRF-trained recurrent segmentation
encoder. The nn.Module wrappers here are deliberately thin: every piece of
non-trivial maths (CRF forward/Viterbi, mLSTM/sLSTM scans, similarity and
novelty features) lives in seg_cores.py and is unit-tested there against
brute-force references.

Pipeline (all at the full 50 fps frame rate -- no temporal downsampling):

  pose  -> STGCN x2 -> flatten -> Linear                       p_t  (d_model)
  hamer -> [temporal smoothing] -> MLP adapter                 s_t  (hamer_proj_dim)
  dino  -> MLP adapter                                         d_t  (dinov2_proj_dim)
  streams (p, s, d) -> cosine SSM per stream -> row similarity (+-K) and
        checkerboard novelty (several scales) -> Linear        sim_t (d_sim)
  concat(p, s, d, sim) -> mixer MLP                            (d_model)
  -> dilated residual conv blocks (local boundary sensor)
  -> context layer: BiLSTM | Bi-xLSTM | none
  -> LayerNorm -> Linear(3) emissions
  -> (optional) linear-chain CRF over {O, I, B} with O->I forbidden

Label convention (project-wide): 0 = Outside, 1 = Inside, 2 = Begin.
forward() returns (emissions (B,3,T), features (B,d_model,T)), the same
layout as every other model in this project, so the existing exporter and
decoder study can use it unchanged.
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seg_cores as C
from seg_ops import get_torch_ops
from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock

TX = get_torch_ops()


# ---------------------------------------------------------------- CRF layer --
class LinearChainCRF(nn.Module):
    """Linear-chain CRF over {O, I, B}. O->I is forbidden (a sign cannot start
    with 'Inside' after 'Outside') via a large fixed negative score, so the
    grammar is enforced in training AND decoding. Windows may legitimately
    start in I (a window can cut a sign), so the start scores are unconstrained."""

    def __init__(self, num_tags=3, forbidden=((C.O, C.I),), forbid_penalty=-20.0):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        pen = torch.tensor(C.forbid_mask(num_tags, forbidden) * forbid_penalty, dtype=torch.float32)
        self.register_buffer("forbid_penalty_matrix", pen)

    def effective_transitions(self):
        return self.transitions + self.forbid_penalty_matrix

    def nll(self, emissions, labels):
        """emissions (B,T,K) float, labels (B,T) ints -> scalar mean NLL per frame."""
        return C.crf_nll_per_frame(TX, emissions.float(), labels, self.effective_transitions(), self.start, self.end)

    @torch.no_grad()
    def decode(self, emissions):
        return C.crf_viterbi(TX, emissions.float(), self.effective_transitions(), self.start, self.end)

    def export_params(self):
        """Raw learned parameters (the O->I penalty is re-applied by every decoder)."""
        return {"transitions": self.transitions.detach().cpu().tolist(),
                "start": self.start.detach().cpu().tolist(), "end": self.end.detach().cpu().tolist()}


# ------------------------------------------------------------- xLSTM layers --
class MLSTMCell(nn.Module):
    """Matrix-memory xLSTM cell (causal). See seg_cores.mlstm_parallel."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by xlstm_heads"
        self.H, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.gate_i = nn.Linear(d_model, n_heads)
        self.gate_f = nn.Linear(d_model, n_heads)
        self.gate_o = nn.Linear(d_model, d_model)
        self.head_norm = nn.LayerNorm(self.dh)
        self.out = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.gate_i.bias.zero_()
            self.gate_f.bias.copy_(torch.linspace(3.0, 6.0, n_heads))   # start by remembering

    def forward(self, x):                                   # (B,T,D)
        B, T, D = x.shape

        def split(t):                                       # (B,T,D) -> (B,H,T,dh)
            return t.view(B, T, self.H, self.dh).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        i_pre = self.gate_i(x).transpose(1, 2)              # (B,H,T)
        f_pre = self.gate_f(x).transpose(1, 2)
        h = C.mlstm_parallel(TX, q, k, v, i_pre, f_pre)     # (B,H,T,dh)
        h = self.head_norm(h.transpose(1, 2)).reshape(B, T, D)
        return self.out(torch.sigmoid(self.gate_o(x)) * h)


class SLSTMCell(nn.Module):
    """Scalar-memory xLSTM cell with recurrent memory mixing (causal)."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by xlstm_heads"
        self.H, self.dh = n_heads, d_model // n_heads
        self.inp = nn.Linear(d_model, 4 * d_model)          # gates z, i, f, o
        self.R = nn.Parameter(torch.randn(4, n_heads, self.dh, self.dh) * (0.1 / math.sqrt(self.dh)))
        self.head_norm = nn.LayerNorm(self.dh)
        self.out = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.inp.bias[2 * d_model:3 * d_model].fill_(3.0)   # forget gate bias

    def forward(self, x):                                   # (B,T,D)
        B, T, D = x.shape
        pre = self.inp(x).view(B, T, 4, self.H, self.dh)
        h = C.slstm_scan(TX, pre, self.R)                   # (B,T,H,dh)
        return self.out(self.head_norm(h).reshape(B, T, D))


class BiXLSTMLayer(nn.Module):
    """Pre-norm residual block: bidirectional xLSTM cell (forward cell + a
    separate cell run on the time-reversed sequence) merged by a Linear, then a
    position-wise feed-forward. kind: 'm' (matrix memory) or 's' (scalar memory)."""

    def __init__(self, d_model, kind, n_heads=4, dropout=0.1, ffn_mult=2):
        super().__init__()
        cell = {"m": MLSTMCell, "s": SLSTMCell}[kind]
        self.norm1 = nn.LayerNorm(d_model)
        self.fwd = cell(d_model, n_heads)
        self.bwd = cell(d_model, n_heads)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        y = torch.cat([self.fwd(h), torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])], dim=-1)
        x = x + self.drop(self.merge(y))
        return x + self.drop(self.ffn(self.norm2(x)))


class BiXLSTMStack(nn.Module):
    def __init__(self, d_model, n_layers, pattern="mmms", n_heads=4, dropout=0.1):
        super().__init__()
        assert set(pattern) <= {"m", "s"} and len(pattern) > 0, "xlstm_pattern must be a string of 'm'/'s'"
        kinds = [pattern[i % len(pattern)] for i in range(n_layers)]
        self.layers = nn.ModuleList([BiXLSTMLayer(d_model, k, n_heads, dropout) for k in kinds])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class BiLSTMContext(nn.Module):
    def __init__(self, d_model, n_layers, dropout):
        super().__init__()
        assert d_model % 2 == 0
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=n_layers, bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, x):
        return self.lstm(x)[0]


# ------------------------------------------------ similarity / smoothing / MLP --
class SimilarityNovelty(nn.Module):
    """Per-frame relational features computed from one or more embedding
    streams: for each stream the cosine self-similarity matrix over the window
    yields (i) the similarity of each frame to its +-K neighbours (TransNetV2)
    and (ii) checkerboard-kernel novelty at several scales (Foote 2000). The
    embeddings are the LEARNED stream outputs, so gradients shape them into a
    space where 'different sign' means 'dissimilar'. Streams are kept separate
    so the model can see WHICH cue changed (handshape vs trajectory)."""

    def __init__(self, n_streams, K=16, scales=(2, 4, 8, 16), d_out=64, dropout=0.1):
        super().__init__()
        self.K, self.scales = int(K), tuple(int(s) for s in scales)
        feat = n_streams * ((2 * self.K + 1) + len(self.scales))
        self.proj = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, d_out), nn.GELU(), nn.Dropout(dropout))
        self._bands = {}

    def _get_bands(self, T):
        if T not in self._bands:
            self._bands[T] = [C.make_novelty_bands(T, L) for L in self.scales]
        return self._bands[T]

    def forward(self, streams):                             # list of (B,T,D_k)
        feats = []
        for e in streams:
            S = C.cosine_ssm(TX, e)
            feats.append(C.row_similarity(TX, S, self.K))
            feats.append(C.checkerboard_novelty(TX, S, self._get_bands(e.shape[1])))
        return self.proj(torch.cat(feats, dim=-1))


class TemporalSmooth(nn.Module):
    """Fixed depthwise Gaussian smoothing over time, replicate-padded. Applied to
    frozen per-frame hand estimates, whose frame-to-frame jitter would otherwise
    be amplified by the similarity/derivative features downstream."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "hamer_smooth must be odd (or 0 to disable)"
        r = kernel_size // 2
        xs = np.arange(-r, r + 1, dtype=np.float64)
        w = np.exp(-0.5 * (xs / (max(r, 1) / 2.0)) ** 2)
        w = w / w.sum()
        self.r, self.channels = r, channels
        self.register_buffer("weight", torch.tensor(w, dtype=torch.float32).view(1, 1, -1).repeat(channels, 1, 1))

    def forward(self, x):                                   # (B,C,T)
        if self.r == 0:
            return x
        return F.conv1d(F.pad(x, (self.r, self.r), mode="replicate"), self.weight, groups=self.channels)


def make_mlp(in_dim, out_dim, n_layers, dropout):
    layers, d = [], in_dim
    for _ in range(max(1, n_layers)):
        layers += [nn.Linear(d, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)]
        d = out_dim
    return nn.Sequential(*layers)


class LocalConvBlock(nn.Module):
    def __init__(self, d, dilation, dropout):
        super().__init__()
        assert d % 8 == 0
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # (B,d,T)
        return x + self.drop(F.gelu(self.norm(self.conv(x))))


# ---------------------------------------------------------------- main model --
class STGCN_Novelty(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256, n_layers=4,
                 backbone="xlstm", xlstm_pattern="mmms", xlstm_heads=4, dropout=0.2,
                 n_local_blocks=3, adapter_layers=2,
                 use_similarity=True, similarity_K=16, novelty_scales=(2, 4, 8, 16), d_sim=64,
                 use_crf=True, crf_forbid_penalty=-20.0, num_classes=3,
                 hamer_dim=None, hamer_proj_dim=128, hamer_smooth=0,
                 dinov2_dim=None, dinov2_proj_dim=128):
        super().__init__()
        if backbone not in ("bilstm", "xlstm", "none"):
            raise ValueError(f"backbone must be 'bilstm', 'xlstm' or 'none', got {backbone!r}")
        if num_classes != 3:
            raise ValueError("this model is BIO-specific (num_classes must be 3)")
        self.hamer_dim, self.dinov2_dim = hamer_dim, dinov2_dim

        A = SkeletonGraph(num_vertices=num_vertices).A
        self.stgcn_blocks = nn.Sequential(STGCNBlock(in_channels, stgcn_channels, A),
                                          STGCNBlock(stgcn_channels, stgcn_channels, A))
        self.pose_proj = nn.Sequential(nn.Linear(num_vertices * stgcn_channels, d_model), nn.LayerNorm(d_model),
                                       nn.GELU(), nn.Dropout(dropout))
        fused = d_model
        n_streams = 1
        if hamer_dim is not None:
            self.hamer_smooth = TemporalSmooth(hamer_dim, hamer_smooth) if hamer_smooth else None
            self.hamer_adapter = make_mlp(hamer_dim, hamer_proj_dim, adapter_layers, dropout)
            fused += hamer_proj_dim
            n_streams += 1
        if dinov2_dim is not None:
            self.dinov2_adapter = make_mlp(dinov2_dim, dinov2_proj_dim, adapter_layers, dropout)
            fused += dinov2_proj_dim
            n_streams += 1
        self.similarity = None
        if use_similarity:
            self.similarity = SimilarityNovelty(n_streams, similarity_K, novelty_scales, d_sim, dropout)
            fused += d_sim

        self.mixer = nn.Sequential(nn.Linear(fused, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.local_blocks = nn.ModuleList([LocalConvBlock(d_model, 2 ** i, dropout) for i in range(n_local_blocks)])
        if backbone == "bilstm":
            self.context = BiLSTMContext(d_model, n_layers, dropout)
        elif backbone == "xlstm":
            self.context = BiXLSTMStack(d_model, n_layers, xlstm_pattern, xlstm_heads, dropout)
        else:
            self.context = nn.Identity()
        self.out_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.crf = LinearChainCRF(num_classes, forbid_penalty=crf_forbid_penalty) if use_crf else None

    def forward(self, x, hamer=None, dinov2=None):
        B, _, T, _ = x.shape
        feat = self.stgcn_blocks(x).permute(0, 2, 3, 1).contiguous().view(B, T, -1)
        p = self.pose_proj(feat)                            # (B,T,d_model)
        streams, parts = [p], [p]
        if self.hamer_dim is not None:
            if hamer is None:
                raise ValueError("model built with hamer_dim but forward() got no `hamer` tensor")
            h = self.hamer_smooth(hamer) if self.hamer_smooth is not None else hamer
            s = self.hamer_adapter(h.permute(0, 2, 1))      # (B,T,hamer_proj_dim)
            streams.append(s); parts.append(s)
        if self.dinov2_dim is not None:
            if dinov2 is None:
                raise ValueError("model built with dinov2_dim but forward() got no `dinov2` tensor")
            d = self.dinov2_adapter(dinov2.permute(0, 2, 1))
            streams.append(d); parts.append(d)
        if self.similarity is not None:
            parts.append(self.similarity(streams))
        z = self.mixer(torch.cat(parts, dim=-1)).transpose(1, 2)    # (B,d,T)
        for blk in self.local_blocks:
            z = blk(z)
        z = self.context(z.transpose(1, 2))                 # (B,T,d)
        z = self.out_norm(z)
        emissions = self.classifier(z)                      # (B,T,3)
        return emissions.permute(0, 2, 1), z.permute(0, 2, 1)

    def decode(self, logits):
        """logits: the (B,3,T) tensor forward() returned -> (B,T) predicted tags
        (CRF Viterbi when the model has a CRF, plain argmax otherwise)."""
        E = logits.permute(0, 2, 1).float()
        return self.crf.decode(E) if self.crf is not None else E.argmax(-1)

    def export_extra(self):
        return {"crf": self.crf.export_params()} if self.crf is not None else {}