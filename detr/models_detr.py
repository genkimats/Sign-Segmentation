"""
DETR-style set-prediction model for sign/phrase segmentation. Instead of
classifying every frame (BIO tagging, everything else in this codebase), an
ENCODER contextualizes the full, unchunked sequence, and a small FIXED set of
learned "segment queries" cross-attend to that encoded sequence via a
Transformer DECODER -- each query directly predicts ONE (confidence, start,
end) segment. Trained via bipartite (Hungarian) matching against
ground-truth segments -- see detr_loss.py.

This is a genuinely different paradigm from every other model in this
codebase: no window size, no overlap, no per-frame BIO smoothing, no
chunked/streaming inference distinction. The whole point is to retire that
entire class of problem by operating on the full sequence directly.

CORRECTED DESIGN (see chat history): this originally used a standard
Transformer encoder (full self-attention, O(T^2) in sequence length). That
is categorically infeasible here -- this corpus's full, unchunked videos run
up to 100,000+ frames, and a T=104,141 self-attention matrix would need tens
of gigabytes per attention head per layer, on any GPU. The encoder is now a
bidirectional LSTM instead: O(T) linear cost regardless of length, no
positional encoding needed (recurrence inherently encodes frame order,
unlike permutation-invariant self-attention), and directly motivated by this
project's own accumulated evidence -- BiLSTM has shown repeated, independent
robustness across every scaling axis tested elsewhere in this codebase
(d_model, n_layers, streaming-inference length). The DECODER stays a
standard Transformer decoder: its cross-attention cost is O(num_queries x T)
-- LINEAR in T, not quadratic -- so attention is still the right tool there;
it was specifically the ENCODER's self-attention among all T frames that was
the danger, not attention in general.

Encoder: the same STGCNBlock spatial encoder used everywhere else in this
codebase, then a bidirectional LSTM stack. Optional HaMeR/DINOv2 fusion,
same pattern as every other model here.

Designed for batch_size=1 -- see dataset_segments.py's docstring for why.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# This file lives in Sign-Segmentation/detr/, but needs the existing
# Sign-Segmentation/src/ package (SkeletonGraph, STGCNBlock), which is NOT
# moving into detr/. Add the project root to sys.path explicitly, computed
# from THIS FILE's own location, so this resolves correctly no matter what
# directory the terminal is in when the importing script (train_detr.py) is
# actually run.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.graph import SkeletonGraph
from src.stgcn import STGCNBlock


class STGCN_DETR(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, stgcn_channels=64, d_model=256,
                 encoder_lstm_layers=4, num_decoder_layers=4, num_queries=4500,
                 nhead=8, dim_feedforward=1024, dropout=0.2,
                 hamer_dim=None, hamer_proj_dim=64, dinov2_dim=None, dinov2_proj_dim=128,
                 memory_pool_stride=16):
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

        # ENCODER: bidirectional LSTM, O(T) not O(T^2) -- see module docstring.
        self.encoder_lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=encoder_lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if encoder_lstm_layers > 1 else 0
        )
        self.encoder_proj = nn.Linear(d_model * 2, d_model)  # bidirectional output -> d_model

        # Pools the encoder's output along TIME before the decoder cross-attends
        # to it. This is the actual fix for a real memory problem, not a tuning
        # knob: decoder cross-attention needs num_queries x T x nhead x
        # num_decoder_layers activations retained simultaneously for backprop --
        # at T~100,000 (this corpus's longest videos) that's ~60 GB regardless
        # of num_queries, nearly 4x a 16 GB GPU. Pooling by 16x brings T down to
        # ~6,500, and total cross-attention memory down to ~3.75 GB. This does
        # NOT reduce prediction precision: start/end are still predicted as
        # continuous sigmoid outputs, not tied to the memory's temporal
        # resolution -- the decoder just gets a coarser (but still full-video)
        # view of context, the same way video transformers commonly use
        # temporal patches/tokens rather than per-frame attention.
        self.memory_pool_stride = memory_pool_stride
        self.memory_pool = nn.AvgPool1d(kernel_size=memory_pool_stride, stride=memory_pool_stride, ceil_mode=True)

        # Learned query embeddings -- one per potential segment, matching DETR's
        # learned "object queries" convention (here: "segment queries"). Must
        # comfortably exceed the maximum number of signs/phrases in any single
        # video in your data -- confirmed via diagnose_segment_counts.py, NOT
        # guessed; see the training script's own setup check for this too.
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

        # Gradient checkpointing for the spatial encoder and LSTM: this
        # project's full videos run 100,000+ frames, and even O(T)-linear
        # operations (STGCN convolutions, LSTM recurrence) produce per-
        # timestep activations that ALL need to be retained for backprop
        # across every layer -- at this length that alone can exceed a 16 GB
        # GPU, even with the decoder's cross-attention already fixed via
        # memory_pool_stride (that fix addresses a DIFFERENT bottleneck,
        # downstream of these two calls -- it cannot help here, which is
        # exactly why the earlier OOM persisted unchanged after adding it).
        # Checkpointing recomputes each forward pass during backward instead
        # of storing every intermediate, trading roughly 30-50% more compute
        # time for a large memory reduction.
        feat = checkpoint(self.stgcn_blocks, x, use_reentrant=False)
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

        feat = self.feature_proj(feat + 1e-5)  # (1, T, d_model)

        # Defensive: cuDNN's LSTM implementation can be picky about memory
        # layout in ways plain PyTorch ops don't always guarantee end-to-end
        # (especially once hamer/dinov2 concatenation is in the mix) --
        # .contiguous() on an already-contiguous tensor is a no-op, so this
        # costs nothing when it's not needed and prevents a real failure mode
        # when it is.
        feat = feat.contiguous()

        # Same checkpointing rationale as the spatial encoder above. Wrapped
        # in a small function because checkpoint() works most reliably when
        # the checkpointed callable returns a plain tensor rather than
        # nn.LSTM's native (output, (h_n, c_n)) nested-tuple return -- h_n/c_n
        # aren't needed downstream anyway.
        def _run_lstm(inp):
            out, _ = self.encoder_lstm(inp)
            return out

        lstm_out = checkpoint(_run_lstm, feat, use_reentrant=False)
        memory = self.encoder_proj(lstm_out)    # (1, T, d_model)
        memory = self.memory_pool(memory.transpose(1, 2)).transpose(1, 2)  # (1, T_pooled, d_model)

        queries = self.query_embed.unsqueeze(0)  # (1, num_queries, d_model)
        decoded = self.decoder(queries, memory)  # (1, num_queries, d_model)

        confidence_logits = self.confidence_head(decoded).squeeze(-1)  # (1, num_queries)
        spans = torch.sigmoid(self.span_head(decoded))                 # (1, num_queries, 2), each in [0,1]

        return confidence_logits, spans