"""
skeleton_encoders.py -- drop-in alternatives to the two-block ST-GCN of each keypoint stream
of MultiStreamSegmenter (stream_encoder = "stgcn" | "blockgcn" | "skateformer" | "linear").

Every encoder maps (B, C_in, T, V) -> (B, channels, T, V) at the input frame rate, so the
existing vertex flatten / part pooling, branch projection and fusion are unchanged. Depth
and width default to the ST-GCN stream's (2 layers, 64 channels), so a comparison tests the
DESIGN of the spatial-temporal block, not its size.

  blockgcn     BlockGCN (Zhou et al., CVPR 2024). Per layer:
                 H' = ReLU( BN( sum_k (A_k + B_k) (H + C)_k W_k ) + H )   then multi-scale TCN
               A_k  learnable adjacency per channel group (init: skeleton graph)
               B_k  static topology encoding: a learned scalar per shortest-path hop distance
               C    dynamic topology encoding from 0-dim persistent homology of each frame's
                    joint-distance graph (Vietoris-Rips): the death time of each joint's
                    component = its MST joining distance, vectorized with learnable Gaussian
                    structure elements (d' = 64) and projected per layer
               W_k  block-diagonal (grouped) channel projection, K = 8 groups
               Deviation: the persistence is computed per FRAME (time-varying topology, as a
               segmenter needs per-frame features), not once per clip.

  skateformer  SkateFormer (Do & Kim, ECCV 2024). Per block:
                 [x_gc, x_tc, x_msa] = Split(Linear(LN(x)))       C/4, C/4, C/2 channels
                 x_gc  <- G-Conv (learnable adjacency per group)
                 x_tc  <- T-Conv (grouped temporal conv)
                 x_msa <- Skate-MSA: 4 channel quarters, each attending inside one partition
                          type: {neighbouring joints = same body part, distant joints = same
                          position across parts} x {local motion = N consecutive frames,
                          global motion = every N-th frame}
                 x <- x + Linear(Concat(...));   x <- x + FFN(LN(x))
               Parts of unequal size are padded and masked; T is padded to a multiple of N
               and masked. Input lift = 3 Linear layers + learnable joint embedding + fixed
               sinusoidal frame embedding (Skate-Embedding without the multi-person part).

  linear       per-vertex Linear only (2 layers): no graph and no temporal mixing (the 2023
               model's flatten + Linear, kept stream-separated).
  tcn          the SAME two ST-GCN blocks with the adjacency replaced by the identity: the
               9-frame temporal convolution stays, the spatial graph mixing is removed
               (built in models.py, since STGCNBlock lives there).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MediaPipe layout of the 65 body + hand vertices (dataset order): pose 0-22, left hand
# 23-43, right hand 44-64 (hand: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring,
# 17-20 pinky). SkateFormer partitions: small anatomical parts, ordered outward, so a
# "distant" partition gathers the same position of every part (e.g. all fingertips).
def _hand_parts(off):
    return [[off + 0, off + 1, off + 2, off + 3, off + 4],                  # wrist + thumb
            [off + 5, off + 6, off + 7, off + 8], [off + 9, off + 10, off + 11, off + 12],
            [off + 13, off + 14, off + 15, off + 16], [off + 17, off + 18, off + 19, off + 20]]


SKATE_BODY_PARTS = ([[0, 9, 10], [1, 2, 3, 7], [4, 5, 6, 8],              # nose+mouth, L / R eye+ear
                     [11, 13, 15, 17, 19, 21], [12, 14, 16, 18, 20, 22]]  # L / R arm, outward
                    + _hand_parts(23) + _hand_parts(44))


def disjoint_parts(parts, num_vertices):
    """Each vertex in exactly one part (first occurrence wins); missing vertices -> own part."""
    seen, out = set(), []
    for p in parts:
        q = [v for v in p if v not in seen and v < num_vertices]
        seen.update(q)
        if q:
            out.append(q)
    rest = [v for v in range(num_vertices) if v not in seen]
    if rest:
        out.append(rest)
    return out


def hop_distance(A, max_hop):
    """Shortest-path hop counts on the (binary) skeleton graph of a normalized adjacency A;
    unreachable or > max_hop -> max_hop + 1."""
    adj = (np.asarray(A) > 0).astype(np.int64)
    np.fill_diagonal(adj, 0)
    V = adj.shape[0]
    dist = np.full((V, V), max_hop + 1, dtype=np.int64)
    for s in range(V):
        dist[s, s] = 0
        frontier, d = [s], 0
        while frontier and d < max_hop:
            d += 1
            nxt = []
            for u in frontier:
                for w in np.nonzero(adj[u])[0]:
                    if dist[s, w] > d:
                        dist[s, w] = d
                        nxt.append(w)
            frontier = nxt
    return dist


# ============================================================================== BlockGCN
@torch.no_grad()
def mst_death_times(coords):
    """
    0-dim persistent homology of the Vietoris-Rips filtration of each frame's joints:
    the multiset of death times equals the minimum-spanning-tree edge lengths. Each joint is
    assigned the length of the MST edge that joins it (Prim from joint 0; joint 0 = the
    infinite bar, 0). coords (N, V, c) -> (N, V).
    """
    N, V, _ = coords.shape
    D = torch.cdist(coords.float(), coords.float())                      # (N, V, V)
    death = torch.zeros(N, V, device=coords.device)
    in_tree = torch.zeros(N, V, dtype=torch.bool, device=coords.device)
    in_tree[:, 0] = True
    best = D[:, 0].clone()
    rows = torch.arange(N, device=coords.device)
    for _ in range(V - 1):
        j = best.masked_fill(in_tree, float("inf")).argmin(dim=1)        # next joint to join
        death[rows, j] = best[rows, j]
        in_tree[rows, j] = True
        best = torch.minimum(best, D[rows, j])
    return death


class PersistenceEncoding(nn.Module):
    """Death times -> d' learnable Gaussian structure elements (Hofer et al. style)."""

    def __init__(self, dim=64):
        super().__init__()
        self.mu = nn.Parameter(torch.linspace(0.0, 2.0, dim))
        self.log_prec = nn.Parameter(torch.full((dim,), math.log(25.0)))
        self.dim = dim

    def forward(self, x):                                                 # x (B, c, T, V) coords
        B, c, T, V = x.shape
        death = mst_death_times(x.permute(0, 2, 3, 1).reshape(B * T, V, c))
        d = torch.log1p(death).view(B, T, V, 1)
        s = torch.exp(-(d - self.mu) ** 2 * self.log_prec.exp())          # (B, T, V, dim)
        return s.permute(0, 3, 1, 2).contiguous()                         # (B, dim, T, V)


class MultiScaleTCN(nn.Module):
    """Three dilated temporal branches + a 1x1 branch (CTR-GCN / BlockGCN style)."""

    def __init__(self, channels, dropout=0.0):
        super().__init__()
        b = channels // 4

        def branch(k, dil):
            return nn.Sequential(nn.Conv2d(channels, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, (k, 1), padding=((k - 1) // 2 * dil, 0), dilation=(dil, 1)),
                                 nn.BatchNorm2d(b))
        self.branches = nn.ModuleList([branch(5, 1), branch(5, 2), branch(5, 3),
                                       nn.Sequential(nn.Conv2d(channels, channels - 3 * b, 1),
                                                     nn.BatchNorm2d(channels - 3 * b))])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return F.relu(x + self.drop(torch.cat([br(x) for br in self.branches], dim=1)))


class BlockGCLayer(nn.Module):
    def __init__(self, channels, A, hops, groups=8, max_hop=6, ph_dim=0, dropout=0.0):
        super().__init__()
        if channels % groups:
            raise ValueError(f"BlockGCN: channels ({channels}) must be divisible by groups ({groups}).")
        V = A.shape[0]
        self.groups = groups
        self.A = nn.Parameter(torch.tensor(np.asarray(A), dtype=torch.float32).repeat(groups, 1, 1))
        self.register_buffer("hops", torch.as_tensor(hops, dtype=torch.long), persistent=False)
        self.hop_emb = nn.Parameter(torch.zeros(groups, max_hop + 2))     # B_k: 0 at init
        self.W = nn.Conv2d(channels, channels, 1, groups=groups)          # block-diagonal W_k
        self.ph_proj = nn.Conv2d(ph_dim, channels, 1) if ph_dim else None  # f_theta per layer
        self.bn = nn.BatchNorm2d(channels)
        self.tcn = MultiScaleTCN(channels, dropout)

    def forward(self, x, ph=None):                                        # x (B, C, T, V)
        h = x if (ph is None or self.ph_proj is None) else x + self.ph_proj(ph)
        B, C, T, V = h.shape
        h = self.W(h).view(B, self.groups, C // self.groups, T, V)
        adj = self.A + self.hop_emb[:, self.hops]                        # (K, V, V)
        h = torch.einsum("bkctv,kwv->bkctw", h, adj).reshape(B, C, T, V)
        x = F.relu(self.bn(h) + x)
        return self.tcn(x)


class BlockGCNEncoder(nn.Module):
    def __init__(self, in_channels, A, channels=64, layers=2, groups=8, use_ph=True, ph_dim=64,
                 max_hop=6, dropout=0.0):
        super().__init__()
        V = A.shape[0]
        self.embed = nn.Sequential(nn.Conv2d(in_channels, channels, 1), nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.joint_pe = nn.Parameter(torch.zeros(1, channels, 1, V))
        nn.init.trunc_normal_(self.joint_pe, std=0.02)
        self.ph = PersistenceEncoding(ph_dim) if use_ph else None
        hops = hop_distance(A, max_hop)
        self.layers = nn.ModuleList([BlockGCLayer(channels, A, hops, groups, max_hop,
                                                  ph_dim if use_ph else 0, dropout) for _ in range(layers)])

    def forward(self, x):
        ph = self.ph(x) if self.ph is not None else None
        h = self.embed(x) + self.joint_pe
        for layer in self.layers:
            h = layer(h, ph)
        return h


# =========================================================================== SkateFormer
def sinusoid(T, C, device):
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, C, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / C))
    pe = torch.zeros(T, C, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe


class SkateMSA(nn.Module):
    """Four partition types, each on its own quarter of the channels."""
    # (group axes, token axes) over the layout (B, M, N, K, L, c): M = time segments, N =
    # frames per segment, K = parts, L = position inside a part.
    TYPES = (((1, 3), (2, 4)),    # 1 neighbouring joints x local motion   tokens (N, L)
             ((1, 4), (2, 3)),    # 2 distant joints     x local motion   tokens (N, K)
             ((2, 3), (1, 4)),    # 3 neighbouring joints x global motion  tokens (M, L)
             ((2, 4), (1, 3)))    # 4 distant joints     x global motion  tokens (M, K)

    def __init__(self, dim, head_dim=8, dropout=0.0):
        super().__init__()
        if dim % 4:
            raise ValueError(f"Skate-MSA width ({dim}) must be divisible by 4.")
        self.c = dim // 4
        self.heads = max(1, self.c // head_dim)
        if self.c % self.heads:
            self.heads = 1
        self.qkv = nn.ModuleList([nn.Linear(self.c, 3 * self.c) for _ in range(4)])
        self.drop = dropout

    def forward(self, x, valid):
        """x (B, M, N, K, L, 4c); valid (M, N, K, L) bool. Returns the same shape."""
        B, M, N, K, L, _ = x.shape
        outs = []
        for i, (g_axes, t_axes) in enumerate(self.TYPES):
            xi = x[..., i * self.c:(i + 1) * self.c]
            perm = (0,) + g_axes + t_axes + (5,)
            xp = xi.permute(*perm)
            gshape, tshape = xp.shape[1:3], xp.shape[3:5]
            G, S = gshape[0] * gshape[1], tshape[0] * tshape[1]
            q, k, v = self.qkv[i](xp.reshape(B * G, S, self.c)).chunk(3, dim=-1)
            q, k, v = (t.view(B * G, S, self.heads, -1).transpose(1, 2) for t in (q, k, v))
            vm = valid.permute(*(a - 1 for a in g_axes + t_axes)).reshape(1, G, S)
            mask = torch.zeros(1, G, S, device=x.device, dtype=q.dtype)
            mask = mask.masked_fill(~vm, torch.finfo(q.dtype).min / 4)
            mask = mask.expand(B, G, S).reshape(B * G, 1, 1, S)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               dropout_p=self.drop if self.training else 0.0)
            o = o.transpose(1, 2).reshape(B, *gshape, *tshape, self.c)
            inv = [0] * 6
            for new, old in enumerate(perm):
                inv[old] = new
            outs.append(o.permute(*inv))
        return torch.cat(outs, dim=-1)


class SkateBlock(nn.Module):
    def __init__(self, dim, A, gc_groups=4, t_kernel=7, ffn_ratio=4, head_dim=8, dropout=0.0):
        super().__init__()
        if dim % 8:
            raise ValueError(f"SkateFormer width ({dim}) must be divisible by 8.")
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.inp = nn.Linear(dim, dim)
        q = dim // 4
        self.gc_groups = gc_groups if q % gc_groups == 0 else 1
        self.gc_adj = nn.Parameter(torch.tensor(np.asarray(A), dtype=torch.float32).repeat(self.gc_groups, 1, 1))
        self.tconv = nn.Conv1d(q, q, t_kernel, padding=t_kernel // 2, groups=self.gc_groups)
        self.msa = SkateMSA(dim // 2, head_dim, dropout)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * ffn_ratio, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, layout):
        """x (B, T, V, C) tokens; layout = SkateFormerEncoder partition helpers."""
        B, T, V, C = x.shape
        q = C // 4
        y = self.inp(self.norm1(x))
        x_gc, x_tc, x_msa = y[..., :q], y[..., q:2 * q], y[..., 2 * q:]
        g = self.gc_groups
        x_gc = torch.einsum("btvgc,gwv->btwgc", x_gc.reshape(B, T, V, g, q // g), self.gc_adj).reshape(B, T, V, q)
        x_tc = self.tconv(x_tc.permute(0, 2, 3, 1).reshape(B * V, q, T)).view(B, V, q, T).permute(0, 3, 1, 2)
        x_msa = layout.unpartition(self.msa(layout.partition(x_msa), layout.valid), T)
        x = x + self.drop(self.out(torch.cat([x_gc, x_tc, x_msa], dim=-1)))
        return x + self.drop(self.ffn(self.norm2(x)))


class _SkateLayout:
    """Joint/time partition bookkeeping for one forward pass."""

    def __init__(self, enc, T, device):
        N = enc.local_frames
        self.N, self.M = N, -(-T // N)
        self.Tp = self.M * N
        self.enc = enc
        t_valid = (torch.arange(self.Tp, device=device) < T).view(self.M, N)
        self.valid = t_valid[:, :, None, None] & enc.joint_valid[None, None]   # (M, N, K, L)

    def partition(self, x):                                               # (B, T, V, c)
        B, T, V, c = x.shape
        pad = x.new_zeros(B, T, 1, c)
        xp = torch.cat([x, pad], dim=2)[:, :, self.enc.joint_index]       # (B, T, K*L, c)
        if self.Tp > T:
            xp = torch.cat([xp, xp.new_zeros(B, self.Tp - T, xp.shape[2], c)], dim=1)
        return xp.view(B, self.M, self.N, self.enc.K, self.enc.L, c)

    def unpartition(self, xp, T):
        B, c = xp.shape[0], xp.shape[-1]
        x = xp.reshape(B, self.Tp, self.enc.K * self.enc.L, c)[:, :T]
        return x[:, :, self.enc.vertex_slot]                              # (B, T, V, c)


class SkateFormerEncoder(nn.Module):
    def __init__(self, in_channels, A, parts, channels=64, layers=2, local_frames=32,
                 head_dim=8, t_kernel=7, ffn_ratio=4, dropout=0.0):
        super().__init__()
        V = A.shape[0]
        parts = disjoint_parts(parts, V)
        self.K, self.L = len(parts), max(len(p) for p in parts)
        index = torch.full((self.K, self.L), V, dtype=torch.long)          # V = the zero pad token
        slot = torch.zeros(V, dtype=torch.long)
        for k, p in enumerate(parts):
            for l, v in enumerate(p):
                index[k, l] = v
                slot[v] = k * self.L + l
        self.register_buffer("joint_index", index.view(-1), persistent=False)
        self.register_buffer("joint_valid", index != V, persistent=False)
        self.register_buffer("vertex_slot", slot, persistent=False)
        self.local_frames = int(local_frames)
        self.lift = nn.Sequential(nn.Linear(in_channels, channels), nn.GELU(),
                                  nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
        self.joint_emb = nn.Parameter(torch.zeros(V, channels))
        nn.init.trunc_normal_(self.joint_emb, std=0.02)
        self.blocks = nn.ModuleList([SkateBlock(channels, A, 4, t_kernel, ffn_ratio, head_dim, dropout)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):                                                 # (B, C_in, T, V)
        B, _, T, V = x.shape
        h = self.lift(x.permute(0, 2, 3, 1))                              # (B, T, V, C)
        h = h + self.joint_emb + sinusoid(T, h.shape[-1], x.device)[None, :, None]
        layout = _SkateLayout(self, T, x.device)
        for blk in self.blocks:
            h = blk(h, layout)
        return self.norm(h).permute(0, 3, 1, 2).contiguous()              # (B, C, T, V)


# ================================================================================ linear
class LinearVertexEncoder(nn.Module):
    """Per-vertex Linear (+ BN, ReLU) x layers: no spatial graph, no temporal mixing."""

    def __init__(self, in_channels, channels=64, layers=2):
        super().__init__()
        mods, c = [], in_channels
        for _ in range(max(1, layers)):
            mods += [nn.Conv2d(c, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
            c = channels
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


STREAM_ENCODERS = ("stgcn", "tcn", "blockgcn", "skateformer", "linear")


def build_stream_encoder(kind, in_channels, A, parts, channels=64, layers=2, dropout=0.0, **opts):
    A = np.asarray(A)
    if kind == "blockgcn":
        return BlockGCNEncoder(in_channels, A, channels, layers, groups=opts.get("blockgcn_groups", 8),
                               use_ph=opts.get("blockgcn_ph", True), dropout=dropout)
    if kind == "skateformer":
        return SkateFormerEncoder(in_channels, A, parts, channels, layers,
                                  local_frames=opts.get("skate_local_frames", 32), dropout=dropout)
    if kind == "linear":
        return LinearVertexEncoder(in_channels, channels, layers)
    raise ValueError(f"stream_encoder must be one of {STREAM_ENCODERS}, got '{kind}'.")