"""
fact_model.py -- FACT adapted to BIO phrase segmentation, plus the frozen-feature BiLSTM reference.

FACT: Lu & Elhamifar, "FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action
Segmentation", CVPR 2024. Building blocks below (MSTCN2, X2Y_map, SALayer, SCALayer, SCADecoder,
SADecoder, the temporal down/up-sampling, the matching losses) follow the official code
https://github.com/ZijiaLewisLu/CVPR2024-FACT (MIT License, Copyright 2024 Zijia Lu), simplified to
batch size 1 and stripped of the transcript mode.

What is CHANGED for phrases (and why)
  1. Two class spaces. The frame branch predicts B/I/O (3 classes, like every model in this
     project); the action tokens predict O-segment / phrase / null. FACT's original builds its
     target segments from runs of the same class, which would merge two ADJACENT phrases (same
     class "phrase", no O between them) into one -- exactly the continuous boundaries that are
     hardest. Here the segments come from the BIO labels (fact_core.bio_to_segments): every B
     starts a new phrase, every O run is its own segment.
  2. Frame losses = this project's R0 loss: class-weighted cross-entropy (weights [1.0, 1.1, 8.5])
     + count loss (src/boundary_losses.py), on every block's frame output.
  3. FACT's temporal smoothing loss is applied to the phrase/non-phrase MARGINAL, not to the
     BIO probabilities -- smoothing a 1-frame B spike would fight the B class.
  4. The temporal-pooling block (UpdateBlockTDU) pools over segments of the predicted BIO with
     B-runs collapsed (fact_core.tdu_segments), so adjacent predicted phrases are pooled apart.
  5. Decoding (fact_core.token_decode): a new phrase starts where the owning token changes.
  6. Small fixes: segment weights are indexed by the MATCHED segments (the original multiplies a
     (frames x matched) matrix by an all-segments vector, which only works when every segment
     is matched and all weights are equal); channel masking is a manual per-channel mask (newer
     PyTorch treats Dropout2d on 3-D input differently); the soft IoU uses a closed form.

Shapes follow FACT: frame features (T, 1, H), action features (A, 1, H), batch size 1 inside.
"""
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from fact_phrase import fact_core as core
from src.boundary_losses import boundary_aware_loss, soft_weighted_ce

N_FRAME_CLS = 3          # O, I, B
N_TOK_CLS = 2            # O-segment, phrase   (+1 null)

FACT_DEFAULTS = {
    "model": "fact",
    "block": "iuUU",          # i input block, u update block, U update block with temporal pooling
    "ntoken": "auto",         # auto = 1.25 x the most segments in any training window, multiple of 8
    "hid_dim": 256,
    "f_dim": 128, "f_layers": 10, "u_f_layers": 8,
    "a_dim": 128, "a_ffdim": 256, "a_nhead": 4, "a_layers": 4, "u_a_layers": 1, "s_layers": 1,
    "dropout": 0.1,
    "cmr": 0.3,               # channel masking rate on the input features
    "fpos": False,            # sinusoidal frame positions (FACT default for its configs: off)
    "mwt": 0.1,               # weight of the frame branch when decoding tokens
    "tm_use": True, "tm_t": 30, "tm_m": 5, "tm_p": 0.05,   # FACT time masking (training only)
    "pc": 0.2, "a2fc": 1.0,   # matching cost weights: class probability, soft IoU
    "sw": 5.0,                # smoothing loss weight
    "tok_weights": [1.0, 1.0],  # token class weights O-segment, phrase
    "nullw": "auto",
    "class_weights": [1.0, 1.1, 8.5],
    "count_loss_weight": 2.0, "count_loss_radius": 5, "count_loss_tau": 0.1, "count_loss_mode": "argmax",
}

BILSTM_DEFAULTS = {
    "model": "frozen_bilstm",
    "d_model": 256, "n_layers": 4, "dropout": 0.2, "input_dropout": 0.2,
    "class_weights": [1.0, 1.1, 8.5],
    "count_loss_weight": 2.0, "count_loss_radius": 5, "count_loss_tau": 0.1, "count_loss_mode": "argmax",
}


# ============================================================================ FACT building blocks
def time_mask(feature, T, num_masks, p):
    """FACT time masking: up to num_masks spans of < T frames (and <= p of the sequence) set to 0."""
    feature = feature.clone()
    L = feature.shape[0]
    for _ in range(num_masks):
        t = min(int(p * L), random.randrange(0, T))
        if t <= 0 or L - t <= 0:
            continue
        t0 = random.randrange(0, L - t)
        feature[t0:t0 + t] = 0
    return feature


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000, empty=False):
        super().__init__()
        self.d_model, self.empty = d_model, empty
        self._build(max_len)

    def _build(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        if not self.empty:
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x):
        if x.size(0) > self.pe.shape[0]:
            self._build(x.size(0) + 10)
            self.pe = self.pe.to(x.device)
        return self.pe[:x.size(0)]


class MSTCN2(nn.Module):
    """MS-TCN++ style dual dilated layers (FACT basic.MSTCN2). In/out: (T, 1, C)."""

    def __init__(self, dim, num_f_maps, out_dim, num_layers, dropout=0.5, in_map=True):
        super().__init__()
        self.num_layers = num_layers
        self.in_map = in_map
        if in_map:
            self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        else:
            assert dim == num_f_maps, (dim, num_f_maps)
        self.conv_dilated_1 = nn.ModuleList(
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers))
        self.conv_dilated_2 = nn.ModuleList(
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i) for i in range(num_layers))
        self.conv_fusion = nn.ModuleList(nn.Conv1d(2 * num_f_maps, num_f_maps, 1) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(num_f_maps, out_dim, 1)

    def forward(self, x):
        x = x.permute(1, 2, 0)                                   # (1, C, T)
        f = self.conv_1x1_in(x) if self.in_map else x
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            if i != self.num_layers - 1:
                f = self.dropout(f)
            f = f + f_in
        return self.conv_out(f).permute(2, 0, 1)                 # (T, 1, out)


def add_positional_encoding(tensor, pos):
    if pos is None:
        return tensor
    d = pos.size(-1)
    tensor = tensor.clone()
    tensor[:, :, :d] = tensor[:, :, :d] + pos
    return tensor


class X2Y_map(nn.Module):
    """Single-head cross-attention Y <- X; keeps attention logits / weights for the losses."""

    def __init__(self, x_dim, y_dim, y_outdim, head_dim, dropout=0.5, kq_pos=False):
        super().__init__()
        self.kq_pos = kq_pos
        self.X_K = nn.Linear(x_dim, head_dim)
        self.X_V = nn.Linear(x_dim, head_dim)
        self.Y_Q = nn.Linear(y_dim, head_dim)
        self.Y_W = nn.Linear(y_dim + head_dim, y_outdim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_feature, Y_feature, X_pos=None, Y_pos=None):
        xk = self.X_K(add_positional_encoding(X_feature, X_pos) if (X_pos is not None and self.kq_pos) else X_feature)
        xv = self.X_V(X_feature)
        yq = self.Y_Q(add_positional_encoding(Y_feature, Y_pos) if (Y_pos is not None and self.kq_pos) else Y_feature)
        attn_logit = torch.einsum("xbd,ybd->byx", xk, yq) / math.sqrt(xk.shape[-1])
        self.attn_logit = attn_logit                             # (1, Y, X)
        attn = torch.softmax(attn_logit, dim=-1)
        attn_feat = torch.einsum("byx,xbh->ybh", attn, xv)
        out = self.Y_W(self.dropout(torch.cat([Y_feature, attn_feat], dim=-1)))
        self.attn = attn.unsqueeze(1)                            # (1, 1, Y, X)
        return out


class SALayer(nn.Module):
    def __init__(self, q_dim, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(q_dim, nhead, dropout=attn_dropout)
        self.linear1 = nn.Linear(q_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, q_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(q_dim), nn.LayerNorm(q_dim)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, tgt, key, value, query_pos=None, key_pos=None):
        q = add_positional_encoding(tgt, query_pos)
        k = add_positional_encoding(key, key_pos)
        tgt2, _ = self.multihead_attn(q, k, value, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm2(tgt + self.dropout2(tgt2))


class SCALayer(nn.Module):
    """Self-attention among tokens + cross-attention tokens <- frames + FFN (DETR-style decoder layer)."""

    def __init__(self, action_dim, frame_dim, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(action_dim, nhead, dropout=attn_dropout)
        self.multihead_attn = nn.MultiheadAttention(action_dim, nhead, kdim=frame_dim, vdim=frame_dim,
                                                    dropout=attn_dropout)
        self.linear1 = nn.Linear(action_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, action_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(action_dim), nn.LayerNorm(action_dim), nn.LayerNorm(action_dim)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = k = add_positional_encoding(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, _ = self.multihead_attn(add_positional_encoding(tgt, query_pos),
                                      add_positional_encoding(memory, pos), memory, need_weights=False)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class SCADecoder(nn.Module):
    def __init__(self, a_dim, out_dim, layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = norm
        self.out_linear = nn.Linear(a_dim, out_dim)

    def forward(self, tgt, memory, pos=None, query_pos=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, pos=pos, query_pos=query_pos)
        if self.norm is not None:
            out = self.norm(out)
        return self.out_linear(out)


class SADecoder(nn.Module):
    def __init__(self, a_dim, out_dim, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.out_linear = nn.Linear(a_dim, out_dim)

    def forward(self, tgt, pos=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, out, out, query_pos=pos, key_pos=pos)
        return self.out_linear(out)


class TemporalDownsampleUpsample:
    """Frames <-> segments for a list of (start, end) inclusive segments covering [0, T)."""

    def __init__(self, segs, device):
        lens = [e - s + 1 for s, e in segs]
        self.segs = segs
        self.num_seg = len(segs)
        self.seg_label = torch.as_tensor(np.repeat(np.arange(len(segs)), lens), dtype=torch.long, device=device)
        self.seg_lens = torch.as_tensor(lens, dtype=torch.float32, device=device)

    def feature_frame2seg(self, frame_feature):
        _, b, h = frame_feature.shape
        out = torch.zeros(self.num_seg, b, h, device=frame_feature.device, dtype=frame_feature.dtype)
        out.index_add_(0, self.seg_label, frame_feature)
        return out / self.seg_lens[:, None, None]

    def feature_seg2frame(self, seg_feature):
        return seg_feature[self.seg_label]

    def attn_seg2frame(self, seg_attn):                          # (1, S, A) -> (1, T, A)
        return seg_attn[0, self.seg_label].unsqueeze(0)


def smooth_loss_logp(logp):
    """FACT temporal smoothing on log-probabilities (B, T, C): clamped squared frame differences."""
    return torch.clamp((logp[:, 1:] - logp[:, :-1]) ** 2, min=0, max=16).mean()


def phrase_marginal_logp(frame_clogit):
    """(T, 1, 3) BIO logits -> (1, T, 2) log [P(O), P(I) + P(B)]."""
    logp = F.log_softmax(frame_clogit.squeeze(1).float(), dim=-1)
    m = torch.stack([logp[:, core.O_TAG], torch.logsumexp(logp[:, 1:], dim=-1)], dim=-1)
    return m.unsqueeze(0)


# ============================================================================ matching criterion
class MatchCriterionBIO:
    """FACT's MatchCriterion with BIO-derived segments (see the module docstring)."""

    def __init__(self, cfg, ntoken, nullw, device):
        self.cfg = cfg
        self.device = device
        self.bio_w = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
        tw = list(cfg["tok_weights"]) + [float(nullw)]
        self.cweight = torch.tensor(tw, dtype=torch.float32, device=device)       # token classes (3)
        self.tok_w = torch.tensor(cfg["tok_weights"], dtype=torch.float32, device=device)

    def set_label(self, bio):
        """bio: (T,) LongTensor at the model's frame rate."""
        seg_label, transcript, _ = core.bio_to_segments(bio.detach().cpu().numpy())
        self.bio = bio
        self.seg_label_np = seg_label
        self.seg_label = torch.as_tensor(seg_label, device=self.device)
        self.transcript = torch.as_tensor(transcript, device=self.device)
        self.S = len(transcript)
        T = len(seg_label)
        self.onehot_bio = F.one_hot(bio, N_FRAME_CLS).float()                    # (T, 3)
        self.onehot_seg = torch.zeros(T, self.S, device=self.device)
        self.onehot_seg[torch.arange(T, device=self.device), self.seg_label] = 1.0
        self.sweight = self.tok_w[self.transcript]                                 # (S,)

    # ---- matching
    def match(self, action_clogit, a2f_attn):
        """action_clogit (A, 1, 3) logits, a2f_attn (1, T, A). Returns (token idx, segment idx)."""
        with torch.no_grad():
            prob = torch.softmax(action_clogit.squeeze(1).float(), dim=-1)          # (A, 3)
            cost = -self.cfg["pc"] * prob[:, self.transcript].cpu().numpy()        # (A, S)
            if self.cfg["a2fc"] > 0:
                iou = core.soft_iou(a2f_attn[0].float().cpu().numpy(), self.seg_label_np, self.S)
                cost = cost - self.cfg["a2fc"] * iou
        aind, sind = linear_sum_assignment(cost)
        return (torch.as_tensor(aind, dtype=torch.long, device=self.device),
                torch.as_tensor(sind, dtype=torch.long, device=self.device))

    # ---- losses
    def frame_loss(self, frame_clogit):
        """R0 frame loss on (T, 1, 3) logits: weighted CE + count loss (src/boundary_losses.py)."""
        logits = frame_clogit.squeeze(1).t().unsqueeze(0).float()                  # (1, 3, T)
        targets = self.onehot_bio.t().unsqueeze(0)                                  # (1, 3, T)
        return boundary_aware_loss(logits, targets, self.bio_w, soft=False,
                                   count_weight=self.cfg["count_loss_weight"], radius=self.cfg["count_loss_radius"],
                                   tau=self.cfg["count_loss_tau"], mode=self.cfg["count_loss_mode"])

    def frame_loss_tdu(self, seg_clogit, tdu):
        """Weighted soft CE of pooled segments against the BIO share of their frames."""
        zoomed = torch.zeros(tdu.num_seg, N_FRAME_CLS, device=self.device)
        zoomed.index_add_(0, tdu.seg_label, self.onehot_bio)
        zoomed = zoomed / tdu.seg_lens[:, None]
        return soft_weighted_ce(seg_clogit.squeeze(1).t().unsqueeze(0), zoomed.t().unsqueeze(0), self.bio_w)

    def action_token_loss(self, match, action_clogit):
        aind, sind = match
        A = action_clogit.shape[0]
        clabel = torch.full((A,), core.TOK_NULL, dtype=torch.long, device=self.device)
        clabel[aind] = self.transcript[sind]
        return F.cross_entropy(action_clogit.squeeze(1).float(), clabel, weight=self.cweight)

    def cross_attn_loss(self, match, attn, dim):
        """attn (1, T, A) logits; dim=2: each frame's distribution over the matched tokens,
        dim=1: each matched token's distribution over frames."""
        aind, sind = match
        tgt = self.onehot_seg[:, sind]                                             # (T, m)
        logp = torch.log_softmax(attn[0, :, aind].float(), dim=dim - 1)
        loss = -(logp * tgt) * self.sweight[sind]
        return loss.sum() / self.onehot_seg.sum()

    def cross_attn_loss_tdu(self, match, attn, tdu, dim):
        aind, sind = match
        zoomed = torch.zeros(tdu.num_seg, self.S, device=self.device)
        zoomed.index_add_(0, tdu.seg_label, self.onehot_seg)
        zoomed = zoomed / tdu.seg_lens[:, None]
        tgt = zoomed[:, sind]
        logp = torch.log_softmax(attn[0, :, aind].float(), dim=dim - 1)
        loss = -(logp * tgt) * self.sweight[sind]
        return loss.sum() / zoomed.sum()


# ============================================================================ FACT blocks
class Block(nn.Module):
    @staticmethod
    def process_feature(feature, nclass):
        """The last nclass channels are class logits; they are replaced by their softmax."""
        clogit = feature[:, :, -nclass:]
        feature = torch.cat([feature[:, :, :-nclass], torch.softmax(clogit, dim=-1)], dim=-1)
        return feature, clogit

    def create_cross_attention(self, c, outdim):
        return X2Y_map(c["hid_dim"], c["hid_dim"], outdim, head_dim=c["hid_dim"], dropout=c["dropout"], kq_pos=True)

    def frame_smooth(self):
        return smooth_loss_logp(phrase_marginal_logp(self.frame_clogit))


class InputBlock(Block):
    def __init__(self, c, in_dim):
        super().__init__()
        self.c = c
        self.frame_branch = MSTCN2(in_dim, c["f_dim"], c["hid_dim"], c["f_layers"], dropout=c["dropout"], in_map=True)
        layer = SCALayer(c["a_dim"], c["hid_dim"], c["a_nhead"], c["a_ffdim"], dropout=c["dropout"],
                         attn_dropout=c["dropout"])
        self.action_branch = SCADecoder(c["a_dim"], c["hid_dim"], layer, c["a_layers"], norm=nn.LayerNorm(c["a_dim"]))

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        frame_feature, self.frame_clogit = self.process_feature(self.frame_branch(frame_feature), N_FRAME_CLS)
        action_feature = self.action_branch(action_feature, frame_feature, pos=frame_pos, query_pos=action_pos)
        action_feature, self.action_clogit = self.process_feature(action_feature, N_TOK_CLS + 1)
        return frame_feature, action_feature

    def compute_loss(self, crit, match):
        return (crit.frame_loss(self.frame_clogit) + crit.action_token_loss(match, self.action_clogit)
                + self.c["sw"] * self.frame_smooth())


class UpdateBlock(Block):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.frame_branch = MSTCN2(c["f_dim"], c["f_dim"], c["hid_dim"], c["u_f_layers"], dropout=c["dropout"],
                                   in_map=False)
        self.f2a_layer = self.create_cross_attention(c, c["a_dim"])
        sa = SALayer(c["a_dim"], c["a_nhead"], dim_feedforward=c["a_ffdim"], dropout=c["dropout"],
                     attn_dropout=c["dropout"])
        self.action_branch = SADecoder(c["a_dim"], c["hid_dim"], sa, c["u_a_layers"])
        self.a2f_layer = self.create_cross_attention(c, c["f_dim"])

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        action_feature = self.f2a_layer(frame_feature, action_feature, X_pos=frame_pos, Y_pos=action_pos)
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, self.action_clogit = self.process_feature(action_feature, N_TOK_CLS + 1)
        frame_feature = self.a2f_layer(action_feature, frame_feature, X_pos=action_pos, Y_pos=frame_pos)
        frame_feature, self.frame_clogit = self.process_feature(self.frame_branch(frame_feature), N_FRAME_CLS)
        self.a2f_attn = self.a2f_layer.attn[0]                               # (1, T, A)
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)      # (1, A, T)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)      # (1, T, A)
        return frame_feature, action_feature

    def compute_loss(self, crit, match):
        f2a_t = torch.transpose(self.f2a_attn_logit, 1, 2)                   # (1, T, A)
        smooth = (smooth_loss_logp(F.log_softmax(self.a2f_attn_logit, dim=2))
                  + smooth_loss_logp(F.log_softmax(f2a_t, dim=2)) + self.frame_smooth())
        return (crit.action_token_loss(match, self.action_clogit)
                + crit.cross_attn_loss(match, f2a_t, dim=1) + crit.cross_attn_loss(match, self.a2f_attn_logit, dim=2)
                + crit.frame_loss(self.frame_clogit) + self.c["sw"] * smooth)


class UpdateBlockTDU(Block):
    """Update block with temporal pooling over the segments of the incoming BIO prediction."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        h = c["hid_dim"]
        self.frame_branch = MSTCN2(c["f_dim"], c["f_dim"], h, c["u_f_layers"], dropout=c["dropout"], in_map=False)
        self.seg_update = nn.GRU(h, h // 2, c["s_layers"], bidirectional=True)
        self.seg_combine = nn.Linear(h, h)
        self.f2a_layer = self.create_cross_attention(c, c["a_dim"])
        sa = SALayer(c["a_dim"], c["a_nhead"], dim_feedforward=c["a_ffdim"], dropout=c["dropout"],
                     attn_dropout=c["dropout"])
        self.action_branch = SADecoder(c["a_dim"], h, sa, c["u_a_layers"])
        self.a2f_layer = self.create_cross_attention(c, c["f_dim"])
        self.sf_merge = nn.Sequential(nn.Linear(h + c["f_dim"], c["f_dim"]), nn.ReLU())

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        pred = frame_feature[:, 0, -N_FRAME_CLS:].argmax(-1).detach().cpu().numpy()
        self.tdu = tdu = TemporalDownsampleUpsample(core.tdu_segments(pred), frame_feature.device)
        seg_feature, _ = self.seg_update(tdu.feature_frame2seg(frame_feature))
        seg_feature = self.seg_combine(torch.relu(seg_feature))
        seg_feature, self.seg_clogit = self.process_feature(seg_feature, N_FRAME_CLS)

        centers = torch.as_tensor([(s + e) // 2 for s, e in tdu.segs], dtype=torch.long, device=frame_feature.device)
        seg_pos = frame_pos[centers]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, self.action_clogit = self.process_feature(action_feature, N_TOK_CLS + 1)
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)

        frame_feature = self.sf_merge(torch.cat([tdu.feature_seg2frame(seg_feature), frame_feature], dim=-1))
        frame_feature, self.frame_clogit = self.process_feature(self.frame_branch(frame_feature), N_FRAME_CLS)

        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)      # (1, A, S)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)      # (1, S, A)
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])           # (1, T, A)
        return frame_feature, action_feature

    def compute_loss(self, crit, match):
        f2a_t = torch.transpose(self.f2a_attn_logit, 1, 2)                   # (1, S, A)
        return ((crit.frame_loss(self.frame_clogit) + crit.frame_loss_tdu(self.seg_clogit, self.tdu)) / 2
                + crit.action_token_loss(match, self.action_clogit)
                + crit.cross_attn_loss_tdu(match, f2a_t, self.tdu, dim=1)
                + crit.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)
                + self.c["sw"] * self.frame_smooth())


class FACTPhrase(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        self.cfg = c = dict(FACT_DEFAULTS, **cfg)
        if not isinstance(c["ntoken"], int):
            raise ValueError("ntoken must be resolved to an int before building the model (train_fact.py does this).")
        if c["block"][0] != "i" or any(b not in "uU" for b in c["block"][1:]):
            raise ValueError(f"block must be 'i' followed by u/U, got {c['block']!r}")
        self.frame_pe = PositionalEncoding(c["hid_dim"], empty=not c["fpos"])
        self.action_query = nn.Parameter(torch.randn(c["ntoken"], 1, c["a_dim"]))
        blocks = [InputBlock(c, in_dim)]
        blocks += [UpdateBlock(c) if t == "u" else UpdateBlockTDU(c) for t in c["block"][1:]]
        self.block_list = nn.ModuleList(blocks)
        self.criterion = None                                               # set by attach_criterion

    def attach_criterion(self, nullw, device):
        self.criterion = MatchCriterionBIO(self.cfg, self.cfg["ntoken"], nullw, device)

    def _forward(self, x):
        """x: (T, D) features of ONE window."""
        c = self.cfg
        seq = x.unsqueeze(1)                                                # (T, 1, D)
        frame_pe = self.frame_pe(seq)
        if self.training and c["cmr"] > 0:
            keep = (torch.rand(1, 1, seq.shape[-1], device=seq.device) >= c["cmr"]).to(seq.dtype)
            seq = seq * keep / (1.0 - c["cmr"])
        if self.training and c["tm_use"]:
            seq = time_mask(seq, c["tm_t"], c["tm_m"], c["tm_p"])
        action_pe = self.action_query
        action_feature = torch.zeros_like(action_pe)
        frame_feature = seq
        for block in self.block_list:
            frame_feature, action_feature = block(frame_feature, action_feature, frame_pe, action_pe)

    def loss(self, x, bio):
        """x (T, D), bio (T,) LongTensor -> scalar loss (FACT: mean over blocks)."""
        self._forward(x)
        crit = self.criterion
        crit.set_label(bio)
        last = self.block_list[-1]
        a2f = last.a2f_attn if hasattr(last, "a2f_attn") else None
        if a2f is None:                                                     # 'i' only: match by class
            a2f = torch.full((1, x.shape[0], self.cfg["ntoken"]), 1.0 / self.cfg["ntoken"], device=x.device)
        match = crit.match(last.action_clogit, a2f)
        losses = [b.compute_loss(crit, match) for b in self.block_list]
        return sum(losses) / len(losses)

    @torch.no_grad()
    def predict(self, x):
        """x (T, D) -> (head_probs (3, T) float32, token_bio (T,) int64), numpy."""
        self._forward(x)
        last = self.block_list[-1]
        logits = last.frame_clogit.squeeze(1).float()                     # (T, 3)
        head = torch.softmax(logits, dim=-1).t().cpu().numpy().astype(np.float32)
        if hasattr(last, "a2f_attn"):
            token = core.token_decode(last.action_clogit.squeeze(1).float().cpu().numpy(),
                                      last.a2f_attn[0].float().cpu().numpy(), logits.cpu().numpy(), self.cfg["mwt"])
        else:
            token = head.argmax(0).astype(np.int64)
        return head, token


# ============================================================================ frozen BiLSTM reference
class FrozenBiLSTM(nn.Module):
    """The R0 encoder (BiLSTM 4 x 256, bidirectional, linear classifier) on frozen fused features.
    input_dropout stands in for the dropout R0's fusion projection applies before its BiLSTM."""

    def __init__(self, cfg, in_dim):
        super().__init__()
        self.cfg = c = dict(BILSTM_DEFAULTS, **cfg)
        self.in_drop = nn.Dropout(c["input_dropout"])
        self.lstm = nn.LSTM(in_dim, c["d_model"], num_layers=c["n_layers"], batch_first=True,
                            dropout=c["dropout"] if c["n_layers"] > 1 else 0.0, bidirectional=True)
        self.classifier = nn.Linear(2 * c["d_model"], N_FRAME_CLS)
        self.bio_w = None

    def forward(self, x):                                                   # (B, T, D) -> (B, 3, T)
        h, _ = self.lstm(self.in_drop(x))
        return self.classifier(h).permute(0, 2, 1)

    def loss_batch(self, x, targets):
        """x (B, T, D), targets (B, 3, T) one-hot with all-zero padding columns."""
        c = self.cfg
        if self.bio_w is None or self.bio_w.device != x.device:
            self.bio_w = torch.tensor(c["class_weights"], dtype=torch.float32, device=x.device)
        return boundary_aware_loss(self(x).float(), targets, self.bio_w, soft=False,
                                   count_weight=c["count_loss_weight"], radius=c["count_loss_radius"],
                                   tau=c["count_loss_tau"], mode=c["count_loss_mode"])

    @torch.no_grad()
    def predict_batch(self, x):
        return torch.softmax(self(x).float(), dim=1).cpu().numpy().astype(np.float32)


def build_model(cfg, in_dim):
    if cfg["model"] == "fact":
        return FACTPhrase(cfg, in_dim)
    if cfg["model"] == "frozen_bilstm":
        return FrozenBiLSTM(cfg, in_dim)
    raise ValueError(f"unknown model {cfg['model']!r}")


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
