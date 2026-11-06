"""
Sign-head building blocks for the multi-stream phrase model (SIGN_HEAD_STUDY in queue_train_phrase.py).

Why: every phrase starts at a sign start, the remaining phrase errors are mostly about WHERE starts
are placed, and gold sign timing / identity add information the video model does not extract
(thesis_error_stats.py, section 4). These modules let the phrase decision use the sign head.

  soft_tempo_features   differentiable sign-timing channels from the sign head's probabilities:
                        P(O), P(I), P(B), log pause before each frame, the pause relative to the
                        window's typical pause, and relative to the largest pause within +-1 s
  starts_from_labels    sign starts of gold sign BIO targets (training, teacher forcing)
  starts_from_logits    sign starts predicted by the sign head (argmax, duplicates < min_dist removed)
  SignLevelDecoder      idea 4: one feature vector per sign (mean over the sign + its first frame +
                        the timing channels at its start), a BiLSTM over the SEQUENCE OF SIGNS, and
                        one logit per sign = "a phrase starts at this sign"
  segment_targets       which candidate sign starts are phrase starts (each gold phrase start goes to
                        its nearest candidate within +-tol frames)
  combine_with_frames   frame-level output for the usual evaluation / decoders: P(B) = the sign-level
                        probability at sign starts and 0 elsewhere, P(O) / P(I) from the frame head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

O_TAG, I_TAG, B_TAG = 0, 1, 2
TEMPO_DIM = 6


def soft_tempo_features(sign_probs, max_pause=100, local=50, eps=1e-4):
    """
    sign_probs (B, 3, T) softmax of the sign head -> (B, 6, T):
      p_O, p_I, p_B,
      log1p(L), L = expected length of the run of sign-O frames that ends just BEFORE frame t
               (soft: sum over d of prod P(O) over the d frames before t, d < max_pause = 2 s),
      log1p(L) - its P(B)-weighted mean over the window (pause relative to this signer's pauses
               in the window; 512 frames ~ 10 s),
      log1p(L) - the largest P(B)-weighted log pause within +-local frames (is this the biggest
               pause around here?).
    """
    pO = sign_probs[:, O_TAG].clamp_min(eps)
    B, T = pO.shape
    c = torch.cumsum(torch.log(pO), dim=1)                              # c[t] = sum_{j<=t} log pO
    c0 = F.pad(c, (1, 0))                                               # c0[k] = c[k-1], c0[0] = 0
    t_idx = torch.arange(T, device=pO.device)
    L = torch.zeros_like(pO)
    for d in range(min(int(max_pause), T)):                             # run of O over [t-d, t]
        k = t_idx - d
        valid = (k >= 0).to(pO.dtype)
        L = L + torch.exp((c - c0[:, k.clamp_min(0)]).clamp(max=0.0)) * valid
    L_prev = F.pad(L[:, :-1], (1, 0))                                    # pause ending at t-1
    logL = torch.log1p(L_prev)
    pB = sign_probs[:, B_TAG]
    ref = (pB * logL).sum(1, keepdim=True) / (pB.sum(1, keepdim=True) + eps)
    lmax = F.max_pool1d((pB * logL)[:, None], kernel_size=2 * local + 1, stride=1, padding=local)[:, 0]
    return torch.stack([sign_probs[:, O_TAG], sign_probs[:, I_TAG], pB, logL, logL - ref, logL - lmax], dim=1)


def _starts_of_classes(cls, valid=None):
    """cls (T,) class ids -> LongTensor of segment starts (a B frame, or I right after O / the window start)."""
    T = cls.shape[0]
    prev = torch.cat([torch.zeros(1, dtype=cls.dtype, device=cls.device), cls[:-1]])
    start = (cls == B_TAG) | ((cls == I_TAG) & (prev == O_TAG))
    start[0] = start[0] | (cls[0] == I_TAG)
    start = start & ~((cls == B_TAG) & (prev == B_TAG))                 # a run of B = one start
    if valid is not None:
        start = start & valid
    return start.nonzero(as_tuple=True)[0]


def _thin(idx, min_dist):
    if len(idx) <= 1 or min_dist <= 1:
        return idx
    keep = [int(idx[0])]
    for t in idx[1:].tolist():
        if t - keep[-1] >= min_dist:
            keep.append(t)
    return torch.tensor(keep, dtype=torch.long, device=idx.device)


def starts_from_labels(sign_targets):
    """sign_targets (B, T) with -100 padding -> list of LongTensor."""
    out = []
    for row in sign_targets:
        valid = row >= 0
        out.append(_starts_of_classes(row.clamp_min(0), valid))
    return out


@torch.no_grad()
def starts_from_logits(sign_logits, min_dist=3):
    """sign_logits (B, 3, T) -> list of LongTensor (argmax starts, closer than min_dist merged)."""
    cls = sign_logits.argmax(dim=1)
    return [_thin(_starts_of_classes(row), min_dist) for row in cls]


def segment_targets(starts, phrase_targets, tol=4):
    """
    starts: list of LongTensor (candidate sign starts); phrase_targets (B, T) hard phrase BIO with
    -100 padding. Returns a list of float tensors (one 0/1 per candidate): each gold phrase start is
    given to its nearest candidate within +-tol frames (at most one positive per gold start).
    """
    out = []
    for s, row in zip(starts, phrase_targets):
        y = torch.zeros(len(s), device=row.device)
        if len(s):
            gold = _starts_of_classes(row.clamp_min(0), row >= 0)
            for g in gold.tolist():
                dist = (s - g).abs()
                j = int(torch.argmin(dist))
                if int(dist[j]) <= tol:
                    y[j] = 1.0
        out.append(y)
    return out


class SignLevelDecoder(nn.Module):
    def __init__(self, emb_dim, hidden=128, dropout=0.2, tempo_dim=TEMPO_DIM):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * emb_dim + tempo_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, emb, tempo, starts):
        """emb (B, T, E), tempo (B, 6, T), starts list of LongTensor -> list of (K_b,) logits."""
        B, T, E = emb.shape
        feats, lens = [], []
        for b in range(B):
            s = starts[b]
            if len(s) == 0:
                feats.append(emb.new_zeros(1, 2 * E + tempo.shape[1]))
                lens.append(0)
                continue
            ends = torch.cat([s[1:], torch.tensor([T], device=s.device)])
            cs = F.pad(torch.cumsum(emb[b], dim=0), (0, 0, 1, 0))           # (T + 1, E)
            mean = (cs[ends] - cs[s]) / (ends - s).clamp_min(1)[:, None].to(emb.dtype)
            feats.append(torch.cat([mean, emb[b, s], tempo[b, :, s].T], dim=1))
            lens.append(len(s))
        x = self.inp(pad_sequence(feats, batch_first=True))                  # (B, Kmax, H)
        lengths = torch.tensor([max(1, n) for n in lens])
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        logits = self.out(h)[..., 0]                                         # (B, Kmax)
        return [logits[b, :n] for b, n in enumerate(lens)]


def combine_with_frames(frame_logits, starts, seg_logits, eps=1e-6):
    """
    frame_logits (B, 3, T), starts / seg_logits lists -> log-probabilities (B, 3, T) whose softmax is:
    P(B) = sigmoid(sign-level logit) at candidate starts and 0 elsewhere; P(O), P(I) = the frame head's
    O / I split of the remaining mass. Usable by every existing evaluation and decoder.
    """
    p = F.softmax(frame_logits.float(), dim=1)
    pb = torch.zeros_like(p[:, B_TAG])
    for b, (s, lg) in enumerate(zip(starts, seg_logits)):
        if len(s):
            pb[b, s] = torch.sigmoid(lg.float())
    oi = p[:, :B_TAG] / p[:, :B_TAG].sum(1, keepdim=True).clamp_min(eps)
    out = torch.cat([oi * (1.0 - pb)[:, None], pb[:, None]], dim=1)
    return torch.log(out.clamp_min(eps))