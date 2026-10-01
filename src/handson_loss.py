"""
src/handson_loss.py -- loss for the 2025 Hands-On recipe: frame-level BIO cross-entropy + sign-level CTC.

    total = CE(bio_logits, hard_labels) + ctc_weight * CTC_per_token(ctc_logits, signs-from-BIO)

(additive, as the paper's "unified loss combining frame-level and gloss-level objectives"; its weighting is not
stated, so ctc_weight is a hyper-parameter.) The CTC is computed on a DEDICATED CTC head's logits
(B, T', num_tokens + 1), at the Transformer's (downsampled) frame rate; blank = class 0. See handson_ctc_core.py for
how targets are derived from BIO tags and why some windows are excluded. This is independent of
src/loss.py's UnifiedCTCLoss.

forward(bio_logits (B,3,T), ctc_logits (B,T',K), hard_labels (B,T) long) -> (total, ce, ctc)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.handson_ctc_core import count_signs, ctc_feasible
except ImportError:                       # allows running the file from its own folder
    from handson_ctc_core import count_signs, ctc_feasible


class HandsOnCTCLoss(nn.Module):
    def __init__(self, ctc_weight=0.5, class_weights=None, blank=0, token=1):
        super().__init__()
        self.ctc_weight = ctc_weight
        self.class_weights = class_weights        # optional tensor (3,), already on the right device
        self.blank, self.token = blank, token
        self.n_seen = 0                           # windows seen / excluded from CTC (for logging)
        self.n_excluded = 0

    def excluded_fraction(self):
        return self.n_excluded / max(self.n_seen, 1)

    def forward(self, bio_logits, ctc_logits, hard_labels):
        ce = F.cross_entropy(bio_logits, hard_labels, weight=self.class_weights)
        lab = hard_labels.detach().cpu().numpy()
        B, T_out = lab.shape[0], ctc_logits.shape[1]
        n_signs = [count_signs(lab[b]) for b in range(B)]
        keep = [b for b in range(B) if ctc_feasible(n_signs[b], T_out)]
        self.n_seen += B
        self.n_excluded += B - len(keep)
        if not keep:
            return ce, ce, ce.new_zeros(())

        dev = ctc_logits.device
        log_probs = F.log_softmax(ctc_logits.float(), dim=-1)                   # (B, T', K)
        lp = log_probs[torch.as_tensor(keep, device=dev)].permute(1, 0, 2).contiguous()   # (T', n, K)
        tgt_len = torch.tensor([n_signs[b] for b in keep], dtype=torch.long, device=dev)
        n_tok = int(tgt_len.sum().item())
        if n_tok == 0:                                                          # every kept window has no sign
            per = -lp[:, :, self.blank].sum(0)                                  # all-blank path is the only path
        else:
            targets = torch.full((n_tok,), self.token, dtype=torch.long, device=dev)
            in_len = torch.full((len(keep),), T_out, dtype=torch.long, device=dev)
            per = F.ctc_loss(lp, targets, in_len, tgt_len, blank=self.blank,
                             reduction="none", zero_infinity=False)
        ctc = (per / tgt_len.clamp(min=1).float()).mean()                       # per-token NLL, mean over windows
        return ce + self.ctc_weight * ctc, ce, ctc