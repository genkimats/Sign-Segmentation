import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Standard Cross-Entropy
# ==========================================
class StandardCrossEntropyLoss(nn.Module):
    """Treats all classes equally."""
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets)

# ==========================================
# 2. Weighted Cross-Entropy
# ==========================================
class WeightedCrossEntropyLoss(nn.Module):
    """Applies a manual multiplier to specific classes to combat imbalance."""
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        self.weights = self.weights.to(inputs.device)
        return F.cross_entropy(inputs, targets, weight=self.weights)

# ==========================================
# 3. Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.tensor([0.1, 0.4, 0.5])
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==========================================
# 4. Boundary Contrastive Loss (BCL)
# ==========================================
class BoundaryContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, targets):
        B, C, T = embeddings.shape
        loss = 0.0
        valid_batches = 0

        for b in range(B):
            begin_mask = (targets[b] == 2)
            other_mask = (targets[b] != 2)

            if not begin_mask.any() or not other_mask.any():
                continue

            begin_emb = embeddings[b, :, begin_mask].transpose(0, 1)
            other_emb = embeddings[b, :, other_mask].transpose(0, 1)

            begin_mean = begin_emb.mean(dim=0)
            other_mean = other_emb.mean(dim=0)

            dist = F.pairwise_distance(begin_mean.unsqueeze(0), other_mean.unsqueeze(0))
            batch_loss = torch.clamp(self.margin - dist, min=0.0)
            
            loss += batch_loss.sum()
            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return loss / valid_batches

# ==========================================
# 5. Combined Boundary Loss (Focal + BCL)
# ==========================================
class CombinedBoundaryLoss(nn.Module):
    def __init__(self, focal_gamma=2.0, contrastive_weight=0.15, margin=1.0):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.bcl = BoundaryContrastiveLoss(margin=margin)
        self.contrastive_weight = contrastive_weight

    def forward(self, logits, embeddings, soft_targets):
        hard_targets = torch.argmax(soft_targets, dim=1)
        
        loss_focal = self.focal(logits, hard_targets)
        loss_bcl = self.bcl(embeddings, hard_targets)
        
        total_loss = loss_focal + (self.contrastive_weight * loss_bcl)
        return total_loss, loss_focal, loss_bcl

# ==========================================
# 6. Unified CTC Loss
# ==========================================
class UnifiedCTCLoss(nn.Module):
    def __init__(self, blank_idx=0, ctc_weight=0.5):
        super().__init__()
        self.ce = StandardCrossEntropyLoss()
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        self.ctc_weight = ctc_weight

    def extract_ctc_targets(self, hard_targets):
        """
        Builds CORRECT CTC targets: the per-frame BIO sequence collapsed by
        removing consecutive duplicate frames (standard CTC target
        construction -- the same operation CTC decoding itself performs in
        reverse), then removing the blank class (0=Outside) entirely, since
        CTC targets must NEVER contain the blank index -- it's implicitly
        available for the aligner to insert anywhere at near-zero cost, not
        something to predict explicitly.

        Example: a frame sequence "OOOBIIIOOOBIIIOOO" (O/B/I = 0/1/2)
        collapses to "OBIOBIO", then drops the O's/blanks, giving the final
        CTC target "BIBI" == [1,2,1,2] -- preserving the actual B->I
        structure for each sign. (Previously this discarded the B->I
        structure entirely and just emitted a run of 1's, one per detected
        sign -- e.g. [1,1] for two signs, regardless of their actual
        content. That's a degenerate target that gives CTC almost nothing
        real to align against, which is the most likely reason CTC hurt
        results last time, not that CTC itself is a bad fit here.)
        """
        ctc_targets = []
        target_lengths = []
        B, T = hard_targets.shape

        for b in range(B):
            seq = hard_targets[b]

            change_points = torch.ones_like(seq, dtype=torch.bool)
            change_points[1:] = seq[1:] != seq[:-1]
            collapsed = seq[change_points]

            non_blank = collapsed[collapsed != 0]

            ctc_targets.append(non_blank)
            target_lengths.append(non_blank.numel())

        target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long, device=hard_targets.device)

        non_empty = [t for t in ctc_targets if t.numel() > 0]
        if not non_empty:
            return torch.tensor([], dtype=torch.long, device=hard_targets.device), target_lengths_tensor

        return torch.cat(non_empty).long(), target_lengths_tensor

    def forward(self, logits, hard_targets):
        loss_ce = self.ce(logits, hard_targets)
        ctc_targets, target_lengths = self.extract_ctc_targets(hard_targets)
        
        if ctc_targets.numel() == 0:
            return loss_ce, loss_ce, torch.tensor(0.0, device=logits.device)
            
        log_probs = F.log_softmax(logits, dim=1)
        log_probs = log_probs.permute(2, 0, 1)
        
        T_len, B_size, _ = log_probs.shape
        # KNOWN REMAINING ISSUE: this assumes every sample in the batch fills the
        # full window (T_len), which is WRONG for the last (padded) window of any
        # video shorter than a clean multiple of window_size -- dataset.py pads
        # incomplete windows with Outside/blank frames, so this counts padding as
        # real input. Fixing this properly needs the actual per-sample valid
        # length threaded through from dataset.py's __getitem__ into train.py and
        # then here -- padding is labeled identically to genuine Outside content,
        # so it can't be reliably inferred from hard_targets alone at this point.
        # Left as full-window here (same behavior as before) since fixing it
        # requires a wider pipeline change; flagging clearly rather than
        # silently leaving it undocumented.
        input_lengths = torch.full((B_size,), T_len, dtype=torch.long, device=logits.device)
        
        loss_ctc = self.ctc(log_probs, ctc_targets, input_lengths, target_lengths)
        total_loss = (1 - self.ctc_weight) * loss_ce + self.ctc_weight * loss_ctc
        return total_loss, loss_ce, loss_ctc

# ==========================================
# 7. Smoothing Truncated MSE Loss
# ==========================================
class SmoothingTruncatedMSELoss(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, logits):
        probs = F.softmax(logits, dim=1)
        diff = probs[:, :, 1:] - probs[:, :, :-1]
        mse = diff ** 2
        tmse = torch.clamp(mse, max=self.threshold)
        return tmse.mean()

class WeightedCE_TMSE_Loss(nn.Module):
    def __init__(self, weights, tmse_weight=0.15, threshold=0.1):
        super().__init__()
        self.wce = WeightedCrossEntropyLoss(weights=weights)
        self.tmse = SmoothingTruncatedMSELoss(threshold=threshold)
        self.tmse_weight = tmse_weight

    def forward(self, logits, targets):
        loss_wce = self.wce(logits, targets)
        loss_tmse = self.tmse(logits)
        return loss_wce + (self.tmse_weight * loss_tmse)

# ==========================================
# 8. Weighted Negative Log-Likelihood
# ==========================================
class WeightedNLLLoss(nn.Module):
    """
    Negative Log-Likelihood Loss with class weights to combat extreme imbalance.
    Expects raw logits (B, C, T) and converts them to log probabilities internally.
    """
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        self.weights = self.weights.to(logits.device)
        
        # NLL requires Log-Probabilities, so we apply LogSoftmax to the class dimension (dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        return F.nll_loss(log_probs, targets, weight=self.weights)