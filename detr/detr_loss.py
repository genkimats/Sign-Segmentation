"""
Bipartite (Hungarian) matching loss for DETR-style set prediction, adapted
for 1D temporal segments instead of 2D bounding boxes.

For each forward call: builds a cost matrix between all N_pred predicted
segments and N_true ground-truth segments (classification cost + L1 span
cost + IoU cost), finds the OPTIMAL one-to-one assignment via the Hungarian
algorithm (scipy.optimize.linear_sum_assignment), then computes the final
loss: matched pairs get a "real segment" classification target plus span
regression loss; unmatched predictions get a "no object" classification
target and no span loss (there's nothing to regress against).

Requires scipy (pip install scipy if not already present -- very likely
already is, given how common a dependency it is).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class DETRSegmentLoss(nn.Module):
    def __init__(self, class_weight=1.0, l1_weight=5.0, iou_weight=2.0, no_object_weight=0.1):
        super().__init__()
        self.class_weight = class_weight
        self.l1_weight = l1_weight
        self.iou_weight = iou_weight
        # Downweights the classification loss for unmatched ("no object") queries --
        # there are usually far more of these than real segments per video, so
        # without downweighting, the loss would be dominated by "predict no object"
        # rather than genuine segment localization. Matches the original DETR
        # paper's own convention for exactly this imbalance.
        self.no_object_weight = no_object_weight

    @staticmethod
    def span_iou(spans_a, spans_b):
        """spans_a: (Na, 2), spans_b: (Nb, 2), both (start_frac, end_frac) in
        [0,1]. Returns (Na, Nb) pairwise IoU matrix."""
        start_a, end_a = spans_a[:, 0:1], spans_a[:, 1:2]  # (Na,1)
        start_b, end_b = spans_b[:, 0], spans_b[:, 1]      # (Nb,)

        inter_start = torch.maximum(start_a, start_b.unsqueeze(0))  # (Na, Nb)
        inter_end = torch.minimum(end_a, end_b.unsqueeze(0))
        intersection = (inter_end - inter_start).clamp(min=0)

        union_a = end_a - start_a          # (Na,1)
        union_b = end_b - start_b          # (Nb,)
        union = union_a + union_b.unsqueeze(0) - intersection

        return intersection / union.clamp(min=1e-6)

    def forward(self, confidence_logits, pred_spans, true_segments_frac):
        """
        confidence_logits: (1, N_pred)
        pred_spans: (1, N_pred, 2)
        true_segments_frac: (N_true, 2) -- ground truth for this ONE video (batch_size=1)

        Returns (total_loss, class_loss, l1_loss, iou_loss) -- the three
        components returned separately too, so you can log them individually
        during training (exactly the "log both raw components before trusting
        the combined loss" practice recommended for the CTC loss a few
        messages back -- same principle applies to any multi-term loss).
        """
        confidence_logits = confidence_logits[0]  # (N_pred,)
        pred_spans = pred_spans[0]                # (N_pred, 2)
        N_pred = confidence_logits.shape[0]
        N_true = true_segments_frac.shape[0]

        confidence_probs = torch.sigmoid(confidence_logits)

        if N_true == 0:
            # No ground-truth segments at all (shouldn't normally happen -- the
            # dataset skips videos with zero segments -- but handled defensively):
            # every prediction should say "no object".
            zero_loss = torch.tensor(0.0, device=confidence_logits.device)
            class_loss = F.binary_cross_entropy(confidence_probs, torch.zeros_like(confidence_probs))
            return class_loss, class_loss, zero_loss, zero_loss

        # --- Build cost matrix (N_pred, N_true) ---
        cost_class = -confidence_probs.unsqueeze(1).expand(N_pred, N_true)
        cost_l1 = torch.cdist(pred_spans, true_segments_frac, p=1)
        iou = self.span_iou(pred_spans, true_segments_frac)
        cost_iou = -iou

        cost_matrix = self.class_weight * cost_class + self.l1_weight * cost_l1 + self.iou_weight * cost_iou

        # Hungarian matching runs on CPU/numpy -- detach and move, matching is
        # not itself a differentiable operation (only the resulting matched
        # losses, computed below on the original tensors, carry gradients).
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        pred_indices, true_indices = linear_sum_assignment(cost_matrix_np)
        pred_indices = torch.as_tensor(pred_indices, device=confidence_logits.device, dtype=torch.long)
        true_indices = torch.as_tensor(true_indices, device=confidence_logits.device, dtype=torch.long)

        # Classification targets: 1 for matched queries, 0 for unmatched
        class_targets = torch.zeros(N_pred, device=confidence_logits.device)
        class_targets[pred_indices] = 1.0

        class_sample_weights = torch.full((N_pred,), self.no_object_weight, device=confidence_logits.device)
        class_sample_weights[pred_indices] = 1.0

        class_loss = F.binary_cross_entropy(confidence_probs, class_targets, weight=class_sample_weights)

        # Span losses -- ONLY for matched pairs (nothing to regress unmatched
        # predictions against).
        matched_pred_spans = pred_spans[pred_indices]
        matched_true_spans = true_segments_frac[true_indices]

        l1_loss = F.l1_loss(matched_pred_spans, matched_true_spans)
        matched_iou = self.span_iou(matched_pred_spans, matched_true_spans).diagonal()
        iou_loss = (1 - matched_iou).mean()

        total_loss = (self.class_weight * class_loss + self.l1_weight * l1_loss + self.iou_weight * iou_loss)

        return total_loss, class_loss, l1_loss, iou_loss