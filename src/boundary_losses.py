"""
Training losses for the two boundary studies (train_phrase.py, loss_function "weighted_ce").

  soft_weighted_ce      cross-entropy against SOFT targets (B, 3, T), class-weighted exactly
                        like PyTorch's weighted mean: sum_t sum_c w_c q_tc (-log p_tc) /
                        sum_t sum_c w_c q_tc. With one-hot targets it equals
                        F.cross_entropy(logits, hard, weight=w). Padding frames (all-zero
                        target) drop out automatically.
                        Used by the AMBIGUITY study: targets from make_soft_labels.py, where a
                        phrase start in a context the annotation itself treats inconsistently
                        (e.g. after palm-up) is not a hard 1 and a non-start there not a hard 0.

  boundary_count_loss   "one start per boundary": over every window of 2r+1 frames, the
                        number of frames the decoder would call B should equal the number of
                        gold phrase starts in that window (0 or 1). "Would call B" is the soft
                        argmax indicator s_t = sigmoid((P(B) - max(P(O), P(I))) / tau), so two
                        B peaks a few frames apart -- the stutter that duplicate removal fixes
                        after the fact -- count ~2 and are penalised, while ONE peak anywhere
                        within +-r of the gold start is (almost) free: the loss is tolerant to
                        small offsets, like the boundary metric. Squared error, averaged over
                        valid frames. Used by the COUNT study.
                        mode="mass" uses s_t = P(B) instead; on toy cases that barely separates
                        a stutter (2 x 0.6) from a single clean peak, hence the default "argmax".
                        The sigmoid saturates for clearly-not-B frames, so the term mostly acts
                        on frames near the B decision boundary (extra B frames); finding missed
                        boundaries is left to the cross-entropy term.

  boundary_aware_loss   soft_weighted_ce (or plain weighted CE) + count_weight * count loss.
"""
import torch
import torch.nn.functional as F

B_TAG = 2


def soft_weighted_ce(logits, targets, weight=None):
    """logits, targets: (B, 3, T); weight: (3,) or None."""
    logp = F.log_softmax(logits.float(), dim=1)
    q = targets.float()
    if weight is not None:
        q_w = q * weight.float().view(1, -1, 1)
    else:
        q_w = q
    den = q_w.sum().clamp_min(1e-8)
    return -(q_w * logp).sum() / den


def boundary_count_loss(logits, targets, radius=5, tau=0.1, mode="argmax"):
    """
    logits, targets: (B, 3, T). Gold starts = frames whose target argmax is B (works for hard
    and soft targets as long as soft targets keep the gold class on top). Padding frames
    (all-zero target) are excluded from both the predicted and the gold counts.
    """
    prob = F.softmax(logits.float(), dim=1)
    if mode == "mass":
        p_b = prob[:, B_TAG]                                             # (B, T)
    elif mode == "argmax":
        other = torch.maximum(prob[:, 0], prob[:, 1])
        p_b = torch.sigmoid((prob[:, B_TAG] - other) / float(tau))      # soft "argmax is B"
    else:
        raise ValueError(f"count loss mode must be 'argmax' or 'mass', got '{mode}'")
    valid = (targets.sum(dim=1) > 0).float()                             # (B, T)
    gold_b = (targets.argmax(dim=1) == B_TAG).float() * valid
    k = 2 * int(radius) + 1
    ker = torch.ones(1, 1, k, device=logits.device, dtype=p_b.dtype)
    pred_count = F.conv1d((p_b * valid).unsqueeze(1), ker, padding=int(radius)).squeeze(1)
    gold_count = F.conv1d(gold_b.unsqueeze(1), ker, padding=int(radius)).squeeze(1)
    err = (pred_count - gold_count) ** 2 * valid
    return err.sum() / valid.sum().clamp_min(1.0)


def boundary_aware_loss(logits, targets, weight=None, soft=True, count_weight=0.0, radius=5,
                        tau=0.1, mode="argmax"):
    """
    soft=True  : soft_weighted_ce against the targets as given.
    soft=False : weighted CE against argmax(targets) (= the usual training loss).
    count_weight > 0 adds count_weight * boundary_count_loss.
    """
    if soft:
        loss = soft_weighted_ce(logits, targets, weight)
    else:
        hard = targets.argmax(dim=1).masked_fill(targets.sum(dim=1) == 0, -100)
        loss = F.cross_entropy(logits.float(), hard, weight=None if weight is None else weight.float(),
                               ignore_index=-100)
    if count_weight and count_weight > 0:
        loss = loss + float(count_weight) * boundary_count_loss(logits, targets, radius, tau, mode)
    return loss