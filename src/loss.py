import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.tensor([0.1, 0.4, 0.5])
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if targets.dim() == inputs.dim():
            alpha_t = torch.einsum('bct,c->bt', targets, self.alpha)
        else:
            alpha_t = self.alpha[targets]
            
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

class BoundaryContrastiveLoss(nn.Module):
    """
    Forces the latent embeddings of 'Begin' frames to cluster together,
    while pushing 'Inside' and 'Outside' frames away from the Begin centroid.
    """
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin 

    def forward(self, embeddings, targets):
        if targets.dim() == 3:
            hard_targets = torch.argmax(targets, dim=1)
        else:
            hard_targets = targets
            
        B, D, T = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1).reshape(-1, D) 
        hard_targets = hard_targets.reshape(-1) 
        
        # --- FIX 1: Safe Normalization ---
        # eps prevents division by zero if an embedding collapses
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        
        b_mask = (hard_targets == 2)
        non_b_mask = (hard_targets != 2)
        
        b_embeddings = embeddings[b_mask]
        non_b_embeddings = embeddings[non_b_mask]
        
        # If no boundaries exist in this batch, return 0 loss gracefully
        if len(b_embeddings) == 0 or len(non_b_embeddings) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        b_centroid = b_embeddings.mean(dim=0, keepdim=True)
        # --- FIX 2: Safe Normalization on Centroid ---
        b_centroid = F.normalize(b_centroid, p=2, dim=1, eps=1e-8)
        
        # --- FIX 3: Clamping Cosine Similarity ---
        # Floating point inaccuracies can sometimes cause cosine sim to be 1.0000001
        # which can cause NaN in subsequent backward gradient calculations.
        pull_sim = F.cosine_similarity(b_embeddings, b_centroid)
        pull_sim = torch.clamp(pull_sim, min=-0.999, max=0.999)
        pull_loss = (1.0 - pull_sim).mean()
        
        push_sim = F.cosine_similarity(non_b_embeddings, b_centroid)
        push_sim = torch.clamp(push_sim, min=-0.999, max=0.999)
        push_loss = F.relu(push_sim - self.margin).mean()
        
        return pull_loss + push_loss

class CombinedBoundaryLoss(nn.Module):
    def __init__(self, focal_gamma=2.0, contrastive_weight=0.15):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.contrastive = BoundaryContrastiveLoss()
        
        # Lambda weight: How much importance to give the Contrastive Loss
        self.lambda_w = contrastive_weight 

    def forward(self, logits, embeddings, targets):
        loss_f = self.focal(logits, targets)
        loss_c = self.contrastive(embeddings, targets)
        
        total_loss = loss_f + (self.lambda_w * loss_c)
        return total_loss, loss_f, loss_c