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
        # Ensure weights is a float tensor
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        # Move weights to the same device as inputs (GPU)
        self.weights = self.weights.to(inputs.device)
        return F.cross_entropy(inputs, targets, weight=self.weights)

# ==========================================
# 3. Focal Loss & BCL (Your existing work)
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
        
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        
        b_mask = (hard_targets == 2)
        non_b_mask = (hard_targets != 2)
        
        b_embeddings = embeddings[b_mask]
        non_b_embeddings = embeddings[non_b_mask]
        
        if len(b_embeddings) == 0 or len(non_b_embeddings) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        b_centroid = b_embeddings.mean(dim=0, keepdim=True)
        b_centroid = F.normalize(b_centroid, p=2, dim=1, eps=1e-8)
        
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
        self.lambda_w = contrastive_weight 

    def forward(self, logits, embeddings, targets):
        loss_f = self.focal(logits, targets)
        loss_c = self.contrastive(embeddings, targets)
        total_loss = loss_f + (self.lambda_w * loss_c)
        return total_loss, loss_f, loss_c