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
        """
        inputs: (B, 3, T) logits
        targets: Can now be Soft Labels (B, 3, T) OR Hard Labels (B, T)
        """
        self.alpha = self.alpha.to(inputs.device)
        
        # PyTorch F.cross_entropy automatically handles soft targets (probabilities)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Dynamically apply alpha weighting depending on target type
        if targets.dim() == inputs.dim():
            # Soft Labels: Weight is the dot product of probabilities and alpha
            alpha_t = torch.einsum('bct,c->bt', targets, self.alpha)
        else:
            # Hard Labels (Fallback): Standard integer indexing
            alpha_t = self.alpha[targets]
            
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss