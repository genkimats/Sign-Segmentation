import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Tensor of weights for each class (e.g., [0.1, 0.4, 0.5])
        gamma: Focusing parameter. Higher = more focus on hard examples.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Default alpha weighting if none provided:
        # Class 0 (Out) gets low weight, Class 1 (In) medium, Class 2 (Begin) high
        if alpha is None:
            self.alpha = torch.tensor([0.1, 0.4, 0.5])
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        """
        inputs: logits from model of shape (Batch, Classes, SequenceLength) -> (B, 3, 1000)
        targets: ground truth of shape (Batch, SequenceLength) -> (B, 1000)
        """
        # Move alpha to the same device as inputs
        self.alpha = self.alpha.to(inputs.device)
        
        # Calculate standard cross entropy loss (unreduced)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probabilities for the true classes
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha[targets]
        
        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss