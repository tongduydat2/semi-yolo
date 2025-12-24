"""
Loss functions for SSOD training
Total Loss = Loss_Supervised + Lambda * Loss_Unsupervised
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class SSODLoss:
    """
    Combined loss for Semi-Supervised Object Detection.
    """
    
    def __init__(self,
                 lambda_unsup: float = 2.0,
                 burn_in_epochs: int = 15):
        """
        Args:
            lambda_unsup: Weight for unsupervised loss (Lambda)
            burn_in_epochs: Number of epochs with Lambda=0
        """
        self.lambda_unsup = lambda_unsup
        self.burn_in_epochs = burn_in_epochs
        
        self.loss_history = {
            'supervised': [],
            'unsupervised': [],
            'total': []
        }
        
    def get_current_lambda(self, epoch: int) -> float:
        """Get current unsupervised weight based on epoch."""
        if epoch < self.burn_in_epochs:
            return 0.0
        return self.lambda_unsup
        
    def compute_total_loss(self,
                          loss_sup: torch.Tensor,
                          loss_unsup: Optional[torch.Tensor] = None,
                          epoch: int = 0) -> torch.Tensor:
        """
        Compute total SSOD loss.
        
        Args:
            loss_sup: Supervised loss from labeled data
            loss_unsup: Unsupervised loss from pseudo-labels
            epoch: Current epoch (for burn-in)
            
        Returns:
            Combined loss
        """
        current_lambda = self.get_current_lambda(epoch)
        
        if loss_unsup is not None and current_lambda > 0:
            total_loss = loss_sup + current_lambda * loss_unsup
            unsup_val = loss_unsup.item()
        else:
            total_loss = loss_sup
            unsup_val = 0.0
        
        # Log
        self.loss_history['supervised'].append(loss_sup.item())
        self.loss_history['unsupervised'].append(unsup_val)
        self.loss_history['total'].append(total_loss.item())
        
        return total_loss
    
    def get_average_losses(self, window: int = 100) -> Dict[str, float]:
        """Get average losses over recent window."""
        def avg(lst):
            recent = lst[-window:]
            return sum(recent) / max(1, len(recent))
        
        return {
            'supervised': avg(self.loss_history['supervised']),
            'unsupervised': avg(self.loss_history['unsupervised']),
            'total': avg(self.loss_history['total'])
        }
    
    def reset(self):
        """Reset loss history."""
        self.loss_history = {'supervised': [], 'unsupervised': [], 'total': []}


class ConsistencyLoss(nn.Module):
    """Consistency loss between Student and Teacher predictions."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.mse = nn.MSELoss()
        
    def forward(self, student_preds: torch.Tensor, teacher_preds: torch.Tensor) -> torch.Tensor:
        student_soft = student_preds / self.temperature
        teacher_soft = teacher_preds / self.temperature
        return self.mse(student_soft, teacher_soft)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
