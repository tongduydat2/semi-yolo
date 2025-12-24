"""
Exponential Moving Average (EMA) for Teacher Model updates.
Teacher weights: W_T = alpha * W_T + (1 - alpha) * W_S
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Any


class EMAUpdater:
    """
    Exponential Moving Average updater for Teacher-Student framework.
    The Teacher model is updated as a slow-moving average of the Student.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 decay: float = 0.999,
                 warmup_steps: int = 0):
        """
        Args:
            model: The Student model to create EMA copy from
            decay: EMA decay rate (alpha), higher = slower updates
            warmup_steps: Number of steps before starting EMA updates
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        
        # Create EMA model (Teacher) as deep copy
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        
        # Freeze EMA model gradients
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
            
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA (Teacher) parameters from Student.
        
        Formula: W_ema = decay * W_ema + (1 - decay) * W_model
        
        Args:
            model: Source model (Student)
        """
        self.step += 1
        
        # Skip updates during warmup
        if self.step <= self.warmup_steps:
            return
            
        # Compute effective decay (gradual ramp up)
        if self.step <= self.warmup_steps + 1000:
            progress = (self.step - self.warmup_steps) / 1000
            decay = min(self.decay, 1 - (1 - self.decay) * (1 + progress) / 2)
        else:
            decay = self.decay
        
        # Update EMA parameters
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        for name in model_params:
            if name in ema_params:
                ema_params[name].data.mul_(decay).add_(
                    model_params[name].data, alpha=1 - decay
                )
        
        # Update buffers (BatchNorm running stats, etc.)
        model_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.ema_model.named_buffers())
        
        for name in model_buffers:
            if name in ema_buffers:
                ema_buffers[name].data.copy_(model_buffers[name].data)
    
    def get_model(self) -> nn.Module:
        """Get the EMA model (Teacher)."""
        return self.ema_model
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
            'step': self.step,
            'warmup_steps': self.warmup_steps
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict['decay']
        self.step = state_dict['step']
        self.warmup_steps = state_dict['warmup_steps']


def copy_weights(source: nn.Module, target: nn.Module):
    """Copy weights from source model to target model."""
    target.load_state_dict(source.state_dict())
