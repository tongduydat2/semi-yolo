"""
Selective EMA for Semi-Supervised Learning.
Extends Ultralytics ModelEMA to only update trainable parameters.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics.utils.torch_utils import ModelEMA


class SelectiveModelEMA(ModelEMA):
    """
    Exponential Moving Average (EMA) that respects frozen parameters.
    
    Only updates parameters where requires_grad=True, making it efficient
    for semi-supervised learning with frozen backbone.
    
    Mathematical formulation:
    For each trainable parameter p:
        θ_ema(t) = decay * θ_ema(t-1) + (1 - decay) * θ_student(t)
    
    For frozen parameters:
        θ_ema(t) = θ_ema(t-1)  (no update)
    
    Args:
        model: The student model to track
        decay: EMA decay rate ∈ [0, 1]. Higher = slower update
        tau: Time constant for decay scheduler (unused in base implementation)
        updates: Number of EMA updates performed (for decay scheduling)
    
    Example:
        >>> model = DetectionModel(cfg)
        >>> # Freeze backbone
        >>> for i in range(10):
        >>>     for p in model.model[i].parameters():
        >>>         p.requires_grad = False
        >>> 
        >>> # Create selective EMA - only updates non-frozen layers
        >>> ema = SelectiveModelEMA(model, decay=0.999)
        >>> 
        >>> # During training
        >>> loss.backward()
        >>> optimizer.step()
        >>> ema.update(model)  # Only updates trainable parameters
    """
    
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """
        Initialize Selective EMA.
        
        Args:
            model: Student model to create EMA from
            decay: EMA decay coefficient
            tau: Temperature for decay scheduling
            updates: Initial update count
        """
        super().__init__(model, decay=decay, tau=tau, updates=updates)
        self._log_frozen_status(model)
    
    def _log_frozen_status(self, model):
        """Log which parameters are frozen for debugging."""
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        frozen_params = total_params - trainable_params
        frozen_pct = 100.0 * frozen_params / total_params if total_params > 0 else 0.0
        
        from ultralytics.utils import LOGGER, colorstr
        LOGGER.info(
            colorstr('SelectiveEMA: ') + 
            f'Tracking {trainable_params:,}/{total_params:,} trainable params '
            f'({100 - frozen_pct:.1f}% trainable, {frozen_pct:.1f}% frozen)'
        )
    
    def update(self, model):
        """
        Update EMA parameters selectively based on requires_grad.
        
        Mathematical operation:
            For trainable parameters (requires_grad=True):
                θ_ema ← decay * θ_ema + (1 - decay) * θ_student
            
            For frozen parameters (requires_grad=False):
                θ_ema ← θ_ema  (no change)
        
        Complexity: O(P_trainable) where P_trainable = number of trainable parameters
        
        Args:
            model: Student model with potentially frozen parameters
        """
        with torch.no_grad():
            # Standard EMA decay scheduling (from parent class)
            self.updates += 1
            decay = self.decay(self.updates)
            
            # Selective update: only process trainable parameters
            updated_count = 0
            skipped_count = 0
            
            for ema_param, model_param in zip(
                self.ema.parameters(), 
                model.parameters()
            ):
                if model_param.requires_grad:
                    # Standard EMA update for trainable parameters
                    # θ_ema = decay * θ_ema + (1 - decay) * θ_model
                    ema_param.copy_(
                        ema_param * decay + model_param * (1.0 - decay)
                    )
                    updated_count += 1
                else:
                    # Skip frozen parameters - no update needed
                    skipped_count += 1
            
            # Also update buffers (batch norm running stats, etc.)
            # Buffers don't have requires_grad, so always update
            for ema_buffer, model_buffer in zip(
                self.ema.buffers(), 
                model.buffers()
            ):
                ema_buffer.copy_(model_buffer)
    
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        Update EMA attributes (from parent class).
        
        Copies non-parameter attributes from student to teacher.
        Useful for things like class names, model configuration, etc.
        
        Args:
            model: Student model
            include: Attributes to forcibly include
            exclude: Attributes to exclude
        """
        super().update_attr(model, include=include, exclude=exclude)


def create_selective_ema(model, decay=0.9999, tau=2000, updates=0):
    """
    Factory function to create SelectiveModelEMA.
    
    Args:
        model: Student model
        decay: EMA decay rate ∈ [0, 1]
        tau: Decay scheduler temperature
        updates: Initial update count
    
    Returns:
        SelectiveModelEMA instance
    
    Example:
        >>> ema = create_selective_ema(model, decay=0.999)
    """
    return SelectiveModelEMA(model, decay=decay, tau=tau, updates=updates)
