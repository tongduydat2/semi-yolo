import torch
from torch import Tensor
from typing import Tuple, Dict, List

from .base import BaseFilter, register_filter


@register_filter('dsat')
class DSATFilter(BaseFilter):
    """
    Dynamic Self-Adaptive Threshold (DSAT) filter from S4OD paper.
    
    S4OD approach: Find optimal threshold by analyzing F1-score curve
    at different confidence levels. The threshold that maximizes F1
    is selected for pseudo-label filtering.
    
    Reference: S4OD - Semi-Supervised learning for Single-Stage Object Detection
    """

    def __init__(
        self,
        num_classes: int = 80,
        initial_threshold: float = 0.5,
        momentum: float = 0.99,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        threshold_steps: int = 9,
        uncertainty_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.momentum = momentum
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_steps = threshold_steps
        self.initial_threshold = initial_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.class_thresholds = torch.ones(num_classes) * initial_threshold

    def __call__(self, predictions: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        """Filter predictions by class-specific adaptive thresholds and uncertainty."""
        cls_scores = predictions['cls_scores']
        labels = predictions.get('labels')
        uncertainties = predictions.get('uncertainties')
        
        if cls_scores.dim() == 1:
            cls_max = cls_scores
            cls_idx = labels if labels is not None else torch.zeros_like(cls_scores, dtype=torch.long)
        else:
            cls_max, cls_idx = cls_scores.max(dim=-1)
        
        device = cls_max.device
        thresholds = self.class_thresholds.to(device)
        per_sample_threshold = thresholds[cls_idx]
        
        conf_mask = cls_max >= per_sample_threshold
        
        if uncertainties is not None:
            if isinstance(self.uncertainty_threshold, Tensor) and self.uncertainty_threshold.dim() > 0:
                per_sample_unc_thresh = self.uncertainty_threshold.to(device)[cls_idx]
                unc_mask = uncertainties <= per_sample_unc_thresh
            else:
                unc_mask = uncertainties <= self.uncertainty_threshold
            mask = conf_mask & unc_mask
        else:
            mask = conf_mask

        return mask, cls_max

    def update(self, epoch: int, total_epochs: int):
        """Update epoch info."""
        pass

    def update_from_f1(self, per_class_f1: Tensor):
        """
        Update thresholds from per-class F1 scores.
        Higher F1 → higher threshold, Lower F1 → lower threshold.
        """
        if per_class_f1 is None or len(per_class_f1) != self.num_classes:
            return
            
        normalized_f1 = per_class_f1.float().clamp(0, 1)
        target_thresholds = self.min_threshold + normalized_f1 * (self.max_threshold - self.min_threshold)
        
        self.class_thresholds = (
            self.momentum * self.class_thresholds +
            (1 - self.momentum) * target_thresholds.cpu()
        )
        self.class_thresholds = self.class_thresholds.clamp(self.min_threshold, self.max_threshold)
        self.uncertainty_threshold = self.class_thresholds / 2

    def reset(self):
        """Reset thresholds to initial values."""
        self.class_thresholds = torch.ones(self.num_classes) * self.initial_threshold

    def __repr__(self):
        return f"DSATFilter(num_classes={self.num_classes}, threshold={self.initial_threshold})"