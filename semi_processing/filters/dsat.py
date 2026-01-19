import torch
from torch import Tensor
from typing import Tuple, Dict

from .base import BaseFilter, register_filter


@register_filter('dsat')
class DSATFilter(BaseFilter):
    """
    Dynamic Self-Adaptive Threshold filter for classification.
    
    Dynamically adjusts per-class thresholds based on training progress
    to balance quality vs quantity of pseudo-labels.
    """

    def __init__(
        self,
        num_classes: int = 80,
        initial_threshold: float = 0.5,
        momentum: float = 0.99,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.momentum = momentum
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.class_thresholds = torch.ones(num_classes) * initial_threshold
        self.current_epoch = 0
        self.total_epochs = 100

    def __call__(self, predictions: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Filter predictions by class-specific adaptive thresholds.

        Args:
            predictions: Dict with 'cls_scores' [N, num_classes]

        Returns:
            mask: Boolean tensor [N]
            scores: Classification scores [N]
        """
        cls_scores = predictions['cls_scores']
        cls_max, cls_idx = cls_scores.max(dim=-1)

        device = cls_max.device
        thresholds = self.class_thresholds.to(device)

        per_sample_threshold = thresholds[cls_idx]
        mask = cls_max >= per_sample_threshold

        return mask, cls_max

    def update(self, epoch: int, total_epochs: int):
        """
        Update thresholds based on training progress.
        Early training: lower thresholds (more samples)
        Late training: higher thresholds (quality focus)
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        progress = epoch / max(total_epochs, 1)
        target_percentile = 50 + 40 * progress

        base = self.min_threshold + (self.max_threshold - self.min_threshold) * (target_percentile / 100)

        self.class_thresholds = (
            self.momentum * self.class_thresholds +
            (1 - self.momentum) * base
        )
        self.class_thresholds = self.class_thresholds.clamp(self.min_threshold, self.max_threshold)

    def update_from_predictions(self, predictions: Dict[str, Tensor]):
        """
        Optionally update thresholds based on prediction distribution.
        Call this during training to adapt to actual data.
        """
        cls_scores = predictions['cls_scores']
        cls_max, cls_idx = cls_scores.max(dim=-1)

        progress = self.current_epoch / max(self.total_epochs, 1)
        target_percentile = 50 + 40 * progress

        for c in range(self.num_classes):
            class_mask = cls_idx == c
            if class_mask.sum() > 0:
                class_scores = cls_max[class_mask]
                percentile_val = target_percentile / 100
                new_threshold = torch.quantile(class_scores, percentile_val)

                self.class_thresholds[c] = (
                    self.momentum * self.class_thresholds[c] +
                    (1 - self.momentum) * new_threshold.cpu()
                )

        self.class_thresholds = self.class_thresholds.clamp(self.min_threshold, self.max_threshold)

    def reset(self):
        """Reset thresholds to initial values."""
        self.class_thresholds = torch.ones(self.num_classes) * 0.5
