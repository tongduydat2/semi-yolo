import torch
from torch import Tensor
from typing import Tuple, Dict

from .base import BaseFilter, register_filter


@register_filter('tal_alignment')
class TALAlignmentFilter(BaseFilter):
    """
    Task-Aligned Learning (TAL) Alignment Score filter.
    
    Combines classification and localization quality:
    alignment_score = cls_score^alpha * iou_score^beta
    
    Higher beta (6.0) weights localization quality more heavily.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        alpha: float = 0.5,
        beta: float = 6.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

    def __call__(self, predictions: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Filter predictions by TAL alignment score.

        Args:
            predictions: Dict with:
                - 'cls_scores': [N, num_classes]
                - 'iou_pred': [N] predicted IoU scores

        Returns:
            mask: Boolean tensor [N]
            scores: Alignment scores [N]
        """
        cls_scores = predictions['cls_scores']
        iou_pred = predictions.get('iou_pred')

        cls_max = cls_scores.max(dim=-1).values

        if iou_pred is None:
            alignment = cls_max ** self.alpha
        else:
            iou_pred = iou_pred.clamp(0, 1)
            alignment = (cls_max ** self.alpha) * (iou_pred ** self.beta)

        mask = alignment >= self.threshold

        return mask, alignment

    def update(self, epoch: int, total_epochs: int):
        """Optionally adjust threshold based on training progress."""
        pass
