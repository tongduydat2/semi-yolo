import torch
from torch import Tensor
from typing import Tuple, Dict

from .base import BaseFilter, register_filter


@register_filter('dfl_entropy')
class DFLEntropyFilter(BaseFilter):
    """
    DFL (Distribution Focal Loss) Entropy filter for regression uncertainty.
    
    YOLOv11 predicts bounding box coordinates as distributions over discrete bins.
    Sharp distribution (low entropy) = confident prediction
    Flat distribution (high entropy) = uncertain prediction
    """

    def __init__(
        self,
        threshold: float = 0.5,
        num_bins: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.num_bins = num_bins
        self.max_entropy = torch.log(torch.tensor(float(num_bins)))

    def __call__(self, predictions: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Filter predictions by DFL distribution entropy.

        Args:
            predictions: Dict with 'box_dist' [N, 4, num_bins]
                        Probability distribution for each coordinate

        Returns:
            mask: Boolean tensor [N] (True = low uncertainty, keep)
            scores: Confidence scores [N] (1 - normalized_entropy)
        """
        box_dist = predictions.get('box_dist')

        if box_dist is None:
            n = len(predictions['boxes'])
            return torch.ones(n, dtype=torch.bool), torch.ones(n)

        entropy = self._compute_entropy(box_dist)
        normalized_entropy = entropy / self.max_entropy.to(entropy.device)
        confidence = 1.0 - normalized_entropy
        mask = normalized_entropy < self.threshold

        return mask, confidence

    def _compute_entropy(self, box_dist: Tensor) -> Tensor:
        """
        Compute entropy of box distributions.

        Args:
            box_dist: [N, 4, num_bins] probability distributions

        Returns:
            entropy: [N] average entropy across 4 coordinates
        """
        epsilon = 1e-8
        log_dist = torch.log(box_dist + epsilon)
        entropy_per_coord = -torch.sum(box_dist * log_dist, dim=-1)
        avg_entropy = entropy_per_coord.mean(dim=-1)

        return avg_entropy

    def update(self, epoch: int, total_epochs: int):
        """Optionally adjust threshold based on training progress."""
        pass
