from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
import torch
from torch import Tensor


class BaseFilter(ABC):
    """Abstract base class for pseudo-label filters."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def __call__(self, predictions: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Filter predictions and return mask and scores.

        Args:
            predictions: Dict containing 'boxes', 'cls_scores', etc.

        Returns:
            mask: Boolean tensor indicating which predictions to keep
            scores: Confidence/quality scores for kept predictions
        """
        pass

    def update(self, epoch: int, total_epochs: int):
        """Update internal state (e.g., thresholds). Override if needed."""
        pass

    def reset(self):
        """Reset filter state. Override if needed."""
        pass


class FilterChain:
    """Chains multiple filters and applies them sequentially."""

    def __init__(self, filters: List[BaseFilter]):
        self.filters = filters

    def __call__(self, predictions: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        """
        Apply all filters sequentially.

        Args:
            predictions: Dict with 'boxes', 'cls_scores', 'box_dist', 'iou_pred'

        Returns:
            Filtered predictions dict with 'boxes', 'labels', 'scores', 'stats'
        """
        if len(predictions.get('boxes', [])) == 0:
            return {
                'boxes': torch.empty(0, 4),
                'labels': torch.empty(0, dtype=torch.long),
                'scores': torch.empty(0),
                'stats': {'total': 0, 'final': 0}
            }

        device = predictions['boxes'].device
        combined_mask = torch.ones(len(predictions['boxes']), dtype=torch.bool, device=device)
        combined_scores = torch.ones(len(predictions['boxes']), device=device)
        stats = {'total': len(predictions['boxes'])}

        for i, f in enumerate(self.filters):
            mask, scores = f(predictions)
            mask = mask.to(device)
            scores = scores.to(device)
            combined_mask &= mask
            combined_scores *= scores
            stats[f'after_{f.__class__.__name__}'] = combined_mask.sum().item()

        cls_scores = predictions['cls_scores']
        cls_max, cls_idx = cls_scores.max(dim=-1)

        stats['final'] = combined_mask.sum().item()

        return {
            'boxes': predictions['boxes'][combined_mask],
            'labels': cls_idx[combined_mask],
            'scores': cls_max[combined_mask],
            'stats': stats
        }

    def update(self, epoch: int, total_epochs: int):
        """Update all filters."""
        for f in self.filters:
            f.update(epoch, total_epochs)

    def add_filter(self, filter: BaseFilter):
        """Add a filter to the chain."""
        self.filters.append(filter)


FILTER_REGISTRY: Dict[str, type] = {}


def register_filter(name: str):
    """Decorator to register a filter class."""
    def decorator(cls):
        FILTER_REGISTRY[name] = cls
        return cls
    return decorator


def get_filter(name: str, **kwargs) -> BaseFilter:
    """Get a filter instance by name."""
    if name not in FILTER_REGISTRY:
        raise ValueError(f"Filter '{name}' not found. Available: {list(FILTER_REGISTRY.keys())}")
    return FILTER_REGISTRY[name](**kwargs)


def build_filter_chain(config: List[Dict[str, Any]]) -> FilterChain:
    """
    Build FilterChain from config.

    Args:
        config: List of dicts with 'name' and 'params' keys
            Example: [{'name': 'dsat', 'params': {'num_classes': 80}}]
    """
    filters = []
    for item in config:
        name = item['name']
        params = item.get('params', {})
        filters.append(get_filter(name, **params))
    return FilterChain(filters)
