from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
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

        Returns:
            mask: Boolean tensor indicating which predictions to keep
            scores: Confidence scores for predictions
        """
        pass

    def update(self, epoch: int, total_epochs: int):
        """Update internal state. Override if needed."""
        pass

    def reset(self):
        """Reset filter state. Override if needed."""
        pass


class FilterChain:
    """Chains multiple filters and applies them sequentially."""

    def __init__(self, filters: List[BaseFilter]):
        self.filters = filters

    def __call__(self, predictions: Dict[str, Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        """Apply all filters sequentially. Returns (mask, filtered_scores)."""
        if len(predictions.get('boxes', [])) == 0:
            return torch.zeros(0, dtype=torch.bool), torch.zeros(0)

        device = predictions['boxes'].device
        combined_mask = torch.ones(len(predictions['boxes']), dtype=torch.bool, device=device)

        for f in self.filters:
            mask, _ = f(predictions)
            combined_mask &= mask.to(device)

        cls_scores = predictions['cls_scores']
        if cls_scores.dim() == 1:
            cls_max = cls_scores
        else:
            cls_max = cls_scores.max(dim=-1).values

        return combined_mask, cls_max[combined_mask]

    def update(self, epoch: int, total_epochs: int):
        """Update all filters."""
        for f in self.filters:
            f.update(epoch, total_epochs)

    def __repr__(self):
        filters_str = ", ".join([repr(f) for f in self.filters])
        return f"FilterChain([{filters_str}])"


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
    """Build FilterChain from config list."""
    filters = []
    for item in config:
        name = item['name']
        params = item.get('params', {})
        filters.append(get_filter(name, **params))
    return FilterChain(filters)
