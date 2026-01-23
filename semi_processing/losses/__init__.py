"""
Semi-Processing Losses Module

Custom loss functions for semi-supervised learning with YOLO.
"""

from .bg_penalty_loss import (
    v8DetectionLossWithBgPenalty,
    AdaptiveBgPenaltyScheduler,
)

__all__ = [
    'v8DetectionLossWithBgPenalty',
    'AdaptiveBgPenaltyScheduler',
]
