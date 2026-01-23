"""
Custom YOLO Detection Loss with Background Penalty for Semi-Supervised Learning.

This module extends the standard YOLOv8 detection loss to include a background penalty term
that penalizes the model when it predicts high confidence for any class on background regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Tuple

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import LOGGER


class v8DetectionLossWithBgPenalty(v8DetectionLoss):
    """
    YOLOv8 Detection Loss with Background Penalty.
    
    Extends the standard v8DetectionLoss by adding a penalty term for background anchors
    where the model predicts high confidence for any class. This is particularly useful
    in semi-supervised learning where pseudo-labels may contain false positives.
    
    Mathematical Formulation:
    -------------------------
    For background anchor i (where target_scores_i = 0 for all classes):
        L_bg_penalty^i = λ_bg * max_c(σ(pred_scores_i^c))
    
    Where:
        - σ: sigmoid function
        - pred_scores_i^c: predicted score for class c at anchor i
        - λ_bg: background penalty weight (hyperparameter)
    
    Total classification loss:
        L_cls = L_BCE + (1/N_bg) * Σ_{i∈bg} L_bg_penalty^i
    
    Attributes:
        lambda_bg (float): Weight for background penalty term. Range: [0.0, 3.0]
                          - 0.0: No penalty (standard BCE)
                          - 0.5-1.0: Moderate penalty (recommended for semi-SSL)
                          - 1.0-2.0: Strong penalty (noisy pseudo-labels)
                          - >2.0: Very aggressive (may miss objects)
        use_focal_bg (bool): If True, use focal loss style for background penalty
        gamma (float): Focal loss gamma parameter for background (default: 2.0)
    """
    
    def __init__(
        self, 
        model, 
        tal_topk: int = 10,
        lambda_bg: float = 1.0,
        use_focal_bg: bool = False,
        gamma: float = 2.0,
    ):
        """
        Initialize v8DetectionLossWithBgPenalty.
        
        Args:
            model: YOLO model (must be de-paralleled)
            tal_topk: Top-k for task-aligned assignment
            lambda_bg: Background penalty weight, controls strength of penalty
            use_focal_bg: Whether to use focal loss style for background
            gamma: Focal loss gamma parameter
        """
        # Only pass model and tal_topk to parent (tal_topk2 not supported in all versions)
        super(v8DetectionLossWithBgPenalty, self).__init__(model, tal_topk)
        self.lambda_bg = lambda_bg
        self.use_focal_bg = use_focal_bg
        self.gamma = gamma
        
        LOGGER.info(f'Loss: Using background penalty (λ_bg={lambda_bg}, focal={use_focal_bg})')
    
    def compute_cls_loss_with_bg_penalty(
        self, 
        pred_scores: torch.Tensor,      # (B, num_anchors, num_classes)
        target_scores: torch.Tensor,    # (B, num_anchors, num_classes)
        target_scores_sum: torch.Tensor # scalar
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute classification loss with background penalty.
        
        This method computes the standard BCE loss and adds a penalty term for
        background anchors where the model predicts high confidence.
        
        Args:
            pred_scores: Predicted class scores (logits), shape (B, H*W, C)
            target_scores: Target scores from TAL assignment, shape (B, H*W, C)
                          Background anchors have target_scores = 0 for all classes
            target_scores_sum: Sum of all target_scores for normalization
            
        Returns:
            total_cls_loss: Classification loss including background penalty
            bg_penalty: Background penalty value (for logging)
            
        Shape Analysis:
            - pred_scores: (B, N, C) where B=batch, N=num_anchors, C=num_classes
            - target_scores: (B, N, C)
            - is_background: (B, N) - boolean mask
            - max_class_prob: (B, N) - max probability across classes
        """
        dtype = pred_scores.dtype
        
        # Standard BCE loss (element-wise)
        # Shape: (B, N, C)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        
        # Identify background anchors: sum(target_scores) ≈ 0 for all classes
        # Shape: (B, N)
        is_background = target_scores.sum(dim=-1) < 1e-6
        
        # Compute background penalty
        bg_penalty = torch.tensor(0.0, device=pred_scores.device, dtype=dtype)
        
        if is_background.any():
            # Get predicted probabilities
            # Shape: (B, N, C)
            pred_probs = pred_scores.sigmoid()
            
            if self.use_focal_bg:
                # Focal loss style: penalize hard negatives more
                # L_focal_bg = -Σ_c (1-p_c)^γ log(1-p_c)
                # This focuses on background anchors where model is confident
                one_minus_prob = 1.0 - pred_probs  # (B, N, C)
                focal_weight = one_minus_prob.pow(self.gamma)
                
                # Log term for numerical stability
                # clamp to avoid log(0)
                log_term = torch.log((one_minus_prob + 1e-7).clamp(min=1e-7))
                
                # Focal loss per class
                focal_bg_loss = -(focal_weight * log_term)  # (B, N, C)
                
                # Sum over classes, average over background anchors
                bg_penalty = focal_bg_loss[is_background].sum() / max(is_background.sum(), 1)
                
            else:
                # Simple max penalty: penalize maximum class probability
                # Intuition: If model predicts ANY class with high confidence on background,
                # it should be penalized
                
                # Max class probability for each anchor
                # Shape: (B, N)
                max_class_prob, _ = pred_probs.max(dim=-1)
                
                # Apply penalty only on background anchors
                # Average over background anchors
                bg_penalty = max_class_prob[is_background].sum() / max(is_background.sum(), 1)
            
            # Weight by lambda_bg
            bg_penalty = self.lambda_bg * bg_penalty
        
        # Total classification loss
        # BCE loss normalized by target_scores_sum (TAL normalization)
        # Background penalty normalized by number of background anchors
        total_cls_loss = bce_loss.sum() / target_scores_sum + bg_penalty
        
        return total_cls_loss, bg_penalty
    
    def get_assigned_targets_and_loss(
        self, 
        preds: dict[str, torch.Tensor], 
        batch: dict[str, Any]
    ) -> Tuple:
        """
        Calculate loss for box, cls, and dfl with background penalty.
        
        This method overrides the parent to use custom classification loss.
        
        Args:
            preds: Model predictions dict with keys 'boxes', 'scores', 'feats'
            batch: Batch data dict with keys 'batch_idx', 'cls', 'bboxes'
            
        Returns:
            assigned_data: Tuple of (fg_mask, target_gt_idx, target_bboxes, 
                                    anchor_points, stride_tensor)
            loss: Tensor of shape (3,) with [box_loss, cls_loss, dfl_loss]
            loss_detach: Detached copy of loss for logging
        """
        # Initialize loss: [box, cls, dfl]
        loss = torch.zeros(3, device=self.device)
        
        # Extract predictions
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        
        # Get anchor points and stride
        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # Prepare targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # Decode predicted bboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        # Task-Aligned Assignment
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # ===== MODIFIED: Custom Classification Loss with Background Penalty =====
        loss[1], bg_penalty = self.compute_cls_loss_with_bg_penalty(
            pred_scores, 
            target_scores, 
            target_scores_sum
        )
        
        # Store bg_penalty for logging (attach to loss object as attribute)
        # This allows access in training loop for monitoring
        if not hasattr(self, '_bg_penalty_history'):
            self._bg_penalty_history = []
        self._bg_penalty_history.append(bg_penalty.item())
        
        # Bbox loss (unchanged)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        
        # Apply loss gains
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain (includes bg_penalty)
        loss[2] *= self.hyp.dfl  # dfl gain
        
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
    
    def get_bg_penalty_stats(self) -> dict[str, float]:
        """
        Get statistics about background penalty over recent batches.
        
        Returns:
            stats: Dictionary with 'mean', 'max', 'min' bg_penalty values
        """
        if not hasattr(self, '_bg_penalty_history') or len(self._bg_penalty_history) == 0:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0}
        
        history = self._bg_penalty_history[-100:]  # Last 100 batches
        return {
            'mean': sum(history) / len(history),
            'max': max(history),
            'min': min(history),
        }
    
    def reset_bg_penalty_history(self):
        """Reset background penalty history (call at epoch end)."""
        if hasattr(self, '_bg_penalty_history'):
            self._bg_penalty_history = []


# ============================================================================
# Adaptive Background Penalty Scheduler
# ============================================================================

class AdaptiveBgPenaltyScheduler:
    """
    Scheduler for adaptive background penalty weight during training.
    
    Gradually increases λ_bg during semi-supervised phase to avoid
    sudden changes that may destabilize training.
    
    Schedules:
    ----------
    1. 'constant': Fixed λ_bg throughout training
    2. 'linear': Linear warmup from 0 to λ_bg_max
    3. 'step': Step increase at specific epochs
    4. 'cosine': Cosine annealing schedule
    """
    
    def __init__(
        self,
        lambda_bg_max: float = 1.0,
        warmup_epochs: int = 5,
        schedule: str = 'linear',
        burn_in_epochs: int = 0,
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            lambda_bg_max: Maximum background penalty weight
            warmup_epochs: Number of epochs to warm up to max
            schedule: Schedule type ('constant', 'linear', 'step', 'cosine')
            burn_in_epochs: Epochs before semi-supervised phase (no penalty)
        """
        self.lambda_bg_max = lambda_bg_max
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule
        self.burn_in_epochs = burn_in_epochs
    
    def get_lambda_bg(self, epoch: int) -> float:
        """
        Get background penalty weight for current epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            lambda_bg: Background penalty weight for this epoch
        """
        if epoch < self.burn_in_epochs:
            return 0.0
        
        effective_epoch = epoch - self.burn_in_epochs
        
        if self.schedule == 'constant':
            return self.lambda_bg_max
        
        elif self.schedule == 'linear':
            if effective_epoch >= self.warmup_epochs:
                return self.lambda_bg_max
            return self.lambda_bg_max * (effective_epoch / self.warmup_epochs)
        
        elif self.schedule == 'step':
            # Step increase at warmup_epochs
            if effective_epoch >= self.warmup_epochs:
                return self.lambda_bg_max
            return self.lambda_bg_max * 0.5
        
        elif self.schedule == 'cosine':
            import math
            if effective_epoch >= self.warmup_epochs:
                return self.lambda_bg_max
            progress = effective_epoch / self.warmup_epochs
            return self.lambda_bg_max * (1 - math.cos(progress * math.pi)) / 2
        
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def __repr__(self):
        return (f"AdaptiveBgPenaltyScheduler(max={self.lambda_bg_max}, "
                f"warmup={self.warmup_epochs}, schedule='{self.schedule}')")
