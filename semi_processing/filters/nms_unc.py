"""
S4OD NMS-Unc: Non-Maximum Suppression with Uncertainty estimation.

Computes regression uncertainty based on standard deviation of redundant boxes.
Used for filtering pseudo-labels in semi-supervised object detection.

Reference: S4OD - Semi-Supervised learning for Single-Stage Object Detection
"""

import torch
from torch import Tensor
from typing import Tuple, List, Optional


def nms_unc(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
    max_det: int = 300,
    uncertainty_threshold: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    S4OD NMS with Uncertainty estimation.
    
    Computes bounding box regression uncertainty based on the standard deviation
    of redundant (suppressed) boxes during NMS process.
    
    Args:
        boxes: [N, 4] bounding boxes in xyxy format
        scores: [N] confidence scores  
        labels: [N] class labels
        iou_threshold: IoU threshold for NMS
        conf_threshold: Confidence threshold for filtering
        max_det: Maximum detections to keep
        uncertainty_threshold: Threshold for uncertainty filtering (normalized 0-1)
    
    Returns:
        kept_boxes: [K, 4] kept bounding boxes
        kept_scores: [K] kept confidence scores
        kept_labels: [K] kept class labels
        uncertainties: [K] regression uncertainty per box (lower is better)
    """
    if len(boxes) == 0:
        return boxes, scores, labels, torch.zeros(0, device=boxes.device)
    
    device = boxes.device
    
    conf_mask = scores >= conf_threshold
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    labels = labels[conf_mask]
    
    if len(boxes) == 0:
        return boxes, scores, labels, torch.zeros(0, device=device)
    
    unique_classes = labels.unique()
    
    all_kept_indices = []
    all_uncertainties = []
    
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = torch.where(cls_mask)[0]
        
        if len(cls_boxes) == 0:
            continue
        
        sorted_indices = cls_scores.argsort(descending=True)
        cls_boxes = cls_boxes[sorted_indices]
        cls_scores = cls_scores[sorted_indices]
        cls_indices = cls_indices[sorted_indices]
        
        kept_mask = torch.ones(len(cls_boxes), dtype=torch.bool, device=device)
        uncertainties = torch.zeros(len(cls_boxes), device=device)
        
        for i in range(len(cls_boxes)):
            if not kept_mask[i]:
                continue
            
            ious = box_iou_single(cls_boxes[i], cls_boxes[i+1:])
            suppress_mask = ious >= iou_threshold
            
            redundant_boxes = cls_boxes[i+1:][suppress_mask]
            
            if len(redundant_boxes) > 0:
                all_related_boxes = torch.cat([cls_boxes[i:i+1], redundant_boxes], dim=0)
                
                center_x = (all_related_boxes[:, 0] + all_related_boxes[:, 2]) / 2
                center_y = (all_related_boxes[:, 1] + all_related_boxes[:, 3]) / 2
                width = all_related_boxes[:, 2] - all_related_boxes[:, 0]
                height = all_related_boxes[:, 3] - all_related_boxes[:, 1]
                
                std_x = center_x.std() / (width.mean() + 1e-8)
                std_y = center_y.std() / (height.mean() + 1e-8)
                std_w = width.std() / (width.mean() + 1e-8)
                std_h = height.std() / (height.mean() + 1e-8)
                
                uncertainty = (std_x + std_y + std_w + std_h) / 4.0
                uncertainties[i] = uncertainty
            else:
                uncertainties[i] = 0.0
            
            suppress_indices = torch.where(suppress_mask)[0] + i + 1
            kept_mask[suppress_indices] = False
        
        kept_in_cls = torch.where(kept_mask)[0]
        for idx in kept_in_cls:
            all_kept_indices.append(cls_indices[idx])
            all_uncertainties.append(uncertainties[idx])
    
    if len(all_kept_indices) == 0:
        return (
            torch.zeros(0, 4, device=device),
            torch.zeros(0, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, device=device)
        )
    
    kept_indices = torch.stack(all_kept_indices)
    uncertainties = torch.stack(all_uncertainties)
    
    kept_boxes = boxes[kept_indices]
    kept_scores = scores[kept_indices]
    kept_labels = labels[kept_indices]
    
    if isinstance(uncertainty_threshold, Tensor) and uncertainty_threshold.dim() > 0:
        per_sample_unc_thresh = uncertainty_threshold.to(device)[kept_labels]
        unc_mask = uncertainties <= per_sample_unc_thresh
    else:
        unc_mask = uncertainties <= uncertainty_threshold
    kept_boxes = kept_boxes[unc_mask]
    kept_scores = kept_scores[unc_mask]
    kept_labels = kept_labels[unc_mask]
    uncertainties = uncertainties[unc_mask]
    
    if len(kept_boxes) > max_det:
        topk_indices = kept_scores.argsort(descending=True)[:max_det]
        kept_boxes = kept_boxes[topk_indices]
        kept_scores = kept_scores[topk_indices]
        kept_labels = kept_labels[topk_indices]
        uncertainties = uncertainties[topk_indices]
    
    return kept_boxes, kept_scores, kept_labels, uncertainties


def box_iou_single(box: Tensor, boxes: Tensor) -> Tensor:
    """
    Compute IoU between one box and multiple boxes.
    
    Args:
        box: [4] single box in xyxy format
        boxes: [N, 4] multiple boxes in xyxy format
    
    Returns:
        ious: [N] IoU values
    """
    if len(boxes) == 0:
        return torch.zeros(0, device=box.device)
    
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    
    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union_area = box_area + boxes_area - inter_area
    
    return inter_area / (union_area + 1e-8)


def nms_unc_batched(
    prediction: Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    nc: int = 0,
    uncertainty_threshold: float = 0.5,
) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    NMS-Unc with same input format as ultralytics.utils.nms.non_max_suppression.
    
    Args:
        prediction: Model output tensor with shape (batch, 4+nc, num_boxes).
        conf_thres: Confidence threshold for filtering.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections per image.
        nc: Number of classes.
        uncertainty_threshold: Threshold for uncertainty filtering.
    
    Returns:
        List of (boxes, scores, labels, uncertainties) tuples per image.
    """
    from ultralytics.utils.ops import xywh2xyxy
    
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    device = prediction.device
    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    
    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    
    results = []
    
    for xi in range(bs):
        x = prediction[xi]
        
        box = x[:, :4]
        cls_scores = x[:, 4:4+nc]
        conf, cls_idx = cls_scores.max(dim=1)
        
        mask = conf > conf_thres
        if mask.sum() == 0:
            results.append((
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), device=device),
            ))
            continue
        
        boxes = box[mask]
        scores = conf[mask]
        labels = cls_idx[mask]
        
        kept_boxes, kept_scores, kept_labels, uncertainties = nms_unc(
            boxes, scores, labels,
            iou_threshold=iou_thres,
            conf_threshold=conf_thres,
            max_det=max_det,
            uncertainty_threshold=uncertainty_threshold,
        )
        
        results.append((kept_boxes, kept_scores, kept_labels, uncertainties))
    
    return results
