"""
Metrics calculation for SSOD evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes in YOLO format.
    
    Args:
        box1, box2: Boxes in format [x_center, y_center, width, height] (normalized)
        
    Returns:
        IoU value
    """
    # Convert to x1, y1, x2, y2
    def yolo_to_xyxy(box):
        x_c, y_c, w, h = box
        return [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]
    
    b1 = yolo_to_xyxy(box1)
    b2 = yolo_to_xyxy(box2)
    
    # Intersection
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union
    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / max(union_area, 1e-10)


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision using 11-point interpolation.
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def calculate_metrics(predictions: List[Dict],
                     ground_truths: List[Dict],
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate detection metrics.
    
    Args:
        predictions: List of prediction dicts with 'boxes_yolo', 'classes', 'confidences'
        ground_truths: List of GT dicts with 'boxes_yolo', 'classes'
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dict with precision, recall, mAP
    """
    all_scores = []
    all_matches = []
    total_gt = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = np.array(pred.get('boxes_yolo', []))
        pred_confs = np.array(pred.get('confidences', []))
        gt_boxes = np.array(gt.get('boxes_yolo', []))
        
        if len(pred_boxes) == 0:
            total_gt += len(gt_boxes)
            continue
            
        if len(gt_boxes) == 0:
            all_scores.extend(pred_confs.tolist())
            all_matches.extend([0] * len(pred_confs))
            continue
        
        total_gt += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)
        
        # Sort predictions by confidence
        order = np.argsort(-pred_confs)
        
        for idx in order:
            pred_box = pred_boxes[idx, 1:5] if pred_boxes.shape[1] == 5 else pred_boxes[idx]
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                box = gt_box[1:5] if len(gt_box) == 5 else gt_box
                iou = calculate_iou(pred_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                all_matches.append(1)
            else:
                all_matches.append(0)
            
            all_scores.append(pred_confs[idx])
    
    if not all_scores:
        return {'precision': 0, 'recall': 0, 'mAP50': 0, 'f1': 0}
    
    # Sort by score
    order = np.argsort(-np.array(all_scores))
    all_matches = np.array(all_matches)[order]
    
    # Calculate precision/recall curve
    tp_cumsum = np.cumsum(all_matches)
    fp_cumsum = np.cumsum(1 - all_matches)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / max(total_gt, 1)
    
    # Calculate metrics
    ap = calculate_ap(recalls, precisions)
    final_precision = precisions[-1] if len(precisions) > 0 else 0
    final_recall = recalls[-1] if len(recalls) > 0 else 0
    f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-10)
    
    return {
        'precision': float(final_precision),
        'recall': float(final_recall),
        'mAP50': float(ap),
        'f1': float(f1)
    }


def compare_models(model_a_predictions: List[Dict],
                  model_b_predictions: List[Dict],
                  ground_truths: List[Dict]) -> Dict[str, Dict]:
    """
    Compare two models (e.g., supervised-only vs SSOD).
    
    Returns:
        Dict with metrics for both models
    """
    metrics_a = calculate_metrics(model_a_predictions, ground_truths)
    metrics_b = calculate_metrics(model_b_predictions, ground_truths)
    
    improvement = {
        k: metrics_b[k] - metrics_a[k] 
        for k in metrics_a
    }
    
    return {
        'model_a': metrics_a,
        'model_b': metrics_b,
        'improvement': improvement
    }
