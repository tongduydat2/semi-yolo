"""
Pseudo-Label Generator with Adaptive Thresholding
Sinh nhãn giả từ Teacher model với cơ chế tự điều chỉnh ngưỡng
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque


class PseudoLabeler:
    """
    Generate and manage pseudo-labels from Teacher predictions.
    Implements adaptive thresholding based on prediction statistics.
    """
    
    def __init__(self,
                 base_threshold: float = 0.75,
                 adaptive: bool = True,
                 tau_min: float = 0.65,
                 tau_max: float = 0.80,
                 target_pseudo_ratio: float = 0.3,
                 history_size: int = 100):
        """
        Args:
            base_threshold: Initial confidence threshold (Tau)
            adaptive: Whether to use adaptive thresholding
            tau_min: Minimum threshold value
            tau_max: Maximum threshold value
            target_pseudo_ratio: Target ratio of images with pseudo-labels
            history_size: Number of batches to track for adaptation
        """
        self.threshold = base_threshold
        self.adaptive = adaptive
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.target_pseudo_ratio = target_pseudo_ratio
        
        self.pseudo_count_history = deque(maxlen=history_size)
        self.total_count_history = deque(maxlen=history_size)
        
    def filter_predictions(self,
                          predictions: List[Dict[str, Any]]) -> Tuple[List[Dict], Dict]:
        """
        Filter Teacher predictions to create pseudo-labels.
        
        Args:
            predictions: Raw predictions from Teacher model
            
        Returns:
            Filtered pseudo-labels and statistics dict
        """
        filtered_labels = []
        stats = {
            'total_predictions': 0,
            'kept_predictions': 0,
            'images_with_labels': 0,
            'current_threshold': self.threshold
        }
        
        for pred in predictions:
            filtered = {
                'boxes_yolo': [],
                'classes': [],
                'confidences': []
            }
            
            boxes_yolo = pred.get('boxes_yolo', [])
            confidences = pred.get('confidences', [])
            
            for box, conf in zip(boxes_yolo, confidences):
                stats['total_predictions'] += 1
                
                if conf >= self.threshold:
                    filtered['boxes_yolo'].append(box)
                    filtered['classes'].append(int(box[0]))
                    filtered['confidences'].append(conf)
                    stats['kept_predictions'] += 1
            
            if filtered['boxes_yolo']:
                stats['images_with_labels'] += 1
                
            filtered_labels.append(filtered)
        
        # Track history
        self.pseudo_count_history.append(stats['images_with_labels'])
        self.total_count_history.append(len(predictions))
        
        # Adapt threshold
        if self.adaptive:
            self._adapt_threshold()
        
        return filtered_labels, stats
    
    def _adapt_threshold(self):
        """
        Adapt threshold based on pseudo-label generation rate.
        - Too few: Lower threshold
        - Too many (noisy): Raise threshold
        """
        if len(self.pseudo_count_history) < 10:
            return
            
        recent_pseudo = sum(list(self.pseudo_count_history)[-10:])
        recent_total = sum(list(self.total_count_history)[-10:])
        
        if recent_total == 0:
            return
            
        current_ratio = recent_pseudo / recent_total
        adjustment = 0.01
        
        if current_ratio < self.target_pseudo_ratio * 0.5:
            self.threshold = max(self.tau_min, self.threshold - adjustment)
        elif current_ratio > self.target_pseudo_ratio * 1.5:
            self.threshold = min(self.tau_max, self.threshold + adjustment)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        if not self.pseudo_count_history:
            return {'avg_ratio': 0, 'threshold': self.threshold}
            
        total_pseudo = sum(self.pseudo_count_history)
        total_images = sum(self.total_count_history)
        
        return {
            'avg_ratio': total_pseudo / max(1, total_images),
            'threshold': self.threshold,
            'total_pseudo_labels': total_pseudo
        }
    
    def reset(self):
        """Reset tracking statistics."""
        self.pseudo_count_history.clear()
        self.total_count_history.clear()


class NMSRefiner:
    """
    Non-Maximum Suppression refiner for pseudo-labels.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        
    def refine(self, labels: Dict[str, List]) -> Dict[str, List]:
        """Apply NMS to refine pseudo-labels."""
        boxes_yolo = labels.get('boxes_yolo', [])
        if not boxes_yolo:
            return labels
            
        confidences = np.array(labels['confidences'])
        boxes_yolo = np.array(boxes_yolo)
        
        # Convert YOLO to xyxy for NMS
        boxes_xyxy = self._yolo_to_xyxy(boxes_yolo[:, 1:5])
        classes = boxes_yolo[:, 0].astype(int)
        
        keep_indices = []
        
        for cls in np.unique(classes):
            cls_mask = classes == cls
            cls_boxes = boxes_xyxy[cls_mask]
            cls_confs = confidences[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            if len(cls_boxes) == 0:
                continue
                
            order = cls_confs.argsort()[::-1]
            keep = []
            
            while order.size > 0:
                i = order[0]
                keep.append(cls_indices[i])
                
                if order.size == 1:
                    break
                    
                ious = self._compute_iou(cls_boxes[order[0]], cls_boxes[order[1:]])
                remaining = np.where(ious < self.iou_threshold)[0]
                order = order[remaining + 1]
            
            keep_indices.extend(keep)
        
        keep_indices = sorted(keep_indices)
        
        return {
            'boxes_yolo': [labels['boxes_yolo'][i] for i in keep_indices],
            'classes': [labels['classes'][i] for i in keep_indices],
            'confidences': [labels['confidences'][i] for i in keep_indices]
        }
    
    def _yolo_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert YOLO format to xyxy (assuming image size 1.0)."""
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)
    
    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        return intersection / np.maximum(union, 1e-10)


def save_pseudo_labels(labels: List[Dict], image_paths: List[str], output_dir: str):
    """Save pseudo-labels in YOLO format."""
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_path, label in zip(image_paths, labels):
        img_name = Path(img_path).stem
        label_file = output_path / f"{img_name}.txt"
        
        with open(label_file, 'w') as f:
            for box in label.get('boxes_yolo', []):
                # YOLO format: class x_center y_center width height
                line = ' '.join(map(str, box))
                f.write(line + '\n')
