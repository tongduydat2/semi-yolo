"""
Lightweight validator for DSAT per-class F1 computation.
Runs on teacher model to extract per-class F1 scores for adaptive thresholding.
"""

import torch
from torch import Tensor
from typing import Optional, Dict, List
import numpy as np

from ultralytics.utils import LOGGER, TQDM, nms
from ultralytics.utils.metrics import box_iou


class DSATValidator:
    """
    Lightweight validator for computing per-class F1 scores.
    Used for updating DSAT adaptive thresholds.
    """
    
    def __init__(
        self,
        nc: int,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.001,
        device: str = 'cuda',
    ):
        self.nc = nc
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.tp = {i: 0 for i in range(self.nc)}
        self.fp = {i: 0 for i in range(self.nc)}
        self.fn = {i: 0 for i in range(self.nc)}
    
    @torch.no_grad()
    def validate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tensor:
        """
        Run validation on model and return per-class F1 scores.
        
        Args:
            model: Model to validate (should be teacher EMA model).
            dataloader: Validation dataloader.
        
        Returns:
            Tensor of shape (nc,) containing per-class F1 scores.
        """
        self.reset()
        model.eval()
        
        for batch in TQDM(dataloader, desc="DSAT Validation"):
            imgs = batch['img'].to(self.device).float() / 255
            
            preds = model(imgs)
            preds = nms.non_max_suppression(
                preds,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                max_det=300,
            )
            
            self._update_stats(preds, batch)
        
        return self._compute_f1()
    
    def _update_stats(self, preds: List[Tensor], batch: Dict):
        """Update TP/FP/FN for each class."""
        batch_idx = batch['batch_idx']
        gt_cls = batch['cls'].squeeze(-1)
        gt_bboxes = batch['bboxes']
        
        for si, pred in enumerate(preds):
            idx = batch_idx == si
            gt_cls_i = gt_cls[idx].to(self.device)
            gt_bboxes_i = gt_bboxes[idx].to(self.device)
            
            if len(gt_bboxes_i) > 0:
                img_h, img_w = batch['img'].shape[2:]
                gt_bboxes_i = self._xywhn_to_xyxy(gt_bboxes_i, img_w, img_h)
            
            if len(pred) == 0:
                for c in gt_cls_i.long().tolist():
                    self.fn[c] += 1
                continue
            
            pred_bboxes = pred[:, :4]
            pred_cls = pred[:, 5].long()
            
            if len(gt_bboxes_i) == 0:
                for c in pred_cls.tolist():
                    self.fp[c] += 1
                continue
            
            iou = box_iou(gt_bboxes_i, pred_bboxes)
            
            matched_gt = set()
            matched_pred = set()
            
            for pi in range(len(pred)):
                pc = pred_cls[pi].item()
                best_iou = 0
                best_gi = -1
                
                for gi in range(len(gt_bboxes_i)):
                    if gi in matched_gt:
                        continue
                    if gt_cls_i[gi].item() != pc:
                        continue
                    if iou[gi, pi] > best_iou and iou[gi, pi] >= 0.5:
                        best_iou = iou[gi, pi]
                        best_gi = gi
                
                if best_gi >= 0:
                    matched_gt.add(best_gi)
                    matched_pred.add(pi)
                    self.tp[pc] += 1
                else:
                    self.fp[pc] += 1
            
            for gi in range(len(gt_bboxes_i)):
                if gi not in matched_gt:
                    gc = gt_cls_i[gi].long().item()
                    self.fn[gc] += 1
    
    def _xywhn_to_xyxy(self, boxes: Tensor, w: int, h: int) -> Tensor:
        """Convert normalized xywh to xyxy pixel coordinates."""
        x, y, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _compute_f1(self) -> Tensor:
        """Compute per-class F1 scores."""
        f1_scores = torch.zeros(self.nc)
        
        for c in range(self.nc):
            tp = self.tp[c]
            fp = self.fp[c]
            fn = self.fn[c]
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores[c] = f1
        
        return f1_scores
