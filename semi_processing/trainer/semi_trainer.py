"""
Semi-Supervised Trainer for YOLOv11.
Extends Ultralytics BaseTrainer for semi-supervised learning.
"""

from __future__ import annotations

import time
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.cfg import get_cfg, DEFAULT_CFG

from filters.base import FilterChain, build_filter_chain
from data.semi_dataset import SemiDataModule
from augmentation.configs import WEAK_AUG, STRONG_AUG, merge_aug_config


class SemiTrainer(DetectionTrainer):
    """
    Semi-Supervised Trainer extending Ultralytics DetectionTrainer.
    
    Implements teacher-student paradigm with:
    - Teacher: EMA of student, generates pseudo-labels (weak aug)
    - Student: Trained on labeled + pseudo-labeled data (strong aug)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if cfg is None and overrides is not None:
            cfg = overrides
            overrides = {}
        
        self.semi_cfg = cfg.pop('semi', {}) if isinstance(cfg, dict) else {}
        self.augment_cfg = cfg.pop('augment', {}) if isinstance(cfg, dict) else {}
        
        self.weak_aug = merge_aug_config(WEAK_AUG, self.augment_cfg.get('weak', {}))
        self.strong_aug = merge_aug_config(STRONG_AUG, self.augment_cfg.get('strong', {}))
        
        if isinstance(cfg.get('data'), dict):
            cfg['data'] = cfg['data'].get('train', 'coco128.yaml')
        
        base_cfg = vars(DEFAULT_CFG).copy()
        for key, value in cfg.items():
            base_cfg[key] = value
        
        for key, value in self.strong_aug.items():
            if key not in cfg:
                base_cfg[key] = value
        
        super().__init__(base_cfg, overrides or {}, _callbacks)

        self.burn_in_epochs = self.semi_cfg.get('burn_in', 5)
        self.lambda_unsup = self.semi_cfg.get('lambda_unsup', 1.0)
        self.lambda_warmup = self.semi_cfg.get('lambda_warmup', 5)
        self.ema_decay = self.semi_cfg.get('ema_decay', 0.999)

        self.teacher = None
        self.filter_chain = None
        self.semi_data = None

        self.in_burn_in = True
        
        LOGGER.info(colorstr('Semi-SSL: ') + f'Weak aug: mosaic={self.weak_aug.get("mosaic")}, mixup={self.weak_aug.get("mixup")}')
        LOGGER.info(colorstr('Semi-SSL: ') + f'Strong aug: mosaic={self.strong_aug.get("mosaic")}, mixup={self.strong_aug.get("mixup")}')

    def _setup_train(self):
        """Setup training with teacher model and data module."""
        super()._setup_train()

        self.teacher = ModelEMA(self.model, decay=self.ema_decay)
        LOGGER.info(colorstr('Semi-SSL: ') + f'Teacher initialized with EMA decay={self.ema_decay}')

        filter_config = self.semi_cfg.get('filters', [
            {'name': 'dsat', 'params': {'num_classes': self.data.get('nc', 80)}},
            {'name': 'dfl_entropy', 'params': {'threshold': 0.5}},
            {'name': 'tal_alignment', 'params': {'threshold': 0.3}},
        ])
        self.filter_chain = build_filter_chain(filter_config)
        LOGGER.info(colorstr('Semi-SSL: ') + f'Filter chain: {[f.__class__.__name__ for f in self.filter_chain.filters]}')

        labeled_path = self.semi_cfg.get('labeled_path', self.args.data)
        unlabeled_path = self.semi_cfg.get('unlabeled_path')

        if unlabeled_path:
            self.semi_data = SemiDataModule(
                labeled_path=labeled_path,
                unlabeled_path=unlabeled_path,
                imgsz=self.args.imgsz,
                batch_size=self.args.batch,
                hyp=self.strong_aug,
            )
            self.semi_data.setup()
            LOGGER.info(colorstr('Semi-SSL: ') + 
                f'Data: {self.semi_data.num_labeled} labeled, {self.semi_data.num_unlabeled} unlabeled')

    def _do_train(self):
        """Main training loop with semi-supervised learning."""
        if self.world_size > 1:
            self._setup_ddp()

        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100)

        self.run_callbacks('on_train_start')
        LOGGER.info(colorstr('Semi-SSL: ') + f'Burn-in for {self.burn_in_epochs} epochs')
        
        self.last_opt_step = -1
        self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.in_burn_in = epoch < self.burn_in_epochs

            self.model.train()
            self.run_callbacks('on_train_epoch_start')

            if hasattr(self.train_loader.dataset, 'close_mosaic') and epoch == self.epochs - self.args.close_mosaic:
                self.train_loader.dataset.close_mosaic(self.args)

            self.filter_chain.update(epoch, self.epochs)

            pbar = enumerate(self.train_loader)
            self.tloss = None

            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                ni = i + nb * epoch

                if ni <= nw:
                    self._warmup(ni, nw)

                with torch.autocast(self.device.type, enabled=self.amp):
                    batch = self.preprocess_batch(batch)
                    loss_sup, loss_items_sup = self.model(batch)

                    if self.in_burn_in or self.semi_data is None:
                        loss = loss_sup
                        loss_unsup = torch.tensor(0.0, device=self.device)
                    else:
                        loss_unsup = self._compute_unsup_loss()
                        lambda_u = self._get_lambda_unsup(epoch)
                        loss = loss_sup + lambda_u * loss_unsup
                    
                    self.loss_items = loss_items_sup
                    self.loss = loss.sum()

                self.scaler.scale(loss.sum()).backward()

                if ni - self.last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.last_opt_step = ni

                if not self.in_burn_in:
                    self.teacher.update(self.model)

                self._update_loss(loss_items_sup, loss_unsup if not self.in_burn_in else None)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{i}': x['lr'] for i, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')

            if RANK in {-1, 0}:
                final = epoch + 1 == self.epochs
                self.metrics, self.fitness = self.validate()
                self.save_model()
                self.run_callbacks('on_model_save')

        self.run_callbacks('on_train_end')

    def _compute_unsup_loss(self) -> Tensor:
        """Compute unsupervised loss using pseudo-labels from teacher."""
        if self.semi_data is None:
            return torch.tensor(0.0, device=self.device)

        _, unlabeled_batch = self.semi_data.get_batch()
        unlabeled_imgs = unlabeled_batch['img'].to(self.device).float() / 255
        batch_size = unlabeled_imgs.shape[0]
        img_size = unlabeled_imgs.shape[2]

        with torch.no_grad():
            teacher_model = self.teacher.ema
            teacher_model.eval()
            teacher_preds = teacher_model(unlabeled_imgs)

        pseudo_results = self._extract_predictions_per_image(teacher_preds, img_size)
        
        all_boxes = []
        all_cls = []
        all_batch_idx = []
        total_boxes = 0
        
        for img_idx, (boxes, labels, scores) in enumerate(pseudo_results):
            if len(boxes) > 0:
                mask, _ = self._apply_filters(boxes, labels, scores)
                if mask.sum() > 0:
                    boxes_xywh = self._xyxy_to_xywhn(boxes[mask], img_size)
                    all_boxes.append(boxes_xywh)
                    all_cls.append(labels[mask].float())
                    all_batch_idx.append(torch.full((mask.sum(),), img_idx, device=self.device))
                    total_boxes += mask.sum().item()
        
        if total_boxes == 0:
            return torch.tensor(0.0, device=self.device)
        
        pseudo_batch = {
            'img': unlabeled_batch['img'].to(self.device).float(),
            'cls': torch.cat(all_cls),
            'bboxes': torch.cat(all_boxes),
            'batch_idx': torch.cat(all_batch_idx),
        }

        loss_unsup, _ = self.model(pseudo_batch)
        return loss_unsup
    
    def _extract_predictions_per_image(self, preds, img_size: int):
        """Extract predictions per image from model output."""
        from ultralytics.utils.nms import non_max_suppression
        
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        results = non_max_suppression(
            preds,
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=100,
        )
        
        per_image_results = []
        for r in results:
            if len(r) > 0:
                boxes = r[:, :4]
                scores = r[:, 4]
                labels = r[:, 5].long()
                per_image_results.append((boxes, labels, scores))
            else:
                per_image_results.append((
                    torch.empty(0, 4, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, device=self.device)
                ))
        return per_image_results
    
    def _apply_filters(self, boxes, labels, scores):
        """Apply confidence filter (simplified for stability)."""
        conf_thresh = 0.5
        mask = scores >= conf_thresh
        return mask, scores[mask] if mask.sum() > 0 else scores
    
    def _xyxy_to_xywhn(self, boxes, img_size):
        """Convert xyxy pixel to xywh normalized."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2 / img_size
        cy = (y1 + y2) / 2 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        return torch.stack([cx, cy, w, h], dim=1)

    def _get_lambda_unsup(self, epoch: int) -> float:
        """Get unsupervised loss weight with warmup."""
        if epoch < self.burn_in_epochs:
            return 0.0

        effective_epoch = epoch - self.burn_in_epochs
        if effective_epoch < self.lambda_warmup:
            return self.lambda_unsup * (effective_epoch / self.lambda_warmup)

        return self.lambda_unsup

    def _update_loss(self, loss_items_sup: Tensor, loss_unsup: Optional[Tensor] = None):
        """Update loss tracking."""
        if self.tloss is None:
            self.tloss = loss_items_sup
        else:
            self.tloss = (self.tloss * self.epoch + loss_items_sup) / (self.epoch + 1)

    def _warmup(self, ni: int, nw: int):
        """Learning rate warmup."""
        xi = [0, nw]
        for j, x in enumerate(self.optimizer.param_groups):
            x['lr'] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
