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
from filters.dsat import DSATFilter
from validators.dsat_validator import DSATValidator
from ultralytics.utils import TQDM
from data.semi_dataset import ThermalAugmentation

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
        self.aug_noise = ThermalAugmentation()
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

    def _init_semi_data(self):
        """Initialize semi-supervised data module."""
        if hasattr(self, 'semi_data') and self.semi_data is not None:
            return

        labeled_path = self.semi_cfg.get('labeled_path', self.args.data)
        unlabeled_path = self.semi_cfg.get('unlabeled_path')

        if unlabeled_path:
            self.semi_data = SemiDataModule(
                labeled_path=labeled_path,
                unlabeled_path=unlabeled_path,
                imgsz=self.args.imgsz,
                batch_size=self.args.batch,
                weak_hyp=self.weak_aug,    # For teacher pseudo-label generation
                strong_hyp=self.strong_aug, # For student training
            )
            # self.data should be available now (loaded by check_det_dataset in super)
            self.semi_data.setup({'data': self.data})
            LOGGER.info(colorstr('Semi-SSL: ') + 
                f'Data: {self.semi_data.num_labeled} labeled, {self.semi_data.num_unlabeled} unlabeled')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        labeled_loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        
        if mode == 'train':
            self._init_semi_data()
            
        return labeled_loader

    def _do_train(self):
        """Main training loop with semi-supervised learning."""
        if self.world_size > 1:
            self._setup_ddp()

        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100)
        
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()

        self.run_callbacks('on_train_start')
        LOGGER.info(colorstr('Semi-SSL: ') + f'Burn-in for {self.burn_in_epochs} epochs')
        if self.semi_data:
            LOGGER.info(colorstr('Semi-SSL: ') + f'Labeled: {self.semi_data.num_labeled}, Unlabeled: {self.semi_data.num_unlabeled}')
        LOGGER.info(colorstr('Semi-SSL: ') + f'{self.filter_chain}')
        self.last_opt_step = -1
        self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.model.train()
            self.run_callbacks('on_train_epoch_start')
            self.filter_chain.update(epoch, self.epochs)

            self.in_burn_in = epoch < self.burn_in_epochs
            if not self.in_burn_in:
                self._update_dsat_from_metrics()
            pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                ni = i + nb * epoch
                lambda_u = self._get_lambda_unsup(epoch)
                
                if ni <= nw:
                    self._warmup(ni, nw)

                with torch.autocast(self.device.type, enabled=self.amp):
                    batch = self.preprocess_batch(batch)
                    if not self.in_burn_in:
                        imgs = batch["img"]
                        batch["img"] = self.aug_noise(imgs)
                    loss_sup, loss_items_sup = self.model(batch)

                    if self.in_burn_in or self.semi_data is None:
                        loss = loss_sup
                        loss_unsup = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                        loss_items_unsup = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                    else:
                        loss_unsup, loss_items_unsup = self._compute_unsup_loss()
                        loss = loss_sup + lambda_u * loss_unsup

                    self.loss_items = loss_items_sup + lambda_u * loss_items_unsup
                    self.loss = loss.sum()
                self.scaler.scale(self.loss).backward()

                if ni - self.last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.last_opt_step = ni

                if not self.in_burn_in:
                    self.teacher.update(self.model)

                self._update_loss(self.loss_items)

                self.run_callbacks('on_train_batch_end')
                pbar.set_postfix(
                    epoch=f'{epoch + 1}/{self.epochs}',
                    mode='Burn-in' if self.in_burn_in else 'Semi',
                    loss=f'{self.loss.item():.4f}',
                    loss_sup=f'{loss_sup.sum().item():.4f}',
                    loss_unsup=f'{loss_unsup.sum().item():.4f}' if not self.in_burn_in else '-',
                    lambda_u = f'{lambda_u}'
                )
                pbar.update()
                if i > nb and self.in_burn_in:
                    break
            pbar.close()

            self.lr = {f'lr/pg{i}': x['lr'] for i, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')
            
            if RANK in {-1, 0}:
                final = epoch + 1 == self.epochs
                self.metrics, self.fitness = self.validate()
                self.save_model()
                self.run_callbacks('on_model_save')

        self.run_callbacks('on_train_end')

    def _compute_unsup_loss(self):
        """
        Compute unsupervised loss using pseudo-labels from teacher.
        """
        if self.semi_data is None:
            LOGGER.warning(colorstr('Semi-SSL DEBUG: ') + 'semi_data is None!')
            return torch.tensor([0.0, 0.0, 0.0], device=self.device), torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # Step 1: Get batches
        unlabeled_weak, unlabeled_strong = self.semi_data.get_unsup_batch()

        if unlabeled_weak['img'].max() > 1.0: 
            unlabeled_weak['img'] = unlabeled_weak['img'].to(self.device).float() / 255.0
        weak_imgs = unlabeled_weak['img']
        img_size = weak_imgs.shape[2]
        batch_size = weak_imgs.shape[0]

        # Step 2: Teacher inference
        with torch.no_grad():
            teacher_model = self.teacher.ema
            teacher_model.eval()
            teacher_preds = teacher_model(weak_imgs)
        
        pseudo_results = self._extract_predictions_per_image(teacher_preds, img_size)
        
        all_boxes = []
        all_cls = []
        all_batch_idx = []
        total_boxes = 0
        total_before_filter = 0
        total_after_filter = 0
 
        for img_idx, (boxes, scores, labels, uncertainties) in enumerate(pseudo_results):
            total_before_filter += len(boxes)
            if len(boxes) > 0:
                predictions = {
                    'boxes': boxes,
                    'labels': labels,
                    'cls_scores': scores,
                    'uncertainties': uncertainties,
                }
                mask, _ = self.filter_chain(predictions)
                total_after_filter += mask.sum().item()
                
                if mask.sum().item() > 0:
                    boxes_xywh = self._xyxy_to_xywhn(boxes[mask], img_size)
                    all_boxes.append(boxes_xywh)
                    all_cls.append(labels[mask].float())
                    all_batch_idx.append(torch.full((mask.sum(),), img_idx, device=self.device))
                    total_boxes += mask.sum().item()

        if total_boxes == 0:
            all_boxes = [torch.zeros((0, 4), device=self.device)]
            all_cls = [torch.zeros(0, device=self.device)]
            all_batch_idx = [torch.zeros(0, device=self.device)]
        
        if unlabeled_strong['img'].max() > 1.0: 
            unlabeled_strong['img'] = unlabeled_strong['img'].to(self.device).float() / 255.0

        pseudo_batch = {
            'img': unlabeled_strong['img'],
            'cls': torch.cat(all_cls),
            'bboxes': torch.cat(all_boxes),
            'batch_idx': torch.cat(all_batch_idx),
        }
        loss_unsup, loss_items_unsup = self.model(pseudo_batch)

        return loss_unsup, loss_items_unsup
    
    def _extract_predictions_per_image(self, preds, img_size: int):
        """Extract predictions per image using NMS-Unc.
        
        Note: uncertainty_threshold=1.0 to disable filtering here.
        Uncertainty filtering is done later in DSAT filter with better control.
        """
        from filters.nms_unc import nms_unc_batched
        
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        per_image_results = nms_unc_batched(
            preds,
            conf_thres=0.01,  # Low threshold to keep more boxes
            iou_thres=0.5,
            max_det=100,
            nc=self.model.nc,
            uncertainty_threshold=1.0,  # Disabled - filter later in DSAT
        )
        
        return per_image_results
    
    def _apply_filters(self, teacher_pred):
        """Apply filter chain from config."""
        predictions = {
            'boxes': teacher_pred[:, :4],
            'labels': teacher_pred[:, 5].long(),
            'cls_scores': teacher_pred[:, 4],
        }
        mask, filtered_scores = self.filter_chain(predictions)
        return mask, filtered_scores
    
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

    def _update_dsat_from_metrics(self):
        """Update DSAT thresholds based on per-class F1 scores from teacher validation."""
        if self.teacher is None or self.in_burn_in:
            return
        
        for f in self.filter_chain.filters:
            if isinstance(f, DSATFilter):
                LOGGER.info(colorstr('Semi-SSL: ') + 'Running DSAT validation on teacher...')
                per_class_f1 = self._validate_teacher_f1()
                if per_class_f1 is not None:
                    f.update_from_f1(per_class_f1)
                    LOGGER.info(colorstr('Semi-SSL: ') + f'Updated DSAT thresholds: {f.class_thresholds.tolist()[:5]}...')
                break
    
    def _validate_teacher_f1(self):
        """Run lightweight validation on teacher model to get per-class F1."""
        try:
            validator = DSATValidator(
                nc=self.model.nc,
                iou_threshold=0.5,
                conf_threshold=0.001,
                device=self.device,
            )
            per_class_f1 = validator.validate(
                model=self.teacher.ema,
                dataloader=self.test_loader,
            )
            return per_class_f1
        except Exception as e:
            LOGGER.warning(f'DSAT validation failed: {e}')
            return None

    def _extract_per_class_f1_from_metrics(self, metrics):
        """Extract per-class F1 scores from validation metrics."""
        try:
            if hasattr(metrics, 'results_dict'):
                results = metrics.results_dict
                per_class_f1 = []
                for i in range(self.model.nc):
                    key = f'metrics/f1_per_class_{i}'
                    if key in results:
                        per_class_f1.append(results[key])
                if len(per_class_f1) == self.model.nc:
                    return torch.tensor(per_class_f1)
            
            if hasattr(metrics, 'box'):
                box_metrics = metrics.box
                if hasattr(box_metrics, 'f1') and box_metrics.f1 is not None:
                    f1 = box_metrics.f1
                    if len(f1) == self.model.nc:
                        return torch.tensor(f1)
                        
        except Exception as e:
            LOGGER.warning(f'Could not extract per-class F1: {e}')
        
        return None
