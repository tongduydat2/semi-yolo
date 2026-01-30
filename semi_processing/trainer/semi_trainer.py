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
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.cfg import get_cfg, DEFAULT_CFG

from filters.base import FilterChain, build_filter_chain
from data.semi_dataset import SemiDataModule
from augmentation.configs import WEAK_AUG, STRONG_AUG, merge_aug_config
from filters.dsat import DSATFilter
from validators.dsat_validator import DSATValidator
from ultralytics.utils import TQDM
from data.semi_dataset import ThermalAugmentation
from tqdm import tqdm
from losses.bg_penalty_loss import (
    v8DetectionLossWithBgPenalty,
    AdaptiveBgPenaltyScheduler
)
from utils.selective_ema import SelectiveModelEMA
from utils.eval_watcher import EvalDataWatcher
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
        self.thermal_aug = ThermalAugmentation()
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
        self.ema_tau = self.semi_cfg.get('ema_tau', 2000)
        
        # Background penalty configuration
        self.lambda_bg = self.semi_cfg.get('lambda_bg', 1.0)
        self.lambda_bg_warmup = self.semi_cfg.get('lambda_bg_warmup', 5)
        self.lambda_bg_schedule = self.semi_cfg.get('lambda_bg_schedule', 'linear')
        self.use_focal_bg = self.semi_cfg.get('use_focal_bg', False)
        
        # Initialize background penalty scheduler
        self.bg_penalty_scheduler = AdaptiveBgPenaltyScheduler(
            lambda_bg_max=self.lambda_bg,
            warmup_epochs=self.lambda_bg_warmup,
            schedule=self.lambda_bg_schedule,
            burn_in_epochs=self.burn_in_epochs,
        )
        
        # Layer freezing configuration
        self.freeze_cfg = self.semi_cfg.get('freeze', {})
        self.freeze_enabled = self.freeze_cfg.get('enabled', False)
        self.freeze_backbone = self.freeze_cfg.get('freeze_backbone', False)
        self.freeze_layers = self.freeze_cfg.get('freeze_layers', [])
        
        self.teacher = None
        self.filter_chain = None
        self.semi_data = None

        self.in_burn_in = True
        self.last_debug_epoch = -1
        
        # Phase-specific checkpoint tracking
        self.best_fitness_burnin = 0.0  # Best fitness during burn-in
        self.best_fitness_semi = 0.0     # Best fitness during semi-supervised
        
        LOGGER.info(colorstr('Semi-SSL: ') + f'Weak aug: mosaic={self.weak_aug.get("mosaic")}, mixup={self.weak_aug.get("mixup")}')
        LOGGER.info(colorstr('Semi-SSL: ') + f'Strong aug: mosaic={self.strong_aug.get("mosaic")}, mixup={self.strong_aug.get("mixup")}')
        LOGGER.info(colorstr('Semi-SSL: ') + 
                   f'Background penalty: λ_bg={self.lambda_bg}, warmup={self.lambda_bg_warmup}, '
                   f'schedule={self.lambda_bg_schedule}, focal={self.use_focal_bg}')

    def _setup_train(self):
        """Setup training with teacher model and data module."""
        super()._setup_train()
        
        # Replace default loss with custom loss including background penalty
        current_lambda_bg = self.bg_penalty_scheduler.get_lambda_bg(0)
        self.model.criterion = v8DetectionLossWithBgPenalty(
            self.model,
            tal_topk=10,
            lambda_bg=current_lambda_bg,
            use_focal_bg=self.use_focal_bg,
        )
        LOGGER.info(colorstr('Semi-SSL: ') + 
                   f'Using v8DetectionLossWithBgPenalty (λ_bg={current_lambda_bg:.3f})')

        self.teacher = SelectiveModelEMA(self.model, decay=self.ema_decay, tau=self.ema_tau)
        LOGGER.info(colorstr('Semi-SSL: ') + f'Teacher initialized with SelectiveEMA decay={self.ema_decay}')

        filter_config = self.semi_cfg.get('filters', [
            {'name': 'dsat', 'params': {'num_classes': self.data.get('nc', 80)}},
            {'name': 'dfl_entropy', 'params': {'threshold': 0.5}},
            {'name': 'tal_alignment', 'params': {'threshold': 0.3}},
        ])
        self.filter_chain = build_filter_chain(filter_config)
        LOGGER.info(colorstr('Semi-SSL: ') + f'Filter chain: {[f.__class__.__name__ for f in self.filter_chain.filters]}')
        
        # Freeze layers if configured
        self._freeze_layers()
        
        # Initialize eval data watcher for human-in-the-loop
        self._init_eval_watcher()


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

        # [FIX LOOP] Xác định số batch dựa trên tập Unlabeled (Lớn nhất)
        if self.semi_data and self.semi_data.unlabeled_loader_weak:
            nb = len(self.semi_data.unlabeled_loader_weak)
        else:
            nb = len(self.train_loader)
            
        nw = max(round(self.args.warmup_epochs * nb), 100)
        
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()

        self.run_callbacks('on_train_start')
        LOGGER.info(colorstr('Semi-SSL: ') + f'Burn-in for {self.burn_in_epochs} epochs')
        if self.semi_data:
            LOGGER.info(colorstr('Semi-SSL: ') + f'Labeled: {self.semi_data.num_labeled}, Unlabeled: {self.semi_data.num_unlabeled}')
        
        self.last_opt_step = -1
        self.optimizer.zero_grad()
        self.best_loss = float('inf')
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            # Update background penalty weight for current epoch
            current_lambda_bg = self.bg_penalty_scheduler.get_lambda_bg(epoch)
            if hasattr(self.model, 'criterion') and hasattr(self.model.criterion, 'lambda_bg'):
                self.model.criterion.lambda_bg = current_lambda_bg
                if epoch % 5 == 0 or epoch == self.burn_in_epochs:
                    LOGGER.info(colorstr('Semi-SSL: ') + f'Epoch {epoch}: λ_bg={current_lambda_bg:.3f}')

            # Sync Epoch cho DataModule
            if self.semi_data:
                self.semi_data.set_epoch(epoch)
            
            self.model.train()
            self.run_callbacks('on_train_epoch_start')
            
            if self.filter_chain:
                self.filter_chain.update(epoch, self.epochs)

            self.in_burn_in = epoch < self.burn_in_epochs
            
            # Update DSAT thresholds (nếu có)
            if not self.in_burn_in:
                self._update_dsat_from_metrics()
            
            
            # [FIX LOOP] Lặp theo số batch của Unlabeled Data
            # Xác định số iterations tối đa cho epoch này
            if self.in_burn_in:
                # Burn-in: chỉ lặp qua labeled data
                max_iters = len(self.train_loader)
            else:
                # Semi-supervised: lặp qua unlabeled data (lớn hơn)
                max_iters = nb
                
            pbar = tqdm(range(max_iters), total=max_iters, bar_format='{l_bar}{bar:10}{r_bar}')
            self.tloss = None

            for i in pbar:
                self.run_callbacks('on_train_batch_start')
                ni = i + nb * epoch
                lambda_u = self._get_lambda_unsup(epoch)

                # 1. Lấy dữ liệu (Đã đồng bộ)
                # Giả định SemiDataModule đã có hàm get_semi_batch()
                # Nếu chưa có, bạn cần gộp get_batch() và get_unsup_batch() như hướng dẫn trước
                batch_labeled, batch_u_weak, batch_u_strong = self.semi_data.get_semi_batch()
            
                # 2. Tiền xử lý (Preprocess) & Augmentation
                with torch.autocast(self.device.type, enabled=self.amp):
                    # Preprocess: Chuyển lên GPU, chia 255 (nếu cần)
                    batch_labeled = self.preprocess_batch(batch_labeled)
                    
                    # [FIX FLOAT] Xử lý thủ công cho Unlabeled nếu preprocess mặc định không chạy
                    if isinstance(batch_u_weak['img'], torch.Tensor) and batch_u_weak['img'].dtype == torch.uint8:
                         batch_u_weak['img'] = batch_u_weak['img'].float() / 255.0
                    elif batch_u_weak['img'].max() > 1.0:
                         batch_u_weak['img'] = batch_u_weak['img'].float() / 255.0
                    batch_u_weak['img'] = batch_u_weak['img'].to(self.device)

                    if isinstance(batch_u_strong['img'], torch.Tensor) and batch_u_strong['img'].dtype == torch.uint8:
                         batch_u_strong['img'] = batch_u_strong['img'].float() / 255.0
                    elif batch_u_strong['img'].max() > 1.0:
                         batch_u_strong['img'] = batch_u_strong['img'].float() / 255.0
                    batch_u_strong['img'] = batch_u_strong['img'].to(self.device)

                    # 3. Augmentation thêm (Thermal Noise) - Chỉ khi hết Burn-in
                    if not self.in_burn_in:
                        batch_labeled['img'] = self.thermal_aug(batch_labeled['img'])
                        # batch_u_strong['img'] = self.thermal_aug(batch_u_strong['img'])
                
                    # Warmup
                    if ni <= nw:
                        self._warmup(ni, nw)

                    # 4. Supervised Forward
                    loss_sup, loss_items_sup = self.model(batch_labeled)

                    # 5. Unsupervised Forward
                    if self.in_burn_in or self.semi_data is None:
                        loss = loss_sup
                        loss_unsup = torch.tensor(0.0, device=self.device)
                        loss_items_unsup = torch.zeros(3, device=self.device)
                    else:
                        loss_unsup, loss_items_unsup, total_boxes = self._compute_unsup_loss(batch_u_weak, batch_u_strong)
                        loss = loss_sup + lambda_u * loss_unsup

                    # 6. Backward
                    self.loss_items = loss_items_sup + lambda_u * loss_items_unsup
                    self.loss = loss.sum() # Đảm bảo là scalar
                
                self.scaler.scale(self.loss).backward()

                if ni - self.last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.last_opt_step = ni

                self.teacher.update(self.model)

                self._update_loss(self.loss_items)

                self.run_callbacks('on_train_batch_end')
                pbar.set_postfix(
                    epoch=f'{epoch + 1}/{self.epochs}',
                    mode='Burn-in' if self.in_burn_in else 'Semi',
                    loss=f'{self.loss.item():.4f}',
                    loss_sup=f'{loss_sup.sum().item():.4f}',
                    loss_unsup=f'{loss_unsup.sum().item():.4f}' if not self.in_burn_in else '-',
                    lambda_u=f'{lambda_u:.2f}',
                    lambda_bg=f'{current_lambda_bg:.2f}',
                    total_boxes=f'{total_boxes:.2f}' if not self.in_burn_in else '-',
                )
                
            # End Batch Loop
            pbar.close()
            
            # Log background penalty statistics
            if hasattr(self.model, 'criterion') and hasattr(self.model.criterion, 'get_bg_penalty_stats'):
                bg_stats = self.model.criterion.get_bg_penalty_stats()
                if bg_stats['mean'] > 0:
                    LOGGER.info(colorstr('Semi-SSL: ') + 
                               f'Epoch {epoch} BG Penalty Stats: '
                               f'mean={bg_stats["mean"]:.4f}, max={bg_stats["max"]:.4f}, min={bg_stats["min"]:.4f}')
                self.model.criterion.reset_bg_penalty_history()

            self.lr = {f'lr/pg{i}': x['lr'] for i, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')
            
            if RANK in {-1, 0}:
                final = epoch + 1 == self.epochs
                self.metrics, self.fitness = self.validate()
                self.save_model()
                self.run_callbacks('on_model_save')
                
                # Phase-specific checkpoint saving
                self._save_phase_checkpoints(epoch, final)

        self.run_callbacks('on_train_end')

    def _compute_unsup_loss(self, batch_u_weak, batch_u_strong):
        """
        Compute unsupervised loss using pseudo-labels from teacher.
        Falls back to consistency regularization when no pseudo-labels available.
        """
        if self.semi_data is None:
            return torch.tensor(0.0, device=self.device), torch.zeros(3, device=self.device), 0

        # Teacher inference (trên ảnh Weak)
        with torch.no_grad():
            # Teacher EMA model
            teacher_model = self.teacher.ema
            teacher_model.eval()
            teacher_preds = teacher_model(batch_u_weak['img'])
        
        # Lấy kích thước ảnh để normalize box
        img_size = batch_u_weak['img'].shape[2] # H hoặc W (giả sử vuông)
        
        # Extract predictions per image
        pseudo_results = self._extract_predictions_per_image(teacher_preds, img_size)
        
        all_boxes = []
        all_cls = []
        all_batch_idx = []
        total_boxes = 0

        # Loop qua từng ảnh để lọc và gom Pseudo-Label
        for img_idx, (boxes, scores, labels, uncertainties) in enumerate(pseudo_results):
            if len(boxes) > 0:
                predictions = {
                    'boxes': boxes,
                    'labels': labels,
                    'cls_scores': scores,
                    'uncertainties': uncertainties,
                }
                # Lọc qua FilterChain (DSAT...)
                if self.filter_chain:
                    mask, _ = self.filter_chain(predictions)
                else:
                    mask = scores > 0.5 # Fallback nếu không có filter

                # [DEBUG] Visualize Pseudo-Labels (Once per epoch)
                if img_idx == 0 and self.epoch > self.last_debug_epoch:
                    self.last_debug_epoch = self.epoch
                    self._save_debug_visualization(
                        batch_u_weak['img'][0], 
                        boxes, labels, scores, mask,
                        img_size,
                        f"epoch_{self.epoch}_batch_0.jpg"
                    )

                if mask.sum().item() > 0:
                    # Convert sang xywh normalized (YOLO format)
                    boxes_xywh = self._xyxy_to_xywhn(boxes[mask], img_size)
                    all_boxes.append(boxes_xywh)
                    all_cls.append(labels[mask].float())
                    all_batch_idx.append(torch.full((mask.sum(),), img_idx, device=self.device))
                    total_boxes += mask.sum().item()

        # [FIX LOSS] Xử lý batch rỗng (Background)
        if total_boxes == 0:
            return torch.tensor([0.0, 0.0, 0.0], device=self.device), torch.tensor([0.0, 0.0, 0.0], device=self.device), 0
       
        final_boxes = torch.cat(all_boxes)
        final_cls = torch.cat(all_cls)
        final_batch_idx = torch.cat(all_batch_idx)

        # Tạo Pseudo Batch cho Student
        pseudo_batch = {
            'img': batch_u_strong['img'],
            'cls': final_cls.view(-1, 1), # Shape (N, 1)
            'bboxes': final_boxes,        # Shape (N, 4)
            'batch_idx': final_batch_idx.view(-1, 1) # Shape (N, 1)
        }
        
        # Student Forward & Loss
        loss_unsup, loss_items_unsup = self.model(pseudo_batch)

        return loss_unsup, loss_items_unsup, total_boxes
    
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
            max_det=1000,
            nc=self.model.nc,
            uncertainty_threshold=1.0,  # Disabled - filter later in DSAT
        )
        
        return per_image_results
    
    def _save_debug_visualization(self, img_tensor, boxes, labels, scores, mask, img_size, filename):
        """Save a visualization of predicted pseudo-labels for debugging."""
        try:
            # Create debug directory
            debug_dir = self.save_dir / 'debug_pseudo'
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert image tensor to numpy
            if img_tensor.max() <= 1.0:
                img_tensor = img_tensor * 255.0
            
            img_numpy = img_tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            img_numpy = np.ascontiguousarray(img_numpy) # CV2 needs contiguous
            
            # Apply mask to get final pseudo-labels
            if mask is not None:
                boxes = boxes[mask]
                labels = labels[mask]
                scores = scores[mask]
            
            annotator = Annotator(img_numpy, line_width=2, example=str(self.model.names))
            
            if len(boxes) > 0:
                for j, box in enumerate(boxes):
                    cls = int(labels[j])
                    conf = float(scores[j])
                    class_name = self.model.names[cls] if hasattr(self.model, 'names') else str(cls)
                    label = f"{class_name} {conf:.2f}"
                    annotator.box_label(box, label, color=colors(cls, True))
            
            # Save
            import cv2
            save_path = debug_dir / filename
            cv2.imwrite(str(save_path), annotator.result())
            
        except Exception as e:
            LOGGER.warning(f"Failed to save debug visualization: {e}")
    
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
        if epoch < self.burn_in_epochs - 1:
            return 0.0

        effective_epoch = epoch - self.burn_in_epochs + 1
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
    
    def _init_eval_watcher(self):
        """Initialize EvalDataWatcher for human-in-the-loop eval reload."""
        try:
            # Get validation data path from self.data
            val_path = self.data.get('val')
            if val_path:
                from pathlib import Path
                val_dir = Path(val_path)
                if val_dir.exists():
                    self._eval_watcher = EvalDataWatcher(eval_dir=val_dir)
                    counts = self._eval_watcher.get_file_counts()
                    LOGGER.info(colorstr('Semi-SSL: ') + 
                               f'Human-in-the-loop enabled: monitoring {val_dir} '
                               f'({counts["images"]} images, {counts["labels"]} labels)')
                else:
                    LOGGER.warning(colorstr('Semi-SSL: ') + 
                                  f'Eval watcher disabled: {val_dir} not found')
                    self._eval_watcher = None
            else:
                self._eval_watcher = None
        except Exception as e:
            LOGGER.warning(f'Failed to initialize eval watcher: {e}')
            self._eval_watcher = None
    
    def validate(self):
        """Run validation with human-in-the-loop eval data reload support.
        
        If eval data directory has changed (human added/modified files),
        reload the validation dataloader before running validation.
        
        Returns:
            Tuple[Dict, float]: metrics dict and fitness score
        """
        # Check for eval data changes (human-in-the-loop)
        if hasattr(self, '_eval_watcher') and self._eval_watcher is not None:
            if self._eval_watcher.check_changed():
                LOGGER.info(colorstr('Semi-SSL: ') + 
                           '🔄 Eval data changed! Reloading validation dataloader...')
                
                # Get updated file counts
                counts = self._eval_watcher.get_file_counts()
                LOGGER.info(colorstr('Semi-SSL: ') + 
                           f'New eval data: {counts["images"]} images, {counts["labels"]} labels')
                
                # Reload validation dataloader
                batch_size = self.args.batch // max(self.world_size, 1)
                self.test_loader = self.get_dataloader(
                    self.data.get('val') or self.data.get('test'),
                    batch_size=batch_size * 2,
                    rank=RANK,
                    mode='val',
                )
                
                # Re-create validator with new dataloader
                self.validator = self.get_validator()
                LOGGER.info(colorstr('Semi-SSL: ') + 
                           '✅ Validation dataloader reloaded successfully')
        
        # Call parent validate()
        return super().validate()

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

    def _save_phase_checkpoints(self, epoch: int, is_final: bool):
        """Save phase-specific checkpoints (best/last for burn-in and semi phases).
        
        Args:
            epoch: Current epoch number
            is_final: Whether this is the final epoch
        """
        import shutil
        from pathlib import Path
        
        weights_dir = Path(self.save_dir) / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        current_fitness = float(self.fitness) if self.fitness else 0.0
        
        # Burn-in phase (epochs 0 to burn_in_epochs-1)
        if self.in_burn_in:
            # 1. Best in burn-in
            if current_fitness > self.best_fitness_burnin:
                self.best_fitness_burnin = current_fitness
                best_burnin_path = weights_dir / 'best_burnin.pt'
                if (weights_dir / 'last.pt').exists():
                    shutil.copy2(weights_dir / 'last.pt', best_burnin_path)
                    LOGGER.info(colorstr('Semi-SSL: ') + 
                               f'Saved best burn-in checkpoint (fitness={current_fitness:.4f}) → {best_burnin_path}')
            
            # 2. Last in burn-in (save at end of burn-in phase)
            if epoch == self.burn_in_epochs - 1:
                last_burnin_path = weights_dir / 'last_burnin.pt'
                if (weights_dir / 'last.pt').exists():
                    shutil.copy2(weights_dir / 'last.pt', last_burnin_path)
                    LOGGER.info(colorstr('Semi-SSL: ') + 
                               f'Saved last burn-in checkpoint (epoch={epoch}) → {last_burnin_path}')
                    
        # Semi-supervised phase (epochs burn_in_epochs to end)
        else:
            # 3. Best in semi
            if current_fitness > self.best_fitness_semi:
                self.best_fitness_semi = current_fitness
                best_semi_path = weights_dir / 'best_semi.pt'
                if (weights_dir / 'last.pt').exists():
                    shutil.copy2(weights_dir / 'last.pt', best_semi_path)
                    LOGGER.info(colorstr('Semi-SSL: ') + 
                               f'Saved best semi checkpoint (fitness={current_fitness:.4f}) → {best_semi_path}')
            
            # 4. Last in semi (save at final epoch)
            if is_final:
                last_semi_path = weights_dir / 'last_semi.pt'
                if (weights_dir / 'last.pt').exists():
                    shutil.copy2(weights_dir / 'last.pt', last_semi_path)
                    LOGGER.info(colorstr('Semi-SSL: ') + 
                               f'Saved last semi checkpoint (epoch={epoch}) → {last_semi_path}')
            current_semi_path = weights_dir / f'current_semi_{epoch}.pt'
            if (weights_dir / 'last.pt').exists():
                shutil.copy2(weights_dir / 'last.pt', current_semi_path)
                LOGGER.info(colorstr('Semi-SSL: ') + 
                           f'Saved current semi checkpoint (epoch={epoch}) → {current_semi_path}')
                           
    def _freeze_layers(self):
        """Freeze specified layers by setting requires_grad=False.
        
        YOLO model structure (typical YOLOv11n):
        - Layers 0-9: Backbone (CSPDarknet)
        - Layers 10-15: Neck (PAN-FPN)
        - Layer 16+: Head (Detect)
        """
        if not self.freeze_enabled:
            return
        
        frozen_count = 0
        total_params = 0
        frozen_params = 0
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            # Backbone is typically first 10 layers in YOLOv11
            backbone_layers = 10
            for i in range(min(backbone_layers, len(self.model.model))):
                for param in self.model.model[i].parameters():
                    if param.requires_grad:  # Only freeze if not already frozen
                        param.requires_grad = False
                        frozen_params += param.numel()
                frozen_count += 1
            LOGGER.info(colorstr('Freeze: ') + f'Froze backbone (layers 0-{min(backbone_layers, len(self.model.model))-1})')
        
        # Freeze specific layers
        if self.freeze_layers:
            layers_to_freeze = set()  # Use set to avoid duplicates
            
            for layer_spec in self.freeze_layers:
                if isinstance(layer_spec, int):
                    # Single layer
                    layers_to_freeze.add(layer_spec)
                elif isinstance(layer_spec, str) and '-' in layer_spec:
                    # Range (e.g., "0-5")
                    try:
                        start, end = map(int, layer_spec.split('-'))
                        layers_to_freeze.update(range(start, end + 1))
                    except ValueError:
                        LOGGER.warning(f'Invalid layer range spec: {layer_spec}')
            
            # Apply freezing
            additional_frozen = 0
            for layer_idx in sorted(layers_to_freeze):
                if layer_idx < len(self.model.model):
                    for param in self.model.model[layer_idx].parameters():
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_params += param.numel()
                    additional_frozen += 1
                else:
                    LOGGER.warning(f'Layer {layer_idx} does not exist (model has {len(self.model.model)} layers)')
            
            if additional_frozen > 0:
                LOGGER.info(colorstr('Freeze: ') + 
                           f'Froze {additional_frozen} additional layers: {sorted(layers_to_freeze)}')
        
        # Count total parameters
        for param in self.model.parameters():
            total_params += param.numel()
        
        if frozen_count > 0 or len(self.freeze_layers) > 0:
            frozen_pct = 100.0 * frozen_params / total_params if total_params > 0 else 0.0
            LOGGER.info(colorstr('Freeze: ') + 
                       f'Total: {frozen_params:,} / {total_params:,} params frozen ({frozen_pct:.1f}%)')
