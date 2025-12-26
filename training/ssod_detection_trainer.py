"""
Custom SSOD Detection Trainer kế thừa từ Ultralytics.
Key: EMA update mỗi batch + Teacher model riêng biệt.
"""

import torch
from copy import deepcopy
from pathlib import Path
from ultralytics.models.yolo.detect import DetectionTrainer


class SSODDetectionTrainer(DetectionTrainer):
    """
    Extended DetectionTrainer với Teacher-Student framework.
    
    Features:
    - Teacher model = EMA of Student (updated every batch)
    - Supports switching between labeled/pseudo dataloaders
    - Pseudo-label generation với Teacher
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # Ensure cfg is not None - use DEFAULT_CFG if None
        if cfg is None:
            cfg = {}
        super().__init__(cfg, overrides, _callbacks)
        
        # Teacher model (sẽ được init sau khi model sẵn sàng)
        self.teacher = None
        self.teacher_ema_decay = 0.999
        
    def setup_model(self):
        """Init Student và Teacher models."""
        super().setup_model()
        
        # Teacher = copy của Student ban đầu
        print("Initializing Teacher model (EMA copy of Student)...")
        self.teacher = deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        print(f"Teacher initialized with {sum(p.numel() for p in self.teacher.parameters())} params")
        
    def optimizer_step(self):
        """Override để add Teacher EMA update mỗi batch."""
        # Standard optimizer step (Student EMA update included)
        super().optimizer_step()
        
        # Update Teacher với weighted average từ Student
        if self.teacher is not None:
            self._update_teacher_ema()
    
    @torch.no_grad()
    def _update_teacher_ema(self):
        """
        Update Teacher: T = decay*T + (1-decay)*S
        Called every optimizer step (every batch).
        """
        decay = self.teacher_ema_decay
        for t_param, s_param in zip(
            self.teacher.parameters(), 
            self.model.parameters()
        ):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1-decay)
        
        # Also update buffers (BatchNorm running stats)
        for t_buf, s_buf in zip(
            self.teacher.buffers(),
            self.model.buffers()
        ):
            t_buf.data.copy_(s_buf.data)
    
    def generate_pseudo_labels(self, images_dir, output_dir, threshold=0.5):
        """
        Generate pseudo-labels từ Teacher model.
        
        Args:
            images_dir: Directory chứa unlabeled images
            output_dir: Directory để save pseudo-labels
            threshold: Confidence threshold
            
        Returns:
            Dict với stats về pseudo-labels
        """
        from ultralytics import YOLO
        
        self.teacher.eval()
        images_path = Path(images_dir)
        labels_path = Path(output_dir)
        labels_path.mkdir(parents=True, exist_ok=True)
        
        # Tạo temporary YOLO wrapper cho Teacher để dùng predict
        # Đây là trick để không cần implement inference thủ công
        temp_yolo = YOLO(self.args.model)
        temp_yolo.model = self.teacher
        
        total_predictions = 0
        images_with_labels = 0
        
        # Get all images
        image_files = list(images_path.glob("*.jpg")) + \
                     list(images_path.glob("*.jpeg")) + \
                     list(images_path.glob("*.png"))
        
        print(f"Generating pseudo-labels for {len(image_files)} images (threshold={threshold:.3f})...")
        
        for img_file in image_files:
            # Inference với Teacher
            results = temp_yolo.predict(
                source=str(img_file),
                conf=threshold,
                verbose=False,
                device=self.device
            )
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                result = results[0]
                labels = self._convert_to_yolo_format(result)
                
                if labels:
                    # Save label file
                    label_file = labels_path / f"{img_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(labels))
                    
                    total_predictions += len(labels)
                    images_with_labels += 1
        
        print(f"Pseudo-label stats: {total_predictions} boxes, {images_with_labels} images")
        
        return {
            'total_predictions': total_predictions,
            'images_with_labels': images_with_labels,
            'threshold': threshold
        }
    
    def _convert_to_yolo_format(self, result):
        """Convert prediction to YOLO label format."""
        labels = []
        if result.boxes is not None:
            img_h, img_w = result.orig_shape
            for box in result.boxes:
                cls = int(box.cls.item())
                xyxy = box.xyxy[0].cpu().numpy()
                x_center = (xyxy[0] + xyxy[2]) / 2 / img_w
                y_center = (xyxy[1] + xyxy[3]) / 2 / img_h
                width = (xyxy[2] - xyxy[0]) / img_w
                height = (xyxy[3] - xyxy[1]) / img_h
                labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return labels
    
    def get_teacher(self):
        """Get Teacher model for external use."""
        return self.teacher
