"""
Teacher-Student Framework for SSOD
Implements Mean Teacher architecture with YOLOv11

REFACTORED: Teacher model IS the EMA model (not separate copies)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Optional, Dict, Any, List
from pathlib import Path
from copy import deepcopy


class TeacherStudentFramework:
    """
    Mean Teacher framework for Semi-Supervised Object Detection.
    
    IMPORTANT: Teacher model is directly the EMA model, not a separate copy.
    This ensures:
    - No architecture mismatch between Student and Teacher
    - EMA updates are directly reflected in pseudo-label generation
    - No redundant model copies
    
    Components:
    - Student Model: YOLO object, trained with backpropagation
    - Teacher Model: EMA copy of Student's internal model, generates pseudo-labels
    """
    
    def __init__(self,
                 model_path: str = "yolo11n.pt",
                 ema_decay: float = 0.999,
                 device: str = "cuda",
                 num_classes: int = 1):
        """
        Initialize Teacher-Student framework.
        
        Args:
            model_path: Path to pretrained YOLO weights
            ema_decay: EMA decay rate for Teacher updates (Alpha)
            device: Device to run models on
            num_classes: Number of object classes
        """
        self.device = device
        self.ema_decay = ema_decay
        self.num_classes = num_classes
        self.model_path = model_path
        
        # Initialize Student model (YOLO wrapper)
        print(f"Loading Student model from: {model_path}")
        self.student = YOLO(model_path)
        
        # Teacher IS the EMA model (direct reference, not separate YOLO object)
        # This is the key change - we don't create a separate YOLO object
        print(f"Initializing Teacher (EMA model)...")
        self.teacher_model = self._create_ema_model(self.student.model)
        
        # EMA step counter (persistent across epochs)
        self.ema_step = 0
        self.warmup_steps = 100  # Warmup before EMA stabilizes
        
        print(f"Framework initialized on device: {device}")
        
    def _create_ema_model(self, source_model: nn.Module) -> nn.Module:
        """Create EMA model as deep copy of source."""
        ema_model = deepcopy(source_model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model
    
    @torch.no_grad()
    def update_teacher(self):
        """
        Update Teacher weights using EMA from Student.
        Formula: W_T = decay * W_T + (1 - decay) * W_S
        
        This updates the EMA model directly (no copy needed).
        """
        self.ema_step += 1
        
        # Compute effective decay (gradual ramp up during warmup)
        if self.ema_step <= self.warmup_steps:
            # During warmup, use lower decay to learn faster
            progress = self.ema_step / self.warmup_steps
            decay = self.ema_decay * progress + 0.9 * (1 - progress)
        else:
            decay = self.ema_decay
        
        student_params = dict(self.student.model.named_parameters())
        teacher_params = dict(self.teacher_model.named_parameters())
        
        updated_count = 0
        for name in student_params:
            if name in teacher_params:
                teacher_params[name].data.mul_(decay).add_(
                    student_params[name].data, alpha=1 - decay
                )
                updated_count += 1
        
        # Also update buffers (BatchNorm running stats)
        student_buffers = dict(self.student.model.named_buffers())
        teacher_buffers = dict(self.teacher_model.named_buffers())
        
        for name in student_buffers:
            if name in teacher_buffers:
                teacher_buffers[name].data.copy_(student_buffers[name].data)
        
        print(f"EMA Teacher update (step {self.ema_step}): {updated_count} params, decay={decay:.4f}")
        
    @torch.no_grad()
    def generate_pseudo_labels(self, 
                               images: List[str],
                               confidence_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Generate pseudo-labels using Teacher (EMA) model.
        
        IMPORTANT: Uses self.teacher_model directly (not a YOLO wrapper).
        This requires manual inference implementation.
        
        Args:
            images: List of image paths
            confidence_threshold: Minimum confidence to keep predictions (Tau)
            
        Returns:
            List of pseudo-labels for each image
        """
        import cv2
        import numpy as np
        
        self.teacher_model.eval()
        pseudo_labels = []
        
        # Process images in batches
        for img_path in images:
            # Read and preprocess image
            img = cv2.imread(str(img_path))
            if img is None:
                pseudo_labels.append({'boxes_yolo': [], 'classes': [], 'confidences': []})
                continue
                
            orig_h, orig_w = img.shape[:2]
            
            # Use Student's YOLO wrapper for prediction on Teacher model
            # Temporarily swap models
            original_model = self.student.model
            self.student.model = self.teacher_model
            
            try:
                results = self.student.predict(
                    source=img_path,
                    conf=confidence_threshold,
                    verbose=False,
                    device=self.device
                )
            finally:
                # Restore original Student model
                self.student.model = original_model
            
            labels = {
                'boxes_yolo': [],
                'classes': [],
                'confidences': [],
                'boxes_xyxy': []
            }
            
            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                img_h, img_w = result.orig_shape
                
                for i in range(len(result.boxes)):
                    conf = result.boxes.conf[i].item()
                    
                    if conf >= confidence_threshold:
                        xyxy = result.boxes.xyxy[i].cpu().numpy()
                        cls = int(result.boxes.cls[i].item())
                        
                        # Convert to YOLO format (normalized)
                        x_center = (xyxy[0] + xyxy[2]) / 2 / img_w
                        y_center = (xyxy[1] + xyxy[3]) / 2 / img_h
                        width = (xyxy[2] - xyxy[0]) / img_w
                        height = (xyxy[3] - xyxy[1]) / img_h
                        
                        labels['boxes_xyxy'].append(xyxy)
                        labels['boxes_yolo'].append([cls, x_center, y_center, width, height])
                        labels['classes'].append(cls)
                        labels['confidences'].append(conf)
            
            pseudo_labels.append(labels)
            
        return pseudo_labels
    
    def train_mode(self):
        """Set Student to train mode, Teacher to eval mode."""
        self.student.model.train()
        self.teacher_model.eval()
        
    def eval_mode(self):
        """Set both models to eval mode."""
        self.student.model.eval()
        self.teacher_model.eval()
    
    def get_student(self) -> YOLO:
        """Get Student YOLO model."""
        return self.student
    
    def get_teacher_model(self) -> nn.Module:
        """Get Teacher (EMA) model directly."""
        return self.teacher_model
    
    def update_student_weights(self, weights_path: str):
        """
        Update Student model with new trained weights.
        
        IMPORTANT: 
        - Reload Student from weights
        - Sync Teacher (EMA) model to match Student architecture
        - EMA step counter is preserved (not reset)
        
        Args:
            weights_path: Path to trained weights (.pt file)
        """
        print(f"Updating Student weights from: {weights_path}")
        
        # Load new Student weights
        self.student = YOLO(weights_path)
        
        # Recreate Teacher (EMA) model from new Student
        # This is necessary because YOLO fuses BatchNorm during training
        self.teacher_model = self._create_ema_model(self.student.model)
        
        print(f"Student and Teacher (EMA) synced. EMA step: {self.ema_step}")
    
    def save_checkpoint(self, path: str, epoch: int, extra_info: Optional[Dict] = None):
        """Save training checkpoint with both models and EMA state."""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.model.state_dict(),
            'teacher_state_dict': self.teacher_model.state_dict(),
            'ema_step': self.ema_step,
            'ema_decay': self.ema_decay
        }
        
        if extra_info:
            checkpoint.update(extra_info)
            
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student.model.load_state_dict(checkpoint['student_state_dict'])
        self.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        self.ema_step = checkpoint.get('ema_step', 0)
        self.ema_decay = checkpoint.get('ema_decay', self.ema_decay)
            
        print(f"Checkpoint loaded from {path}, epoch {checkpoint['epoch']}, ema_step {self.ema_step}")
        return checkpoint['epoch']
    
    def save_teacher(self, path: str):
        """Save Teacher (EMA) model state dict."""
        torch.save(self.teacher_model.state_dict(), path)
        print(f"Teacher model saved to: {path}")
    
    def load_teacher(self, path: str) -> bool:
        """Load Teacher (EMA) model state dict."""
        if Path(path).exists():
            state_dict = torch.load(path, map_location=self.device)
            self.teacher_model.load_state_dict(state_dict)
            print(f"Teacher model loaded from: {path}")
            return True
        return False
    
    def export_student(self, path: str, format: str = "pt"):
        """Export trained Student model."""
        self.student.save(path)
        print(f"Student model exported to {path}")
