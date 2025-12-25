"""
Teacher-Student Framework for SSOD
Implements Mean Teacher architecture with YOLOv11
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Optional, Dict, Any, List
from pathlib import Path

from .ema import EMAUpdater


class TeacherStudentFramework:
    """
    Mean Teacher framework for Semi-Supervised Object Detection.
    
    Components:
    - Student Model: Trained with backpropagation on labeled + pseudo-labeled data
    - Teacher Model: Updated via EMA from Student, generates stable pseudo-labels
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
        
        # Initialize Student model
        print(f"Loading Student model from: {model_path}")
        self.student = YOLO(model_path)
        
        # Initialize Teacher as copy of Student
        print(f"Initializing Teacher model (EMA copy)...")
        self.teacher = YOLO(model_path)
        
        # Freeze Teacher gradients
        for param in self.teacher.model.parameters():
            param.requires_grad_(False)
            
        # EMA updater for syncing Student -> Teacher
        self.ema_updater = EMAUpdater(
            self.student.model, 
            decay=self.ema_decay
        )
        
        print(f"Framework initialized on device: {device}")
        
    def update_teacher(self):
        """
        Update Teacher weights using EMA from Student.
        W_T = alpha * W_T + (1 - alpha) * W_S
        
        Note: After YOLO training, BatchNorm may be fused into Conv layers,
        causing state_dict mismatch. We rebuild Teacher from Student weights.
        """
        # First update EMA model from Student
        self.ema_updater.update(self.student.model)
        
        # Rebuild Teacher from EMA weights safely
        # Instead of load_state_dict (which fails on architecture mismatch),
        # we copy matching parameters individually
        ema_state = self.ema_updater.get_model().state_dict()
        teacher_state = self.teacher.model.state_dict()
        
        # Copy only matching keys with matching shapes
        updated_keys = 0
        skipped_keys = 0
        for key in teacher_state:
            if key in ema_state and teacher_state[key].shape == ema_state[key].shape:
                teacher_state[key] = ema_state[key]
                updated_keys += 1
            else:
                skipped_keys += 1
        
        # Load the updated state dict
        self.teacher.model.load_state_dict(teacher_state)
        print(f"EMA Teacher update: {updated_keys} keys updated, {skipped_keys} skipped")
        
    @torch.no_grad()
    def generate_pseudo_labels(self, 
                               images: List[str],
                               confidence_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Generate pseudo-labels using Teacher model.
        
        Args:
            images: List of image paths or batch of images
            confidence_threshold: Minimum confidence to keep predictions (Tau)
            
        Returns:
            List of pseudo-labels for each image
        """
        self.teacher.model.eval()
        
        # Run Teacher inference
        results = self.teacher.predict(
            source=images,
            conf=confidence_threshold,
            verbose=False,
            device=self.device
        )
        
        pseudo_labels = []
        
        for result in results:
            labels = {
                'boxes': [],
                'classes': [],
                'confidences': [],
                'boxes_xyxy': [],
                'boxes_yolo': []  # [class, x_center, y_center, w, h] normalized
            }
            
            if result.boxes is not None and len(result.boxes) > 0:
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
        self.teacher.model.eval()
        
    def eval_mode(self):
        """Set both models to eval mode."""
        self.student.model.eval()
        self.teacher.model.eval()
        
    def save_checkpoint(self, 
                        path: str, 
                        epoch: int,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        extra_info: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.model.state_dict(),
            'teacher_state_dict': self.teacher.model.state_dict(),
            'ema_state_dict': self.ema_updater.state_dict()
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if extra_info:
            checkpoint.update(extra_info)
            
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, 
                        path: str,
                        optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """Load training checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student.model.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.model.load_state_dict(checkpoint['teacher_state_dict'])
        self.ema_updater.load_state_dict(checkpoint['ema_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        print(f"Checkpoint loaded from {path}, epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def get_student(self) -> YOLO:
        """Get Student model."""
        return self.student
    
    def get_teacher(self) -> YOLO:
        """Get Teacher model."""
        return self.teacher
    
    def update_student_weights(self, weights_path: str):
        """
        Update Student model with new weights.
        Also reloads Teacher from same weights to maintain architecture consistency.
        
        CRITICAL: YOLO fuses BatchNorm into Conv layers during training.
        If we don't reload Teacher, ~47% of weights cannot be updated via EMA
        due to architecture mismatch, causing Teacher to fail at predictions.
        
        Args:
            weights_path: Path to trained weights (.pt file)
        """
        print(f"Updating Student weights from: {weights_path}")
        
        # Load the trained weights into Student model
        self.student = YOLO(weights_path)
        
        # CRITICAL: Reload Teacher from same weights to match architecture
        # This ensures 100% of keys can be updated via EMA
        self.teacher = YOLO(weights_path)
        
        # Freeze Teacher gradients (Teacher never trains directly)
        for param in self.teacher.model.parameters():
            param.requires_grad_(False)
        
        # Recreate EMA updater to track new Student architecture
        self.ema_updater = EMAUpdater(
            self.student.model,
            decay=self.ema_decay
        )
        
        print(f"Student and Teacher updated from: {weights_path}")
    
    def save_teacher(self, path: str):
        """
        Save Teacher model to file.
        This preserves EMA-accumulated knowledge across epochs.
        
        Args:
            path: Path to save Teacher weights
        """
        self.teacher.save(path)
        print(f"Teacher model saved to: {path}")
    
    def load_teacher(self, path: str) -> bool:
        """
        Load Teacher model from file.
        
        Args:
            path: Path to saved Teacher weights
            
        Returns:
            True if loaded successfully, False otherwise
        """
        from pathlib import Path
        if Path(path).exists():
            self.teacher = YOLO(path)
            # Freeze Teacher gradients
            for param in self.teacher.model.parameters():
                param.requires_grad_(False)
            print(f"Teacher model loaded from: {path}")
            return True
        return False
    
    def update_student_weights_preserve_teacher(self, weights_path: str, teacher_path: str = None):
        """
        Update Student model with new weights, preserving Teacher if saved.
        
        This method:
        1. Loads new Student weights
        2. Tries to load saved Teacher (preserves EMA knowledge)
        3. Falls back to reload Teacher from Student weights if no saved Teacher
        
        Args:
            weights_path: Path to trained Student weights
            teacher_path: Path to saved Teacher weights (optional)
        """
        print(f"Updating Student weights from: {weights_path}")
        
        # Load the trained weights into Student model
        self.student = YOLO(weights_path)
        
        # Try to load saved Teacher (preserves EMA knowledge)
        teacher_loaded = False
        if teacher_path:
            teacher_loaded = self.load_teacher(teacher_path)
        
        # Fallback: reload Teacher from Student weights if no saved Teacher
        if not teacher_loaded:
            print("No saved Teacher found, reloading from Student weights...")
            self.teacher = YOLO(weights_path)
            for param in self.teacher.model.parameters():
                param.requires_grad_(False)
        
        # Recreate EMA updater to track new Student architecture
        self.ema_updater = EMAUpdater(
            self.student.model,
            decay=self.ema_decay
        )
        
        print(f"Student updated. Teacher {'preserved' if teacher_loaded else 'reset'}.")
    
    def export_student(self, path: str, format: str = "pt"):
        """Export trained Student model."""
        self.student.save(path)
        print(f"Student model exported to {path}")
