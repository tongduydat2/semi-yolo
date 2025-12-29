"""
SSOD Trainer V3 - Merged Dataset Approach (Option C)

Key idea: 
- Merge labeled + pseudo-labeled data into ONE dataset
- Single YOLO.train() call per epoch
- No separate supervised/unsupervised phases
- Simple and effective

This avoids:
- Optimizer reset between phases
- Scheduler reset
- Catastrophic forgetting
"""

import yaml
import logging
import shutil
import torch
from pathlib import Path
from copy import deepcopy
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SSODTrainerV3:
    """
    Simplified SSOD Trainer using merged dataset approach.
    
    Flow per epoch:
    1. Generate pseudo-labels with Teacher
    2. Merge pseudo-labels INTO labeled dataset (temporary)
    3. Train Student on merged dataset (single train call)
    4. Update Teacher via EMA
    5. Cleanup merged data
    
    Benefits:
    - Single optimizer, single scheduler
    - No state reset between phases  
    - Labeled data always present (prevents forgetting)
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.student = None
        self.teacher = None
        self.ema_decay = self.config.get('ssod', {}).get('ema_rate', 0.999)
        
        # Paths
        self.merged_dir = self.output_dir / "merged_train"
        
        logger.info("SSOD Trainer V3 (Merged Dataset) initialized")
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_models(self):
        """Initialize Student and Teacher models."""
        model_cfg = self.config['model']
        
        logger.info(f"Loading models from: {model_cfg['base_model']}")
        
        # Student = YOLO wrapper for training
        self.student = YOLO(model_cfg['base_model'])
        
        # Teacher = Copy of student for pseudo-label generation
        self.teacher = YOLO(model_cfg['base_model'])
        
        # Freeze Teacher
        for param in self.teacher.model.parameters():
            param.requires_grad_(False)
        self.teacher.model.eval()
        
        logger.info("Student and Teacher models initialized")
    
    def _update_teacher_ema(self):
        """Update Teacher weights via EMA from Student."""
        decay = self.ema_decay
        
        with torch.no_grad():
            student_state = self.student.model.state_dict()
            teacher_state = self.teacher.model.state_dict()
            
            for key in teacher_state:
                if key in student_state and teacher_state[key].shape == student_state[key].shape:
                    # Ensure both tensors on same device
                    s_tensor = student_state[key]
                    t_tensor = teacher_state[key]
                    
                    if t_tensor.device != s_tensor.device:
                        t_tensor = t_tensor.to(s_tensor.device)
                    
                    teacher_state[key] = decay * t_tensor + (1 - decay) * s_tensor
            
            self.teacher.model.load_state_dict(teacher_state)
        
        logger.info(f"Teacher updated via EMA (decay={decay})")
    
    def _generate_pseudo_labels(self, threshold: float):
        """Generate pseudo-labels using Teacher model."""
        data_cfg = self.config['data']
        unlabeled_dir = Path(data_cfg['unlabeled']['images'])
        
        pseudo_labels_dir = self.output_dir / "pseudo_labels"
        pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_files = list(unlabeled_dir.glob("*.jpg")) + \
                     list(unlabeled_dir.glob("*.jpeg")) + \
                     list(unlabeled_dir.glob("*.png"))
        
        logger.info(f"Generating pseudo-labels for {len(image_files)} images (threshold={threshold:.3f})...")
        
        total_boxes = 0
        images_with_labels = 0
        
        for img_file in image_files:
            results = self.teacher.predict(
                source=str(img_file),
                conf=threshold,
                verbose=False,
                device=0
            )
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                result = results[0]
                img_h, img_w = result.orig_shape
                
                labels = []
                for box in result.boxes:
                    cls = int(box.cls.item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    x_center = (xyxy[0] + xyxy[2]) / 2 / img_w
                    y_center = (xyxy[1] + xyxy[3]) / 2 / img_h
                    width = (xyxy[2] - xyxy[0]) / img_w
                    height = (xyxy[3] - xyxy[1]) / img_h
                    labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                if labels:
                    label_file = pseudo_labels_dir / f"{img_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(labels))
                    total_boxes += len(labels)
                    images_with_labels += 1
        
        logger.info(f"Pseudo-labels: {total_boxes} boxes, {images_with_labels} images")
        
        return {
            'total_boxes': total_boxes,
            'images_with_labels': images_with_labels,
            'dir': pseudo_labels_dir
        }
    
    def _create_merged_dataset(self, pseudo_labels_dir: Path):
        """
        Merge labeled data + pseudo-labeled data into single dataset.
        
        Creates:
        - merged_train/images/ (symlinks to labeled + unlabeled images)
        - merged_train/labels/ (symlinks to labeled + pseudo labels)
        """
        data_cfg = self.config['data']
        
        # Clean old merged data
        if self.merged_dir.exists():
            shutil.rmtree(self.merged_dir)
        
        merged_images = self.merged_dir / "images"
        merged_labels = self.merged_dir / "labels"
        merged_images.mkdir(parents=True, exist_ok=True)
        merged_labels.mkdir(parents=True, exist_ok=True)
        
        # Source paths
        if data_cfg.get('skip_colorization', False):
            labeled_images = Path(data_cfg['labeled']['images'])
            labeled_labels = Path(data_cfg['labeled']['labels'])
        else:
            labeled_images = Path(data_cfg['fake_ironred']['images'])
            labeled_labels = Path(data_cfg['fake_ironred']['labels'])
        
        unlabeled_images = Path(data_cfg['unlabeled']['images'])
        
        copied_labeled = 0
        copied_pseudo = 0
        
        # Copy labeled data
        for img_file in labeled_images.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_file = labeled_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(img_file, merged_images / img_file.name)
                    shutil.copy2(label_file, merged_labels / label_file.name)
                    copied_labeled += 1
        
        # Copy pseudo-labeled data
        for label_file in pseudo_labels_dir.glob("*.txt"):
            img_name = label_file.stem
            for ext in ['.jpg', '.jpeg', '.png']:
                img_src = unlabeled_images / f"{img_name}{ext}"
                if img_src.exists():
                    # Prefix to avoid name collision
                    new_img_name = f"pseudo_{img_src.name}"
                    new_label_name = f"pseudo_{label_file.name}"
                    shutil.copy2(img_src, merged_images / new_img_name)
                    shutil.copy2(label_file, merged_labels / new_label_name)
                    copied_pseudo += 1
                    break
        
        logger.info(f"Merged dataset: {copied_labeled} labeled + {copied_pseudo} pseudo = {copied_labeled + copied_pseudo} total")
        
        return copied_labeled + copied_pseudo
    
    def _create_data_yaml(self) -> str:
        """Create data.yaml for merged dataset."""
        model_cfg = self.config['model']
        data_cfg = self.config['data']
        
        # Class names
        if 'class_names' in model_cfg:
            class_names = model_cfg['class_names']
        else:
            class_names = {i: f'class_{i}' for i in range(model_cfg['num_classes'])}
        
        val_path = Path(data_cfg['val']['images']).resolve()
        
        data_yaml = {
            'train': str((self.merged_dir / "images").resolve()),
            'val': str(val_path),
            'names': class_names
        }
        
        yaml_path = self.output_dir / "merged_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        return str(yaml_path)
    
    def train(self):
        """Main training loop."""
        train_cfg = self.config['training']
        ssod_cfg = self.config.get('ssod', {})
        
        max_epochs = train_cfg['max_epochs']
        burn_in_epochs = train_cfg['burn_in_epochs']
        tau_start = ssod_cfg.get('tau_start', 0.5)
        tau_end = ssod_cfg.get('tau_end', 0.75)
        
        logger.info("=" * 50)
        logger.info("Starting SSOD Training V3 (Merged Dataset)")
        logger.info(f"Max epochs: {max_epochs}")
        logger.info(f"Burn-in epochs: {burn_in_epochs}")
        logger.info("=" * 50)
        
        # Initialize models
        self._init_models()
        
        for epoch in range(max_epochs):
            is_burnin = epoch < burn_in_epochs
            phase = "BURN-IN" if is_burnin else "SSOD"
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{max_epochs} - Phase: {phase}")
            logger.info("=" * 50)
            
            if is_burnin:
                # Burn-in: Train only on labeled data
                data_yaml = self._create_labeled_yaml()
            else:
                # SSOD: Generate pseudo-labels and merge
                progress = (epoch - burn_in_epochs) / max(max_epochs - burn_in_epochs, 1)
                threshold = tau_start + (tau_end - tau_start) * progress
                
                logger.info(f"Generating pseudo-labels (threshold={threshold:.3f})...")
                pseudo_stats = self._generate_pseudo_labels(threshold)
                
                if pseudo_stats['images_with_labels'] > 0:
                    self._create_merged_dataset(pseudo_stats['dir'])
                    data_yaml = self._create_data_yaml()
                else:
                    logger.warning("No pseudo-labels, using labeled data only")
                    data_yaml = self._create_labeled_yaml()
            
            # Train Student (single call per epoch!)
            logger.info("Training Student...")
            self.student.train(
                data=data_yaml,
                epochs=1,
                batch=train_cfg['batch_size'],
                imgsz=self.config['model']['imgsz'],
                device=0,
                exist_ok=True,
                project=str(self.output_dir),
                name=f"epoch_{epoch}",
            )
            
            # Update Teacher via EMA
            self._update_teacher_ema()
            
            # Load latest weights into Student for next iteration
            latest_weights = self.output_dir / f"epoch_{epoch}" / "weights" / "last.pt"
            if latest_weights.exists():
                logger.info(f"Loading weights from: {latest_weights}")
                self.student = YOLO(str(latest_weights))
                # Also update Teacher to stay close to Student
                self.teacher = YOLO(str(latest_weights))
                for param in self.teacher.model.parameters():
                    param.requires_grad_(False)
                self.teacher.model.eval()
            
            # Cleanup
            pseudo_dir = self.output_dir / "pseudo_labels"
            if pseudo_dir.exists():
                shutil.rmtree(pseudo_dir)
            if self.merged_dir.exists():
                shutil.rmtree(self.merged_dir)
        
        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info("=" * 50)
    
    def _create_labeled_yaml(self) -> str:
        """Create data.yaml for labeled data only (burn-in phase)."""
        model_cfg = self.config['model']
        data_cfg = self.config['data']
        
        if data_cfg.get('skip_colorization', False):
            train_path = Path(data_cfg['labeled']['images']).resolve()
        else:
            train_path = Path(data_cfg['fake_ironred']['images']).resolve()
        
        val_path = Path(data_cfg['val']['images']).resolve()
        
        if 'class_names' in model_cfg:
            class_names = model_cfg['class_names']
        else:
            class_names = {i: f'class_{i}' for i in range(model_cfg['num_classes'])}
        
        data_yaml = {
            'train': str(train_path),
            'val': str(val_path),
            'names': class_names
        }
        
        yaml_path = self.output_dir / "labeled_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        return str(yaml_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SSOD Training V3 (Merged Dataset)')
    parser.add_argument('--config', '-c', type=str, default='config/ssod_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    trainer = SSODTrainerV3(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
