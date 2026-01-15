"""
Iterative Semi-Supervised Training with PVF-10 Synthetic Data Anchor
=====================================================================

Algorithm:
1. Generate synthetic images from PVF-10 (overlay defects on panels)
2. Train YOLO on synthetic + real labeled data
3. Use trained model to label real unlabeled images (with consistency check)
4. Add high-confidence labels to training set
5. Detect distribution collapse and adjust synthetic ratio
6. Repeat until convergence

Key Features:
- Synthetic data injection prevents confirmation bias
- Consistency filter removes noisy predictions
- Distribution collapse detection prevents drift to background
"""

import os
import sys
import yaml
import shutil
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
from copy import deepcopy

import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pseudo_dataset_PV.generate import SyntheticGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iterative_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClassDistributionGuard:
    """Monitor detection rate and alert on distribution collapse."""
    
    def __init__(self, 
                 min_detection_rate: float = 0.05,
                 collapse_threshold: float = 0.5,
                 history_size: int = 10):
        """
        Args:
            min_detection_rate: Minimum acceptable detection rate
            collapse_threshold: Alert if rate drops by this fraction
            history_size: Number of iterations to track
        """
        self.min_detection_rate = min_detection_rate
        self.collapse_threshold = collapse_threshold
        self.history = deque(maxlen=history_size)
    
    def check(self, detection_rate: float) -> Tuple[bool, str]:
        """
        Check if distribution is collapsing.
        
        Returns:
            (is_ok, message)
        """
        # First iterations - just record
        if len(self.history) < 2:
            self.history.append(detection_rate)
            return True, "OK (warming up)"
        
        # Check absolute minimum
        if detection_rate < self.min_detection_rate:
            return False, f"COLLAPSE: rate {detection_rate:.3f} < min {self.min_detection_rate}"
        
        # Check relative drop
        recent_avg = np.mean(list(self.history)[-3:])
        if detection_rate < recent_avg * (1 - self.collapse_threshold):
            return False, f"COLLAPSE: rate {detection_rate:.3f} dropped > {self.collapse_threshold*100}% from {recent_avg:.3f}"
        
        self.history.append(detection_rate)
        return True, "OK"
    
    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        if not self.history:
            return {'history': [], 'avg': 0}
        return {
            'history': list(self.history),
            'avg': np.mean(list(self.history)),
            'min': min(self.history),
            'max': max(self.history)
        }


class ConsistencyFilter:
    """Filter predictions by consistency across augmentations."""
    
    def __init__(self,
                 n_augmentations: int = 4,
                 min_consistency: float = 0.67,
                 iou_threshold: float = 0.5):
        """
        Args:
            n_augmentations: Number of augmented predictions
            min_consistency: Fraction of augmentations that must agree
            iou_threshold: IoU threshold for matching boxes
        """
        self.n_augmentations = n_augmentations
        self.min_consistency = min_consistency
        self.iou_threshold = iou_threshold
    
    def filter(self, model: YOLO, image_path: str, 
               conf_threshold: float = 0.5) -> List[Dict]:
        """
        Predict with multiple augmentations and return consistent boxes.
        
        Args:
            model: YOLO model
            image_path: Path to image
            conf_threshold: Confidence threshold
            
        Returns:
            List of consistent detection dicts
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        all_predictions = []
        
        # Augmentation functions
        augmentations = [
            lambda x: x,  # Original
            lambda x: cv2.flip(x, 1),  # Horizontal flip
            lambda x: self._adjust_brightness(x, 0.8),  # Darker
            lambda x: self._adjust_brightness(x, 1.2),  # Brighter
        ]
        
        for i, aug_fn in enumerate(augmentations[:self.n_augmentations]):
            aug_image = aug_fn(image.copy())
            
            # Predict
            results = model.predict(
                source=aug_image,
                conf=conf_threshold,
                verbose=False
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = []
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    
                    # Inverse transform for flipped images
                    if i == 1:  # Horizontal flip
                        w = image.shape[1]
                        xyxy = np.array([w - xyxy[2], xyxy[1], w - xyxy[0], xyxy[3]])
                    
                    boxes.append({
                        'xyxy': xyxy,
                        'cls': cls,
                        'conf': conf
                    })
                all_predictions.append(boxes)
            else:
                all_predictions.append([])
        
        # Find consistent boxes
        if not all_predictions or not all_predictions[0]:
            return []
        
        consistent_boxes = []
        for base_box in all_predictions[0]:
            vote_count = 1
            
            for other_preds in all_predictions[1:]:
                for other_box in other_preds:
                    if (base_box['cls'] == other_box['cls'] and 
                        self._compute_iou(base_box['xyxy'], other_box['xyxy']) > self.iou_threshold):
                        vote_count += 1
                        break
            
            # Check if enough votes
            if vote_count >= len(all_predictions) * self.min_consistency:
                consistent_boxes.append(base_box)
        
        return consistent_boxes
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-10)


class IterativeSemiTrainer:
    """
    Iterative Semi-Supervised Training with PVF-10 Synthetic Data Anchor.
    
    Main Loop:
    1. Generate synthetic images from PVF-10
    2. Merge with real labeled data
    3. Train model
    4. Predict on unlabeled with consistency check
    5. Filter by confidence and add to labeled set
    6. Check for distribution collapse
    7. Adjust synthetic ratio and repeat
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.distribution_guard = ClassDistributionGuard(
            min_detection_rate=self.config['collapse']['min_detection_rate'],
            collapse_threshold=self.config['collapse']['threshold']
        )
        
        self.consistency_filter = ConsistencyFilter(
            n_augmentations=self.config['consistency']['n_augmentations'],
            min_consistency=self.config['consistency']['min_consistency'],
            iou_threshold=self.config['consistency']['iou_threshold']
        )
        
        # Synthetic generator
        self.synthetic_generator = None
        self._init_synthetic_generator()
        
        # Model
        self.model = None
        
        # Training state
        self.iteration = 0
        self.synthetic_ratio = self.config['synthetic']['initial_ratio']
        
        # Paths
        self.real_labeled_dir = self.output_dir / "real_labeled"
        self.real_labeled_dir.mkdir(exist_ok=True)
        (self.real_labeled_dir / "images").mkdir(exist_ok=True)
        (self.real_labeled_dir / "labels").mkdir(exist_ok=True)
        
        logger.info("IterativeSemiTrainer initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set defaults
        defaults = {
            'max_iterations': 10,
            'epochs_per_iter': 10,
            'batch_size': 16,
            'imgsz': 640,
            'synthetic': {
                'initial_ratio': 0.8,
                'min_ratio': 0.2,
                'decay_rate': 0.15,
                'num_images_per_iter': 1000,
            },
            'consistency': {
                'n_augmentations': 4,
                'min_consistency': 0.67,
                'iou_threshold': 0.5,
            },
            'collapse': {
                'min_detection_rate': 0.05,
                'threshold': 0.5,
            },
            'pseudo_label_threshold': 0.75,
        }
        
        # Merge defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k not in config[key]:
                        config[key][k] = v
        
        return config
    
    def _init_synthetic_generator(self):
        """Initialize synthetic data generator."""
        synth_config = self.config.get('synthetic_generator', {})
        
        if 'config' in synth_config and 'obb_weights' in synth_config:
            try:
                self.synthetic_generator = SyntheticGenerator(
                    synth_config['config'],
                    synth_config['obb_weights']
                )
                logger.info("Synthetic generator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize synthetic generator: {e}")
                self.synthetic_generator = None
    
    def _init_model(self):
        """Initialize YOLO model."""
        model_path = self.config.get('model', {}).get('weights', 'yolo11n.pt')
        self.model = YOLO(model_path)
        logger.info(f"Model initialized from: {model_path}")
    
    def _compute_synthetic_count(self) -> int:
        """Compute number of synthetic images needed based on current ratio."""
        real_count = len(list((self.real_labeled_dir / "images").glob("*.jpg")))
        
        if real_count == 0:
            return self.config['synthetic']['num_images_per_iter']
        
        # Compute to achieve desired ratio
        # synthetic / (synthetic + real) = ratio
        # synthetic = ratio * real / (1 - ratio)
        target = int(self.synthetic_ratio * real_count / (1 - self.synthetic_ratio))
        return max(target, self.config['synthetic']['num_images_per_iter'] // 2)
    
    def _generate_synthetic_data(self) -> Path:
        """Generate synthetic images for current iteration."""
        synthetic_dir = self.output_dir / f"synthetic_iter_{self.iteration}"
        
        if self.synthetic_generator is None:
            logger.warning("Synthetic generator not available, skipping generation")
            synthetic_dir.mkdir(parents=True, exist_ok=True)
            return synthetic_dir
        
        n_images = self._compute_synthetic_count()
        logger.info(f"Generating {n_images} synthetic images (ratio={self.synthetic_ratio:.2f})")
        
        # Update output directory
        self.synthetic_generator.output_dir = synthetic_dir
        self.synthetic_generator.writer.images_dir = synthetic_dir / "images"
        self.synthetic_generator.writer.labels_dir = synthetic_dir / "labels"
        self.synthetic_generator.writer.masks_dir = synthetic_dir / "masks"
        
        for d in [synthetic_dir, synthetic_dir / "images", 
                  synthetic_dir / "labels", synthetic_dir / "masks"]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Generate
        class_dist = self.config.get('class_distribution', {
            'Hot-Spot': n_images // 6,
            'Broken-Cell': n_images // 6,
            'Shadow': n_images // 6,
            'Junction-Box-Heat': n_images // 6,
            'Debris-Cover': n_images // 6,
            'Substring-Open-Circuit': n_images // 6,
        })
        
        try:
            self.synthetic_generator.generate(
                num_images=n_images,
                class_distribution=class_dist
            )
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}")
        
        return synthetic_dir
    
    def _create_merged_dataset(self, synthetic_dir: Path) -> Path:
        """Merge synthetic and real labeled data."""
        merged_dir = self.output_dir / f"merged_iter_{self.iteration}"
        merged_images = merged_dir / "images"
        merged_labels = merged_dir / "labels"
        
        # Clean old merged data
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        
        merged_images.mkdir(parents=True, exist_ok=True)
        merged_labels.mkdir(parents=True, exist_ok=True)
        
        count_synthetic = 0
        count_real = 0
        
        # Copy synthetic data
        synth_images = synthetic_dir / "images"
        synth_labels = synthetic_dir / "labels"
        
        if synth_images.exists():
            for img_file in synth_images.glob("*.jpg"):
                label_file = synth_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(img_file, merged_images / f"synth_{img_file.name}")
                    shutil.copy2(label_file, merged_labels / f"synth_{label_file.name}")
                    count_synthetic += 1
        
        # Copy real labeled data
        real_images = self.real_labeled_dir / "images"
        real_labels = self.real_labeled_dir / "labels"
        
        for img_file in real_images.glob("*.jpg"):
            label_file = real_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(img_file, merged_images / f"real_{img_file.name}")
                shutil.copy2(label_file, merged_labels / f"real_{label_file.name}")
                count_real += 1
        
        logger.info(f"Merged dataset: {count_synthetic} synthetic + {count_real} real = {count_synthetic + count_real} total")
        
        return merged_dir
    
    def _create_data_yaml(self, train_dir: Path) -> str:
        """Create YOLO data.yaml for training."""
        class_names = self.config.get('class_names', {
            0: 'Hot-Spot',
            1: 'Broken-Cell',
            2: 'Shadow',
            3: 'Junction-Box-Heat',
            4: 'Debris-Cover',
            5: 'Substring-Open-Circuit',
        })
        
        # Get val path - fallback to train if not exists
        val_path = self.config.get('val_images', None)
        
        if val_path is None or not Path(val_path).exists():
            # Use train as val (not ideal but allows training without separate val)
            val_path = str(train_dir / "images")
            logger.warning(f"No validation set found, using train data for validation")
        else:
            val_path = str(Path(val_path).resolve())
        
        data_yaml = {
            'train': str((train_dir / "images").resolve()),
            'val': val_path,
            'names': class_names
        }
        
        yaml_path = self.output_dir / f"data_iter_{self.iteration}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        return str(yaml_path)

    
    def _train_model(self, data_yaml: str):
        """Train model for one iteration."""
        logger.info(f"Training model (iteration {self.iteration})...")
        
        self.model.train(
            data=data_yaml,
            epochs=self.config['epochs_per_iter'],
            batch=self.config['batch_size'],
            imgsz=self.config['imgsz'],
            device=0,
            exist_ok=True,
            project=str(self.output_dir),
            name=f"train_iter_{self.iteration}",
        )
        
        # Load best weights
        best_weights = self.output_dir / f"train_iter_{self.iteration}" / "weights" / "best.pt"
        if best_weights.exists():
            self.model = YOLO(str(best_weights))
            logger.info(f"Loaded best weights from: {best_weights}")
    
    def _predict_unlabeled(self) -> Tuple[int, int]:
        """
        Predict on unlabeled images with consistency filter.
        
        Returns:
            (n_labeled, n_total_unlabeled)
        """
        unlabeled_dir = Path(self.config['unlabeled_images'])
        threshold = self.config['pseudo_label_threshold']
        
        # Load previously labeled images to skip
        labeled_tracking_file = self.output_dir / "labeled_images.txt"
        labeled_set = set()
        if labeled_tracking_file.exists():
            with open(labeled_tracking_file, 'r') as f:
                labeled_set = set(line.strip() for line in f)
        
        image_files = list(unlabeled_dir.glob("*.jpg")) + \
                     list(unlabeled_dir.glob("*.png")) + \
                     list(unlabeled_dir.glob("*.JPG"))
        
        # Filter out already labeled
        image_files = [f for f in image_files if f.name not in labeled_set]
        
        logger.info(f"Predicting on {len(image_files)} unlabeled images...")
        
        n_labeled = 0
        newly_labeled = []
        
        for img_path in tqdm(image_files, desc="Pseudo-labeling"):
            # Get consistent predictions
            boxes = self.consistency_filter.filter(
                self.model, 
                str(img_path),
                conf_threshold=threshold
            )
            
            if not boxes:
                continue
            
            # Filter by confidence
            high_conf_boxes = [b for b in boxes if b['conf'] >= threshold]
            
            if not high_conf_boxes:
                continue
            
            # Save to real_labeled
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            labels = []
            for box in high_conf_boxes:
                xyxy = box['xyxy']
                x_center = (xyxy[0] + xyxy[2]) / 2 / w
                y_center = (xyxy[1] + xyxy[3]) / 2 / h
                width = (xyxy[2] - xyxy[0]) / w
                height = (xyxy[3] - xyxy[1]) / h
                labels.append(f"{box['cls']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Copy image and save labels
            new_name = f"iter{self.iteration}_{img_path.name}"
            shutil.copy2(img_path, self.real_labeled_dir / "images" / new_name)
            
            label_file = self.real_labeled_dir / "labels" / f"{Path(new_name).stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(labels))
            
            # Track labeled image (instead of deleting)
            newly_labeled.append(img_path.name)
            
            n_labeled += 1
        
        # Update tracking file
        with open(labeled_tracking_file, 'a') as f:
            for name in newly_labeled:
                f.write(f"{name}\n")
        
        return n_labeled, len(image_files)
    
    def _update_synthetic_ratio(self, collapse_detected: bool):
        """Update synthetic ratio based on training progress."""
        if collapse_detected:
            # Increase synthetic ratio on collapse
            self.synthetic_ratio = min(
                self.synthetic_ratio + 0.2,
                self.config['synthetic']['initial_ratio']
            )
            logger.warning(f"Collapse detected, increasing synthetic ratio to {self.synthetic_ratio:.2f}")
        else:
            # Normal decay
            self.synthetic_ratio = max(
                self.config['synthetic']['min_ratio'],
                self.synthetic_ratio - self.config['synthetic']['decay_rate']
            )
    
    def run_iteration(self) -> bool:
        """
        Run one training iteration.
        
        Returns:
            True if iteration completed successfully
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {self.iteration}")
        logger.info(f"{'='*60}")
        
        # Step 1: Generate synthetic data
        synthetic_dir = self._generate_synthetic_data()
        
        # Step 2: Merge datasets
        merged_dir = self._create_merged_dataset(synthetic_dir)
        
        # Step 3: Create data.yaml
        data_yaml = self._create_data_yaml(merged_dir)
        
        # Step 4: Train model
        self._train_model(data_yaml)
        
        # Step 5: Predict on unlabeled
        n_labeled, n_total = self._predict_unlabeled()
        
        # Step 6: Check distribution
        detection_rate = n_labeled / max(n_total, 1)
        is_ok, message = self.distribution_guard.check(detection_rate)
        logger.info(f"Distribution check: {message} (rate={detection_rate:.3f})")
        
        # Step 7: Update synthetic ratio
        self._update_synthetic_ratio(not is_ok)
        
        # Step 8: Cleanup
        if self.config.get('cleanup', True):
            if synthetic_dir.exists():
                shutil.rmtree(synthetic_dir)
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
        
        logger.info(f"Iteration {self.iteration} complete: {n_labeled} new labels added")
        
        self.iteration += 1
        return is_ok
    
    def train(self, max_iterations: Optional[int] = None):
        """
        Run full training loop.
        
        Args:
            max_iterations: Override max iterations from config
        """
        if max_iterations is None:
            max_iterations = self.config['max_iterations']
        
        logger.info("="*60)
        logger.info("ITERATIVE SEMI-SUPERVISED TRAINING")
        logger.info("="*60)
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Initial synthetic ratio: {self.synthetic_ratio}")
        
        # Initialize model
        self._init_model()
        
        # Training loop
        for i in range(max_iterations):
            success = self.run_iteration()
            
            # Check if unlabeled is exhausted
            unlabeled_dir = Path(self.config['unlabeled_images'])
            remaining = len(list(unlabeled_dir.glob("*.jpg")))
            
            if remaining == 0:
                logger.info("All unlabeled images have been labeled!")
                break
            
            logger.info(f"Remaining unlabeled: {remaining}")
        
        # Save final model
        final_model = self.output_dir / "final_model.pt"
        if self.model:
            self.model.save(str(final_model))
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Final model saved to: {final_model}")
        logger.info(f"Distribution stats: {self.distribution_guard.get_stats()}")
        logger.info("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Iterative Semi-Supervised Training')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--iterations', '-n', type=int, default=None,
                       help='Max iterations (override config)')
    
    args = parser.parse_args()
    
    trainer = IterativeSemiTrainer(args.config)
    trainer.train(max_iterations=args.iterations)


if __name__ == '__main__':
    main()
