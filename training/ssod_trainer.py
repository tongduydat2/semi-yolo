"""
SSOD Training Loop - Core implementation
Main entry point for Semi-Supervised Object Detection training
"""

import torch
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import os
import sys
import glob
import shutil

# Add parent directory to path for imports when running directly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanupManager:
    """
    Manage disk space by cleaning up temporary files after each training cycle.
    Prevents disk space from growing unbounded during long training runs.
    """
    
    def __init__(self, output_dir: str, keep_best_checkpoints: int = 3):
        """
        Args:
            output_dir: Training output directory
            keep_best_checkpoints: Number of best checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.keep_best = keep_best_checkpoints
        
    def cleanup_pseudo_labels(self):
        """Remove pseudo-labels from previous epoch."""
        pseudo_dir = self.output_dir / "pseudo_labels"
        if pseudo_dir.exists():
            shutil.rmtree(pseudo_dir)
            logger.info(f"Cleaned up pseudo-labels: {pseudo_dir}")
    
    def cleanup_old_checkpoints(self):
        """Keep only the best N checkpoints, remove older ones."""
        checkpoints = sorted(
            self.output_dir.glob("ssod_checkpoint_epoch*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) > self.keep_best:
            for ckpt in checkpoints[:-self.keep_best]:
                ckpt.unlink()
                logger.info(f"Removed old checkpoint: {ckpt.name}")
                
        # Also clean old student models
        student_models = sorted(
            self.output_dir.glob("student_epoch*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        if len(student_models) > self.keep_best:
            for model in student_models[:-self.keep_best]:
                model.unlink()
                logger.info(f"Removed old student model: {model.name}")
    
    def cleanup_temp_yaml(self):
        """Remove temporary data.yaml files."""
        for yaml_file in self.output_dir.glob("*_data.yaml"):
            yaml_file.unlink()
    
    def cleanup_training_runs(self):
        """Remove YOLO training run directories (logs, weights, etc.)."""
        patterns = ["supervised_epoch*", "unsupervised_epoch*"]
        for pattern in patterns:
            for run_dir in self.output_dir.glob(pattern):
                if run_dir.is_dir():
                    shutil.rmtree(run_dir)
                    logger.info(f"Removed training run: {run_dir.name}")
                
    def run_epoch_cleanup(self):
        """Run cleanup operations after each epoch."""
        self.cleanup_pseudo_labels()
        self.cleanup_temp_yaml()
        self.cleanup_training_runs()
        
    def run_full_cleanup(self):
        """Run all cleanup operations including checkpoint pruning."""
        self.run_epoch_cleanup()
        self.cleanup_old_checkpoints()
        
    def get_disk_usage(self) -> Dict[str, float]:
        """Get current disk usage of output directory in MB."""
        total_size = 0
        for f in self.output_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        return {
            'total_mb': total_size / (1024 * 1024),
            'path': str(self.output_dir)
        }


class SSODTrainer:
    """
    Semi-Supervised Object Detection Trainer.
    
    Implements Mean Teacher framework:
    - Burn-in phase: Supervised only (Teacher learns basic features)
    - SSOD phase: Supervised + Pseudo-labels from Teacher
    - EMA updates: Smooth Teacher updates from Student
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer from config file.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config = self._load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_epoch = 0
        
        logger.info(f"Initializing SSOD Trainer on device: {self.device}")
        
        self._init_framework()
        self._init_pseudo_labeler()
        self._init_loss()
        self._setup_output_dir()
        self._init_cleanup_manager()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    
    def _init_framework(self):
        """Initialize Teacher-Student framework."""
        from models.teacher_student import TeacherStudentFramework
        
        model_cfg = self.config['model']
        ssod_cfg = self.config['ssod']
        
        self.framework = TeacherStudentFramework(
            model_path=model_cfg['base_model'],
            ema_decay=ssod_cfg['ema_rate'],
            device=self.device,
            num_classes=model_cfg['num_classes']
        )
        logger.info("Teacher-Student framework initialized")
        
    def _init_pseudo_labeler(self):
        """Initialize pseudo-label generator."""
        from training.pseudo_labeler import PseudoLabeler, NMSRefiner
        
        ssod_cfg = self.config['ssod']
        
        self.pseudo_labeler = PseudoLabeler(
            base_threshold=ssod_cfg['confidence_threshold'],
            adaptive=ssod_cfg['adaptive_threshold'],
            tau_min=ssod_cfg['tau_min'],
            tau_max=ssod_cfg['tau_max']
        )
        self.nms_refiner = NMSRefiner(iou_threshold=ssod_cfg['nms_threshold'])
        logger.info("Pseudo-labeler initialized")
        
    def _init_loss(self):
        """Initialize loss calculator."""
        from training.losses import SSODLoss
        
        train_cfg = self.config['training']
        ssod_cfg = self.config['ssod']
        
        self.loss_calculator = SSODLoss(
            lambda_unsup=ssod_cfg['unsupervised_weight'],
            burn_in_epochs=train_cfg['burn_in_epochs']
        )
        
    def _setup_output_dir(self):
        """Setup output directory for checkpoints."""
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
    def _init_cleanup_manager(self):
        """Initialize cleanup manager for disk space management."""
        cleanup_cfg = self.config.get('cleanup', {})
        self.cleanup_enabled = cleanup_cfg.get('enabled', True)
        keep_best = cleanup_cfg.get('keep_best_checkpoints', 3)
        
        self.cleanup_manager = CleanupManager(
            output_dir=str(self.output_dir),
            keep_best_checkpoints=keep_best
        )
        
        if self.cleanup_enabled:
            logger.info(f"Cleanup enabled: keeping {keep_best} best checkpoints")
        
        # Check if we should skip colorization and use original labeled data
        data_cfg = self.config.get('data', {})
        self.skip_colorization = data_cfg.get('skip_colorization', False)
        
        if self.skip_colorization:
            logger.info("Skip colorization enabled: using original labeled data directly")
        
        # Progressive threshold settings
        ssod_cfg = self.config.get('ssod', {})
        self.progressive_threshold = ssod_cfg.get('progressive_threshold', True)
        self.tau_start = ssod_cfg.get('tau_start', 0.5)
        self.tau_end = ssod_cfg.get('tau_end', 0.75)
        
        if self.progressive_threshold:
            logger.info(f"Progressive threshold enabled: {self.tau_start} -> {self.tau_end}")
    
    def _get_progressive_threshold(self, epoch: int) -> float:
        """
        Calculate confidence threshold that increases over training.
        Starts low (more pseudo-labels) and increases as Teacher improves.
        """
        train_cfg = self.config['training']
        burn_in_epochs = train_cfg['burn_in_epochs']
        max_epochs = train_cfg['max_epochs']
        
        if epoch < burn_in_epochs:
            return self.tau_end  # Not used during burn-in anyway
        
        # Progress from 0 to 1 over SSOD phase
        ssod_epochs = max_epochs - burn_in_epochs
        current_ssod_epoch = epoch - burn_in_epochs
        progress = min(current_ssod_epoch / max(ssod_epochs, 1), 1.0)
        
        # Linear interpolation from tau_start to tau_end
        threshold = self.tau_start + (self.tau_end - self.tau_start) * progress
        
        return threshold
    
    def train(self):
        """
        Main training loop implementing SSOD algorithm.
        """
        train_cfg = self.config['training']
        data_cfg = self.config['data']
        
        max_epochs = train_cfg['max_epochs']
        burn_in_epochs = train_cfg['burn_in_epochs']
        
        logger.info(f"=" * 50)
        logger.info(f"Starting SSOD Training")
        logger.info(f"Max epochs: {max_epochs}")
        logger.info(f"Burn-in epochs: {burn_in_epochs} (supervised only)")
        logger.info(f"=" * 50)
        
        # Get data paths
        labeled_images = data_cfg['fake_ironred']['images']
        unlabeled_images = data_cfg['unlabeled']['images']
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Determine phase
            is_burnin = epoch < burn_in_epochs
            phase = "BURN-IN (Supervised)" if is_burnin else "SSOD (Semi-Supervised)"
            current_lambda = self.loss_calculator.get_current_lambda(epoch)
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{max_epochs} - Phase: {phase}")
            logger.info(f"Lambda (unsup weight): {current_lambda}")
            logger.info(f"{'='*50}")
            
            # === PHASE 1: Train on Labeled Data (Supervised) ===
            logger.info("Phase 1: Training on labeled data...")
            self._train_supervised(epoch)
            
            # === PHASE 2: Generate Pseudo-labels & Train (if not burn-in) ===
            if not is_burnin:
                # Get progressive threshold for current epoch
                current_threshold = self._get_progressive_threshold(epoch)
                logger.info(f"Phase 2: Generating pseudo-labels (threshold={current_threshold:.3f})...")
                
                pseudo_stats = self._generate_pseudo_labels(unlabeled_images, current_threshold)
                logger.info(f"Pseudo-label stats: {pseudo_stats}")
                
                # Only train on pseudo-labels if we have any
                if pseudo_stats.get('with_labels', 0) > 0:
                    logger.info("Phase 3: Training on pseudo-labeled data...")
                    self._train_unsupervised(epoch)
                else:
                    logger.warning("Phase 3: Skipped - no valid pseudo-labels generated")
            
            # === Update Teacher via EMA ===
            self.framework.update_teacher()
            logger.info("Teacher updated via EMA")
            
            # === Logging ===
            avg_losses = self.loss_calculator.get_average_losses(window=50)
            logger.info(f"Avg Losses - Sup: {avg_losses['supervised']:.4f}, "
                       f"Unsup: {avg_losses['unsupervised']:.4f}, "
                       f"Total: {avg_losses['total']:.4f}")
            
            # === Save Checkpoint ===
            if (epoch + 1) % self.config['evaluation']['save_interval'] == 0:
                self._save_checkpoint(epoch)
            
            # === Cleanup to save disk space ===
            if self.cleanup_enabled:
                self.cleanup_manager.run_epoch_cleanup()
                self.cleanup_manager.cleanup_old_checkpoints()
                usage = self.cleanup_manager.get_disk_usage()
                logger.info(f"Disk usage: {usage['total_mb']:.2f} MB")
        
        logger.info("Training completed!")
        self._save_checkpoint(max_epochs - 1, final=True)
        
    def _train_supervised(self, epoch: int):
        """Train Student on labeled data."""
        data_cfg = self.config['data']
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        
        run_name = f"supervised_epoch{epoch}"
        
        # Use YOLO's built-in training
        self.framework.get_student().train(
            data=self._create_data_yaml('labeled'),
            epochs=1,
            batch=train_cfg['batch_size'],
            imgsz=model_cfg['imgsz'],
            device=self.device,
            verbose=False,
            exist_ok=True,
            project=str(self.output_dir),
            name=run_name
        )
        
        # Load trained weights back into Student for next epoch
        trained_weights = self.output_dir / run_name / "weights" / "last.pt"
        if trained_weights.exists():
            self.framework.update_student_weights(str(trained_weights))
            logger.info(f"Loaded trained weights from: {trained_weights}")
        else:
            logger.warning(f"Trained weights not found: {trained_weights}")
        
    def _train_unsupervised(self, epoch: int):
        """Train Student on pseudo-labeled data."""
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        
        run_name = f"unsupervised_epoch{epoch}"
        
        # Train on pseudo-labeled data
        self.framework.get_student().train(
            data=self._create_data_yaml('pseudo'),
            epochs=1,
            batch=train_cfg['batch_size'],
            imgsz=model_cfg['imgsz'],
            device=self.device,
            verbose=False,
            exist_ok=True,
            project=str(self.output_dir),
            name=run_name
        )
        
        # Load trained weights back into Student
        trained_weights = self.output_dir / run_name / "weights" / "last.pt"
        if trained_weights.exists():
            self.framework.update_student_weights(str(trained_weights))
            logger.info(f"Loaded trained weights from: {trained_weights}")
        else:
            logger.warning(f"Trained weights not found: {trained_weights}")
        
    def _generate_pseudo_labels(self, unlabeled_dir: str, threshold: float = None) -> Dict:
        """Generate pseudo-labels using Teacher model with batch processing."""
        from training.pseudo_labeler import save_pseudo_labels
        import torch
        
        # Use provided threshold or default
        if threshold is None:
            threshold = self.pseudo_labeler.threshold
        
        # Get unlabeled images
        image_dir = Path(unlabeled_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_paths:
            logger.warning(f"No images found in {unlabeled_dir}")
            return {'total': 0, 'with_labels': 0}
        
        # Process in batches to avoid OOM
        BATCH_SIZE = 8
        all_predictions = []
        
        logger.info(f"Generating pseudo-labels for {len(image_paths)} images (batch_size={BATCH_SIZE})...")
        
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = [str(p) for p in image_paths[i:i+BATCH_SIZE]]
            
            try:
                batch_preds = self.framework.generate_pseudo_labels(
                    images=batch_paths,
                    confidence_threshold=threshold
                )
                all_predictions.extend(batch_preds)
            except Exception as e:
                logger.warning(f"Error processing batch {i//BATCH_SIZE}: {e}")
                # Add empty predictions for failed batch
                all_predictions.extend([{'boxes_yolo': [], 'classes': [], 'confidences': []} for _ in batch_paths])
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
        
        # Update pseudo_labeler threshold for this epoch
        self.pseudo_labeler.threshold = threshold
        
        # Filter and refine
        filtered, stats = self.pseudo_labeler.filter_predictions(all_predictions)
        refined = [self.nms_refiner.refine(l) for l in filtered]
        
        # Count images with valid labels
        images_with_labels = sum(1 for r in refined if len(r.get('boxes_yolo', [])) > 0)
        stats['with_labels'] = images_with_labels
        stats['threshold'] = threshold
        
        # Save pseudo-labels
        pseudo_label_dir = self.output_dir / "pseudo_labels"
        save_pseudo_labels(refined, [str(p) for p in image_paths], str(pseudo_label_dir))
        
        return stats
    
    def _create_data_yaml(self, mode: str) -> str:
        """Create temporary data.yaml for YOLO training."""
        data_cfg = self.config['data']
        model_cfg = self.config['model']
        
        if mode == 'labeled':
            # Use original labeled data if skip_colorization, otherwise use fake_ironred
            if self.skip_colorization:
                train_path = data_cfg['labeled']['images']
                logger.info("Using original labeled data (skip_colorization=true)")
            else:
                train_path = data_cfg['fake_ironred']['images']
                logger.info("Using fake IronRed data")
        elif mode == 'pseudo':
            # For pseudo-labeled data:
            # Images are from unlabeled, labels are from pseudo_labels directory
            # We need to create a proper YOLO structure
            
            import shutil
            
            pseudo_label_dir = self.output_dir / "pseudo_labels"
            unlabeled_images_path = Path(data_cfg['unlabeled']['images'])
            
            # Create pseudo_train directory with proper structure
            pseudo_train_dir = self.output_dir / "pseudo_train"
            pseudo_images_dir = pseudo_train_dir / "images"
            pseudo_labels_dir = pseudo_train_dir / "labels"
            
            # AUTO-CLEANUP: Remove old pseudo_train to avoid stale symlinks/files
            if pseudo_train_dir.exists():
                shutil.rmtree(pseudo_train_dir)
                logger.info(f"Cleaned up old pseudo_train directory")
            
            # AUTO-CLEANUP: Remove cache files in datasets dir
            unlabeled_parent = unlabeled_images_path.parent
            for cache_file in unlabeled_parent.glob("*.cache"):
                cache_file.unlink()
                logger.info(f"Deleted dataset cache: {cache_file}")
            
            pseudo_images_dir.mkdir(parents=True, exist_ok=True)
            pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy/link images that have pseudo-labels
            image_count = 0
            for label_file in pseudo_label_dir.glob("*.txt"):
                # Find corresponding image
                img_name = label_file.stem
                
                # Try common image extensions
                src_img = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    candidate = unlabeled_images_path / f"{img_name}{ext}"
                    if candidate.exists():
                        src_img = candidate
                        break
                
                if src_img:
                    dst_img = pseudo_images_dir / src_img.name
                    dst_label = pseudo_labels_dir / label_file.name
                    
                    # Create symlink or copy (symlink first, fallback to copy)
                    try:
                        if not dst_img.exists():
                            dst_img.symlink_to(src_img.resolve())
                        if not dst_label.exists():
                            dst_label.symlink_to(label_file.resolve())
                    except Exception:
                        # Fallback to copy
                        if not dst_img.exists():
                            shutil.copy2(src_img, dst_img)
                        if not dst_label.exists():
                            shutil.copy2(label_file, dst_label)
                    
                    image_count += 1
            
            logger.info(f"Prepared {image_count} pseudo-labeled images in {pseudo_train_dir}")
            train_path = str(pseudo_images_dir.resolve())
            logger.info(f"Using pseudo-labeled data: {train_path}")
            
            # Delete old cache files to force YOLO to rescan
            for cache_file in pseudo_train_dir.rglob("*.cache"):
                cache_file.unlink()
                logger.info(f"Deleted cache: {cache_file}")
            
            # Also delete cache in unlabeled dir to prevent confusion
            unlabeled_cache = Path(data_cfg['unlabeled']['images']).parent / "labels.cache"
            if unlabeled_cache.exists():
                unlabeled_cache.unlink()
                logger.info(f"Deleted unlabeled cache: {unlabeled_cache}")
            unlabeled_cache2 = Path(data_cfg['unlabeled']['images']).with_suffix('.cache')
            if unlabeled_cache2.exists():
                unlabeled_cache2.unlink()
                logger.info(f"Deleted unlabeled cache: {unlabeled_cache2}")
        else:
            train_path = data_cfg['unlabeled']['images']
        
        # Get class names from config, or auto-generate if not defined
        if 'class_names' in model_cfg:
            class_names = model_cfg['class_names']
        else:
            # Fallback to auto-generated names
            class_names = {i: f'class_{i}' for i in range(model_cfg['num_classes'])}
        
        # Use absolute paths to avoid confusion
        val_path = str(Path(data_cfg['val']['images']).resolve()) if Path(data_cfg['val']['images']).exists() else data_cfg['val']['images']
        
        data_yaml = {
            'path': '.',
            'train': train_path,  # Use train_path consistently
            'val': val_path,
            'names': class_names
        }
        
        logger.info(f"Data YAML for {mode}: train={data_yaml['train']}")
        logger.info(f"Labels expected at: {str(Path(data_yaml['train']).parent / 'labels')}")
        
        yaml_path = self.output_dir / f"{mode}_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        return str(yaml_path)
    
    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"epoch{epoch + 1}"
        ckpt_path = self.output_dir / f"ssod_checkpoint_{suffix}.pt"
        
        self.framework.save_checkpoint(
            path=str(ckpt_path),
            epoch=epoch,
            extra_info={
                'pseudo_stats': self.pseudo_labeler.get_statistics(),
                'loss_history': self.loss_calculator.get_average_losses()
            }
        )
        
        # Also save Student as standalone YOLO model
        student_path = self.output_dir / f"student_{suffix}.pt"
        self.framework.get_student().save(str(student_path))
        logger.info(f"Student model saved to {student_path}")
        
    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        self.current_epoch = self.framework.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from epoch {self.current_epoch}")


def create_trainer(config_path: str) -> SSODTrainer:
    """Factory function to create SSOD trainer."""
    return SSODTrainer(config_path)
