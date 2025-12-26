"""
SSOD Trainer V2 - Simplified version using SSODDetectionTrainer.
Uses persistent trainer instance with EMA updates every batch.
"""

import yaml
import logging
import shutil
from pathlib import Path

from training.ssod_detection_trainer import SSODDetectionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SSODTrainerV2:
    """
    Simplified SSOD Trainer using SSODDetectionTrainer.
    
    Key Features:
    - Single trainer instance (persistent)
    - EMA updates every batch (automatic in trainer)
    - Supports labeled + pseudo-labeled training
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to YAML config file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trainer instance (created once, reused)
        self.trainer = None
        self.device = 0  # CUDA device
        
        logger.info(f"SSOD Trainer V2 initialized")
        logger.info(f"Output directory: {self.output_dir}")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_data_yaml(self, mode: str) -> str:
        """Create temporary data.yaml for YOLO training."""
        data_cfg = self.config['data']
        model_cfg = self.config['model']
        
        if mode == 'labeled':
            if data_cfg.get('skip_colorization', False):
                train_path = data_cfg['labeled']['images']
            else:
                train_path = data_cfg['fake_ironred']['images']
        elif mode == 'pseudo':
            train_path = str(self.output_dir / "pseudo_train" / "images")
        else:
            train_path = data_cfg['unlabeled']['images']
        
        val_path = data_cfg['val']['images']
        
        # Class names
        if 'class_names' in model_cfg:
            class_names = model_cfg['class_names']
        else:
            class_names = {i: f'class_{i}' for i in range(model_cfg['num_classes'])}
        
        data_yaml = {
            'path': '.',
            'train': train_path,
            'val': val_path,
            'names': class_names
        }
        
        # Save to temp file
        yaml_path = self.output_dir / f"{mode}_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        logger.info(f"Data YAML for {mode}: train={train_path}")
        return str(yaml_path)
    
    def _create_trainer(self) -> SSODDetectionTrainer:
        """Create SSODDetectionTrainer instance."""
        model_cfg = self.config['model']
        train_cfg = self.config['training']
        
        overrides = {
            'model': model_cfg['base_model'],
            'data': self._create_data_yaml('labeled'),
            'epochs': 1,  # Train 1 epoch at a time
            'batch': train_cfg['batch_size'],
            'imgsz': model_cfg['imgsz'],
            'device': self.device,
            'exist_ok': True,
            'project': str(self.output_dir),
            'name': 'ssod_run',
        }
        
        trainer = SSODDetectionTrainer(overrides=overrides)
        return trainer
    
    def _prepare_pseudo_dataset(self, pseudo_labels_dir: Path):
        """
        Prepare pseudo-labeled dataset for training.
        Copies images that have pseudo-labels to pseudo_train directory.
        """
        data_cfg = self.config['data']
        unlabeled_images_dir = Path(data_cfg['unlabeled']['images'])
        
        pseudo_train_dir = self.output_dir / "pseudo_train"
        pseudo_images_dir = pseudo_train_dir / "images"
        pseudo_labels_target = pseudo_train_dir / "labels"
        
        # Clean old data
        if pseudo_train_dir.exists():
            shutil.rmtree(pseudo_train_dir)
        
        pseudo_images_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels_target.mkdir(parents=True, exist_ok=True)
        
        # Copy images and labels
        copied = 0
        for label_file in pseudo_labels_dir.glob("*.txt"):
            img_name = label_file.stem
            
            # Find image
            for ext in ['.jpg', '.jpeg', '.png']:
                img_src = unlabeled_images_dir / f"{img_name}{ext}"
                if img_src.exists():
                    # Copy image and label
                    shutil.copy2(img_src, pseudo_images_dir / img_src.name)
                    shutil.copy2(label_file, pseudo_labels_target / label_file.name)
                    copied += 1
                    break
        
        logger.info(f"Prepared pseudo dataset: {copied} images")
        return copied
    
    def train(self):
        """Main training loop."""
        train_cfg = self.config['training']
        data_cfg = self.config['data']
        ssod_cfg = self.config.get('ssod', {})
        
        max_epochs = train_cfg['max_epochs']
        burn_in_epochs = train_cfg['burn_in_epochs']
        
        logger.info("=" * 50)
        logger.info("Starting SSOD Training V2 (Persistent Trainer)")
        logger.info(f"Max epochs: {max_epochs}")
        logger.info(f"Burn-in epochs: {burn_in_epochs}")
        logger.info("=" * 50)
        
        # === CREATE TRAINER ONCE ===
        logger.info("Creating trainer (will be reused for all epochs)...")
        self.trainer = self._create_trainer()
        
        # Setup model (includes Teacher initialization)
        self.trainer.setup_model()
        
        for epoch in range(max_epochs):
            is_burnin = epoch < burn_in_epochs
            phase = "BURN-IN" if is_burnin else "SSOD"
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{max_epochs} - Phase: {phase}")
            logger.info("=" * 50)
            
            # === PHASE 1: Supervised Training ===
            logger.info("Phase 1: Training on labeled data...")
            
            # Set dataloader to labeled data
            labeled_yaml = self._create_data_yaml('labeled')
            self.trainer.args.data = labeled_yaml
            
            # Train 1 epoch (EMA updates every batch automatically!)
            self.trainer.train()
            
            # === PHASE 2 & 3: Pseudo-labeling (if not burn-in) ===
            if not is_burnin:
                # Calculate progressive threshold
                progress = (epoch - burn_in_epochs) / max(max_epochs - burn_in_epochs, 1)
                tau_start = ssod_cfg.get('tau_start', 0.5)
                tau_end = ssod_cfg.get('tau_end', 0.75)
                threshold = tau_start + (tau_end - tau_start) * progress
                
                logger.info(f"Phase 2: Generating pseudo-labels (threshold={threshold:.3f})...")
                
                # Generate pseudo-labels with Teacher
                pseudo_labels_dir = self.output_dir / "pseudo_labels"
                pseudo_stats = self.trainer.generate_pseudo_labels(
                    images_dir=data_cfg['unlabeled']['images'],
                    output_dir=str(pseudo_labels_dir),
                    threshold=threshold
                )
                
                logger.info(f"Pseudo-label stats: {pseudo_stats}")
                
                if pseudo_stats['images_with_labels'] > 0:
                    logger.info("Phase 3: Training on pseudo-labeled data...")
                    
                    # Prepare pseudo dataset
                    self._prepare_pseudo_dataset(pseudo_labels_dir)
                    
                    # Set dataloader to pseudo data
                    pseudo_yaml = self._create_data_yaml('pseudo')
                    self.trainer.args.data = pseudo_yaml
                    
                    # Train 1 epoch on pseudo data
                    self.trainer.train()
                else:
                    logger.warning("Phase 3: Skipped - no pseudo-labels generated")
            
            # === Save Student model ===
            student_path = self.output_dir / f"student_epoch{epoch}.pt"
            self.trainer.model.save(str(student_path))
            logger.info(f"Student saved to: {student_path}")
            
            # === Cleanup pseudo-labels ===
            pseudo_labels_dir = self.output_dir / "pseudo_labels"
            if pseudo_labels_dir.exists():
                shutil.rmtree(pseudo_labels_dir)
        
        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info("=" * 50)


def main():
    """Entry point for SSOD training V2."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SSOD Training V2')
    parser.add_argument('--config', '-c', type=str, default='config/ssod_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    trainer = SSODTrainerV2(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
