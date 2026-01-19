"""
Semi-YOLOv11: Semi-Supervised Object Detection Training

Usage:
    python main.py --config config/default.yaml
    python main.py --model yolo11n.pt --labeled ./data/labeled --unlabeled ./data/unlabeled
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.config_utils import get_config, SemiConfig
from trainer.semi_trainer import SemiTrainer
from filters.dsat import DSATFilter
from filters.dfl_entropy import DFLEntropyFilter
from filters.tal_alignment import TALAlignmentFilter


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-YOLOv11 Training')

    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path (overrides config)')
    parser.add_argument('--labeled', type=str, default=None,
                        help='Labeled data path')
    parser.add_argument('--unlabeled', type=str, default=None,
                        help='Unlabeled data path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Image size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--project', type=str, default=None,
                        help='Project directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--burn-in', type=int, default=None,
                        help='Burn-in epochs')
    parser.add_argument('--lambda-unsup', type=float, default=None,
                        help='Unsupervised loss weight')
    parser.add_argument('--ema-decay', type=float, default=None,
                        help='EMA decay rate')

    return parser.parse_args()


def build_overrides(args) -> dict:
    """Build overrides dict from CLI arguments."""
    overrides = {}
    
    if args.model:
        overrides['model'] = args.model
    if args.epochs:
        overrides['epochs'] = args.epochs
    if args.batch:
        overrides['batch'] = args.batch
    if args.imgsz:
        overrides['imgsz'] = args.imgsz
    if args.device:
        overrides['device'] = args.device
    if args.project:
        overrides['project'] = args.project
    if args.name:
        overrides['name'] = args.name
    
    semi_overrides = {}
    if args.labeled:
        semi_overrides['labeled_path'] = args.labeled
    if args.unlabeled:
        semi_overrides['unlabeled_path'] = args.unlabeled
    if args.burn_in:
        semi_overrides['burn_in'] = args.burn_in
    if args.lambda_unsup:
        semi_overrides['lambda_unsup'] = args.lambda_unsup
    if args.ema_decay:
        semi_overrides['ema_decay'] = args.ema_decay
    
    if semi_overrides:
        overrides['semi'] = semi_overrides
    
    return overrides


def main():
    args = parse_args()

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    
    overrides = build_overrides(args)
    cfg = get_config(str(config_path) if config_path.exists() else None, overrides)
    
    print("\n" + "="*50)
    print("Semi-YOLOv11 Configuration")
    print("="*50)
    print(f"Model: {cfg['model']}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch: {cfg['batch']}")
    print(f"Image Size: {cfg['imgsz']}")
    print(f"Labeled Path: {cfg['semi'].get('labeled_path', 'N/A')}")
    print(f"Unlabeled Path: {cfg['semi'].get('unlabeled_path', 'N/A')}")
    print(f"Burn-in Epochs: {cfg['semi']['burn_in']}")
    print(f"Lambda Unsup: {cfg['semi']['lambda_unsup']}")
    print(f"EMA Decay: {cfg['semi']['ema_decay']}")
    print(f"Filters: {[f['name'] for f in cfg['semi']['filters']]}")
    print("="*50 + "\n")

    trainer_overrides = {
        'model': cfg['model'],
        'data': cfg.get('data', {}),
        'epochs': cfg['epochs'],
        'batch': cfg['batch'],
        'imgsz': cfg['imgsz'],
        'device': cfg.get('device', '0'),
        'project': cfg.get('project', 'runs/semi'),
        'name': cfg.get('name', 'exp'),
        'semi': cfg['semi'],
    }

    trainer = SemiTrainer(overrides=trainer_overrides)
    trainer.train()

    print(f"\nTraining complete. Results saved to {trainer.save_dir}")


if __name__ == '__main__':
    main()
