"""
SSOD Training Entry Point
Semi-Supervised Object Detection with Gray-to-IronRed Domain Adaptation

Usage:
    python train_ssod.py --config config/ssod_config.yaml
    python train_ssod.py --colorize --input datasets/labeled --output datasets/fake_ironred
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description='SSOD Training for Gray-to-IronRed Domain Adaptation'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # === Train command ===
    train_parser = subparsers.add_parser('train', help='Run SSOD training')
    train_parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/ssod_config.yaml',
        help='Path to config file'
    )
    train_parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # === Colorize command ===
    colorize_parser = subparsers.add_parser('colorize', help='Convert Gray images to IronRed')
    colorize_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory with grayscale images'
    )
    colorize_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for colorized images'
    )
    colorize_parser.add_argument(
        '--labels', '-l',
        type=str,
        default=None,
        help='Labels directory (if separate from images)'
    )
    colorize_parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Config file for custom LUT settings'
    )
    
    # === Analyze command ===
    analyze_parser = subparsers.add_parser('analyze', help='Analyze IronRed spectrum')
    analyze_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Directory with IronRed images to analyze'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        type=str,
        default='spectrum_analysis.png',
        help='Output path for visualization'
    )
    analyze_parser.add_argument(
        '--samples', '-n',
        type=int,
        default=100,
        help='Number of images to sample'
    )
    
    # === Evaluate command ===
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    
    # === Iterative Semi-Training command ===
    iterative_parser = subparsers.add_parser('iterative', 
        help='Run iterative semi-training with PVF-10 synthetic data')
    iterative_parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/iterative_config.yaml',
        help='Path to iterative training config file'
    )
    iterative_parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=None,
        help='Max iterations (override config)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training(args)
    elif args.command == 'iterative':
        run_iterative_training(args)
    elif args.command == 'colorize':
        run_colorization(args)
    elif args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    else:
        parser.print_help()


def run_training(args):
    """Run SSOD training."""
    from training.ssod_trainer import SSODTrainer
    
    print("=" * 60)
    print("SSOD Training - Semi-Supervised Object Detection")
    print("=" * 60)
    
    trainer = SSODTrainer(args.config)
    
    if args.resume:
        trainer.resume(args.resume)
    
    trainer.train()


def run_iterative_training(args):
    """Run iterative semi-training with PVF-10 synthetic data."""
    from training.iterative_semi_trainer import IterativeSemiTrainer
    
    print("=" * 60)
    print("ITERATIVE SEMI-TRAINING")
    print("With PVF-10 Synthetic Data Anchor")
    print("=" * 60)
    
    trainer = IterativeSemiTrainer(args.config)
    trainer.train(max_iterations=args.iterations)


def run_colorization(args):
    """Run colorization pipeline."""
    from data.colorization import IronRedColorizer
    
    print("=" * 60)
    print("Colorization Pipeline - Gray to IronRed")
    print("=" * 60)
    
    if args.config:
        colorizer = IronRedColorizer.from_config_file(args.config)
    else:
        colorizer = IronRedColorizer()
    
    # Visualize LUT first
    lut_path = Path(args.output) / "lut_visualization.png"
    Path(args.output).mkdir(parents=True, exist_ok=True)
    colorizer.visualize_lut(str(lut_path))
    
    # Colorize images
    count = colorizer.colorize_directory(
        input_dir=args.input,
        output_dir=args.output,
        copy_labels=True,
        label_dir=args.labels
    )
    
    print(f"\nColorization complete! {count} images processed.")
    print(f"Output saved to: {args.output}")


def run_analysis(args):
    """Run spectrum analysis."""
    from data.spectrum_analysis import SpectrumAnalyzer
    
    print("=" * 60)
    print("Spectrum Analysis - Analyzing IronRed Images")
    print("=" * 60)
    
    analyzer = SpectrumAnalyzer(sample_count=args.samples)
    analyzer.analyze_directory(args.input)
    analyzer.visualize_spectrum(args.output)
    
    dominant = analyzer.get_dominant_colors()
    print("\nDominant color peaks:")
    for channel, peaks in dominant.items():
        print(f"  {channel}: {peaks}")
    
    print(f"\nVisualization saved to: {args.output}")


def run_evaluation(args):
    """Run model evaluation."""
    from ultralytics import YOLO
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    model = YOLO(args.model)
    results = model.val(data=args.data)
    
    print("\nEvaluation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")


if __name__ == "__main__":
    main()
