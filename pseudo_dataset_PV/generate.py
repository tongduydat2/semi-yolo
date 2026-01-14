"""
Synthetic Defect Generator
==========================
Main script that combines all modules to generate synthetic defect images.

Usage:
    python generate.py \
        --config config.yaml \
        --weights best.pt \
        --num-images 100 \
        --classes "Cell:50,Hot-Spot:30,Cracking:20"
"""

import os
import yaml
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from detector import OBBDetector
from transformer import DefectTransformer
from data_loader import DefectDataLoader
from annotation_writer import AnnotationWriter
from thermal_extractor import ThermalExtractor

import numpy as np
class SyntheticGenerator:
    """Main generator that orchestrates all modules."""
    
    def __init__(self, config_path: str, weights_path: str):
        """
        Initialize generator with config and model.
        
        Args:
            config_path: Path to YAML config file
            weights_path: Path to YOLO OBB model weights
        """
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Validate config
        required_keys = ['panel_images', 'defect_images', 'defect_metadata', 'output_dir']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Initialize modules
        self.detector = OBBDetector(weights_path)
        self.transformer = DefectTransformer()
        self.thermal_extractor = ThermalExtractor()
        self.data_loader = DefectDataLoader(
            self.config['defect_images'],
            self.config['defect_metadata']
        )
        self.writer = AnnotationWriter(self.config['output_dir'])
        
        # Paths
        self.panel_images_dir = Path(self.config['panel_images'])
        self.output_dir = Path(self.config['output_dir'])
        
        # Class mapping
        self.class_to_id = {}
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_defects': 0,
            'defects_per_class': defaultdict(int)
        }
    
    def _parse_class_distribution(self, classes_str: str) -> dict:
        """Parse class distribution string."""
        distribution = {}
        for item in classes_str.split(','):
            parts = item.strip().split(':')
            if len(parts) == 2:
                class_name = parts[0].strip()
                count = int(parts[1].strip())
                
                if class_name not in self.data_loader.defects_by_class:
                    print(f"Warning: Class '{class_name}' not found")
                    continue
                    
                distribution[class_name] = count
        
        self.class_to_id = {cls: idx for idx, cls in enumerate(sorted(distribution.keys()))}
        return distribution
    
    def generate(self, num_images: int, class_distribution: dict,
                 max_per_image: int = 10, conf_threshold: float = 0.5):
        """
        Generate synthetic images.
        
        Args:
            num_images: Number of images to generate
            class_distribution: {class_name: target_count}
            max_per_image: Max defects per image
            conf_threshold: OBB confidence threshold
        """
        print(f"\n{'='*60}")
        print("SYNTHETIC DEFECT GENERATION")
        print(f"{'='*60}")
        print(f"Target images: {num_images}")
        print(f"Max per image: {max_per_image}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Class distribution: {class_distribution}")
        print(f"{'='*60}\n")
        
        # Get panel images
        panel_images = list(self.panel_images_dir.glob('*.jpg')) + \
                       list(self.panel_images_dir.glob('*.png')) + \
                       list(self.panel_images_dir.glob('*.JPG'))
        
        if not panel_images:
            raise ValueError(f"No panel images found in: {self.panel_images_dir}")
        
        print(f"Found {len(panel_images)} panel images")
        
        # Create class_to_id mapping from class_distribution
        self.class_to_id = {cls: i for i, cls in enumerate(class_distribution.keys())}
        print(f"Class mapping: {self.class_to_id}")
        
        # Track counts
        current_counts = {cls: 0 for cls in class_distribution}
        
        pbar = tqdm(total=num_images, desc="Generating")
        image_idx = 0
        attempts = 0
        max_attempts = num_images * 10
        
        while image_idx < num_images and attempts < max_attempts:
            attempts += 1
            
            # Select random panel
            panel_path = random.choice(panel_images)
            
            # Detect OBBs - only use No-Anomaly panels for overlay
            detections = self.detector.detect(str(panel_path), conf_threshold, 
                                               filter_class="No-Anomaly")
            if not detections:
                continue
            
            # Limit detections - random value in range [n/2, n]
            actual_max = random.randint(max(1, max_per_image // 2), max_per_image)
            if len(detections) > actual_max:
                detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                detections = detections[:actual_max]
            
            # Load panel
            panel_img = Image.open(panel_path)
            
            # Check which classes need more
            classes_needed = [cls for cls, target in class_distribution.items()
                             if current_counts[cls] < target]
            
            if not classes_needed:
                break
            
            annotations = []
            result_img = panel_img.copy()
            accumulated_mask = None  # Reset mask for each new image
            
            for obb in detections:
                if not classes_needed:
                    break
                
                target_class = random.choice(classes_needed)
                defects = self.data_loader.get_defects(target_class)
                if not defects:
                    continue
                
                # Get raw thermal range from panel image (TIFF/EXIF)
                thermal_range = self.thermal_extractor.get_thermal_range(str(panel_path))
                
                
                # Load and transform defect
                defect_info = random.choice(defects)
                defect_img = Image.open(defect_info['path'])
                transformed = self.transformer.transform(defect_img, obb)
                
                # Overlay (now also returns binary mask)
                result_img, bbox, defect_mask = self.transformer.overlay(result_img, transformed, obb)
                
                # Accumulate mask (multiple defects on same image)
                if accumulated_mask is None:
                    accumulated_mask = np.zeros((result_img.size[1], result_img.size[0]), dtype=np.uint8)
                accumulated_mask = np.maximum(accumulated_mask, defect_mask)
                
                # Record annotation
                class_id = self.class_to_id[target_class]
                annotations.append((
                    class_id,
                    bbox['x_center'],
                    bbox['y_center'],
                    bbox['width'],
                    bbox['height']
                ))
                
                # Update counts
                current_counts[target_class] += 1
                self.stats['defects_per_class'][target_class] += 1
                self.stats['total_defects'] += 1
                
                classes_needed = [cls for cls, target in class_distribution.items()
                                 if current_counts[cls] < target]
            
            if annotations:
                # Save image
                filename = f"synthetic_{image_idx:05d}"
                result_img.save(self.writer.images_dir / f"{filename}.jpg", quality=95)
                
                # Save annotation
                self.writer.save_annotation(filename, annotations)
                
                # Save binary mask
                import cv2
                cv2.imwrite(str(self.writer.masks_dir / f"{filename}.png"), accumulated_mask)
                
                image_idx += 1
                self.stats['total_images'] += 1
                pbar.update(1)
        
        pbar.close()
        
        # Save log
        self.writer.save_log(
            self.config,
            self.class_to_id,
            {
                'total_images': self.stats['total_images'],
                'total_defects': self.stats['total_defects'],
                'defects_per_class': dict(self.stats['defects_per_class'])
            }
        )
        
        # Save YOLO data.yaml
        class_names = [name for name, _ in sorted(self.class_to_id.items(), key=lambda x: x[1])]
        self.writer.save_data_yaml(class_names)
        
        self._print_summary(class_distribution)
    
    def _print_summary(self, class_distribution: dict):
        """Print generation summary."""
        print(f"\n{'='*60}")
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total images: {self.stats['total_images']}")
        print(f"Total defects: {self.stats['total_defects']}")
        print("\nDefects per class:")
        for cls, count in sorted(self.stats['defects_per_class'].items()):
            target = class_distribution.get(cls, 'N/A')
            print(f"  {cls}: {count} / {target}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic defect images')
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    parser.add_argument('-w', '--weights', required=True, help='YOLO OBB weights')
    parser.add_argument('-n', '--num-images', type=int, required=True, help='Number of images')
    parser.add_argument('--classes', help='Class distribution "Class:count,..." (optional, reads from config)')
    parser.add_argument('--max-per-image', type=int, default=10, help='Max defects per image')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='OBB confidence')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config not found: {args.config}")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights not found: {args.weights}")
        return
    
    generator = SyntheticGenerator(args.config, args.weights)
    
    # Get class distribution from args or config
    if args.classes:
        class_dist = generator._parse_class_distribution(args.classes)
    elif 'class_distribution' in generator.config:
        class_dist = generator.config['class_distribution']
        print(f"Using class distribution from config: {class_dist}")
    else:
        print("Error: No class distribution specified. Use --classes or add to config.yaml")
        return
    
    if not class_dist:
        print("Error: No valid classes specified")
        return
    
    generator.generate(args.num_images, class_dist, args.max_per_image, args.conf_threshold)


if __name__ == '__main__':
    main()
