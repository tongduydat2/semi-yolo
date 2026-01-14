"""
Annotation Writer Module
========================
Handles saving YOLO format annotations.
"""

import json
from pathlib import Path


class AnnotationWriter:
    """Writes YOLO format annotations and logs."""
    
    def __init__(self, output_dir: str):
        """
        Initialize writer.
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.masks_dir = self.output_dir / 'masks'
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
    
    def save_annotation(self, filename: str, annotations: list):
        """
        Save annotations in YOLO bbox format.
        
        Args:
            filename: Base filename (without extension)
            annotations: List of (class_id, x_center, y_center, width, height)
        """
        label_path = self.labels_dir / f"{filename}.txt"
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id, x_c, y_c, w, h = ann
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    
    def save_log(self, config: dict, class_mapping: dict, stats: dict):
        """
        Save generation log to JSON.
        
        Args:
            config: Configuration dict
            class_mapping: Class name to ID mapping
            stats: Generation statistics
        """
        log_path = self.output_dir / 'generation_log.json'
        log_data = {
            'config': config,
            'class_mapping': class_mapping,
            'statistics': stats
        }
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def save_data_yaml(self, class_names: list):
        """
        Save YOLO format data.yaml file.
        
        Args:
            class_names: List of class names in order of class IDs
        """
        import yaml
        
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Saved: {yaml_path}")
