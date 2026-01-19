"""
Configuration utilities for Semi-YOLOv11.
Handles loading and merging YAML configurations.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration from YAML and apply overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: Dict of overrides from CLI args
    
    Returns:
        Merged configuration dict
    """
    default_config = {
        'model': 'yolo11n.pt',
        'imgsz': 640,
        'batch': 16,
        'epochs': 100,
        'workers': 8,
        'device': '0',
        'project': 'runs/semi',
        'name': 'exp',
        'data': {
            'nc': 80,
            'names': [],
        },
        'semi': {
            'labeled_path': None,
            'unlabeled_path': None,
            'burn_in': 5,
            'lambda_unsup': 1.0,
            'lambda_warmup': 5,
            'ema_decay': 0.999,
            'filters': [
                {'name': 'dsat', 'params': {'num_classes': 80}},
                {'name': 'dfl_entropy', 'params': {'threshold': 0.5}},
                {'name': 'tal_alignment', 'params': {'threshold': 0.3}},
            ],
        },
        'augment': {
            'weak': {
                'mosaic': 0.0,
                'mixup': 0.0,
                'fliplr': 0.5,
                'scale': 0.2,
            },
            'strong': {
                'mosaic': 1.0,
                'mixup': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'fliplr': 0.5,
                'scale': 0.5,
                'erasing': 0.3,
            },
        },
    }
    
    if config_path and Path(config_path).exists():
        file_config = load_yaml(config_path)
        config = merge_configs(default_config, file_config)
    else:
        config = default_config
    
    if overrides:
        config = merge_configs(config, overrides)
    
    return config


class SemiConfig:
    """Configuration object with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SemiConfig(value))
            else:
                setattr(self, key, value)
        self._dict = config_dict
    
    def to_dict(self) -> Dict[str, Any]:
        return self._dict
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self._dict[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._dict
