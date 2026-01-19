"""
Augmentation configuration presets for semi-supervised training.

Uses Ultralytics' built-in augmentation system.
Weak augmentation for Teacher, Strong augmentation for Student.
"""

from typing import Dict, Any

WEAK_AUG: Dict[str, Any] = {
    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'degrees': 0.0,
    'translate': 0.0,
    'scale': 0.2,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'bgr': 0.0,
    'mosaic': 0.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'copy_paste_mode': 'flip',
    'auto_augment': '',
    'erasing': 0.0,
    'crop_fraction': 1.0,
}

STRONG_AUG: Dict[str, Any] = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'bgr': 0.0,
    'mosaic': 1.0,
    'mixup': 0.5,
    'copy_paste': 0.3,
    'copy_paste_mode': 'flip',
    'auto_augment': '',
    'erasing': 0.3,
    'crop_fraction': 1.0,
}


def get_weak_transforms() -> Dict[str, Any]:
    """Return weak augmentation config for Teacher."""
    return WEAK_AUG.copy()


def get_strong_transforms() -> Dict[str, Any]:
    """Return strong augmentation config for Student."""
    return STRONG_AUG.copy()


def merge_aug_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge augmentation configs."""
    result = base.copy()
    result.update(overrides)
    return result
