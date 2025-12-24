"""
Augmentation strategies for SSOD
- Weak Augmentation: Cho Teacher (giữ ảnh sạch để dự đoán chính xác)
- Strong Augmentation: Cho Student (làm khó để học robust)
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional, Dict, Any


class WeakAugmentation:
    """
    Weak Augmentation for Teacher Model.
    Only applies gentle transformations to keep predictions accurate.
    """
    
    def __init__(self, 
                 size: Tuple[int, int] = (640, 640),
                 horizontal_flip_prob: float = 0.5):
        self.size = size
        self.horizontal_flip_prob = horizontal_flip_prob
        
    def __call__(self, 
                 image: np.ndarray, 
                 boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply weak augmentation.
        
        Args:
            image: Input image (BGR)
            boxes: Bounding boxes in YOLO format [class, x_center, y_center, w, h] (normalized)
            
        Returns:
            Augmented image and transformed boxes
        """
        h, w = image.shape[:2]
        
        # Resize
        image = cv2.resize(image, self.size)
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = cv2.flip(image, 1)
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip x_center
        
        return image, boxes


class StrongAugmentation:
    """
    Strong Augmentation for Student Model.
    Applies aggressive transformations to force robust learning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        
        self.size = tuple(config.get('resize', [640, 640]))
        self.mosaic_prob = config.get('mosaic_prob', 0.5)
        self.mixup_prob = config.get('mixup', 0.3)
        self.cutout_prob = config.get('cutout', 0.3)
        self.cutout_holes = config.get('cutout_holes', 3)
        self.cutout_ratio = config.get('cutout_ratio', 0.1)
        self.noise_std = config.get('gaussian_noise', 0.1)
        self.color_jitter = config.get('color_jitter', True)
        self.brightness_range = config.get('brightness', [0.8, 1.2])
        self.contrast_range = config.get('contrast', [0.8, 1.2])
        
        self.weak_aug = WeakAugmentation(size=self.size)
        
    def __call__(self, 
                 image: np.ndarray, 
                 boxes: Optional[np.ndarray] = None,
                 extra_images: Optional[List[np.ndarray]] = None,
                 extra_boxes: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply strong augmentation."""
        # Start with weak augmentation
        image, boxes = self.weak_aug(image, boxes)
        
        # Color jitter
        if self.color_jitter:
            image = self._apply_color_jitter(image)
        
        # Gaussian noise
        if self.noise_std > 0:
            image = self._apply_gaussian_noise(image)
        
        # Cutout
        if random.random() < self.cutout_prob:
            image = self._apply_cutout(image)
        
        # Mosaic (requires extra images)
        if extra_images and len(extra_images) >= 3 and random.random() < self.mosaic_prob:
            image, boxes = self._apply_mosaic(image, boxes, extra_images[:3], 
                                              extra_boxes[:3] if extra_boxes else None)
        # Mixup
        elif extra_images and random.random() < self.mixup_prob:
            image, boxes = self._apply_mixup(image, boxes, extra_images[0], 
                                             extra_boxes[0] if extra_boxes else None)
        
        return image, boxes
    
    def _apply_color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jittering suitable for thermal images."""
        brightness = random.uniform(*self.brightness_range)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        contrast = random.uniform(*self.contrast_range)
        mean = np.mean(image)
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        return image
    
    def _apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, self.noise_std * 255, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def _apply_cutout(self, image: np.ndarray) -> np.ndarray:
        """Apply random cutout (holes) to image."""
        h, w = image.shape[:2]
        hole_size = int(min(h, w) * self.cutout_ratio)
        
        for _ in range(self.cutout_holes):
            x = random.randint(0, w - hole_size)
            y = random.randint(0, h - hole_size)
            image[y:y+hole_size, x:x+hole_size] = 0
        return image
    
    def _apply_mosaic(self, image, boxes, extra_images, extra_boxes):
        """Apply 4-image mosaic augmentation."""
        s = self.size[0]
        mosaic_img = np.zeros((s, s, 3), dtype=np.uint8)
        
        xc = int(random.uniform(s * 0.25, s * 0.75))
        yc = int(random.uniform(s * 0.25, s * 0.75))
        
        all_images = [image] + list(extra_images)
        all_boxes = [boxes] + (list(extra_boxes) if extra_boxes else [None] * 3)
        
        combined_boxes = []
        positions = [(0, 0, xc, yc), (xc, 0, s, yc), (0, yc, xc, s), (xc, yc, s, s)]
        
        for i, (x1, y1, x2, y2) in enumerate(positions):
            if i >= len(all_images):
                break
            img = cv2.resize(all_images[i], (x2 - x1, y2 - y1))
            mosaic_img[y1:y2, x1:x2] = img
            
            if all_boxes[i] is not None and len(all_boxes[i]) > 0:
                bx = all_boxes[i].copy()
                # Transform normalized coords to mosaic coords
                scale_x = (x2 - x1) / s
                scale_y = (y2 - y1) / s
                bx[:, 1] = bx[:, 1] * scale_x + x1 / s
                bx[:, 2] = bx[:, 2] * scale_y + y1 / s
                bx[:, 3] = bx[:, 3] * scale_x
                bx[:, 4] = bx[:, 4] * scale_y
                combined_boxes.append(bx)
        
        if combined_boxes:
            combined_boxes = np.vstack(combined_boxes)
        else:
            combined_boxes = boxes
            
        return mosaic_img, combined_boxes
    
    def _apply_mixup(self, image1, boxes1, image2, boxes2):
        """Apply mixup augmentation (blend two images)."""
        lam = random.uniform(0.4, 0.6)
        image2 = cv2.resize(image2, self.size)
        
        mixed = (image1.astype(np.float32) * lam + 
                 image2.astype(np.float32) * (1 - lam))
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        
        combined_boxes = []
        if boxes1 is not None and len(boxes1) > 0:
            combined_boxes.append(boxes1)
        if boxes2 is not None and len(boxes2) > 0:
            combined_boxes.append(boxes2)
            
        if combined_boxes:
            combined_boxes = np.vstack(combined_boxes)
        else:
            combined_boxes = boxes1
            
        return mixed, combined_boxes


if __name__ == "__main__":
    # Test augmentations
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_boxes = np.array([[0, 0.5, 0.5, 0.2, 0.2]])  # YOLO format
    
    weak_aug = WeakAugmentation()
    weak_img, weak_boxes = weak_aug(dummy_img.copy(), dummy_boxes.copy())
    print(f"Weak augmentation output: {weak_img.shape}")
    
    strong_aug = StrongAugmentation()
    strong_img, strong_boxes = strong_aug(dummy_img.copy(), dummy_boxes.copy())
    print(f"Strong augmentation output: {strong_img.shape}")
