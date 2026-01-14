"""
Defect Transformer Module
=========================
Handles resizing, rotating, and overlaying defect images onto panel images.
Simplified version for pre-colored (iron red) defect images.
"""

import cv2
import numpy as np
from PIL import Image


class DefectTransformer:
    """Transforms and overlays defect images onto panels."""
    
    @staticmethod
    def transform(defect_img: Image.Image, obb: dict) -> Image.Image:
        """
        Transform defect image to match OBB size and rotation.
        
        Args:
            defect_img: PIL Image of defect (already colored)
            obb: OBB detection dict with cx, cy, width, height, angle
            
        Returns:
            Transformed PIL Image
        """
        # Resize to match OBB dimensions

        # PIL → OpenCV
        img = np.array(defect_img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray > 20).astype(np.uint8)

        # === Crop theo mask ===
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return defect_img

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        img = img[y_min:y_max+1, x_min:x_max+1]
        mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Resize theo OBB
        target_w = max(int(obb['width']), 10)
        target_h = max(int(obb['height']), 10)

        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Rotate with anti-aliasing (BICUBIC is smoother than BILINEAR)
        angle_degrees = np.degrees(obb['angle'])
        img = Image.fromarray(img)
        img = img.rotate(-angle_degrees, expand=True, resample=Image.Resampling.BICUBIC)

        return img
    
    @staticmethod
    def overlay(panel_img: Image.Image, defect_img: Image.Image, obb: dict) -> tuple:
        """
        Overlay transformed defect onto panel image.
        Direct paste - defect replaces panel pixels.
        
        Args:
            panel_img: PIL Image of panel
            defect_img: Transformed defect image
            obb: OBB detection dict
            
        Returns:
            (modified_panel_img, bbox_dict, mask)
        """
        panel_w, panel_h = panel_img.size
        defect_w, defect_h = defect_img.size
        
        # Calculate paste position (center of OBB)
        paste_x = int(obb['cx'] - defect_w / 2)
        paste_y = int(obb['cy'] - defect_h / 2)
        
        # Calculate crop offsets if defect goes out of bounds
        crop_left = max(0, -paste_x)
        crop_top = max(0, -paste_y)
        crop_right = max(0, (paste_x + defect_w) - panel_w)
        crop_bottom = max(0, (paste_y + defect_h) - panel_h)
        
        # Adjust paste position to stay within bounds
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)
        
        # Calculate actual dimensions after cropping
        actual_w = defect_w - crop_left - crop_right
        actual_h = defect_h - crop_top - crop_bottom
        
        if actual_w <= 0 or actual_h <= 0:
            # Defect completely out of bounds, return original
            mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
            return panel_img, {'x_center': 0.5, 'y_center': 0.5, 'width': 0, 'height': 0}, mask
        
        # Create copy
        result = panel_img.copy()
        if result.mode != 'RGB':
            result = result.convert('RGB')
        
        # Convert to numpy
        result_array = np.array(result)
        panel_region = result_array[paste_y:paste_y+actual_h, paste_x:paste_x+actual_w].copy()
        
        # Get defect crop (crop from defect image based on bounds)
        defect_array = np.array(defect_img)
        if len(defect_array.shape) == 2:
            defect_crop = np.stack([defect_array[crop_top:crop_top+actual_h, crop_left:crop_left+actual_w]] * 3, axis=-1)
        else:
            defect_crop = defect_array[crop_top:crop_top+actual_h, crop_left:crop_left+actual_w]
        
        # ========== REMOVE BLACK BACKGROUND ==========
        # Method 1: Grayscale threshold
        defect_gray = cv2.cvtColor(defect_crop, cv2.COLOR_RGB2GRAY)
        gray_mask = defect_gray > 20  # Fixed low threshold for black pixels
        
        # Method 2: Check all RGB channels are above threshold
        # rgb_mask = np.all(defect_crop > 5, axis=-1)  # All channels > 10
        
        # Combine masks: must pass both checks
        mask = gray_mask.astype(np.float32)
        
        # ========== EDGE SMOOTHING ==========
        # Apply Gaussian blur to mask edges to reduce jagged aliasing
        mask_blurred = cv2.GaussianBlur(mask, (3, 3), 3)
        
        # Alpha blend using smoothed mask
        mask_3ch = np.stack([mask_blurred] * 3, axis=-1)
        final = (panel_region.astype(np.float32) * (1 - mask_3ch) + 
                 defect_crop.astype(np.float32) * mask_3ch)
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # Put back into result
        result_array[paste_y:paste_y+actual_h, paste_x:paste_x+actual_w] = final
        result = Image.fromarray(result_array)
        
        # Create full-size mask for saving (use binary mask, not blurred)
        full_mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
        full_mask[paste_y:paste_y+actual_h, paste_x:paste_x+actual_w] = (gray_mask * 255).astype(np.uint8)
        
        # Calculate bbox (YOLO format: normalized)
        bbox = {
            'x_center': (paste_x + actual_w / 2) / panel_w,
            'y_center': (paste_y + actual_h / 2) / panel_h,
            'width': actual_w / panel_w,
            'height': actual_h / panel_h
        }
        
        return result, bbox, full_mask
