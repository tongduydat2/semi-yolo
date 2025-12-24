"""
Pipeline "Nhuộm màu" - Chuyển đổi Gray sang IronRed
Sử dụng Look-Up Table (LUT) dựa trên đặc trưng vật lý camera nhiệt
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import shutil
import yaml


class IronRedColorizer:
    """Convert grayscale thermal images to IronRed colormap."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize colorizer with temperature range mappings.
        
        Args:
            config: Optional config dict with colorization settings.
                   If None, uses default IronRed palette.
        """
        if config is None:
            # Default IronRed palette
            self.cold_range = (0, 60)
            self.warm_range = (60, 150)
            self.hot_range = (150, 255)
            self.cold_start = (0, 0, 0)
            self.cold_end = (80, 0, 80)
            self.warm_start = (80, 0, 80)
            self.warm_end = (255, 100, 0)
            self.hot_start = (255, 100, 0)
            self.hot_end = (255, 255, 200)
        else:
            self.cold_range = tuple(config.get('cold_range', [0, 60]))
            self.warm_range = tuple(config.get('warm_range', [60, 150]))
            self.hot_range = tuple(config.get('hot_range', [150, 255]))
            self.cold_start = tuple(config.get('cold_color_start', [0, 0, 0]))
            self.cold_end = tuple(config.get('cold_color_end', [80, 0, 80]))
            self.warm_start = tuple(config.get('warm_color_start', [80, 0, 80]))
            self.warm_end = tuple(config.get('warm_color_end', [255, 100, 0]))
            self.hot_start = tuple(config.get('hot_color_start', [255, 100, 0]))
            self.hot_end = tuple(config.get('hot_color_end', [255, 255, 200]))
        
        self.lut = self._build_lut()
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'IronRedColorizer':
        """Create colorizer from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config.get('colorization', {}))
        
    def _build_lut(self) -> np.ndarray:
        """Build Look-Up Table for Gray to IronRed conversion."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        for gray in range(256):
            r, g, b = self._map_gray_to_rgb(gray)
            lut[gray] = [b, g, r]  # OpenCV uses BGR
        return lut
    
    def _interpolate(self, t: float, start: Tuple, end: Tuple) -> Tuple[int, int, int]:
        """Linear interpolation between two RGB colors."""
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        return (r, g, b)
    
    def _map_gray_to_rgb(self, gray: int) -> Tuple[int, int, int]:
        """
        Map single gray value to RGB using IronRed thermal palette.
        
        Logic:
        - 0-60: Black to Dark Purple (Cold regions)
        - 60-150: Purple to Orange-Red (Warm regions)  
        - 150-255: Orange to Yellow/White (Hot regions)
        """
        cold_end = self.cold_range[1]
        warm_end = self.warm_range[1]
        
        if gray <= cold_end:
            t = gray / cold_end if cold_end > 0 else 0
            return self._interpolate(t, self.cold_start, self.cold_end)
        elif gray <= warm_end:
            t = (gray - cold_end) / (warm_end - cold_end)
            return self._interpolate(t, self.warm_start, self.warm_end)
        else:
            hot_end = self.hot_range[1]
            t = (gray - warm_end) / (hot_end - warm_end) if hot_end > warm_end else 1
            t = min(1.0, t)
            return self._interpolate(t, self.hot_start, self.hot_end)
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply LUT to grayscale image.
        
        Args:
            image: Input grayscale or BGR image
            
        Returns:
            Colorized IronRed image (BGR)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return self.lut[gray]
    
    def colorize_directory(self, 
                          input_dir: str, 
                          output_dir: str,
                          copy_labels: bool = True,
                          label_dir: Optional[str] = None) -> int:
        """
        Colorize all images in a directory (YOLO format).
        
        Args:
            input_dir: Path to input images directory
            output_dir: Path to output directory
            copy_labels: Whether to copy corresponding label files
            label_dir: Path to labels directory (if different from input_dir)
            
        Returns:
            Number of images processed
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        out_images = output_path / "images"
        out_labels = output_path / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        if copy_labels:
            out_labels.mkdir(parents=True, exist_ok=True)
        
        # Determine label directory
        if label_dir:
            label_path = Path(label_dir)
        else:
            label_path = input_path.parent / "labels"
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        processed = 0
        
        for img_file in input_path.iterdir():
            if img_file.suffix.lower() not in image_extensions:
                continue
                
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Warning: Could not read {img_file}")
                continue
                
            colorized = self.apply(image)
            cv2.imwrite(str(out_images / img_file.name), colorized)
            
            # Copy label file if exists
            if copy_labels:
                label_file = label_path / (img_file.stem + ".txt")
                if label_file.exists():
                    shutil.copy(str(label_file), str(out_labels / label_file.name))
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} images...")
        
        print(f"Colorization complete. Total: {processed} images")
        return processed
    
    def visualize_lut(self, save_path: Optional[str] = None):
        """Visualize the LUT as a gradient bar."""
        import matplotlib.pyplot as plt
        
        gradient = np.arange(256).reshape(1, 256).astype(np.uint8)
        colorized = self.apply(gradient)
        colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 3))
        
        axes[0].imshow(np.repeat(gradient, 30, axis=0), cmap='gray', aspect='auto')
        axes[0].set_title('Input: Grayscale')
        axes[0].set_xticks([0, 60, 150, 255])
        axes[0].set_xticklabels(['0 (Cold)', '60', '150', '255 (Hot)'])
        axes[0].set_yticks([])
        
        axes[1].imshow(np.repeat(colorized_rgb, 30, axis=0), aspect='auto')
        axes[1].set_title('Output: IronRed')
        axes[1].set_xticks([0, 60, 150, 255])
        axes[1].set_xticklabels(['Black/Purple', 'Purple', 'Orange-Red', 'Yellow/White'])
        axes[1].set_yticks([])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"LUT visualization saved to {save_path}")
        plt.close()
        return fig


def compare_histogram_similarity(fake_iron: np.ndarray, real_iron: np.ndarray) -> float:
    """
    Compare Fake IronRed with Real IronRed using histogram similarity.
    Pass Criteria: > 80% similarity
    """
    similarities = []
    for i in range(3):
        hist_fake = cv2.calcHist([fake_iron], [i], None, [256], [0, 256])
        hist_real = cv2.calcHist([real_iron], [i], None, [256], [0, 256])
        hist_fake = cv2.normalize(hist_fake, hist_fake).flatten()
        hist_real = cv2.normalize(hist_real, hist_real).flatten()
        sim = cv2.compareHist(hist_fake, hist_real, cv2.HISTCMP_CORREL)
        similarities.append(sim)
    return np.mean(similarities)


if __name__ == "__main__":
    # Example usage
    colorizer = IronRedColorizer()
    colorizer.visualize_lut("lut_visualization.png")
    
    # Colorize directory example:
    # colorizer.colorize_directory(
    #     input_dir="datasets/labeled/images",
    #     output_dir="datasets/fake_ironred",
    #     copy_labels=True,
    #     label_dir="datasets/labeled/labels"
    # )
