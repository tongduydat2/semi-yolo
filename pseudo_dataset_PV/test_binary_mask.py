"""
Test script to visualize defect binary mask.
Shows defect images converted to 0/1 based on mean threshold.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def visualize_defect_binary(defect_paths: list, output_dir: str = None):
    """
    Convert defect images to binary (0/1) based on mean threshold.
    Pixels > mean → 1 (white)
    Pixels <= mean → 0 (black)
    """
    fig, axes = plt.subplots(len(defect_paths), 3, figsize=(12, 4 * len(defect_paths)))
    
    if len(defect_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for i, path in enumerate(defect_paths):
        # Load image
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not load: {path}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean
        mean_val = np.mean(gray)
        
        # Create binary mask: 1 where > mean, 0 elsewhere
        binary = (gray > mean_val).astype(np.uint8) * 255
        
        # Display
        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original: {Path(path).name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title(f"Grayscale (mean={mean_val:.1f})")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(binary, cmap='gray')
        axes[i, 2].set_title("Binary (>mean = 1, else = 0)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "defect_binary_visualization.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--defects", nargs="+", required=True, help="Paths to defect images")
    parser.add_argument("--output", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    visualize_defect_binary(args.defects, args.output)
