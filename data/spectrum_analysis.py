"""
Phân tích phổ màu từ tập Unlabeled IronRed
Mục đích: Xác định dải màu chủ đạo để xây dựng LUT chính xác
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class SpectrumAnalyzer:
    """Analyze color spectrum from IronRed thermal images."""
    
    def __init__(self, sample_count: int = 100):
        """
        Args:
            sample_count: Number of images to sample for analysis
        """
        self.sample_count = sample_count
        self.histograms: Dict[str, np.ndarray] = {}
        
    def analyze_directory(self, image_dir: str) -> Dict[str, np.ndarray]:
        """
        Analyze color spectrum from images in directory.
        
        Args:
            image_dir: Path to directory containing IronRed images
            
        Returns:
            Dictionary with R, G, B histogram data
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + \
                      list(image_dir.glob("*.png")) + \
                      list(image_dir.glob("*.jpeg"))
        
        if len(image_files) > self.sample_count:
            indices = np.random.choice(len(image_files), self.sample_count, replace=False)
            image_files = [image_files[i] for i in indices]
        
        r_hist = np.zeros(256, dtype=np.float64)
        g_hist = np.zeros(256, dtype=np.float64)
        b_hist = np.zeros(256, dtype=np.float64)
        total_pixels = 0
        
        print(f"Analyzing {len(image_files)} images...")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            b, g, r = cv2.split(img)
            r_hist += cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
            g_hist += cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
            b_hist += cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
            total_pixels += img.shape[0] * img.shape[1]
        
        if total_pixels > 0:
            r_hist /= total_pixels
            g_hist /= total_pixels
            b_hist /= total_pixels
        
        self.histograms = {'R': r_hist, 'G': g_hist, 'B': b_hist}
        print(f"Analysis complete. Total pixels: {total_pixels:,}")
        return self.histograms
    
    def get_dominant_colors(self) -> Dict[str, List[int]]:
        """Extract dominant color ranges from histograms."""
        dominant = {}
        for channel, hist in self.histograms.items():
            peaks = []
            for i in range(5, 251):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    if hist[i] > np.mean(hist) * 1.5:
                        peaks.append(i)
            dominant[channel] = peaks
        return dominant
    
    def visualize_spectrum(self, save_path: Optional[str] = None):
        """Visualize the color spectrum histogram."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['red', 'green', 'blue']
        
        for ax, (channel, hist), color in zip(axes, self.histograms.items(), colors):
            ax.fill_between(range(256), hist, alpha=0.7, color=color)
            ax.set_title(f'{channel} Channel Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Normalized Frequency')
            ax.set_xlim([0, 255])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Spectrum saved to {save_path}")
        plt.close()
        return fig
        
    def compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms using correlation (0-1, higher is better)."""
        return cv2.compareHist(
            hist1.astype(np.float32), 
            hist2.astype(np.float32), 
            cv2.HISTCMP_CORREL
        )
    
    def compare_images(self, img1_path: str, img2_path: str) -> float:
        """
        Compare histogram similarity between two images.
        
        Args:
            img1_path: Path to first image (e.g., Fake IronRed)
            img2_path: Path to second image (e.g., Real IronRed)
            
        Returns:
            Average similarity across R, G, B channels (0-1)
        """
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images")
        
        similarities = []
        for i in range(3):
            hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarities.append(sim)
        
        return np.mean(similarities)


if __name__ == "__main__":
    # Example usage
    analyzer = SpectrumAnalyzer(sample_count=100)
    
    # Uncomment to analyze your IronRed images:
    # analyzer.analyze_directory("datasets/unlabeled/images")
    # analyzer.visualize_spectrum("spectrum_analysis.png")
    # dominant = analyzer.get_dominant_colors()
    # print("Dominant peaks:", dominant)
