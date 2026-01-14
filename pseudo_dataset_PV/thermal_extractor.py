"""
Thermal Data Extractor Module
==============================
Extract raw thermal min/max from R-JPEG images using DJI Thermal SDK.
"""

import os
import sys
from pathlib import Path

# Add thermal_parser to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'thermal_parser'))

try:
    from thermal import Thermal
    THERMAL_AVAILABLE = True
except ImportError:
    THERMAL_AVAILABLE = False
    print("Warning: thermal module not available. Using grayscale fallback.")


class ThermalExtractor:
    """Extract raw thermal data from R-JPEG images."""
    
    def __init__(self):
        """Initialize thermal parser if available."""
        self._parser = None
        if THERMAL_AVAILABLE:
            try:
                self._parser = Thermal()
                print("Thermal parser initialized (DJI SDK)")
            except Exception as e:
                print(f"Warning: Could not initialize thermal parser: {e}")
                self._parser = None
    
    def get_thermal_range(self, image_path: str) -> tuple:
        """
        Get min/max temperature from R-JPEG image.
        
        Args:
            image_path: Path to thermal image (R-JPEG/TIFF)
            
        Returns:
            (min_temp, max_temp) in Celsius, or (None, None) if failed
        """
        if self._parser is None:
            return None, None
        
        try:
            # Parse thermal data
            temp_array, meta = self._parser.parse(image_path)
            
            # Get min/max temperature
            min_temp = float(temp_array.min())
            max_temp = float(temp_array.max())
            
            return min_temp, max_temp
        except Exception as e:
            print(f"Warning: Could not extract thermal data from {image_path}: {e}")
            return None, None
    
    def get_thermal_array(self, image_path: str):
        """
        Get full temperature array from R-JPEG image.
        
        Args:
            image_path: Path to thermal image
            
        Returns:
            numpy array of temperatures in Celsius, or None if failed
        """
        if self._parser is None:
            return None
        
        try:
            temp_array, meta = self._parser.parse(image_path)
            return temp_array
        except Exception as e:
            print(f"Warning: Could not parse thermal data: {e}")
            return None
    
    @property
    def is_available(self) -> bool:
        """Check if thermal parsing is available."""
        return self._parser is not None
