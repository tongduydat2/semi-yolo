"""
Data Loader Module
==================
Handles loading defect metadata and images with pickle caching.
"""

import json
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict


class DefectDataLoader:
    """Loads and organizes defect images by class with caching support."""
    
    CACHE_DIR = ".cache"
    
    def __init__(self, images_dir: str, metadata_path: str, use_cache: bool = True):
        """
        Initialize data loader.
        
        Args:
            images_dir: Path to defect images directory
            metadata_path: Path to metadata JSON file
            use_cache: Whether to use pickle cache (default: True)
        """
        self.images_dir = Path(images_dir)
        self.metadata_path = Path(metadata_path)
        self.use_cache = use_cache
        self.cache_path = self._get_cache_path()
        
        self.defects_by_class = self._load_with_cache()
    
    def _get_cache_path(self) -> Path:
        """Generate cache file path based on metadata file hash."""
        # Create cache dir next to metadata file
        cache_dir = self.metadata_path.parent / self.CACHE_DIR
        cache_dir.mkdir(exist_ok=True)
        
        # Hash based on metadata path and images dir
        hash_input = f"{self.metadata_path}_{self.images_dir}".encode()
        hash_id = hashlib.md5(hash_input).hexdigest()[:8]
        
        return cache_dir / f"defects_cache_{hash_id}.pkl"
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is newer than metadata file."""
        if not self.cache_path.exists():
            return False
        
        cache_mtime = self.cache_path.stat().st_mtime
        metadata_mtime = self.metadata_path.stat().st_mtime
        
        return cache_mtime > metadata_mtime
    
    def _load_with_cache(self) -> dict:
        """Load metadata with caching support."""
        if self.use_cache and self._is_cache_valid():
            return self._load_from_cache()
        else:
            data = self._load_metadata()
            if self.use_cache:
                self._save_to_cache(data)
            return data
    
    def _load_from_cache(self) -> dict:
        """Load from pickle cache."""
        print(f"Loading from cache: {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Print statistics
        print(data.keys())
        total = sum(len(v) for v in data.values())
        print(f"  Loaded {total} defects in {len(data)} classes (cached)")
        return data
    
    def _save_to_cache(self, data: dict):
        """Save to pickle cache."""
        print(f"Saving cache to: {self.cache_path}")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_metadata(self) -> dict:
        """Load and organize defect images by class (original method)."""
        print(f"Loading defect metadata from: {self.metadata_path}")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        defects_by_class = defaultdict(list)
        
        for img_id, info in metadata.items():
            img_path = self.images_dir / info['image_filepath'].replace('images/', '')
            if not img_path.exists():
                img_path = self.images_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                anomaly_class = info['anomaly_class']
                defects_by_class[anomaly_class].append({
                    'id': img_id,
                    'path': str(img_path),
                    'class': anomaly_class
                })
        
        # Print statistics
        print("\nDefect images per class:")
        for cls, items in sorted(defects_by_class.items()):
            print(f"  {cls}: {len(items)} images")
        
        return dict(defects_by_class)
    
    def clear_cache(self):
        """Delete cache file to force reload."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            print(f"Cache cleared: {self.cache_path}")
    
    def get_classes(self) -> list:
        """Get list of available classes."""
        return list(self.defects_by_class.keys())
    
    def get_defects(self, class_name: str) -> list:
        """Get defects for a specific class."""
        return self.defects_by_class.get(class_name, [])
