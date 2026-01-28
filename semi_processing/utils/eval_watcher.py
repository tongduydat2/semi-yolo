"""
Human-in-the-Loop Eval Data Watcher.

Monitors eval/validation directory for changes during training.
When changes detected (new images, modified labels), triggers dataloader reload.

Complexity: Θ(n) per check where n = number of files in eval directory.
Uses file metadata (mtime, size) to avoid reading file contents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from ultralytics.utils import LOGGER, colorstr


class EvalDataWatcher:
    """
    Watch eval data directory for changes to enable human-in-the-loop updates.
    
    Use case: During long training runs, humans can add/modify eval data,
    and the trainer will automatically reload the validation dataloader.
    
    Attributes:
        eval_dir: Path to eval/val data directory
        labels_dir: Path to labels directory (optional, auto-detected if None)
        _last_fingerprint: Cached fingerprint from previous check
    """
    
    def __init__(
        self,
        eval_dir: str | Path,
        labels_dir: Optional[str | Path] = None,
    ):
        """
        Initialize watcher.
        
        Args:
            eval_dir: Path to images directory (e.g., dataset/images/val)
            labels_dir: Path to labels directory. If None, infers from eval_dir
                       by replacing 'images' with 'labels' in path.
        """
        self.eval_dir = Path(eval_dir)
        
        # Auto-detect labels directory
        if labels_dir is None:
            # Common convention: dataset/images/val -> dataset/labels/val
            eval_str = str(self.eval_dir)
            if 'images' in eval_str:
                self.labels_dir = Path(eval_str.replace('images', 'labels'))
            else:
                self.labels_dir = None
        else:
            self.labels_dir = Path(labels_dir)
        
        self._last_fingerprint: Optional[str] = None
        self._initialized = False
        
        LOGGER.info(colorstr('EvalWatcher: ') + f'Monitoring: {self.eval_dir}')
        if self.labels_dir and self.labels_dir.exists():
            LOGGER.info(colorstr('EvalWatcher: ') + f'Labels dir: {self.labels_dir}')
    
    def compute_fingerprint(self) -> str:
        """
        Compute directory fingerprint from file metadata.
        
        Uses (relative_path, mtime, size) for each file.
        Avoids reading file contents for efficiency.
        
        Returns:
            MD5 hash string representing current state of directories.
        """
        entries = []
        
        # Scan images directory
        if self.eval_dir.exists():
            for p in sorted(self.eval_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                    stat = p.stat()
                    entries.append(f"img|{p.relative_to(self.eval_dir)}|{stat.st_mtime:.6f}|{stat.st_size}")
        
        # Scan labels directory
        if self.labels_dir and self.labels_dir.exists():
            for p in sorted(self.labels_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in {'.txt', '.json', '.xml'}:
                    stat = p.stat()
                    entries.append(f"lbl|{p.relative_to(self.labels_dir)}|{stat.st_mtime:.6f}|{stat.st_size}")
        
        # Also track file count for quick change detection
        entries.append(f"count|{len(entries)}")
        
        return hashlib.md5('\n'.join(entries).encode()).hexdigest()
    
    def check_changed(self) -> bool:
        """
        Check if eval data has changed since last check.
        
        Returns:
            True if data changed (or first run), False otherwise.
            First run returns False to avoid unnecessary reload at start.
        """
        current_fp = self.compute_fingerprint()
        
        if not self._initialized:
            # First run - just record fingerprint, don't trigger reload
            self._last_fingerprint = current_fp
            self._initialized = True
            LOGGER.info(colorstr('EvalWatcher: ') + f'Initial fingerprint: {current_fp[:8]}...')
            return False
        
        if current_fp != self._last_fingerprint:
            LOGGER.info(colorstr('EvalWatcher: ') + 
                       f'Data changed! Old: {self._last_fingerprint[:8]}... → New: {current_fp[:8]}...')
            self._last_fingerprint = current_fp
            return True
        
        return False
    
    def get_file_counts(self) -> dict:
        """Get current file counts for logging."""
        img_count = 0
        lbl_count = 0
        
        if self.eval_dir.exists():
            img_count = sum(1 for p in self.eval_dir.rglob('*') 
                          if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
        
        if self.labels_dir and self.labels_dir.exists():
            lbl_count = sum(1 for p in self.labels_dir.rglob('*')
                          if p.is_file() and p.suffix.lower() in {'.txt', '.json', '.xml'})
        
        return {'images': img_count, 'labels': lbl_count}
