from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class SyntheticAdapter(ABC):
    """
    Abstract adapter for external synthetic data generation modules.
    
    Implement this interface to integrate your synthetic data pipeline.
    """

    @abstractmethod
    def generate(
        self,
        background_img: np.ndarray,
        num_objects: int = 3,
        classes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic image by pasting objects onto background.

        Args:
            background_img: Background image (H, W, C)
            num_objects: Number of objects to paste
            classes: Optional list of class indices to use

        Returns:
            synthetic_img: Composite image (H, W, C)
            boxes: Bounding boxes in xyxy format (N, 4)
            labels: Class labels (N,)
        """
        pass

    @abstractmethod
    def get_available_classes(self) -> List[int]:
        """Return list of available class indices in crop library."""
        pass

    def __len__(self) -> int:
        """Return total number of available crops."""
        return 0


class DummySyntheticAdapter(SyntheticAdapter):
    """Dummy adapter that returns original image without modifications."""

    def generate(
        self,
        background_img: np.ndarray,
        num_objects: int = 3,
        classes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return background_img, np.empty((0, 4)), np.empty(0, dtype=np.int64)

    def get_available_classes(self) -> List[int]:
        return []
