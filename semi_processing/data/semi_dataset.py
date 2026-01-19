from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader, Dataset

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data.utils import FORMATS_HELP_MSG
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from ultralytics.utils import IterableSimpleNamespace

from .synthetic_adapter import SyntheticAdapter, DummySyntheticAdapter


def dict_to_namespace(d: Dict[str, Any]) -> IterableSimpleNamespace:
    """Convert dict to IterableSimpleNamespace, using DEFAULT_CFG as base."""
    base = vars(DEFAULT_CFG).copy()
    base.update(d)
    return IterableSimpleNamespace(**base)


class UnlabeledYOLODataset(YOLODataset):
    """
    YOLODataset variant for unlabeled images.
    Returns images without ground truth labels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_labels(self):
        """Return empty labels for all images."""
        labels = []
        for im_file in self.im_files:
            labels.append({
                'im_file': im_file,
                'shape': (640, 640),
                'cls': torch.empty(0),
                'bboxes': torch.empty(0, 4),
                'segments': [],
                'keypoints': None,
                'normalized': True,
                'bbox_format': 'xywh',
            })
        return labels


class SemiDataModule:
    """
    Data module for semi-supervised training.
    Combines labeled, unlabeled, and optional synthetic data sources.
    """

    def __init__(
        self,
        labeled_path: str,
        unlabeled_path: str,
        imgsz: int = 640,
        batch_size: int = 16,
        labeled_ratio: float = 0.5,
        workers: int = 8,
        synthetic_adapter: Optional[SyntheticAdapter] = None,
        hyp: Optional[Dict[str, Any]] = None,
        augment_labeled: bool = True,
        augment_unlabeled: bool = True,
    ):
        self.labeled_path = Path(labeled_path)
        self.unlabeled_path = Path(unlabeled_path)
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.labeled_ratio = labeled_ratio
        self.workers = workers
        self.synthetic = synthetic_adapter or DummySyntheticAdapter()
        self.hyp = hyp or {}
        self.augment_labeled = augment_labeled
        self.augment_unlabeled = augment_unlabeled

        self.labeled_dataset = None
        self.unlabeled_dataset = None
        self.labeled_loader = None
        self.unlabeled_loader = None

    def setup(self, cfg: Optional[Dict] = None):
        """Initialize datasets and dataloaders."""
        cfg = cfg or {}

        self.labeled_dataset = YOLODataset(
            img_path=str(self.labeled_path / 'images'),
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            augment=self.augment_labeled,
            hyp=dict_to_namespace(self.hyp),
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix='labeled: ',
            task='detect',
            data=cfg.get('data', {}),
        )

        self.unlabeled_dataset = UnlabeledYOLODataset(
            img_path=str(self.unlabeled_path / 'images'),
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            augment=self.augment_unlabeled,
            hyp=dict_to_namespace(self.hyp),
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix='unlabeled: ',
            task='detect',
            data=cfg.get('data', {}),
        )

        labeled_bs = max(1, int(self.batch_size * self.labeled_ratio))
        unlabeled_bs = self.batch_size - labeled_bs

        self.labeled_loader = self._build_dataloader(
            self.labeled_dataset, labeled_bs, shuffle=True
        )
        self.unlabeled_loader = self._build_dataloader(
            self.unlabeled_dataset, unlabeled_bs, shuffle=True
        )

    def _build_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Build dataloader for dataset."""
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=YOLODataset.collate_fn,
        )

    def get_batch(self):
        """
        Get a combined batch of labeled and unlabeled data.

        Returns:
            labeled_batch: Dict with 'img', 'cls', 'bboxes', etc.
            unlabeled_batch: Dict with 'img' only (no labels)
        """
        labeled_batch = next(iter(self.labeled_loader))
        unlabeled_batch = next(iter(self.unlabeled_loader))
        return labeled_batch, unlabeled_batch

    def close_mosaic(self):
        """Disable mosaic augmentation for final epochs."""
        if hasattr(self.labeled_dataset, 'close_mosaic'):
            self.labeled_dataset.close_mosaic(self.hyp)
        if hasattr(self.unlabeled_dataset, 'close_mosaic'):
            self.unlabeled_dataset.close_mosaic(self.hyp)

    @property
    def num_labeled(self) -> int:
        return len(self.labeled_dataset) if self.labeled_dataset else 0

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled_dataset) if self.unlabeled_dataset else 0
