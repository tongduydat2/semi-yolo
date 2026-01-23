from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data.utils import FORMATS_HELP_MSG
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from ultralytics.utils import IterableSimpleNamespace
import torch
import torchvision.transforms.v2 as T

from .synthetic_adapter import SyntheticAdapter, DummySyntheticAdapter


def dict_to_namespace(d: Dict[str, Any]) -> IterableSimpleNamespace:
    """Convert dict to IterableSimpleNamespace, using DEFAULT_CFG as base."""
    base = vars(DEFAULT_CFG).copy()
    base.update(d)
    return IterableSimpleNamespace(**base)


class SyncRandomSampler(Sampler):
    """
    Random sampler that can be synchronized across multiple dataloaders.
    Uses a shared seed that is reset together for all loaders.
    """
    
    def __init__(self, data_source: Dataset, seed: int = 0):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        
    def __iter__(self) -> Iterator[int]:
        # Use seed + epoch for reproducibility per epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.data_source)
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling across synced loaders."""
        self.epoch = epoch


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
    
    Implements proper weak/strong augmentation separation with synchronized sampling:
    - Two loaders for unlabeled data (weak + strong aug)
    - Same shuffle order via synchronized seed
    - Ensures Teacher and Student see the SAME images
    
    Reference: Unbiased Teacher (NeurIPS 2021), FixMatch (NeurIPS 2020)
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
        weak_hyp: Optional[Dict[str, Any]] = None,
        strong_hyp: Optional[Dict[str, Any]] = None,
        hyp: Optional[Dict[str, Any]] = None,  # Backward compatibility
        augment_labeled: bool = True,
        augment_unlabeled: bool = True,
        sync_seed: int = 42,
    ):
        self.labeled_path = Path(labeled_path)
        self.unlabeled_path = Path(unlabeled_path)
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.labeled_ratio = labeled_ratio
        self.workers = workers
        self.synthetic = synthetic_adapter or DummySyntheticAdapter()
        
        # Separate weak/strong augmentation configs
        self.weak_hyp = weak_hyp or hyp or {}
        self.strong_hyp = strong_hyp or hyp or {}
        
        self.augment_labeled = augment_labeled
        self.augment_unlabeled = augment_unlabeled
        self.sync_seed = sync_seed

        # Datasets
        self.labeled_dataset = None
        self.unlabeled_dataset_weak = None
        self.unlabeled_dataset_strong = None
        
        # Loaders
        self.labeled_loader = None
        self.unlabeled_loader_weak = None
        self.unlabeled_loader_strong = None
        
        # Synchronized samplers
        self._weak_sampler = None
        self._strong_sampler = None
        self._current_epoch = 0

    def setup(self, cfg: Optional[Dict] = None):
        """Initialize datasets and dataloaders with synchronized sampling."""
        cfg = cfg or {}

        # Labeled dataset with strong augmentation
        self.labeled_dataset = YOLODataset(
            img_path=str(self.labeled_path / 'images'),
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            augment=self.augment_labeled,
            hyp=dict_to_namespace(self.strong_hyp),
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix='labeled: ',
            task='detect',
            data=cfg.get('data', {}),
        )

        # Unlabeled dataset - WEAK augmentation (for Teacher)
        self.unlabeled_dataset_weak = UnlabeledYOLODataset(
            img_path=str(self.unlabeled_path / 'images'),
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            augment=False,  # Minimal aug for stable teacher predictions
            hyp=dict_to_namespace(self.weak_hyp),
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix='unlabeled_weak: ',
            task='detect',
            data=cfg.get('data', {}),
        )
        
        # Unlabeled dataset - STRONG augmentation (for Student)
        self.unlabeled_dataset_strong = UnlabeledYOLODataset(
            img_path=str(self.unlabeled_path / 'images'),
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            augment=self.augment_unlabeled,
            hyp=dict_to_namespace(self.strong_hyp),
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix='unlabeled_strong: ',
            task='detect',
            data=cfg.get('data', {}),
        )

        # Batch size allocation
        labeled_bs = max(1, int(self.batch_size * self.labeled_ratio))
        unlabeled_bs = max(1, self.batch_size - labeled_bs)

        # Create synchronized samplers with same seed
        self._weak_sampler = SyncRandomSampler(self.unlabeled_dataset_weak, seed=self.sync_seed)
        self._strong_sampler = SyncRandomSampler(self.unlabeled_dataset_strong, seed=self.sync_seed)

        # Build loaders with synchronized samplers
        self.labeled_loader = self._build_dataloader(
            self.labeled_dataset, labeled_bs, shuffle=True
        )
        self.unlabeled_loader_weak = self._build_dataloader(
            self.unlabeled_dataset_weak, unlabeled_bs, 
            shuffle=False, sampler=self._weak_sampler
        )
        self.unlabeled_loader_strong = self._build_dataloader(
            self.unlabeled_dataset_strong, unlabeled_bs, 
            shuffle=False, sampler=self._strong_sampler
        )
        
        # Initialize iterators
        self._labeled_iter = iter(self.labeled_loader)
        self._unlabeled_weak_iter = iter(self.unlabeled_loader_weak)
        self._unlabeled_strong_iter = iter(self.unlabeled_loader_strong)

    def _build_dataloader(
        self, 
        dataset: Dataset, 
        batch_size: int, 
        shuffle: bool = True,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        """Build dataloader for dataset."""
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.workers,
            pin_memory=False,
            collate_fn=YOLODataset.collate_fn,
        )

    def get_semi_batch(self):
        """
            Trả về trọn bộ 3 thành phần cho 1 bước training:
            1. Labeled Batch (Đã tự động lặp lại nếu hết)
            2. Unlabeled Weak (Cho Teacher)
            3. Unlabeled Strong (Cho Student)
        """
        # 1. Lấy Unlabeled (Weak + Strong) - Buộc phải đồng bộ
        try:
            u_weak = next(self._unlabeled_weak_iter)
            u_strong = next(self._unlabeled_strong_iter)
        except StopIteration:
            # Hết epoch của Unlabeled -> Reset cả hai
            self._current_epoch += 1
            self.set_epoch(self._current_epoch)
            u_weak = next(self._unlabeled_weak_iter)
            u_strong = next(self._unlabeled_strong_iter)

        # 2. Lấy Labeled (Tự động lặp lại - Cycle)
        try:
            labeled = next(self._labeled_iter)
        except StopIteration:
            self._labeled_iter = iter(self.labeled_loader)
            labeled = next(self._labeled_iter)
            
        return labeled, u_weak, u_strong

    def set_epoch(self, epoch: int):
        """
        Set epoch for synchronized sampling.
        Call this at the start of each epoch to ensure weak/strong loaders
        iterate the same images in the same order.
        """
        self._current_epoch = epoch
        if self._weak_sampler:
            self._weak_sampler.set_epoch(epoch)
        if self._strong_sampler:
            self._strong_sampler.set_epoch(epoch)
        
        # Reset iterators for new epoch
        self._unlabeled_weak_iter = iter(self.unlabeled_loader_weak)
        self._unlabeled_strong_iter = iter(self.unlabeled_loader_strong)

    def get_batch(self):
        """Get labeled and unlabeled (strong aug) batch. Backward compatible."""
        try:
            unlabeled_batch = next(self._unlabeled_strong_iter)
        except StopIteration:
            self._unlabeled_strong_iter = iter(self.unlabeled_loader_strong)
            unlabeled_batch = next(self._unlabeled_strong_iter)

        try:
            labeled_batch = next(self._labeled_iter)
        except StopIteration:
            self._labeled_iter = iter(self.labeled_loader)
            labeled_batch = next(self._labeled_iter)
        
        return labeled_batch, unlabeled_batch

    def get_unsup_batch(self):
        """
        Get synchronized weak/strong batches for semi-supervised learning.
        
        Returns:
            unlabeled_weak: Batch for teacher inference (minimal aug)
            unlabeled_strong: Batch for student training (strong aug)
            
        Note: Both batches contain the SAME images due to synchronized sampling.
        """
        try:
            unlabeled_weak = next(self._unlabeled_weak_iter)
        except StopIteration:
            # Sync reset both iterators
            self._current_epoch += 1
            self.set_epoch(self._current_epoch)
            unlabeled_weak = next(self._unlabeled_weak_iter)
            
        try:
            unlabeled_strong = next(self._unlabeled_strong_iter)
        except StopIteration:
            self._unlabeled_strong_iter = iter(self.unlabeled_loader_strong)
            unlabeled_strong = next(self._unlabeled_strong_iter)
        
        return unlabeled_weak, unlabeled_strong

    def close_mosaic(self):
        """Disable mosaic augmentation for final epochs."""
        if hasattr(self.labeled_dataset, 'close_mosaic'):
            self.labeled_dataset.close_mosaic(self.strong_hyp)
        if hasattr(self.unlabeled_dataset_weak, 'close_mosaic'):
            self.unlabeled_dataset_weak.close_mosaic(self.weak_hyp)
        if hasattr(self.unlabeled_dataset_strong, 'close_mosaic'):
            self.unlabeled_dataset_strong.close_mosaic(self.strong_hyp)

    @property
    def num_labeled(self) -> int:
        return len(self.labeled_dataset) if self.labeled_dataset else 0

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled_dataset_strong) if self.unlabeled_dataset_strong else 0

class ThermalAugmentation:
    """
    Augmentation pipeline chuyên biệt cho ảnh nhiệt (Thermal Images) trên GPU.
    Chỉ tác động vào Pixel-level (Intensity/Occlusion), giữ nguyên hình học (Geometric).
    
    Bao gồm:
    1. Gaussian Noise: Giả lập nhiễu hạt sensor.
    2. Random Erasing: Giả lập che khuất/mất tín hiệu.
    """
    def __init__(self, 
                 noise_p=0.5,       # Xác suất thêm nhiễu
                 noise_sigma=0.05,  # Cường độ nhiễu (cho ảnh 0-1)
                 erase_p=0.5,       # Xác suất che phủ
                 erase_scale=(0.02, 0.1), # Diện tích vùng che
                 erase_ratio=(0.3, 3.3),  # Tỷ lệ cạnh vùng che
                 erase_value=0.0):        # Giá trị điền vào (0 = lạnh/đen)
        
        self.noise_p = noise_p
        self.noise_sigma = noise_sigma
        
        # Khởi tạo RandomErasing của Torchvision (đã tối ưu cho GPU)
        self.eraser = T.RandomErasing(
            p=erase_p, 
            scale=erase_scale, 
            ratio=erase_ratio, 
            value=erase_value
        )

    def __call__(self, imgs):
        """
        Input: imgs (Tensor) [Batch, Channel, H, W] - range [0.0, 1.0]
        Output: imgs (Tensor) đã augmentation
        """
        # 1. Gaussian Noise Logic
        # Kiểm tra xác suất (áp dụng cho cả batch để nhanh, hoặc per-image tùy logic)
        if self.noise_p > 0 and torch.rand(1, device=imgs.device) < self.noise_p:
            # Tạo nhiễu chuẩn
            noise = torch.randn_like(imgs) * self.noise_sigma
            imgs = imgs + noise

        # 2. Random Erasing (Torchvision tự xử lý xác suất bên trong)
        imgs = self.eraser(imgs)

        # 3. Safety Clamp (QUAN TRỌNG)
        # Đảm bảo giá trị pixel không văng ra khỏi miền [0, 1] gây NaN loss
        return torch.clamp(imgs, 0.0, 1.0)