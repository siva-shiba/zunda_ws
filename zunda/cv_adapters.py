"""Cross Validation用のデータセットアダプタ実装."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .callbacks import LoggingCallback, WandbCallback
from .classification import TouhokuProjectClassificationDataset


class TouhokuClassificationCVAdapter:
    """TouhokuProjectClassificationDataset用のCVアダプタ.

    分類タスク（東北ずん子project画像）向けのCross Validationを実現します.
    """

    def create_dataset(self, cfg: Any) -> TouhokuProjectClassificationDataset:
        """CV用のデータセットを作成."""
        return TouhokuProjectClassificationDataset(
            data_root=cfg.data_root,
            transform=None,
            image_extensions=None,
        )

    def get_labels(
        self,
        dataset: TouhokuProjectClassificationDataset,
        indices: List[int],
    ) -> np.ndarray:
        """指定インデックスに対するラベル配列を取得."""
        labels = []
        for i in indices:
            _, _, label_idx = dataset.samples[i]
            labels.append(label_idx)
        return np.array(labels)

    def get_train_val_indices(
        self,
        dataset: TouhokuProjectClassificationDataset,
    ) -> Tuple[List[int], List[int]]:
        """Train/Val用とTest用（unknownクラス）に分割."""
        train_val_indices = []
        test_indices = []
        unknown_idx = dataset.class_to_idx.get("unknown", None)

        for i in range(len(dataset)):
            _, _, label_idx = dataset.samples[i]
            if unknown_idx is not None and label_idx == unknown_idx:
                test_indices.append(i)
            else:
                train_val_indices.append(i)

        return train_val_indices, test_indices

    def get_class_mappings(
        self,
        dataset: TouhokuProjectClassificationDataset,
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """(class_to_idx, idx_to_class)を取得."""
        return dataset.get_class_to_idx(), dataset.get_idx_to_class()

    def build_transforms(
        self, cfg: Any
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        """(train_transform, val_transform)を構築."""
        train_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ])
        return train_transform, val_transform

    def create_fold_callbacks(
        self, fold_cfg: Any, logger: logging.Logger
    ) -> List[Any]:
        """各Fold用のコールバックを作成."""
        return [
            WandbCallback(fold_cfg, logger=logger),
            LoggingCallback(logger=logger),
        ]


def create_empty_test_loader(cfg: Any) -> DataLoader:
    """CV用の空のtest_loaderを作成（TouhokuProjectClassificationDataset用）."""
    dataset = TouhokuProjectClassificationDataset(
        data_root=cfg.data_root,
        transform=None,
        image_extensions=None,
    )
    return DataLoader(
        Subset(dataset, []),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
