"""データセットアダプタ: オリジナル・公開データセットを学習パイプラインに差し替えるための雛形と実装."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Tuple,
    Type,
)

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .classification import TouhokuProjectClassificationDataset


class DatasetAdapterConfig(Protocol):
    """アダプタが参照する設定のプロトコル. TrainerConfig などが満たす想定."""

    data_root: str
    image_size: int
    batch_size: int
    num_workers: int
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    use_stratified_split: bool
    use_weighted_sampler: bool


# 戻り値の型（build_dataloaders 用）
DataloadersResult = Tuple[
    DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]
]

# レジストリ（デコレータで自動登録）
DATASET_REGISTRY: Dict[str, Type["BaseDatasetAdapter"]] = {}


def register_dataset(name: str):
    """データセットアダプタを DATASET_REGISTRY に登録するデコレータ.

    PyTorch / Detectron2 などでよく使うパターン。クラス定義時に自動で登録される。

    Example:
        @register_dataset("mnist")
        class MNISTDataset(BaseDatasetAdapter):
            ...
    """
    def decorator(cls: Type["BaseDatasetAdapter"]) -> Type["BaseDatasetAdapter"]:
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


class BaseDatasetAdapter(ABC):
    """データセットアダプタの雛形.

    継承して get_in_channels / build_default_transforms / build_dataloaders
    を実装する。
    """

    name: str = ""

    @staticmethod
    @abstractmethod
    def get_in_channels(cfg: DatasetAdapterConfig) -> int:
        """入力画像のチャネル数（1=グレー, 3=RGB）を返す."""
        ...

    @staticmethod
    @abstractmethod
    def build_default_transforms(
        cfg: DatasetAdapterConfig,
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        """(train_transform, val_transform) を返す."""
        ...

    @classmethod
    @abstractmethod
    def build_dataloaders(
        cls,
        cfg: DatasetAdapterConfig,
        logger: Any,
        build_transforms_func: Optional[
            Callable[
                [DatasetAdapterConfig],
                Tuple[transforms.Compose, transforms.Compose],
            ]
        ] = None,
    ) -> DataloadersResult:
        """train/val/test の DataLoader と class_to_idx, idx_to_class を返す."""
        ...


# ---------------------------------------------------------------------------
# 公開データセット用: (image, label) を {"image", "label"} に揃えるラッパー
# ---------------------------------------------------------------------------


class ImageLabelDictDataset(Dataset):
    """(image, label) を返す Dataset を batch['image'], batch['label'] にするラッパー."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, label = self._dataset[idx]
        return {"image": img, "label": label}


def _build_split_dataloaders(
    cfg: DatasetAdapterConfig,
    logger: Any,
    train_full: Dataset,
    val_full: Dataset,
    test_dataset: Dataset,
    classes: list,
) -> DataloadersResult:
    """公開データセットの train を train/val に分割し DataLoader を作る.

    train_full と val_full は同じデータを別の transform で読み込んだもの
    (val に学習用の augmentation がかからないようにするため).
    """
    n_total = len(train_full)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val
    # random_split と同じ並びにして、以前の MNIST 実験と同じ分割を再現する
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_dataset = ImageLabelDictDataset(Subset(train_full, train_idx))
    val_dataset = ImageLabelDictDataset(Subset(val_full, val_idx))
    test_dataset = ImageLabelDictDataset(test_dataset)

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    train_loader = _loader(train_dataset, True)
    val_loader = _loader(val_dataset, False)
    test_loader = _loader(test_dataset, False)

    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_class = {i: name for i, name in enumerate(classes)}

    logger.info(f"学習データ: {len(train_dataset)} サンプル")
    logger.info(f"検証データ: {len(val_dataset)} サンプル")
    logger.info(f"テストデータ: {len(test_dataset)} サンプル")

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class


# ---------------------------------------------------------------------------
# Touhoku（オリジナル）アダプタ
# ---------------------------------------------------------------------------


@register_dataset("touhoku")
class TouhokuDataset(BaseDatasetAdapter):
    """東北プロジェクト画像データセット用アダプタ."""

    name = "touhoku"

    @staticmethod
    def get_in_channels(cfg: DatasetAdapterConfig) -> int:
        return 3

    @staticmethod
    def build_default_transforms(
        cfg: DatasetAdapterConfig,
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        train_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ])
        return train_transform, val_transform

    @classmethod
    def build_dataloaders(
        cls,
        cfg: DatasetAdapterConfig,
        logger: Any,
        build_transforms_func: Optional[
            Callable[
                [DatasetAdapterConfig],
                Tuple[transforms.Compose, transforms.Compose],
            ]
        ] = None,
    ) -> DataloadersResult:
        if build_transforms_func is not None:
            train_transform, val_transform = build_transforms_func(cfg)
        else:
            train_transform, val_transform = cls.build_default_transforms(cfg)

        create_dl = TouhokuProjectClassificationDataset
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class = (
            create_dl.create_classification_train_val_test_dataloaders(
                data_root=cfg.data_root,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
                batch_size=cfg.batch_size,
                shuffle_train=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=val_transform,
                random_seed=cfg.seed,
                exclude_class=getattr(cfg, "exclude_class", ["unknown"]),
                include_class=getattr(cfg, "include_classes", None),
                use_stratified_split=cfg.use_stratified_split,
                use_weighted_sampler=cfg.use_weighted_sampler,
            )
        )

        logger.info(f"学習データ: {len(train_loader.dataset)} サンプル")
        logger.info(f"検証データ: {len(val_loader.dataset)} サンプル")
        logger.info(f"テストデータ: {len(test_loader.dataset)} サンプル")

        return train_loader, val_loader, test_loader, class_to_idx, idx_to_class


# ---------------------------------------------------------------------------
# MNIST アダプタ
# ---------------------------------------------------------------------------


@register_dataset("mnist")
class MNISTDataset(BaseDatasetAdapter):
    """MNIST 用アダプタ."""

    name = "mnist"

    @staticmethod
    def get_in_channels(cfg: DatasetAdapterConfig) -> int:
        return 1

    @staticmethod
    def build_default_transforms(
        cfg: DatasetAdapterConfig,
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        return train_transform, val_transform

    @classmethod
    def build_dataloaders(
        cls,
        cfg: DatasetAdapterConfig,
        logger: Any,
        build_transforms_func: Optional[
            Callable[
                [DatasetAdapterConfig],
                Tuple[transforms.Compose, transforms.Compose],
            ]
        ] = None,
    ) -> DataloadersResult:
        from torchvision.datasets import MNIST

        if build_transforms_func is not None:
            train_transform, val_transform = build_transforms_func(cfg)
        else:
            train_transform, val_transform = cls.build_default_transforms(cfg)

        root = Path(cfg.data_root)
        root.mkdir(parents=True, exist_ok=True)
        train_full = MNIST(root=str(root), train=True, download=True, transform=train_transform)
        val_full = MNIST(root=str(root), train=True, download=False, transform=val_transform)
        test_ds = MNIST(root=str(root), train=False, download=True, transform=val_transform)

        classes = [str(i) for i in range(10)]
        return _build_split_dataloaders(cfg, logger, train_full, val_full, test_ds, classes)


# ---------------------------------------------------------------------------
# CIFAR-10 アダプタ
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@register_dataset("cifar10")
class CIFAR10Dataset(BaseDatasetAdapter):
    """CIFAR-10 用アダプタ.

    ずんだ画像での結果と比較するためのベンチマーク用。
    元画像は 32x32 だが cfg.image_size にリサイズするので、
    ImageNet 系の深いモデル (224) にもそのまま使える。
    """

    name = "cifar10"

    @staticmethod
    def get_in_channels(cfg: DatasetAdapterConfig) -> int:
        return 3

    @staticmethod
    def build_default_transforms(
        cfg: DatasetAdapterConfig,
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        train_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(cfg.image_size, padding=cfg.image_size // 8),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        return train_transform, val_transform

    @classmethod
    def build_dataloaders(
        cls,
        cfg: DatasetAdapterConfig,
        logger: Any,
        build_transforms_func: Optional[
            Callable[
                [DatasetAdapterConfig],
                Tuple[transforms.Compose, transforms.Compose],
            ]
        ] = None,
    ) -> DataloadersResult:
        from torchvision.datasets import CIFAR10

        if build_transforms_func is not None:
            train_transform, val_transform = build_transforms_func(cfg)
        else:
            train_transform, val_transform = cls.build_default_transforms(cfg)

        root = Path(cfg.data_root)
        root.mkdir(parents=True, exist_ok=True)
        train_full = CIFAR10(root=str(root), train=True, download=True, transform=train_transform)
        val_full = CIFAR10(root=str(root), train=True, download=False, transform=val_transform)
        test_ds = CIFAR10(root=str(root), train=False, download=True, transform=val_transform)

        return _build_split_dataloaders(cfg, logger, train_full, val_full, test_ds, list(train_full.classes))
