"""推論用: チェックポイントに保存した学習設定から DataLoader を再構築する."""

from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset_adapters import DATASET_REGISTRY

# train_config を保存していない古いチェックポイント用（当時の inference.py が使っていた設定値）
LEGACY_TRAIN_CONFIG: Dict[str, Any] = {
    "dataset": "touhoku",
    "data_root": "data/touhoku_project_images",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "use_stratified_split": True,
    "include_classes": None,
}

TransformsFunc = Callable[[Any], Tuple[transforms.Compose, transforms.Compose]]


class _RemapLabelDataset(Dataset):
    """batch['label'] をチェックポイント側のクラス番号に読み替えるラッパー."""

    def __init__(self, dataset: Dataset, old_to_new: Dict[int, int]):
        self._dataset = dataset
        self._old_to_new = old_to_new

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self._dataset[idx])
        sample["label"] = self._old_to_new[int(sample["label"])]
        return sample


def _align_labels(
    loaders: Tuple[DataLoader, ...],
    data_class_to_idx: Dict[str, int],
    target_class_to_idx: Dict[str, int],
    logger: Any,
) -> Tuple[DataLoader, ...]:
    """データセットのクラス番号をチェックポイントの番号に合わせる."""
    if data_class_to_idx == target_class_to_idx:
        return loaders
    if set(data_class_to_idx) != set(target_class_to_idx):
        raise ValueError(
            "データセットのクラスがチェックポイントと一致しません: "
            f"data={sorted(data_class_to_idx)}, checkpoint={sorted(target_class_to_idx)}"
        )
    logger.warning("クラス番号がチェックポイントと異なるため、チェックポイント側の番号に読み替えます")
    old_to_new = {i: target_class_to_idx[name] for name, i in data_class_to_idx.items()}
    return tuple(
        DataLoader(
            _RemapLabelDataset(dl.dataset, old_to_new),
            batch_size=dl.batch_size,
            shuffle=False,
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
        )
        for dl in loaders
    )


def get_train_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """チェックポイントから学習設定を取り出す（古い形式は LEGACY_TRAIN_CONFIG で補う）."""
    saved = checkpoint.get("train_config")
    if saved is None:
        # 当時はファイルの走査順がファイルシステム依存だったため、分割は学習時と一致しない可能性がある
        cfg = dict(LEGACY_TRAIN_CONFIG)
        mc = checkpoint.get("model_config") or {}
        if "image_size" in mc:
            cfg["image_size"] = mc["image_size"]
        return cfg
    return dict(saved)


def build_eval_dataloaders(
    train_config: Dict[str, Any],
    logger: Any,
    build_transforms_func: Optional[TransformsFunc] = None,
    *,
    data_root: Optional[str] = None,
    image_size: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: Optional[int] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """学習時と同じデータセット・分割で、評価用 (augmentation なし) の DataLoader を作る.

    Args:
        train_config: 学習時の設定（get_train_config の戻り値）
        logger: ロガー
        build_transforms_func: 学習時に使った transforms 関数 (train, val) を返す.
            None ならデータセットアダプタのデフォルト。val 側をすべての split に使う。
        data_root, image_size, seed: 指定したときだけ学習時の値を上書きする
        class_to_idx: チェックポイントのクラス番号。指定するとラベルをこの番号に揃える
    """
    overrides = {
        "data_root": data_root,
        "image_size": image_size,
        "seed": seed,
    }
    cfg_dict = {**train_config, **{k: v for k, v in overrides.items() if v is not None}}
    cfg_dict.update(
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=False,  # 推論では再サンプリングしない
    )
    cfg_dict.setdefault("seed", 42)
    cfg = SimpleNamespace(**cfg_dict)

    dataset_name = cfg_dict.get("dataset", "touhoku")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知の dataset です: {dataset_name}. 利用可能: {sorted(DATASET_REGISTRY)}"
        )
    adapter = DATASET_REGISTRY[dataset_name]
    logger.info(f"データセット: {dataset_name} (data_root={cfg.data_root})")

    base_func = build_transforms_func or adapter.build_default_transforms

    def eval_only_transforms(c: Any) -> Tuple[transforms.Compose, transforms.Compose]:
        _, val_tf = base_func(c)
        return val_tf, val_tf

    train_loader, val_loader, test_loader, data_class_to_idx, data_idx_to_class = (
        adapter.build_dataloaders(cfg, logger, build_transforms_func=eval_only_transforms)
    )
    if class_to_idx is None:
        return train_loader, val_loader, test_loader, data_class_to_idx, data_idx_to_class

    train_loader, val_loader, test_loader = _align_labels(
        (train_loader, val_loader, test_loader), data_class_to_idx, class_to_idx, logger
    )
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    return train_loader, val_loader, test_loader, dict(class_to_idx), idx_to_class
