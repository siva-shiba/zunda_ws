"""Cross Validation用の汎用モジュール.

モデルに依存せず再利用可能なCVロジックを提供します.
データセット固有の処理はCVDatasetAdapterを実装して注入します.
"""

import json
import logging
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class CVDatasetAdapter(Protocol):
    """Cross Validation用のデータセットアダプタプロトコル.

    データセット固有の処理を抽象化し、run_cross_validationから利用します.
    任意のデータセット/モデルでCVを行うにはこのプロトコルを実装してください.
    """

    def create_dataset(self, cfg: Any) -> Dataset:
        """CV用のデータセットを作成."""
        ...

    def get_labels(self, dataset: Dataset, indices: List[int]) -> np.ndarray:
        """指定インデックスに対するラベル配列を取得（StratifiedKFold用）."""
        ...

    def get_train_val_indices(
        self, dataset: Dataset
    ) -> Tuple[List[int], List[int]]:
        """Train/Val用インデックスとTest用インデックスに分割."""
        ...

    def get_class_mappings(
        self, dataset: Dataset
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """(class_to_idx, idx_to_class)を取得."""
        ...

    def build_transforms(
        self, cfg: Any
    ) -> Tuple[Optional[transforms.Compose], Optional[transforms.Compose]]:
        """(train_transform, val_transform)を構築."""
        ...

    def create_fold_callbacks(
        self, fold_cfg: Any, logger: logging.Logger
    ) -> List[Any]:
        """各Fold用のコールバックリストを作成（WandbCallback等）."""
        ...


def create_cv_dataloaders(
    dataset: Dataset,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int,
    num_workers: int,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
    image_key: str = "image",
) -> Tuple[DataLoader, DataLoader]:
    """Cross Validation用のDataLoaderを作成.

    Args:
        dataset: データセット（Subsetでラップ可能）
        train_indices: 学習用のインデックス
        val_indices: 検証用のインデックス
        batch_size: バッチサイズ
        num_workers: ワーカー数
        train_transform: 学習用の画像変換
        val_transform: 検証用の画像変換
        image_key: サンプル辞書内の画像キー（デフォルト: 'image'）

    Returns:
        (train_loader, val_loader)
    """
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    class TransformSubset:
        def __init__(self, subset: Subset, transform: Any, key: str):
            self.subset = subset
            self.transform = transform
            self.image_key = key

        def __len__(self) -> int:
            return len(self.subset)

        def __getitem__(self, idx: int) -> dict:
            sample = self.subset[idx]
            if (
                isinstance(sample, dict)
                and self.transform
                and self.image_key in sample
            ):
                sample = sample.copy()
                sample[self.image_key] = self.transform(sample[self.image_key])
            return sample

    train_dataset = TransformSubset(train_subset, train_transform, image_key)
    val_dataset = TransformSubset(val_subset, val_transform, image_key)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_cross_validation(
    cfg: Any,
    trainer_class: type,
    logger: logging.Logger,
    adapter: CVDatasetAdapter,
    create_empty_test_loader: Optional[Callable[[Any], DataLoader]] = None,
) -> Dict[str, Any]:
    """Cross Validationを実行.

    Args:
        cfg: TrainerConfig（cv_folds, save_dir, seed等の属性が必要）
        trainer_class: Trainerクラス（fit()メソッドを持つ）
        logger: ロガー
        adapter: データセット固有の処理を提供するアダプタ
        create_empty_test_loader: 空のtest_loaderを作成する関数（任意）
            引数はcfg、CV時はtest_loaderを使用しない

    Returns:
        Cross Validationの結果辞書
    """
    logger.info("=" * 80)
    logger.info("Cross Validationを開始します")
    logger.info(f"Fold数: {cfg.cv_folds}")

    # wandb group名を生成（指定されていない場合）
    if hasattr(cfg, "wandb_group") and cfg.wandb_group:
        wandb_group = cfg.wandb_group
    else:
        if getattr(cfg, "wandb_run_name", None):
            wandb_group = f"{cfg.wandb_run_name}_cv"
        else:
            wandb_group = f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"WANDB Group: {wandb_group}")
    logger.info("=" * 80)

    # データセットを作成
    full_dataset = adapter.create_dataset(cfg)
    class_to_idx, idx_to_class = adapter.get_class_mappings(full_dataset)

    train_val_indices, test_indices = adapter.get_train_val_indices(full_dataset)
    train_val_labels = adapter.get_labels(full_dataset, train_val_indices)

    logger.info(f"Train/Val用データ: {len(train_val_indices)} サンプル")
    logger.info(f"Test用データ: {len(test_indices)} サンプル")

    skf = StratifiedKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed
    )

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(train_val_indices, train_val_labels), start=1
    ):
        logger.info("=" * 80)
        logger.info(f"Fold {fold_idx}/{cfg.cv_folds}")
        logger.info("=" * 80)

        fold_save_dir = Path(cfg.save_dir) / f"fold_{fold_idx}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)

        # 設定をコピーしてFold用に変更
        fold_cfg_dict = {}
        for field in fields(cfg):
            fold_cfg_dict[field.name] = getattr(cfg, field.name)

        fold_cfg_dict["save_dir"] = str(fold_save_dir)
        fold_cfg_dict["seed"] = cfg.seed + fold_idx
        fold_cfg_dict["train_ratio"] = 0.0
        fold_cfg_dict["val_ratio"] = 0.0
        fold_cfg_dict["test_ratio"] = 0.0
        fold_cfg_dict["use_cv"] = True
        fold_cfg_dict["current_fold"] = fold_idx
        wandb_run_name = getattr(cfg, "wandb_run_name", None)
        fold_cfg_dict["wandb_run_name"] = (
            f"{wandb_run_name}_fold{fold_idx}"
            if wandb_run_name
            else f"fold_{fold_idx}"
        )
        fold_cfg_dict["wandb_group"] = wandb_group

        fold_cfg = type(cfg)(**fold_cfg_dict)

        fold_callbacks = adapter.create_fold_callbacks(fold_cfg, logger)
        train_transform, val_transform = adapter.build_transforms(cfg)

        train_indices_actual = [train_val_indices[i] for i in train_idx]
        val_indices_actual = [train_val_indices[i] for i in val_idx]

        train_loader, val_loader = create_cv_dataloaders(
            dataset=full_dataset,
            train_indices=train_indices_actual,
            val_indices=val_indices_actual,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        logger.info(f"Fold {fold_idx} - Train: {n_train}, Val: {n_val}")

        trainer = trainer_class(fold_cfg, logger=logger, callbacks=fold_callbacks)
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader

        if create_empty_test_loader is not None:
            trainer.test_loader = create_empty_test_loader(cfg)
        else:
            trainer.test_loader = DataLoader(
                Subset(full_dataset, []),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

        trainer.class_to_idx = class_to_idx
        trainer.idx_to_class = idx_to_class

        trainer.fit()

        val_acc = trainer.history["val_acc"]
        train_acc = trainer.history["train_acc"]
        fold_result = {
            "fold": fold_idx,
            "best_val_acc": trainer.history.get("best_val_acc", 0.0),
            "best_epoch": trainer.history.get("best_epoch", 0),
            "final_val_acc": val_acc[-1] if val_acc else 0.0,
            "final_train_acc": train_acc[-1] if train_acc else 0.0,
        }
        fold_results.append(fold_result)

        logger.info(
            f"Fold {fold_idx} 完了 - Best Val Acc: {fold_result['best_val_acc']:.4f}"
        )

    cv_results = {
        "cv_folds": cfg.cv_folds,
        "fold_results": fold_results,
        "mean_best_val_acc": np.mean([r["best_val_acc"] for r in fold_results]),
        "std_best_val_acc": np.std([r["best_val_acc"] for r in fold_results]),
        "mean_final_val_acc": np.mean([r["final_val_acc"] for r in fold_results]),
        "std_final_val_acc": np.std([r["final_val_acc"] for r in fold_results]),
        "mean_final_train_acc": np.mean(
            [r["final_train_acc"] for r in fold_results]
        ),
        "std_final_train_acc": np.std(
            [r["final_train_acc"] for r in fold_results]
        ),
    }

    cv_results_path = Path(cfg.save_dir) / "cv_results.json"
    with open(cv_results_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 80)
    logger.info("Cross Validation結果")
    logger.info("=" * 80)
    mean_best = cv_results["mean_best_val_acc"]
    std_best = cv_results["std_best_val_acc"]
    mean_final = cv_results["mean_final_val_acc"]
    std_final = cv_results["std_final_val_acc"]
    mean_train = cv_results["mean_final_train_acc"]
    std_train = cv_results["std_final_train_acc"]
    logger.info(f"Mean Best Val Acc: {mean_best:.4f} ± {std_best:.4f}")
    logger.info(f"Mean Final Val Acc: {mean_final:.4f} ± {std_final:.4f}")
    logger.info(f"Mean Final Train Acc: {mean_train:.4f} ± {std_train:.4f}")
    logger.info("=" * 80)
    logger.info(f"結果を保存: {cv_results_path}")

    return cv_results
