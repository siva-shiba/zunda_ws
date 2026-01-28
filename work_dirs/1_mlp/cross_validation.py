"""Cross Validation用のヘルパー関数とクラス."""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zunda import TouhokuProjectClassificationDataset


def create_cv_dataloaders(
    dataset: TouhokuProjectClassificationDataset,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int,
    num_workers: int,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Cross Validation用のDataLoaderを作成.

    Args:
        dataset: データセット
        train_indices: 学習用のインデックス
        val_indices: 検証用のインデックス
        batch_size: バッチサイズ
        num_workers: ワーカー数
        train_transform: 学習用の画像変換
        val_transform: 検証用の画像変換

    Returns:
        (train_loader, val_loader)
    """
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # 変換を適用するラッパー
    class TransformSubset:
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            sample = self.subset[idx]
            if self.transform and 'image' in sample:
                sample = sample.copy()
                sample['image'] = self.transform(sample['image'])
            return sample

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)

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


def get_labels_from_indices(dataset: TouhokuProjectClassificationDataset, indices: List[int]) -> np.ndarray:
    """指定されたインデックスからラベルを取得.

    Args:
        dataset: データセット
        indices: インデックスのリスト

    Returns:
        ラベルの配列
    """
    labels = []
    for i in indices:
        _, _, label_idx = dataset.samples[i]
        labels.append(label_idx)
    return np.array(labels)


def run_cross_validation(
    cfg,
    trainer_class,
    logger: logging.Logger,
    callbacks: Optional[List] = None,
) -> Dict:
    """Cross Validationを実行.

    Args:
        cfg: TrainerConfig
        trainer_class: Trainerクラス
        logger: ロガー
        callbacks: コールバックのリスト

    Returns:
        Cross Validationの結果辞書
    """
    logger.info("="*80)
    logger.info("Cross Validationを開始します")
    logger.info(f"Fold数: {cfg.cv_folds}")

    # wandb group名を生成（指定されていない場合）
    if hasattr(cfg, 'wandb_group') and cfg.wandb_group:
        wandb_group = cfg.wandb_group
    else:
        # 自動生成: run_nameがある場合はそれを使用、ない場合はタイムスタンプ
        if cfg.wandb_run_name:
            wandb_group = f"{cfg.wandb_run_name}_cv"
        else:
            from datetime import datetime
            wandb_group = f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"WANDB Group: {wandb_group}")
    logger.info("="*80)

    # 全データセットを作成（testセットを除く）
    val_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    # まず全データセットを作成してクラス情報を取得
    full_dataset = TouhokuProjectClassificationDataset(
        data_root=cfg.data_root,
        transform=None,  # 後で適用
        image_extensions=None,
    )
    class_to_idx = full_dataset.get_class_to_idx()
    idx_to_class = full_dataset.get_idx_to_class()

    # testセットを除外（exclude_from_train_valを使用）
    # 全データからunknownクラスを除外してtrain/val用データセットを作成
    train_val_indices = []
    test_indices = []
    unknown_idx = class_to_idx.get('unknown', None)

    for i in range(len(full_dataset)):
        _, _, label_idx = full_dataset.samples[i]
        if unknown_idx is not None and label_idx == unknown_idx:
            test_indices.append(i)
        else:
            train_val_indices.append(i)

    train_val_labels = get_labels_from_indices(full_dataset, train_val_indices)

    logger.info(f"Train/Val用データ: {len(train_val_indices)} サンプル")
    logger.info(f"Test用データ: {len(test_indices)} サンプル")

    # StratifiedKFoldで分割
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    # 各Foldの結果を保存
    fold_results = []
    fold_save_dirs = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_val_indices, train_val_labels), start=1):
        logger.info("="*80)
        logger.info(f"Fold {fold_idx}/{cfg.cv_folds}")
        logger.info("="*80)

        # Fold用の保存ディレクトリ
        fold_save_dir = Path(cfg.save_dir) / f"fold_{fold_idx}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)
        fold_save_dirs.append(fold_save_dir)

        # 設定をコピーしてFold用に変更
        # TrainerConfigは引数として渡されたcfgと同じ型なので、そのまま使用
        # dataclassのcopyを作成
        from dataclasses import fields
        fold_cfg_dict = {}
        for field in fields(cfg):
            fold_cfg_dict[field.name] = getattr(cfg, field.name)

        # Fold用に変更
        fold_cfg_dict['save_dir'] = str(fold_save_dir)
        fold_cfg_dict['seed'] = cfg.seed + fold_idx
        # CVではtrain/val/test比率は使わないので0にするが、
        # Trainer側ではuse_cv=Trueのときに_build_dataloadersを呼ばないようにしている
        fold_cfg_dict['train_ratio'] = 0.0
        fold_cfg_dict['val_ratio'] = 0.0
        fold_cfg_dict['test_ratio'] = 0.0
        # Fold内のTrainerには「CVモード」として振る舞ってもらう
        fold_cfg_dict['use_cv'] = True
        # wandb run名とgroupを設定
        fold_cfg_dict['wandb_run_name'] = f"{cfg.wandb_run_name}_fold{fold_idx}" if cfg.wandb_run_name else f"fold_{fold_idx}"
        fold_cfg_dict['wandb_group'] = wandb_group  # 全Foldで同じgroup名を使用

        # TrainerConfigの型を取得してインスタンスを作成
        fold_cfg = type(cfg)(**fold_cfg_dict)

        # DataLoaderを作成
        train_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        train_loader, val_loader = create_cv_dataloaders(
            dataset=full_dataset,
            train_indices=[train_val_indices[i] for i in train_indices],
            val_indices=[train_val_indices[i] for i in val_indices],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        logger.info(f"Fold {fold_idx} - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

        # Trainerを作成（カスタムデータローダーを使用）
        trainer = trainer_class(fold_cfg, logger=logger, callbacks=callbacks)
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        # test_loaderは空のDataLoaderを作成（CVでは使用しない）
        trainer.test_loader = DataLoader(
            Subset(full_dataset, []),  # 空のSubset
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        trainer.class_to_idx = class_to_idx
        trainer.idx_to_class = idx_to_class

        # 学習を実行
        trainer.fit()

        # Foldの結果を保存
        fold_result = {
            'fold': fold_idx,
            'best_val_acc': trainer.history.get('best_val_acc', 0.0),
            'best_epoch': trainer.history.get('best_epoch', 0),
            'final_val_acc': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0.0,
            'final_train_acc': trainer.history['train_acc'][-1] if trainer.history['train_acc'] else 0.0,
        }
        fold_results.append(fold_result)

        logger.info(f"Fold {fold_idx} 完了 - Best Val Acc: {fold_result['best_val_acc']:.4f}")

    # Cross Validation結果を集計
    cv_results = {
        'cv_folds': cfg.cv_folds,
        'fold_results': fold_results,
        'mean_best_val_acc': np.mean([r['best_val_acc'] for r in fold_results]),
        'std_best_val_acc': np.std([r['best_val_acc'] for r in fold_results]),
        'mean_final_val_acc': np.mean([r['final_val_acc'] for r in fold_results]),
        'std_final_val_acc': np.std([r['final_val_acc'] for r in fold_results]),
        'mean_final_train_acc': np.mean([r['final_train_acc'] for r in fold_results]),
        'std_final_train_acc': np.std([r['final_train_acc'] for r in fold_results]),
    }

    # 結果を保存
    cv_results_path = Path(cfg.save_dir) / f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(cv_results_path, 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)

    logger.info("="*80)
    logger.info("Cross Validation結果")
    logger.info("="*80)
    logger.info(f"Mean Best Val Acc: {cv_results['mean_best_val_acc']:.4f} ± {cv_results['std_best_val_acc']:.4f}")
    logger.info(f"Mean Final Val Acc: {cv_results['mean_final_val_acc']:.4f} ± {cv_results['std_final_val_acc']:.4f}")
    logger.info(f"Mean Final Train Acc: {cv_results['mean_final_train_acc']:.4f} ± {cv_results['std_final_train_acc']:.4f}")
    logger.info("="*80)
    logger.info(f"結果を保存: {cv_results_path}")

    return cv_results
