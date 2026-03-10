"""分類タスク用MLPの学習スクリプト."""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, fields, MISSING
from typing import Dict, Optional, Tuple, Callable, Any
import typing
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from collections import Counter

from zunda import (
    TouhokuProjectClassificationDataset,
    DATASET_REGISTRY,
    TouhokuDataset,
)
from zunda.callbacks import Callback, CallbackRunner, LoggingCallback, WandbCallback
from zunda.cross_validation import run_cross_validation
from zunda.cv_adapters import TouhokuClassificationCVAdapter, create_empty_test_loader
from model import SimpleMLP
from predictor import Predictor

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """ロギングを設定.

    Args:
        log_dir: ログファイルを保存するディレクトリ
        log_level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）

    Returns:
        設定済みのlogger
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 1実行1ファイル
    log_file = log_dir_path / "train.log"

    # ログフォーマット
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ルートロガーを設定
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # 既存のハンドラーをクリア
    logger.handlers.clear()

    # ファイルハンドラー（ファイルに出力）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # ファイルには全ログを記録
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # コンソールハンドラー（標準出力に出力）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    logger.info(f"ログファイル: {log_file}")

    return logger


@dataclass
class TrainerConfig:
    """学習設定."""
    data_root: str
    dataset: str = "touhoku"
    in_channels: Optional[int] = None  # Noneのとき dataset から推定 (mnist->1, touhoku->3)
    image_size: int = 256
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    hidden_size: int = 512
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: Optional[str] = None  # None のときは save_dir に混在（後方互換）
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # 学習対象クラス（None/空なら除外クラス以外をすべて使用、指定時はそのクラスのみ）
    include_classes: Optional[list] = None
    # Cross Validation設定
    use_cv: bool = False
    cv_folds: int = 5
    current_fold: Optional[int] = None  # CV時の現在のFold番号（W&Bで識別用）
    # テスト評価設定
    test_eval_interval: Optional[int] = None  # テスト評価の実行間隔（エポック数）
    # None: 学習終了時のみ, 1: 毎エポック, N: Nエポックごと
    # クラス不均衡対策設定
    use_class_weights: bool = True  # クラス重み付き損失関数を使用するか
    use_weighted_sampler: bool = False  # Weighted Random Samplerを使用するか
    use_stratified_split: bool = True  # Stratified Splitを使用するか（クラス比率を保ったまま分割）
    class_weight_method: str = "balanced"  # クラス重みの計算方法: "balanced", "inverse", "sqrt"
    use_focal_loss: bool = False  # Focal Lossを使用するか
    focal_loss_alpha: float = 0.25  # Focal Lossのalphaパラメータ
    focal_loss_gamma: float = 2.0  # Focal Lossのgammaパラメータ
    # WANDB設定
    use_wandb: bool = True
    wandb_project: str = "mlp"
    wandb_entity: str = "zunda"
    wandb_run_name: str = None
    wandb_group: str = None  # wandbのgroup名（Cross ValidationでFoldをまとめる際に使用）
    wandb_tags: list = None
    upload_checkpoint: bool = False  # チェックポイントのアップロード（デフォルト: False）


class ClassificationTrainer:
    """分類タスクの学習クラス."""

    def __init__(
        self,
        cfg: TrainerConfig,
        logger: logging.Logger = None,
        callbacks: list[Callback] = None,
        build_transforms_func: Optional[Callable[[TrainerConfig], Tuple[transforms.Compose, transforms.Compose]]] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._set_seed(cfg.seed)
        self.build_transforms_func = build_transforms_func

        # データセットアダプタを取得
        if cfg.dataset not in DATASET_REGISTRY:
            raise ValueError(f"未知のdatasetです: {cfg.dataset}. DATASET_REGISTRY に登録してください。")
        self.dataset_cls = DATASET_REGISTRY[cfg.dataset]

        # Callbackランナーを初期化
        self.runner = CallbackRunner(callbacks or [])

        # データローダーを作成（通常学習時のみ）
        if not self.cfg.use_cv:
            (
                self.train_loader,
                self.val_loader,
                self.test_loader,
                self.class_to_idx,
                self.idx_to_class,
            ) = self.dataset_cls.build_dataloaders(
                cfg=self.cfg,
                logger=self.logger,
                build_transforms_func=self.build_transforms_func,
            )
            # クラス重みを計算（損失関数用）
            self.class_weights = self._calculate_class_weights()
        else:
            if self.cfg.dataset != "touhoku":
                raise ValueError("現在 use_cv がサポートされているのは dataset='touhoku' のみです。")
            # Cross Validation時は、クラス情報だけを取得
            temp_dataset = TouhokuProjectClassificationDataset(
                data_root=self.cfg.data_root,
                transform=None,
                image_extensions=None,
            )
            self.class_to_idx = temp_dataset.get_class_to_idx()
            self.idx_to_class = temp_dataset.get_idx_to_class()
            self.class_weights = None  # CV時は後で計算

        # モデルを作成
        in_ch = cfg.in_channels if cfg.in_channels is not None else self.dataset_cls.get_in_channels(cfg)
        input_size = cfg.image_size * cfg.image_size * in_ch
        num_classes = len(self.class_to_idx)
        self.model = SimpleMLP(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_classes=num_classes
        ).to(self.device)

        # オプティマイザー
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # 損失関数（クラス重みまたはFocal Lossを適用）
        if cfg.use_focal_loss:
            from losses import FocalLoss
            if cfg.use_class_weights and hasattr(self, 'class_weights') and self.class_weights is not None:
                self.criterion = FocalLoss(
                    alpha=cfg.focal_loss_alpha,
                    gamma=cfg.focal_loss_gamma,
                    weight=self.class_weights.to(self.device)
                )
            else:
                self.criterion = FocalLoss(
                    alpha=cfg.focal_loss_alpha,
                    gamma=cfg.focal_loss_gamma
                )
            self.logger.info(f"Focal Lossを使用します (alpha={cfg.focal_loss_alpha}, gamma={cfg.focal_loss_gamma})")
        elif cfg.use_class_weights and hasattr(self, 'class_weights') and self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
            self.logger.info(f"クラス重み付き損失関数を使用します")
            self.logger.info(f"クラス重み: {dict(zip(self.idx_to_class.values(), self.class_weights.cpu().numpy()))}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 学習履歴
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_acc": 0.0,
            "best_epoch": 0,
        }

        # チェックポイント保存ディレクトリ
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # 結果（混同行列など）保存ディレクトリ（未指定時は save_dir）
        self.results_dir = Path(cfg.results_dir) if cfg.results_dir else self.save_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # グローバルステップカウンター
        self.global_step = 0

        self.plots = {}
        self.metrics = {}

        # Callbackのon_initを呼ぶ（初期化処理をCallbackに委譲）
        self.runner.call("on_init", self)


    def _set_seed(self, seed: int):
        """乱数シードを設定."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _calculate_class_weights(self) -> Optional[torch.Tensor]:
        """クラス重みを計算.

        Returns:
            クラス重みのテンソル（デバイスはCPU、後でデバイスに移動）
        """
        if not self.cfg.use_class_weights:
            return None

        # 学習データセットからクラス分布を取得
        train_dataset = self.train_loader.dataset
        labels = []

        # データセットからラベルを収集
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            labels.append(sample['label'])

        # クラスごとのサンプル数をカウント
        class_counts = Counter(labels)
        num_classes = len(self.class_to_idx)

        # クラス名とサンプル数をログに出力
        self.logger.info("学習データのクラス分布:")
        for idx in range(num_classes):
            class_name = self.idx_to_class[idx]
            count = class_counts.get(idx, 0)
            self.logger.info(f"  {class_name}: {count} サンプル")

        # クラス重みを計算
        if self.cfg.class_weight_method == "balanced":
            # sklearnのbalanced方式: n_samples / (n_classes * np.bincount(y))
            total_samples = len(labels)
            weights = []
            for idx in range(num_classes):
                count = class_counts.get(idx, 1)  # 0の場合は1に（ゼロ除算回避）
                weight = total_samples / (num_classes * count)
                weights.append(weight)
        elif self.cfg.class_weight_method == "inverse":
            # 逆数方式: 1 / count
            max_count = max(class_counts.values())
            weights = []
            for idx in range(num_classes):
                count = class_counts.get(idx, 1)
                weight = max_count / count
                weights.append(weight)
        elif self.cfg.class_weight_method == "sqrt":
            # 平方根方式: sqrt(max_count / count)
            max_count = max(class_counts.values())
            weights = []
            for idx in range(num_classes):
                count = class_counts.get(idx, 1)
                weight = np.sqrt(max_count / count)
                weights.append(weight)
        else:
            raise ValueError(f"未知のclass_weight_method: {self.cfg.class_weight_method}")

        # テンソルに変換
        class_weights = torch.tensor(weights, dtype=torch.float32)

        return class_weights

    def _calculate_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """精度を計算.

        全体のaccuracy（正解率）を計算します。
        予測クラス = argmax(logits) と正解ラベルが一致する割合です。
        """
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

    def _log_class_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        split_name: str = "val"
    ) -> Dict[str, float]:
        """各クラスごとのメトリクスを計算してログに出力.

        Args:
            preds: 予測ラベル（numpy配列）
            labels: 正解ラベル（numpy配列）
            split_name: データセット名（"train", "val", "test"）

        Returns:
            クラスごとのメトリクス辞書
        """
        # 各クラスごとのprecision, recall, F1-scoreを計算
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )

        # 全体のmacro平均とweighted平均も計算
        precision_macro = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )[0]
        precision_weighted = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )[0]
        recall_macro = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )[1]
        recall_weighted = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )[1]
        f1_macro = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )[2]
        f1_weighted = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )[2]

        # ログに出力
        self.logger.info(f"{'='*80}")
        self.logger.info(f"{split_name.upper()} - クラスごとのメトリクス")
        self.logger.info(f"{'='*80}")
        self.logger.info(
            f"{'クラス名':<20} {'Precision':<12} {'Recall':<12} "
            f"{'F1-score':<12} {'Support':<10}"
        )
        self.logger.info("-" * 80)

        metrics_dict = {}
        for idx, class_name in self.idx_to_class.items():
            if idx < len(precision):
                p = precision[idx]
                r = recall[idx]
                f = f1[idx]
                s = support[idx]
                self.logger.info(
                    f"{class_name:<20} {p:<12.4f} {r:<12.4f} "
                    f"{f:<12.4f} {s:<10}"
                )
                # W&B用のメトリクスも保存
                metrics_dict[f"{split_name}/class_{class_name}/precision"] = float(p)
                metrics_dict[f"{split_name}/class_{class_name}/recall"] = float(r)
                metrics_dict[f"{split_name}/class_{class_name}/f1"] = float(f)

        self.logger.info("-" * 80)
        self.logger.info(
            f"{'Macro Avg':<20} {precision_macro:<12.4f} {recall_macro:<12.4f} "
            f"{f1_macro:<12.4f}"
        )
        self.logger.info(
            f"{'Weighted Avg':<20} {precision_weighted:<12.4f} "
            f"{recall_weighted:<12.4f} {f1_weighted:<12.4f}"
        )
        self.logger.info(f"{'='*80}\n")

        # 全体平均も追加
        metrics_dict[f"{split_name}/precision_macro"] = float(precision_macro)
        metrics_dict[f"{split_name}/recall_macro"] = float(recall_macro)
        metrics_dict[f"{split_name}/f1_macro"] = float(f1_macro)
        metrics_dict[f"{split_name}/precision_weighted"] = float(precision_weighted)
        metrics_dict[f"{split_name}/recall_weighted"] = float(recall_weighted)
        metrics_dict[f"{split_name}/f1_weighted"] = float(f1_weighted)

        return metrics_dict


    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """1エポックの学習."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            # 順伝播
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # 逆伝播
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # 統計情報を更新
            batch_size = images.size(0)
            batch_loss = loss.item()
            batch_acc = self._calculate_accuracy(logits.detach(), labels)
            total_loss += batch_loss * batch_size
            total_acc += batch_acc * batch_size
            n_samples += batch_size

            # バッチ終了時のコールバック
            self.global_step += 1
            self.metrics.update({
                "step": self.global_step,
                "train/loss": batch_loss,
                "train/acc": batch_acc,
            })
            self.runner.call("on_train_batch_end", self)

        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples
        return avg_loss, avg_acc

    @torch.no_grad()
    def _evaluate(
        self, loader: DataLoader, return_predictions: bool = False
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """評価.

        Args:
            loader: データローダー
            return_predictions: 予測とラベルも返すかどうか

        Returns:
            (avg_loss, avg_acc, all_preds, all_labels)
            return_predictions=Falseの場合: (avg_loss, avg_acc, None, None)
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        all_preds = []
        all_labels = []

        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += self._calculate_accuracy(logits, labels) * batch_size
            n_samples += batch_size

            if return_predictions:
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples

        if return_predictions:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            return avg_loss, avg_acc, all_preds, all_labels
        else:
            return avg_loss, avg_acc, None, None

    def fit(self):
        """学習を実行."""
        best_val_acc = 0.0
        best_epoch = 0
        train_end_called = False

        try:
            # 学習開始
            self.runner.call("on_train_start")

            for epoch in range(1, self.cfg.epochs + 1):
                # エポック開始
                self.runner.call("on_epoch_start", epoch)

                # 学習 & 検証
                train_loss, train_acc = self._train_one_epoch(epoch)
                val_loss, val_acc, val_preds, val_labels = self._evaluate(
                    self.val_loader, return_predictions=True
                )

                # 履歴を更新
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # 各クラスごとのメトリクスを計算・表示（最終エポックまたは5エポックごと）
                if epoch == self.cfg.epochs or epoch % 5 == 0:
                    class_metrics = self._log_class_metrics(
                        val_preds, val_labels, split_name="val"
                    )
                    self.metrics.update(class_metrics)

                # 評価終了時のコールバック（metricsを更新してから呼ぶ）
                self.metrics.update({
                    "epoch": epoch,
                    "eval/loss": val_loss,
                    "eval/acc": val_acc,
                })
                self.runner.call("on_eval_end", self)

                # エポック終了時のコールバック（metricsを更新してから呼ぶ）
                self.metrics.update({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "eval/loss": val_loss,
                    "eval/acc": val_acc,
                })

                # ベストモデルを保存
                ckpt_saved = False
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'class_to_idx': self.class_to_idx,
                        'idx_to_class': self.idx_to_class,
                    }
                    checkpoint_path = self.save_dir / 'best_model.pt'
                    torch.save(checkpoint, checkpoint_path)
                    self.metrics["best_val_acc"] = best_val_acc
                    self.metrics["best_epoch"] = best_epoch
                    ckpt_saved = True
                    # 履歴を更新
                    self.history["best_val_acc"] = best_val_acc
                    self.history["best_epoch"] = best_epoch

                self.runner.call("on_epoch_end", self, checkpoint_saved=ckpt_saved)

                # テストデータで評価（設定された間隔で実行）
                test_interval = self.cfg.test_eval_interval
                if (
                    hasattr(self, 'test_loader')
                    and len(self.test_loader.dataset) > 0
                    and test_interval is not None
                    and test_interval > 0
                    and epoch % int(test_interval) == 0
                ):
                    self.logger.info(f"Epoch {epoch} - テストデータで評価...")
                    test_loss, test_acc, test_preds, test_labels = self._evaluate(
                        self.test_loader, return_predictions=True
                    )
                    self.logger.info(
                        f"Epoch {epoch} - Test Loss: {test_loss:.4f} | "
                        f"Test Acc: {test_acc:.4f}"
                    )

                    # テストデータのクラスごとのメトリクスを表示
                    test_class_metrics = self._log_class_metrics(
                        test_preds, test_labels, split_name="test"
                    )

                    self.metrics.update({
                        "epoch": epoch,
                        "test/loss": test_loss,
                        "test/acc": test_acc,
                    })
                    self.metrics.update(test_class_metrics)
                    self.runner.call("on_eval_end", self)

            # 最終モデルを保存
            checkpoint = {
                'epoch': self.cfg.epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class,
            }
            final_checkpoint_path = self.save_dir / 'final_model.pt'
            torch.save(checkpoint, final_checkpoint_path)
            self.logger.info(f"最終モデルを保存しました: {final_checkpoint_path}")

            # テストデータで評価（学習終了時、test_eval_intervalがNoneの場合のみ）
            # test_eval_intervalが設定されている場合は既にエポック中に評価済み
            if (
                hasattr(self, 'test_loader')
                and len(self.test_loader.dataset) > 0
                and self.cfg.test_eval_interval is None
            ):
                self.logger.info("テストデータで評価（学習終了時）...")
                test_loss, test_acc, test_preds, test_labels = self._evaluate(
                    self.test_loader, return_predictions=True
                )
                self.logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

                # テストデータのクラスごとのメトリクスを表示
                test_class_metrics = self._log_class_metrics(
                    test_preds, test_labels, split_name="test"
                )

                self.metrics.update({
                    "epoch": self.cfg.epochs + 1,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                })
                self.metrics.update(test_class_metrics)
                self.runner.call("on_eval_end", self)

            # 学習終了後、confusion matrixを生成（ベストモデルと最終モデルの両方）

            # Predictorを作成
            predictor = Predictor(
                model=self.model,
                device=self.device,
                class_to_idx=self.class_to_idx,
                idx_to_class=self.idx_to_class,
                logger=self.logger,
            )
            if hasattr(self, 'test_loader') and len(self.test_loader.dataset) > 0:
                self.logger.info("学習終了後、confusion matrixを生成中...")

                # 最終モデルでconfusion matrixを生成
                self.logger.info("最終モデルでconfusion matrixを生成中...")
                checksum = sum(p.sum().item() for p in predictor.model.parameters())
                self.logger.info(f"最終モデルの重みチェックサム: {checksum:.6f}")

                final_cm_paths = {}
                for split_name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
                    _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                    cm_path = predictor.create_confusion_matrix(
                        true_labels, pred_labels, split_name,
                        model_type='final',
                        save_dir=self.results_dir,
                        use_timestamp=False,
                    )
                    final_cm_paths[f'{split_name}_cm'] = str(cm_path)
                    self.plots[f'confusion_matrix/final/{split_name}_cm'] = str(cm_path)

                # ベストモデルでconfusion matrixを生成
                best_cm_paths = {}
                best_checkpoint_path = self.save_dir / 'best_model.pt'
                if best_checkpoint_path.exists():
                    self.logger.info("ベストモデルでconfusion matrixを生成中...")
                    best_checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(best_checkpoint['model_state_dict'])

                    checksum = sum(p.sum().item() for p in predictor.model.parameters())
                    self.logger.info(f"ベストモデルの重みチェックサム: {checksum:.6f}")

                    for split_name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
                        _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                        cm_path = predictor.create_confusion_matrix(
                            true_labels, pred_labels, split_name,
                            model_type='best',
                            save_dir=self.results_dir,
                            use_timestamp=False,
                        )
                        best_cm_paths[f'{split_name}_cm'] = str(cm_path)
                        self.plots[f'confusion_matrix/best/{split_name}_cm'] = str(cm_path)
            else:
                # CVの場合はtrain/valのみ
                # final→bestの順で処理
                self.logger.info("学習終了後、confusion matrixを生成中...")


                # 最終モデルでconfusion matrixを生成
                self.logger.info("最終モデルでconfusion matrixを生成中...")
                final_cm_paths = {}
                for split_name, loader in [('train', self.train_loader), ('val', self.val_loader)]:
                    _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                    cm_path = predictor.create_confusion_matrix(
                        true_labels, pred_labels, split_name,
                        model_type='final',
                        save_dir=self.results_dir,
                        use_timestamp=False,
                    )
                    final_cm_paths[f'{split_name}_cm'] = str(cm_path)
                    self.plots[f'confusion_matrix/final/{split_name}_cm'] = str(cm_path)

                # ベストモデルでconfusion matrixを生成
                best_cm_paths = {}
                best_checkpoint_path = self.save_dir / 'best_model.pt'
                if best_checkpoint_path.exists():
                    self.logger.info("ベストモデルでconfusion matrixを生成中...")
                    best_checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(best_checkpoint['model_state_dict'])

                    for split_name, loader in [('train', self.train_loader), ('val', self.val_loader)]:
                        _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                        cm_path = predictor.create_confusion_matrix(
                            true_labels, pred_labels, split_name,
                            model_type='best',
                            save_dir=self.results_dir,
                            use_timestamp=False,
                        )
                        best_cm_paths[f'{split_name}_cm'] = str(cm_path)
                        self.plots[f'confusion_matrix/best/{split_name}_cm'] = str(cm_path)

            self.metrics.update({
                "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else 0.0,
                "final_train_acc": self.history["train_acc"][-1] if self.history["train_acc"] else 0.0,
                "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
                "final_val_acc": self.history["val_acc"][-1] if self.history["val_acc"] else 0.0,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            })

            train_end_called = True
            self.runner.call("on_train_end", self)

        except Exception as e:
            self.runner.call("on_exception", e)
            raise
        finally:
            if not train_end_called:
                self.runner.call("on_train_end", self)


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    """設定ファイル（JSON）を読み込む. path が None の場合は空の辞書を返す."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _parse_override_value(key: str, raw: str, field_type: type) -> Any:
    """上書き文字列をフィールド型に合わせて変換する."""
    raw = raw.strip()
    if raw.lower() in ("none", "null", ""):
        return None
    if field_type is bool:
        return raw.lower() in ("true", "1", "yes")
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is list or (hasattr(field_type, "__origin__") and getattr(field_type, "__origin__") is list):
        if raw.startswith("[") and raw.endswith("]"):
            return [x.strip() for x in raw[1:-1].split(",") if x.strip()]
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


def parse_overrides(override_list: list, config_fields: dict) -> Dict[str, Any]:
    """-o key=value のリストを辞書に変換する（型を適用）. config_fields は field_name -> type の辞書."""
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"上書きは key=value 形式で指定してください: {item}")
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in config_fields:
            raise ValueError(f"未知の設定キー: {key}")
        result[key] = _parse_override_value(key, value, config_fields[key])
    return result


def config_dict_to_trainer_config(d: Dict[str, Any]) -> TrainerConfig:
    """辞書とデフォルトをマージして TrainerConfig を構築する."""
    defaults = {}
    field_types = {}
    for f in fields(TrainerConfig):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.name == "device":
            defaults[f.name] = "cuda" if torch.cuda.is_available() else "cpu"
        field_types[f.name] = f.type
    # Optional[X] -> X に正規化
    for k, t in list(field_types.items()):
        if getattr(t, "__origin__", None) is typing.Union and hasattr(t, "__args__"):
            args = [a for a in t.__args__ if a is not type(None)]
            if len(args) == 1:
                field_types[k] = args[0]
    field_names = {f.name for f in fields(TrainerConfig)}
    merged = {**defaults, **d}
    if "data_root" not in merged or merged["data_root"] is None or merged["data_root"] == "":
        if merged.get("dataset") == "mnist":
            merged["data_root"] = "data"  # MNISTのダウンロード先デフォルト
        else:
            raise ValueError("data_root を設定ファイルまたは -o data_root=... で指定してください")
    # device の遅延評価
    if merged.get("device") in (None, "auto", ""):
        merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # 型変換（辞書の値が文字列のままの場合）
    for k in list(merged.keys()):
        if k not in field_types:
            continue
        t = field_types[k]
        v = merged[k]
        if v is None:
            continue
        if t is bool and isinstance(v, str):
            merged[k] = v.lower() in ("true", "1", "yes")
        elif t is int and isinstance(v, (str, float)):
            merged[k] = int(v)
        elif t is float and isinstance(v, str):
            merged[k] = float(v)
        elif (t is list or getattr(t, "__origin__", None) is list) and isinstance(v, str):
            merged[k] = [x.strip() for x in v.split(",") if x.strip()]
    return TrainerConfig(**{k: v for k, v in merged.items() if k in field_names})


def parse_args():
    """コマンドライン引数を解析. 設定ファイルと -o key=value による上書きのみ受け付ける."""
    parser = argparse.ArgumentParser(
        description="分類タスク用MLPの学習（設定ファイル + 上書き）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 設定ファイルのみ
  python train.py config.json

  # 設定ファイル + 上書き
  python train.py config.json -o epochs=100 -o lr=0.0005

  # 上書きのみ（data_root は必須）
  python train.py -o data_root=/ws/data/images -o epochs=50
"""
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="設定ファイル（JSON）。省略時は全項目を -o またはデフォルトで指定"
    )
    parser.add_argument(
        "-o", "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="設定の上書き（複数指定可）。例: -o epochs=100 -o use_wandb=false"
    )
    return parser.parse_args()


def build_transforms(cfg: TrainerConfig):
    """Cross Validation 用の transforms 構築関数（現状 Touhoku のみ想定）."""
    # CV は Touhoku 前提なので TouhokuDataset のデフォルト transforms を利用
    return TouhokuDataset.build_default_transforms(cfg)


def main():
    """メイン関数."""
    args = parse_args()

    # 1実行分の出力を outputs/(timestamp)/ に集約
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wandb").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # wandb の保存先を run_dir 内に固定（ログ・チェックポイントと対応が取れる）
    os.environ["WANDB_DIR"] = str(run_dir / "wandb")

    # ログは run_dir/train.log に出力
    logger = setup_logging(str(run_dir), "INFO")
    logger.info(f"出力ルート: {run_dir.resolve()}")

    try:
        config_path = Path(args.config) if args.config else None
        base = load_config(config_path)
        field_types = {f.name: f.type for f in fields(TrainerConfig)}
        for k, t in list(field_types.items()):
            if getattr(t, "__origin__", None) is typing.Union and hasattr(t, "__args__"):
                non_none = [a for a in t.__args__ if a is not type(None)]
                if len(non_none) == 1:
                    field_types[k] = non_none[0]
        overrides = parse_overrides(args.overrides or [], field_types)
        cfg = config_dict_to_trainer_config({**base, **overrides})

        data_root_path = Path(cfg.data_root)
        if not data_root_path.is_absolute():
            tmp = {f.name: getattr(cfg, f.name) for f in fields(TrainerConfig)}
            tmp["data_root"] = str(project_root / cfg.data_root)
            cfg = TrainerConfig(**tmp)

        # 保存先を run_dir 配下に統一
        cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(TrainerConfig)}
        cfg_dict["save_dir"] = str(run_dir / "checkpoints")
        cfg_dict["results_dir"] = str(run_dir / "results")
        cfg = TrainerConfig(**cfg_dict)
        if getattr(cfg, "wandb_run_name", None) in (None, ""):
            cfg.wandb_run_name = f"{cfg.wandb_project}_{cfg.dataset}_{run_id}"

        # Callbackを作成
        callbacks = [
            WandbCallback(cfg, logger=logger),
            LoggingCallback(logger=logger),
        ]

        # Cross Validationを使用する場合
        if cfg.use_cv:
            adapter = TouhokuClassificationCVAdapter()

            cv_results = run_cross_validation(
                cfg=cfg,
                trainer_class=ClassificationTrainer,
                logger=logger,
                adapter=adapter,
                create_empty_test_loader=create_empty_test_loader,
                build_transforms_func=build_transforms,
            )
            logger.info("Cross Validationが完了しました")
        else:
            # 通常の学習
            trainer = ClassificationTrainer(
                cfg,
                logger=logger,
                callbacks=callbacks,
                build_transforms_func=build_transforms,
            )
            trainer.fit()
    except Exception as e:
        logger.exception("学習中にエラーが発生しました:")
        raise


if __name__ == "__main__":
    main()
