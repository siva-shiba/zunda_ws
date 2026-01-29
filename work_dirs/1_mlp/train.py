"""分類タスク用MLPの学習スクリプト."""

import sys
import os
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zunda import TouhokuProjectClassificationDataset
from zunda.callbacks import Callback, CallbackRunner, LoggingCallback, WandbCallback
from model import SimpleMLP
from predictor import Predictor
from cross_validation import run_cross_validation


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

    # ログファイル名（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"train_{timestamp}.log"

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
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Cross Validation設定
    use_cv: bool = False
    cv_folds: int = 5
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

    def __init__(self, cfg: TrainerConfig, logger: logging.Logger = None, callbacks: list[Callback] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._set_seed(cfg.seed)

        # Callbackランナーを初期化
        self.runner = CallbackRunner(callbacks or [])

        # データローダーを作成（通常学習時のみ）
        # Cross Validation時は外部でDataLoaderを差し替えるので、ここでは作成しない
        if not self.cfg.use_cv:
            self.train_loader, self.val_loader, self.test_loader, self.class_to_idx, self.idx_to_class = \
                self._build_dataloaders()
        else:
            # Cross Validation時は、クラス情報だけを取得（データローダーは後で設定される）
            temp_dataset = TouhokuProjectClassificationDataset(
                data_root=self.cfg.data_root,
                transform=None,
                image_extensions=None,
            )
            self.class_to_idx = temp_dataset.get_class_to_idx()
            self.idx_to_class = temp_dataset.get_idx_to_class()

        # モデルを作成
        input_size = cfg.image_size * cfg.image_size * 3  # RGB画像
        num_classes = len(self.class_to_idx)
        self.model = SimpleMLP(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_classes=num_classes
        ).to(self.device)

        # オプティマイザー
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # 損失関数
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

    def _build_dataloaders(self):
        """データローダーを構築."""
        # 画像変換を定義
        train_transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # データ拡張
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
        ])

        # train/val/testに分割してDataLoaderを作成
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class = \
            TouhokuProjectClassificationDataset.create_classification_train_val_test_dataloaders(
                data_root=self.cfg.data_root,
                train_ratio=self.cfg.train_ratio,
                val_ratio=self.cfg.val_ratio,
                test_ratio=self.cfg.test_ratio,
                batch_size=self.cfg.batch_size,
                shuffle_train=True,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=val_transform,
                random_seed=self.cfg.seed,
            )

        self.logger.info(f"学習データ: {len(train_loader.dataset)} サンプル")
        self.logger.info(f"検証データ: {len(val_loader.dataset)} サンプル")
        self.logger.info(f"テストデータ: {len(test_loader.dataset)} サンプル")

        return train_loader, val_loader, test_loader, class_to_idx, idx_to_class

    def _calculate_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """精度を計算."""
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()


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
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """評価."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_acc += self._calculate_accuracy(logits, labels) * batch_size
            n_samples += batch_size

        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples
        return avg_loss, avg_acc

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

                # 学習
                train_loss, train_acc = self._train_one_epoch(epoch)

                # 検証
                val_loss, val_acc = self._evaluate(self.val_loader)

                # 履歴を更新
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

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
                checkpoint_saved = False
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
                    checkpoint_saved = True
                    # 履歴を更新
                    self.history["best_val_acc"] = best_val_acc
                    self.history["best_epoch"] = best_epoch

                self.runner.call("on_epoch_end", self, checkpoint_saved=checkpoint_saved)

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

            # テストデータで評価（test_loaderが存在する場合のみ）
            if hasattr(self, 'test_loader') and len(self.test_loader.dataset) > 0:
                self.logger.info("テストデータで評価...")
                test_loss, test_acc = self._evaluate(self.test_loader)
                self.logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

                self.metrics.update({
                    "epoch": self.cfg.epochs + 1,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                })
                self.runner.call("on_eval_end", self)

            # 学習終了後、confusion matrixを生成（ベストモデルと最終モデルの両方）
            # CVの場合はtest_loaderが空なのでスキップ
            if hasattr(self, 'test_loader') and len(self.test_loader.dataset) > 0:
                self.logger.info("学習終了後、confusion matrixを生成中...")

                # Predictorを作成
                predictor = Predictor(
                    model=self.model,
                    device=self.device,
                    class_to_idx=self.class_to_idx,
                    idx_to_class=self.idx_to_class,
                    logger=self.logger,
                )

                # ベストモデルでconfusion matrixを生成
                best_cm_paths = {}
                best_checkpoint_path = self.save_dir / 'best_model.pt'
                if best_checkpoint_path.exists():
                    self.logger.info("ベストモデルでconfusion matrixを生成中...")
                    best_checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(best_checkpoint['model_state_dict'])

                    for split_name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
                        _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                        cm_path = predictor.create_confusion_matrix(
                            true_labels, pred_labels, split_name,
                            model_type='best',
                            save_dir=self.save_dir,
                            use_timestamp=False,
                        )
                        best_cm_paths[f'{split_name}_cm'] = str(cm_path)
                        self.plots[f'confusion_matrix/best/{split_name}_cm'] = str(cm_path)

                # 最終モデルでconfusion matrixを生成
                self.logger.info("最終モデルでconfusion matrixを生成中...")
                self.model.load_state_dict(checkpoint['model_state_dict'])

                final_cm_paths = {}
                for split_name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
                    _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                    cm_path = predictor.create_confusion_matrix(
                        true_labels, pred_labels, split_name,
                        model_type='final',
                        save_dir=self.save_dir,
                        use_timestamp=False,
                    )
                    final_cm_paths[f'{split_name}_cm'] = str(cm_path)
                    self.plots[f'confusion_matrix/final/{split_name}_cm'] = str(cm_path)
            else:
                # CVの場合はtrain/valのみ
                self.logger.info("学習終了後、confusion matrixを生成中...")

                # Predictorを作成
                predictor = Predictor(
                    model=self.model,
                    device=self.device,
                    class_to_idx=self.class_to_idx,
                    idx_to_class=self.idx_to_class,
                    logger=self.logger,
                )

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
                            save_dir=self.save_dir,
                            use_timestamp=False,
                        )
                        best_cm_paths[f'{split_name}_cm'] = str(cm_path)
                        self.plots[f'confusion_matrix/best/{split_name}_cm'] = str(cm_path)

                # 最終モデルでconfusion matrixを生成
                self.logger.info("最終モデルでconfusion matrixを生成中...")
                self.model.load_state_dict(checkpoint['model_state_dict'])

                final_cm_paths = {}
                for split_name, loader in [('train', self.train_loader), ('val', self.val_loader)]:
                    _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                    cm_path = predictor.create_confusion_matrix(
                        true_labels, pred_labels, split_name,
                        model_type='final',
                        save_dir=self.save_dir,
                        use_timestamp=False,
                    )
                    final_cm_paths[f'{split_name}_cm'] = str(cm_path)
                    self.plots[f'confusion_matrix/final/{split_name}_cm'] = str(cm_path)

            self.metrics.update({
                "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else 0.0,
                "final_train_acc": self.history["train_acc"][-1] if self.history["train_acc"] else 0.0,
                "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
                "final_val_acc": self.history["val_acc"][-1] if self.history["val_acc"] else 0.0,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            })

            # test_loaderが存在する場合のみ追加
            if hasattr(self, 'test_loader') and len(self.test_loader.dataset) > 0:
                self.metrics["test_loss"] = test_loss
                self.metrics["test_acc"] = test_acc

            train_end_called = True
            self.runner.call("on_train_end", self)

        except Exception as e:
            self.runner.call("on_exception", e)
            raise
        finally:
            if not train_end_called:
                self.runner.call("on_train_end", self)


def parse_args():
    """コマンドライン引数を解析."""
    parser = argparse.ArgumentParser(description="分類タスク用MLPの学習")
    parser.add_argument(
        "data_root",
        type=str,
        help="データセットのルートディレクトリパス"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="画像サイズ（デフォルト: 256）"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="バッチサイズ（デフォルト: 32）"
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="エポック数（デフォルト: 10）"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学習率（デフォルト: 1e-3）"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="隠れ層のサイズ（デフォルト: 512）"
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=4,
        help="DataLoaderのワーカー数（デフォルト: 4、共有メモリ不足の場合は0を推奨）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="デバイス（デフォルト: cuda if available else cpu）"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="チェックポイント保存ディレクトリ（デフォルト: ./checkpoints）"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="ログファイル保存ディレクトリ（デフォルト: ./logs）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベル（デフォルト: INFO）"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="WANDBを使用する（デフォルト: True）"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="use_wandb",
        help="WANDBを使用しない"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mlp",
        help="WANDBプロジェクト名（デフォルト: zunda-mlp-classification）"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="zunda",
        help="WANDBエンティティ名（デフォルト: zunda）"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WANDBラン名（デフォルト: None、自動生成）"
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="WANDBグループ名（Cross ValidationでFoldをまとめる際に使用、デフォルト: None）"
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="WANDBタグ（スペース区切りで複数指定可能）"
    )
    parser.add_argument(
        "--upload-checkpoint",
        action="store_true",
        help="ベストモデルのチェックポイントをWANDBにアップロード（デフォルト: False）"
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Cross Validationを使用する（デフォルト: False）"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross ValidationのFold数（デフォルト: 5）"
    )
    return parser.parse_args()


def main():
    """メイン関数."""
    args = parse_args()

    # ロギングを設定
    logger = setup_logging(args.log_dir, args.log_level)

    try:
        cfg = TrainerConfig(
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            hidden_size=args.hidden_size,
            num_workers=args.num_workers,
            device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
            seed=args.seed,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            use_cv=args.use_cv,
            cv_folds=args.cv_folds,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tags,
            upload_checkpoint=args.upload_checkpoint,
        )

        # Callbackを作成
        callbacks = [
            WandbCallback(cfg, logger=logger),
            LoggingCallback(logger=logger),
        ]

        # Cross Validationを使用する場合
        if cfg.use_cv:
            cv_results = run_cross_validation(
                cfg=cfg,
                trainer_class=ClassificationTrainer,
                logger=logger,
                callbacks=callbacks,
            )
            logger.info("Cross Validationが完了しました")
        else:
            # 通常の学習
            trainer = ClassificationTrainer(cfg, logger=logger, callbacks=callbacks)
            trainer.fit()
    except Exception as e:
        logger.exception("学習中にエラーが発生しました:")
        raise


if __name__ == "__main__":
    main()
