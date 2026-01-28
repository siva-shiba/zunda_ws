"""ロギング用コールバック."""

import logging
import torch.nn as nn
from typing import Any, Dict

from .base import Callback


class LoggingCallback(Callback):
    """ログ出力を担当するコールバック."""

    def __init__(self, logger: logging.Logger = None):
        """初期化.

        Args:
            logger: 使用するlogger（Noneの場合は標準loggingを使用）
        """
        self.logger = logger or logging.getLogger(__name__)

    def on_init(self, trainer: Any) -> None:
        """Trainer生成直後に設定をログ出力."""
        cfg = trainer.cfg
        num_classes = len(trainer.class_to_idx)

        self.logger.info("=" * 60)
        self.logger.info("学習設定")
        self.logger.info("=" * 60)
        self.logger.info(f"デバイス: {trainer.device}")
        self.logger.info(f"画像サイズ: {cfg.image_size}")
        self.logger.info(f"バッチサイズ: {cfg.batch_size}")
        self.logger.info(f"エポック数: {cfg.epochs}")
        self.logger.info(f"学習率: {cfg.lr}")
        self.logger.info(f"隠れ層サイズ: {cfg.hidden_size}")
        self.logger.info(f"ワーカー数: {cfg.num_workers}")
        self.logger.info(f"クラス数: {num_classes}")
        self.logger.info(f"モデルパラメータ数: {sum(p.numel() for p in trainer.model.parameters()):,}")
        self.logger.info("=" * 60)

        # モデル構造を出力
        self._log_model_structure(trainer)

    def _log_model_structure(self, trainer: Any) -> None:
        """モデル構造をログに出力."""
        self.logger.info("=" * 60)
        self.logger.info("モデル構造")
        self.logger.info("=" * 60)

        # モデルの文字列表現を取得
        model_str = str(trainer.model)
        self.logger.info("\n" + model_str)

        # 各層の詳細情報を出力
        self.logger.info("層の詳細:")
        total_params = 0
        trainable_params = 0

        for name, module in trainer.model.named_modules():
            if len(list(module.children())) == 0:  # リーフノードのみ
                num_params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if num_params > 0 or isinstance(module, (nn.Flatten, nn.ReLU, nn.Dropout)):
                    layer_info = f"  {name}: {type(module).__name__}"

                    # 層の種類に応じた詳細情報
                    if isinstance(module, nn.Linear):
                        layer_info += f" (in_features={module.in_features}, out_features={module.out_features})"
                        if module.bias is not None:
                            layer_info += f", bias=True"
                        else:
                            layer_info += f", bias=False"
                    elif isinstance(module, nn.Flatten):
                        layer_info += " (入力テンソルを1次元に変換)"
                    elif isinstance(module, nn.ReLU):
                        layer_info += " (inplace=True)"
                    elif isinstance(module, nn.Dropout):
                        layer_info += f" (p={module.p})"

                    self.logger.info(layer_info)

                    if num_params > 0:
                        self.logger.info(f"    パラメータ数: {num_params:,} (学習可能: {trainable:,})")
                        total_params += num_params
                        trainable_params += trainable

        self.logger.info("=" * 60)
        self.logger.info(f"総パラメータ数: {total_params:,}")
        self.logger.info(f"学習可能パラメータ数: {trainable_params:,}")
        self.logger.info(f"固定パラメータ数: {total_params - trainable_params:,}")

        # メモリ使用量の推定
        param_size_mb = total_params * 4 / (1024 ** 2)  # float32と仮定
        self.logger.info(f"推定メモリ使用量: {param_size_mb:.2f} MB (パラメータのみ)")
        self.logger.info("=" * 60)

    def on_train_start(self) -> None:
        """学習開始時のログ."""
        self.logger.info("学習開始...")

    def on_epoch_start(self, epoch: int) -> None:
        """エポック開始時のログ（必要に応じてオーバーライド）."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], checkpoint_saved: bool = False) -> None:
        """エポック終了時のログ."""
        train_loss = metrics.get("train/loss", 0.0)
        train_acc = metrics.get("train/acc", 0.0)
        val_loss = metrics.get("eval/loss", 0.0)
        val_acc = metrics.get("eval/acc", 0.0)

        log_msg = (
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # ベストモデルの情報があれば追加
        if checkpoint_saved and "best_val_acc" in metrics:
            best_val_acc = metrics["best_val_acc"]
            best_epoch = metrics.get("best_epoch", epoch)
            log_msg += f" | Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})"
            # self.logger.info(f"ベストモデルを保存しました: checkpoints/best_model.pt (Val Acc: {best_val_acc:.4f})")

        self.logger.info(log_msg)

    def on_train_end(self, metrics: Dict[str, float] = None) -> None:
        """学習終了時のログ."""
        self.logger.info("学習完了!")

        if metrics:
            best_val_acc = metrics.get("best_val_acc", 0.0)
            best_epoch = metrics.get("best_epoch", 0)
            test_loss = metrics.get("test_loss", 0.0)
            test_acc = metrics.get("test_acc", 0.0)

            self.logger.info(f"ベスト検証精度: {best_val_acc:.4f} (Epoch {best_epoch})")

            # 最終サマリー
            self.logger.info("=" * 60)
            self.logger.info("学習結果サマリー")
            self.logger.info("=" * 60)
            if "final_train_loss" in metrics:
                self.logger.info(
                    f"最終エポック - Train Loss: {metrics['final_train_loss']:.4f}, "
                    f"Train Acc: {metrics['final_train_acc']:.4f}"
                )
                self.logger.info(
                    f"最終エポック - Val Loss: {metrics['final_val_loss']:.4f}, "
                    f"Val Acc: {metrics['final_val_acc']:.4f}"
                )
            self.logger.info(f"テスト - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            self.logger.info(f"ベスト検証精度: {best_val_acc:.4f} (Epoch {best_epoch})")
            self.logger.info("=" * 60)

    def on_exception(self, exc: Exception) -> None:
        """例外発生時のログ."""
        self.logger.exception("学習中にエラーが発生しました:")
