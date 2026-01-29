"""WANDB用コールバック."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import wandb

from .base import Callback


def load_wandb_api_key(wandb_file: str = ".wandb") -> Optional[str]:
    """`.wandb`ファイルからWANDB APIキーを読み込む.

    Args:
        wandb_file: `.wandb`ファイルのパス

    Returns:
        APIキー（ファイルが存在しない場合はNone）
    """
    logger = logging.getLogger(__name__)

    # Docker環境では /ws/.wandb を優先的に探す
    docker_wandb_path = Path("/ws/.wandb")
    if docker_wandb_path.exists():
        try:
            with open(docker_wandb_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip().split('\n')[0].split('#')[0].strip()
                if api_key:
                    logger.info(f"WANDB APIキーを読み込みました: {docker_wandb_path}")
                    return api_key
                logger.warning(f"{docker_wandb_path}は空か、有効なAPIキーが含まれていません")
        except Exception as e:
            logger.warning(f"{docker_wandb_path}の読み込みに失敗: {e}")

    # プロジェクトルートからも探す（zundaパッケージの親ディレクトリ）
    project_root = Path(__file__).parent.parent.parent
    wandb_path = project_root / wandb_file

    if wandb_path.exists():
        try:
            with open(wandb_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip().split('\n')[0].split('#')[0].strip()
                if api_key:
                    logger.info(f"WANDB APIキーを読み込みました: {wandb_path}")
                    return api_key
                logger.warning(f"{wandb_path}は空か、有効なAPIキーが含まれていません")
        except Exception as e:
            logger.warning(f"{wandb_path}の読み込みに失敗: {e}")

    logger.warning(f"WANDB APIキーファイルが見つかりませんでした: {wandb_file} または /ws/.wandb")
    return None


class WandbCallback(Callback):
    """WANDBへのログ送信を担当するコールバック."""

    def __init__(self, cfg: Any, logger: logging.Logger = None):
        """初期化.

        Args:
            cfg: TrainerConfigインスタンス
            logger: 使用するlogger（Noneの場合は標準loggingを使用）
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.wandb_run = None
        self._finished = False

    def on_init(self, trainer: Any) -> None:
        """WANDBを初期化."""
        if not self.cfg.use_wandb:
            return

        try:
            # APIキーを読み込む
            api_key = load_wandb_api_key()
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
                masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
                self.logger.info(f"WANDB_API_KEY環境変数が設定されました: {masked_key}")
            elif not os.environ.get("WANDB_API_KEY"):
                self.logger.warning("WANDB_API_KEYが設定されていません。.wandbファイルまたは環境変数を設定してください。")
                return

            # モデル情報を取得
            num_classes = len(trainer.class_to_idx)
            input_size = self.cfg.image_size * self.cfg.image_size * 3
            total_params = sum(p.numel() for p in trainer.model.parameters())
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            model_str = str(trainer.model)

            config = {
                "image_size": self.cfg.image_size,
                "batch_size": self.cfg.batch_size,
                "epochs": self.cfg.epochs,
                "lr": self.cfg.lr,
                "hidden_size": self.cfg.hidden_size,
                "num_workers": self.cfg.num_workers,
                "seed": self.cfg.seed,
                "train_ratio": self.cfg.train_ratio,
                "val_ratio": self.cfg.val_ratio,
                "test_ratio": self.cfg.test_ratio,
                "num_classes": num_classes,
                "input_size": input_size,
                "device": str(self.cfg.device),
                "model_architecture": model_str,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "frozen_params": total_params - trainable_params,
                "model_size_mb": total_params * 4 / (1024 ** 2),
            }

            # WANDBを初期化
            wandb_mode = os.environ.get("WANDB_MODE", "online")
            init_kwargs = {
                "project": self.cfg.wandb_project,
                "entity": self.cfg.wandb_entity,
                "name": self.cfg.wandb_run_name,
                "tags": self.cfg.wandb_tags if self.cfg.wandb_tags else None,
                "config": config,
                "mode": wandb_mode,
            }
            # groupが指定されている場合は追加
            if hasattr(self.cfg, 'wandb_group') and self.cfg.wandb_group:
                init_kwargs["group"] = self.cfg.wandb_group

            self.wandb_run = wandb.init(**init_kwargs)

            if self.wandb_run is not None:
                self.logger.info(f"WANDB初期化完了: {self.wandb_run.url}")
                self.logger.info(f"WANDB Run ID: {self.wandb_run.id}")
                self.logger.info(f"WANDB Run Name: {self.wandb_run.name}")
                if hasattr(self.cfg, 'wandb_group') and self.cfg.wandb_group:
                    self.logger.info(f"WANDB Group: {self.cfg.wandb_group}")

        except Exception as e:
            self.logger.warning(f"WANDBの初期化に失敗しました: {e}")
            self.logger.exception(e)
            self.wandb_run = None

    def on_train_batch_end(self, trainer: Any) -> None:
        """バッチ終了時にWANDBにログ（現在は使用していない）."""
        pass

    def on_eval_end(self, trainer: Any) -> None:
        """評価終了時にWANDBにログ."""
        if not self.wandb_run:
            return

        try:
            epoch = trainer.metrics.get("epoch", 0)
            wandb.log(trainer.metrics, step=epoch)
        except Exception as e:
            self.logger.error(f"WANDBへのログ記録中にエラーが発生しました: {e}")

    def on_epoch_end(self, trainer: Any, checkpoint_saved: bool = False) -> None:
        """エポック終了時にWANDBにログ."""
        if not self.wandb_run:
            return

        try:
            epoch = trainer.metrics.get("epoch", 0)
            epoch_metrics = {
                "epoch": epoch,
                "train/loss": trainer.metrics.get("train/loss", 0.0),
                "train/acc": trainer.metrics.get("train/acc", 0.0),
                "eval/loss": trainer.metrics.get("eval/loss", 0.0),
                "eval/acc": trainer.metrics.get("eval/acc", 0.0),
            }
            wandb.log(epoch_metrics, step=epoch)

            if checkpoint_saved and "best_val_acc" in trainer.metrics:
                best_metrics = {
                    "best_val_acc": trainer.metrics["best_val_acc"],
                    "best_epoch": trainer.metrics.get("best_epoch", epoch)
                }
                wandb.log(best_metrics, step=epoch)

                if self.cfg.upload_checkpoint:
                    checkpoint_path = Path(self.cfg.save_dir) / 'best_model.pt'
                    if checkpoint_path.exists():
                        wandb.save(str(checkpoint_path))
        except Exception as e:
            self.logger.error(f"WANDBへのログ記録中にエラーが発生しました: {e}")

    def on_train_end(self, trainer: Any) -> None:
        """学習終了時にWANDBを終了."""
        if not self.wandb_run or self._finished:
            return

        try:
            self._log_plots(trainer.plots)
            wandb.summary.update({
                "best_val_acc": trainer.metrics.get("best_val_acc", 0.0),
                "best_epoch": trainer.metrics.get("best_epoch", 0),
                "test_acc": trainer.metrics.get("test_acc", 0.0),
                "test_loss": trainer.metrics.get("test_loss", 0.0),
            })

            wandb.finish()
            self._finished = True
            self.logger.info("WANDBを終了しました")
        except Exception as e:
            self.logger.error(f"WANDBの終了処理中にエラーが発生しました: {e}")

    def _log_plots(self, plots: Dict[str, Any]) -> None:
        """plots辞書から可視化データをWANDBに送信.

        Args:
            plots: Trainerやvalidatorが生成した可視化データの辞書
                  キーは階層構造（例: 'confusion_matrix/best/train_cm'）
                  値は画像パス（str）またはwandb.Imageオブジェクト
        """
        if not self.wandb_run:
            return

        try:
            for key, value in plots.items():
                if isinstance(value, str):
                    # 文字列の場合は画像パスとして扱う
                    image_path = Path(value)
                    if image_path.exists() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        image = wandb.Image(str(image_path))
                        wandb.log({key: image})
                        self.logger.info(f"WANDBに画像を送信: {key}")
                    else:
                        self.logger.warning(f"画像ファイルが見つかりません: {value}")
                elif hasattr(value, '__class__') and 'Image' in str(type(value)):
                    # 既にwandb.Imageオブジェクトの場合
                    wandb.log({key: value})
                    self.logger.info(f"WANDBに画像を送信: {key}")
                else:
                    self.logger.warning(f"未対応のplots値の型: {type(value)} (key: {key})")
        except Exception as e:
            self.logger.error(f"WANDBへのplots送信中にエラーが発生しました: {e}")

    def on_exception(self, exc: Exception) -> None:
        """例外発生時にWANDBを終了."""
        self.on_train_end()
