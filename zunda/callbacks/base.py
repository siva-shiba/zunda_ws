"""学習コールバックの基底クラスとランナー."""

import logging
from typing import Any, Dict, List, Optional


class Callback:
    """コールバックの基底クラス.

    すべてのメソッドはno-op（pass）で実装されているため、
    必要なメソッドだけをオーバーライドすればよい。
    """

    def on_init(self, trainer: Any) -> None:
        """Trainer生成直後に呼ばれる.

        Args:
            trainer: ClassificationTrainerインスタンス
        """
        pass

    def on_train_start(self) -> None:
        """学習開始時に呼ばれる."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """エポック開始時に呼ばれる.

        Args:
            epoch: エポック番号（1始まり）
        """
        pass

    def on_train_batch_end(self, step: int, metrics: Dict[str, float]) -> None:
        """1バッチの学習終了時に呼ばれる.

        Args:
            step: グローバルステップ数
            metrics: メトリクス辞書（例: {"train/loss": 0.5, "train/acc": 0.8}）
        """
        pass

    def on_eval_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """評価終了時に呼ばれる.

        Args:
            epoch: エポック番号
            metrics: 評価メトリクス辞書（例: {"eval/loss": 0.4, "eval/acc": 0.85}）
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], checkpoint_saved: bool = False) -> None:
        """エポック終了時に呼ばれる.

        Args:
            epoch: エポック番号
            metrics: エポック全体のメトリクス辞書（train+eval統合）
            checkpoint_saved: ベストモデルが保存されたかどうか
        """
        pass

    def on_train_end(self, metrics: Dict[str, float] = None) -> None:
        """学習終了時に呼ばれる.

        Args:
            metrics: 最終メトリクス辞書（オプション）
        """
        pass

    def on_exception(self, exc: Exception) -> None:
        """例外発生時に呼ばれる.

        Args:
            exc: 発生した例外
        """
        pass


class CallbackRunner:
    """複数のコールバックを順に実行するランナー."""

    def __init__(self, callbacks: List[Callback]):
        """初期化.

        Args:
            callbacks: コールバックのリスト
        """
        self.callbacks = callbacks
        self.logger = logging.getLogger(__name__)

    def call(self, method_name: str, *args, **kwargs) -> None:
        """指定されたメソッドをすべてのコールバックで順に呼び出す.

        Args:
            method_name: 呼び出すメソッド名
            *args: 位置引数
            **kwargs: キーワード引数

        Raises:
            Exception: コールバック実行中に発生した例外（ログ出力後に再raise）
        """
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    method = getattr(callback, method_name)
                    method(*args, **kwargs)
                except Exception as e:
                    self.logger.error(
                        f"Callback {callback.__class__.__name__}.{method_name} でエラーが発生しました: {e}"
                    )
                    self.logger.exception(e)
                    raise
