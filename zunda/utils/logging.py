"""ロギングユーティリティ."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """ロギングを設定.

    Args:
        log_dir: ログファイルを保存するディレクトリ。
                 None の場合はコンソール出力のみ。
        log_level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）

    Returns:
        設定済みのルート logger
    """
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    level = getattr(logging, log_level.upper())

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    # コンソール
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # ファイル（log_dir 指定時のみ）
    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file = log_dir_path / "train.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logger.info("ログファイル: %s", log_file)

    return logger
