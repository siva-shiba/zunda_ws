"""ViT分類タスクの推論スクリプト（bestモデル/最終モデル）."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib

matplotlib.use("Agg")

# プロジェクトルートをパスに追加（ローカル import の前に必要）
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zunda import (
    ClassificationPredictor,
    TouhokuProjectClassificationDataset,
    setup_logging,
)  # noqa: E402

from model import build_vit_model  # noqa: E402


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict:
    """チェックポイントをロード."""

    logger = logging.getLogger(__name__)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

    logger.info(f"チェックポイントをロード中: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint.get("epoch", None)
    val_acc = checkpoint.get("val_acc", None)
    logger.info(f"エポック: {epoch}, Val Acc: {val_acc:.4f}")
    return checkpoint


def create_model_from_checkpoint(
    checkpoint: Dict, device: torch.device
) -> Tuple[nn.Module, Dict]:
    """チェックポイントから ViT モデルを作成."""

    logger = logging.getLogger(__name__)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]
    num_classes = len(class_to_idx)

    mc = checkpoint.get("model_config")
    if mc is not None:
        model_name = mc.get("model_name", "vit_b_16")
        pretrained = mc.get("pretrained", False)
        in_channels = mc.get("in_channels", 3)
        image_size = mc.get("image_size", 224)
        logger.info(
            "モデル設定（checkpoint）: "
            "model_name=%s pretrained=%s in_channels=%s "
            "image_size=%s num_classes=%s",
            model_name,
            pretrained,
            in_channels,
            image_size,
            num_classes,
        )
    else:
        model_name = "vit_b_16"
        pretrained = False
        in_channels = 3
        image_size = 224
        logger.info(
            "model_config がありません。デフォルトで構築: "
            "model_name=%s in_channels=%s image_size=%s",
            model_name,
            in_channels,
            image_size,
        )

    model = build_vit_model(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        image_size=image_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = {
        "image_size": image_size,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    return model, config


def create_dataloaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    class_to_idx: Dict[str, int],
    seed: int = 42,
) -> Tuple:
    """学習/検証/テストデータローダーを作成."""

    logger = logging.getLogger(__name__)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    inference_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    create_dl = (
        TouhokuProjectClassificationDataset.create_classification_train_val_test_dataloaders
    )
    train_loader, val_loader, test_loader, _, _ = create_dl(
        data_root=data_root,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=batch_size,
        shuffle_train=False,
        num_workers=num_workers,
        pin_memory=True,
        train_transform=inference_transform,
        val_transform=inference_transform,
        test_transform=inference_transform,
        random_seed=seed,
    )

    logger.info(f"学習データ: {len(train_loader.dataset)} サンプル")
    logger.info(f"検証データ: {len(val_loader.dataset)} サンプル")
    logger.info(f"テストデータ: {len(test_loader.dataset)} サンプル")
    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT bestモデルを使った推論")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="チェックポイントファイルのパス（例: ./checkpoints/best_model.pt）",
    )
    parser.add_argument(
        "data_root",
        type=str,
        help="データセットのルートディレクトリパス",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="画像サイズ（デフォルト: checkpoint の model_config から取得）",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="バッチサイズ（デフォルト: 32）",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=4,
        help="DataLoaderのワーカー数（デフォルト: 4）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="デバイス（デフォルト: cuda if available else cpu）",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="結果保存ディレクトリ（デフォルト: ./results）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベル（デフォルト: INFO）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(log_dir=None, log_level=args.log_level)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"使用デバイス: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        checkpoint_path = Path(args.checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path, device)

        model, config = create_model_from_checkpoint(checkpoint, device)
        logger.info(f"モデルをロードしました: {checkpoint_path}")

        image_size = (
            args.image_size
            if args.image_size is not None
            else config["image_size"]
        )
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=args.data_root,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_to_idx=config["class_to_idx"],
            seed=args.seed,
        )

        results_dir = Path(args.results_dir)
        predictor = ClassificationPredictor(
            model=model,
            device=device,
            class_to_idx=config["class_to_idx"],
            idx_to_class=config["idx_to_class"],
            logger=logger,
        )

        logger.info("=" * 80)
        logger.info("学習データで推論を実行中...")
        logger.info("=" * 80)
        (
            train_probs,
            train_preds,
            train_labels,
            train_pred_labels,
            train_true_labels,
            train_acc,
        ) = predictor.predict(train_loader, return_probs=True)
        predictor.save_results(
            all_probs=train_probs,
            all_preds=train_preds,
            all_labels=train_labels,
            true_labels=train_true_labels,
            pred_labels=train_pred_labels,
            accuracy=train_acc,
            split="train",
            results_dir=results_dir,
            use_timestamp=True,
        )

        logger.info("=" * 80)
        logger.info("検証データで推論を実行中...")
        logger.info("=" * 80)
        (
            val_probs,
            val_preds,
            val_labels,
            val_pred_labels,
            val_true_labels,
            val_acc,
        ) = predictor.predict(val_loader, return_probs=True)
        predictor.save_results(
            all_probs=val_probs,
            all_preds=val_preds,
            all_labels=val_labels,
            true_labels=val_true_labels,
            pred_labels=val_pred_labels,
            accuracy=val_acc,
            split="val",
            results_dir=results_dir,
            use_timestamp=True,
        )

        logger.info("=" * 80)
        logger.info("テストデータで推論を実行中...")
        logger.info("=" * 80)
        (
            test_probs,
            test_preds,
            test_labels,
            test_pred_labels,
            test_true_labels,
            test_acc,
        ) = predictor.predict(test_loader, return_probs=True)
        predictor.save_results(
            all_probs=test_probs,
            all_preds=test_preds,
            all_labels=test_labels,
            true_labels=test_true_labels,
            pred_labels=test_pred_labels,
            accuracy=test_acc,
            split="test",
            results_dir=results_dir,
            use_timestamp=True,
        )

        print("=" * 80)
        print("全体推論結果サマリー")
        print("=" * 80)
        print(f"学習データ精度: {train_acc:.4f}")
        print(f"検証データ精度: {val_acc:.4f}")
        print(f"テストデータ精度: {test_acc:.4f}")
        print("=" * 80 + "\n")
        logger.info("推論が完了しました")

    except Exception as e:
        logger.exception("推論中にエラーが発生しました: %s", e)
        raise


if __name__ == "__main__":
    main()
