"""bestモデルを使ったテストデータの推論スクリプト."""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zunda import TouhokuProjectClassificationDataset
from model import SimpleMLP


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """ロギングを設定.

    Args:
        log_level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）

    Returns:
        設定済みのlogger
    """
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

    # コンソールハンドラー（標準出力に出力）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict:
    """チェックポイントをロード.

    Args:
        checkpoint_path: チェックポイントファイルのパス
        device: デバイス

    Returns:
        チェックポイント辞書
    """
    logger = logging.getLogger(__name__)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

    logger.info(f"チェックポイントをロード中: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"エポック: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 0.0):.4f}")

    return checkpoint


def create_model_from_checkpoint(checkpoint: Dict, device: torch.device) -> Tuple[nn.Module, Dict]:
    """チェックポイントからモデルを作成.

    Args:
        checkpoint: チェックポイント辞書
        device: デバイス

    Returns:
        (モデル, 設定辞書)
    """
    logger = logging.getLogger(__name__)

    # チェックポイントから必要な情報を取得
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = checkpoint['idx_to_class']
    num_classes = len(class_to_idx)

    # モデルの設定を推測（チェックポイントに保存されていない場合のデフォルト値）
    # 実際のモデル構造から推測する必要がある
    # ここでは、state_dictから推測するか、デフォルト値を使用
    hidden_size = 512  # デフォルト値（必要に応じてチェックポイントに保存する）
    image_size = 256  # デフォルト値

    # state_dictから入力サイズを推測
    first_layer_key = [k for k in checkpoint['model_state_dict'].keys() if 'net.1.weight' in k or 'net.0.weight' in k]
    if first_layer_key:
        # Linear層の重みから入力サイズを取得
        first_layer_weight = checkpoint['model_state_dict'][first_layer_key[0]]
        input_size = first_layer_weight.shape[1]
        # 画像サイズを推測（RGB画像の場合）
        if input_size % 3 == 0:
            image_size = int(np.sqrt(input_size // 3))
    else:
        input_size = image_size * image_size * 3

    logger.info(f"モデル設定: input_size={input_size}, hidden_size={hidden_size}, num_classes={num_classes}")

    # モデルを作成
    model = SimpleMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes
    ).to(device)

    # 重みをロード
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    config = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_classes': num_classes,
        'image_size': image_size,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
    }

    return model, config


def create_dataloaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    class_to_idx: Dict[str, int],
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """学習/検証/テストデータローダーを作成.

    Args:
        data_root: データセットのルートディレクトリ
        image_size: 画像サイズ
        batch_size: バッチサイズ
        num_workers: ワーカー数
        class_to_idx: クラス名からインデックスへのマッピング
        seed: 乱数シード

    Returns:
        (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)

    # 推論用のtransform（データ拡張なし）
    inference_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # 学習/検証/テストデータローダーを作成
    train_loader, val_loader, test_loader, _, _ = \
        TouhokuProjectClassificationDataset.create_classification_train_val_test_dataloaders(
            data_root=data_root,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=batch_size,
            shuffle_train=False,  # 推論時はシャッフル不要
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


@torch.no_grad()
def inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    idx_to_class: Dict[int, str]
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], float]:
    """推論を実行.

    Args:
        model: モデル
        test_loader: テストデータローダー
        device: デバイス
        idx_to_class: インデックスからクラス名へのマッピング

    Returns:
        (予測確率, 予測ラベル, 真のラベル名, 予測ラベル名, 精度)
    """
    logger = logging.getLogger(__name__)
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    logger.info("推論を実行中...")
    for batch_idx, batch in enumerate(test_loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        # 推論
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"処理済み: {batch_idx + 1}/{len(test_loader)} バッチ")

    # 結合
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ラベル名に変換
    true_labels = [idx_to_class[label] for label in all_labels]
    pred_labels = [idx_to_class[pred] for pred in all_preds]

    # 精度を計算
    accuracy = (all_preds == all_labels).mean()

    logger.info(f"推論完了: 精度 = {accuracy:.4f}")

    return all_probs, all_preds, true_labels, pred_labels, accuracy


def save_results(
    results_dir: Path,
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    true_labels: List[str],
    pred_labels: List[str],
    accuracy: float,
    idx_to_class: Dict[int, str],
    class_to_idx: Dict[str, int],
    split: str = "test"
):
    """結果を保存.

    Args:
        results_dir: 結果保存ディレクトリ
        all_probs: 予測確率
        all_preds: 予測ラベル（インデックス）
        true_labels: 真のラベル名
        pred_labels: 予測ラベル名
        accuracy: 精度
        idx_to_class: インデックスからクラス名へのマッピング
        class_to_idx: クラス名からインデックスへのマッピング
        split: データセットの種類（"train", "val", "test"）
    """
    logger = logging.getLogger(__name__)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 分類レポートを保存
    all_class_names = sorted(class_to_idx.keys())
    all_class_indices = [class_to_idx[name] for name in all_class_names]

    report = classification_report(
        true_labels,
        pred_labels,
        labels=all_class_names,
        target_names=all_class_names,
        output_dict=True,
        zero_division=0  # テストデータに含まれていないクラスは0で評価
    )

    report_path = results_dir / f"classification_report_{split}_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'split': split,
            'accuracy': accuracy,
            'classification_report': report
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"分類レポートを保存: {report_path}")

    # 2. 混同行列を保存（画像とCSV）
    # チェックポイントに保存されている全クラスを含める
    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=all_class_names
    )

    # 画像として保存
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=all_class_names,
        yticklabels=all_class_names
    )
    plt.title(f'Confusion Matrix - {split.upper()} (Accuracy: {accuracy:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = results_dir / f"confusion_matrix_{split}_{timestamp}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"混同行列を保存: {cm_path}")

    # CSVとして保存
    cm_df = pd.DataFrame(
        cm,
        index=all_class_names,
        columns=all_class_names
    )
    cm_csv_path = results_dir / f"confusion_matrix_{split}_{timestamp}.csv"
    cm_df.to_csv(cm_csv_path)
    logger.info(f"混同行列（CSV）を保存: {cm_csv_path}")

    # 3. 予測結果を保存（各サンプルの予測確率とラベル）
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': pred_labels,
        'correct': [t == p for t, p in zip(true_labels, pred_labels)]
    })

    # 各クラスの確率を追加
    for i, class_name in enumerate(sorted(class_to_idx.keys())):
        class_idx = class_to_idx[class_name]
        results_df[f'prob_{class_name}'] = all_probs[:, class_idx]

    results_csv_path = results_dir / f"predictions_{split}_{timestamp}.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"予測結果を保存: {results_csv_path}")

    # 4. サマリーを表示
    print("\n" + "="*80)
    print(f"推論結果サマリー - {split.upper()}")
    print("="*80)
    print(f"全体精度: {accuracy:.4f}")
    print(f"\nクラス別精度:")
    for class_name in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[class_name]
        class_mask = np.array([l == class_name for l in true_labels])
        if class_mask.sum() > 0:
            class_acc = (np.array(pred_labels)[class_mask] == np.array(true_labels)[class_mask]).mean()
            print(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} サンプル)")
    print("="*80 + "\n")


def parse_args():
    """コマンドライン引数を解析."""
    parser = argparse.ArgumentParser(description="bestモデルを使ったテストデータの推論")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="チェックポイントファイルのパス（例: ./checkpoints/best_model.pt）"
    )
    parser.add_argument(
        "data_root",
        type=str,
        help="データセットのルートディレクトリパス（例: /ws/data/touhoku_project_images）"
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
        "--num-workers",
        "-w",
        type=int,
        default=4,
        help="DataLoaderのワーカー数（デフォルト: 4）"
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
        "--results-dir",
        type=str,
        default="./results",
        help="結果保存ディレクトリ（デフォルト: ./results）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベル（デフォルト: INFO）"
    )
    return parser.parse_args()


def main():
    """メイン関数."""
    args = parse_args()

    # ロギングを設定
    logger = setup_logging(args.log_level)

    # デバイスを設定
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"使用デバイス: {device}")

    # 乱数シードを設定
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        # チェックポイントをロード
        checkpoint_path = Path(args.checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path, device)

        # モデルを作成
        model, config = create_model_from_checkpoint(checkpoint, device)
        logger.info(f"モデルをロードしました: {checkpoint_path}")

        # 学習/検証/テストデータローダーを作成
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=args.data_root,
            image_size=config['image_size'],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_to_idx=config['class_to_idx'],
            seed=args.seed
        )

        results_dir = Path(args.results_dir)

        # 学習データで推論
        logger.info("="*80)
        logger.info("学習データで推論を実行中...")
        logger.info("="*80)
        train_probs, train_preds, train_true_labels, train_pred_labels, train_acc = inference(
            model=model,
            test_loader=train_loader,
            device=device,
            idx_to_class=config['idx_to_class']
        )
        save_results(
            results_dir=results_dir,
            all_probs=train_probs,
            all_preds=train_preds,
            true_labels=train_true_labels,
            pred_labels=train_pred_labels,
            accuracy=train_acc,
            idx_to_class=config['idx_to_class'],
            class_to_idx=config['class_to_idx'],
            split="train"
        )

        # 検証データで推論
        logger.info("="*80)
        logger.info("検証データで推論を実行中...")
        logger.info("="*80)
        val_probs, val_preds, val_true_labels, val_pred_labels, val_acc = inference(
            model=model,
            test_loader=val_loader,
            device=device,
            idx_to_class=config['idx_to_class']
        )
        save_results(
            results_dir=results_dir,
            all_probs=val_probs,
            all_preds=val_preds,
            true_labels=val_true_labels,
            pred_labels=val_pred_labels,
            accuracy=val_acc,
            idx_to_class=config['idx_to_class'],
            class_to_idx=config['class_to_idx'],
            split="val"
        )

        # テストデータで推論
        logger.info("="*80)
        logger.info("テストデータで推論を実行中...")
        logger.info("="*80)
        test_probs, test_preds, test_true_labels, test_pred_labels, test_acc = inference(
            model=model,
            test_loader=test_loader,
            device=device,
            idx_to_class=config['idx_to_class']
        )
        save_results(
            results_dir=results_dir,
            all_probs=test_probs,
            all_preds=test_preds,
            true_labels=test_true_labels,
            pred_labels=test_pred_labels,
            accuracy=test_acc,
            idx_to_class=config['idx_to_class'],
            class_to_idx=config['class_to_idx'],
            split="test"
        )

        # 全体サマリーを表示
        print("\n" + "="*80)
        print("全体推論結果サマリー")
        print("="*80)
        print(f"学習データ精度: {train_acc:.4f}")
        print(f"検証データ精度: {val_acc:.4f}")
        print(f"テストデータ精度: {test_acc:.4f}")
        print("="*80 + "\n")

        logger.info("推論が完了しました")

    except Exception as e:
        logger.exception("推論中にエラーが発生しました:")
        raise


if __name__ == "__main__":
    main()
