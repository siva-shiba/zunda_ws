"""推論と結果保存を行うPredictorクラス（画像分類用）."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")


class ClassificationPredictor:
    """推論と結果保存を行うクラス（画像分類用）."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_to_idx: Dict[str, int],
        idx_to_class: Dict[int, str],
        logger: Optional[logging.Logger] = None,
    ):
        """初期化.

        Args:
            model: 推論に使用するモデル
            device: デバイス
            class_to_idx: クラス名からインデックスへのマッピング
            idx_to_class: インデックスからクラス名へのマッピング
            logger: ロガー（Noneの場合は標準loggingを使用）
        """
        self.model = model
        self.device = device
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.logger = logger or logging.getLogger(__name__)

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        return_probs: bool = False,
    ) -> Tuple:
        """推論を実行.

        Args:
            loader: データローダー
            return_probs: 予測確率も返すかどうか

        Returns:
            return_probs=False: (予測ラベルidx, 真ラベルidx, 予測名, 真名, 精度)
            return_probs=True: (予測確率, 予測idx, 真idx, 予測名, 真名, 精度)
        """
        self.model.eval()

        all_probs = []
        all_preds = []
        all_labels = []

        self.logger.info("推論を実行中...")
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            if return_probs:
                all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                self.logger.info(
                    "処理済み: %s/%s バッチ",
                    batch_idx + 1, len(loader)
                )

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        true_labels = [self.idx_to_class[label] for label in all_labels]
        pred_labels = [self.idx_to_class[pred] for pred in all_preds]
        accuracy = (all_preds == all_labels).mean()

        self.logger.info("推論完了: 精度 = %.4f", accuracy)

        if return_probs:
            all_probs = np.concatenate(all_probs, axis=0)
            return (
                all_probs, all_preds, all_labels,
                pred_labels, true_labels, accuracy
            )
        return all_preds, all_labels, pred_labels, true_labels, accuracy

    def create_confusion_matrix(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        split: str,
        model_type: Optional[str] = None,
        save_dir: Optional[Path] = None,
        use_timestamp: bool = False,
    ) -> Path:
        """混同行列を作成して保存."""
        if isinstance(true_labels[0], (int, np.integer)):
            true_labels = [self.idx_to_class[label] for label in true_labels]
        if isinstance(pred_labels[0], (int, np.integer)):
            pred_labels = [self.idx_to_class[pred] for pred in pred_labels]

        all_class_names = sorted(self.class_to_idx.keys())
        cm = confusion_matrix(
            true_labels, pred_labels, labels=all_class_names
        )
        accuracy = (np.array(true_labels) == np.array(pred_labels)).mean()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=all_class_names,
            yticklabels=all_class_names
        )
        title = f'Confusion Matrix - {split.upper()}'
        if model_type:
            title += f' ({model_type.upper()})'
        title += f' (Accuracy: {accuracy:.4f})'
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        save_dir = save_dir or Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"confusion_matrix_{split}"
        if model_type:
            filename += f"_{model_type}"
        if use_timestamp:
            filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename += ".png"
        cm_path = save_dir / filename
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("混同行列を保存: %s (Accuracy: %.4f)", cm_path, accuracy)
        return cm_path

    def save_results(
        self,
        all_probs: np.ndarray,
        all_preds: np.ndarray,
        all_labels: np.ndarray,
        true_labels: List[str],
        pred_labels: List[str],
        accuracy: float,
        split: str,
        results_dir: Path,
        save_classification_report: bool = True,
        save_confusion_matrix: bool = True,
        save_confusion_matrix_csv: bool = True,
        save_predictions_csv: bool = True,
        model_type: Optional[str] = None,
        use_timestamp: bool = True,
    ) -> Dict[str, Path]:
        """結果を保存."""
        results_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else None
        )

        if save_classification_report:
            all_class_names = sorted(self.class_to_idx.keys())
            report = classification_report(
                true_labels, pred_labels,
                labels=all_class_names,
                target_names=all_class_names,
                output_dict=True,
                zero_division=0
            )
            filename = f"classification_report_{split}"
            if model_type:
                filename += f"_{model_type}"
            if timestamp:
                filename += f"_{timestamp}"
            filename += ".json"
            report_path = results_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'split': split,
                    'accuracy': accuracy,
                    'classification_report': report
                }, f, indent=2, ensure_ascii=False)
            self.logger.info("分類レポートを保存: %s", report_path)
            saved_paths['classification_report'] = report_path

        if save_confusion_matrix or save_confusion_matrix_csv:
            all_class_names = sorted(self.class_to_idx.keys())
            cm = confusion_matrix(
                true_labels, pred_labels, labels=all_class_names
            )
            if save_confusion_matrix:
                cm_path = self.create_confusion_matrix(
                    all_labels, all_preds, split,
                    model_type=model_type,
                    save_dir=results_dir,
                    use_timestamp=use_timestamp,
                )
                saved_paths['confusion_matrix'] = cm_path
            if save_confusion_matrix_csv:
                cm_df = pd.DataFrame(
                    cm, index=all_class_names, columns=all_class_names
                )
                filename = f"confusion_matrix_{split}"
                if model_type:
                    filename += f"_{model_type}"
                if timestamp:
                    filename += f"_{timestamp}"
                filename += ".csv"
                cm_csv_path = results_dir / filename
                cm_df.to_csv(cm_csv_path)
                self.logger.info("混同行列（CSV）を保存: %s", cm_csv_path)
                saved_paths['confusion_matrix_csv'] = cm_csv_path

        if save_predictions_csv:
            results_df = pd.DataFrame({
                'true_label': true_labels,
                'pred_label': pred_labels,
                'correct': [t == p for t, p in zip(true_labels, pred_labels)]
            })
            for class_name in sorted(self.class_to_idx.keys()):
                class_idx = self.class_to_idx[class_name]
                results_df[f'prob_{class_name}'] = all_probs[:, class_idx]
            filename = f"predictions_{split}"
            if model_type:
                filename += f"_{model_type}"
            if timestamp:
                filename += f"_{timestamp}"
            filename += ".csv"
            results_csv_path = results_dir / filename
            results_df.to_csv(results_csv_path, index=False)
            self.logger.info("予測結果を保存: %s", results_csv_path)
            saved_paths['predictions_csv'] = results_csv_path

        print("\n" + "=" * 80)
        print(f"推論結果サマリー - {split.upper()}")
        if model_type:
            print(f"モデルタイプ: {model_type.upper()}")
        print("=" * 80)
        print(f"全体精度: {accuracy:.4f}")
        print("\nクラス別精度:")
        for class_name in sorted(self.class_to_idx.keys()):
            class_mask = np.array([x == class_name for x in true_labels])
            if class_mask.sum() > 0:
                class_acc = (
                    np.array(pred_labels)[class_mask]
                    == np.array(true_labels)[class_mask]
                ).mean()
                n = class_mask.sum()
                print(f"  {class_name}: {class_acc:.4f} ({n} サンプル)")
        print("=" * 80 + "\n")

        return saved_paths
