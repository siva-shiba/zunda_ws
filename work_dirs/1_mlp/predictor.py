"""推論と結果保存を行うPredictorクラス."""

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class Predictor:
    """推論と結果保存を行うクラス."""

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
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], float]:
        """推論を実行.

        Args:
            loader: データローダー
            return_probs: 予測確率も返すかどうか

        Returns:
            (予測ラベルインデックス, 真のラベルインデックス, 予測ラベル名, 真のラベル名, 精度)
            return_probs=Trueの場合: (予測確率, 予測ラベルインデックス, 真のラベルインデックス, 予測ラベル名, 真のラベル名, 精度)
        """
        self.model.eval()

        all_probs = []
        all_preds = []
        all_labels = []

        self.logger.info("推論を実行中...")
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            # 推論
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            if return_probs:
                all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"処理済み: {batch_idx + 1}/{len(loader)} バッチ")

        # 結合
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # ラベル名に変換
        true_labels = [self.idx_to_class[label] for label in all_labels]
        pred_labels = [self.idx_to_class[pred] for pred in all_preds]

        # 精度を計算
        accuracy = (all_preds == all_labels).mean()

        self.logger.info(f"推論完了: 精度 = {accuracy:.4f}")

        if return_probs:
            all_probs = np.concatenate(all_probs, axis=0)
            return all_probs, all_preds, all_labels, pred_labels, true_labels, accuracy
        else:
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
        """混同行列を作成して保存.

        Args:
            true_labels: 正解ラベル（インデックスまたはクラス名）
            pred_labels: 予測ラベル（インデックスまたはクラス名）
            split: データセットの種類（'train', 'val', 'test'）
            model_type: モデルの種類（'best', 'final'など、Noneの場合はタイトルに含めない）
            save_dir: 保存先ディレクトリ（Noneの場合は現在のディレクトリ）
            use_timestamp: タイムスタンプをファイル名に含めるかどうか

        Returns:
            保存された画像ファイルのパス
        """
        # ラベルがインデックスの場合はクラス名に変換
        if isinstance(true_labels[0], (int, np.integer)):
            true_labels = [self.idx_to_class[label] for label in true_labels]
        if isinstance(pred_labels[0], (int, np.integer)):
            pred_labels = [self.idx_to_class[pred] for pred in pred_labels]

        # クラス名のリストを取得（順序を保持）
        all_class_names = sorted(self.class_to_idx.keys())

        # 混同行列を計算
        cm = confusion_matrix(
            true_labels,
            pred_labels,
            labels=all_class_names
        )

        # 精度を計算
        accuracy = (np.array(true_labels) == np.array(pred_labels)).mean()

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
        title = f'Confusion Matrix - {split.upper()}'
        if model_type:
            title += f' ({model_type.upper()})'
        title += f' (Accuracy: {accuracy:.4f})'
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # ファイル名を生成
        if save_dir is None:
            save_dir = Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"confusion_matrix_{split}"
        if model_type:
            filename += f"_{model_type}"
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename += f"_{timestamp}"
        filename += ".png"

        cm_path = save_dir / filename
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"混同行列を保存: {cm_path} (Accuracy: {accuracy:.4f})")

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
        """結果を保存.

        Args:
            all_probs: 予測確率
            all_preds: 予測ラベル（インデックス）
            all_labels: 真のラベル（インデックス）
            true_labels: 真のラベル名
            pred_labels: 予測ラベル名
            accuracy: 精度
            split: データセットの種類（"train", "val", "test"）
            results_dir: 結果保存ディレクトリ
            save_classification_report: 分類レポートを保存するか
            save_confusion_matrix: 混同行列画像を保存するか
            save_confusion_matrix_csv: 混同行列CSVを保存するか
            save_predictions_csv: 予測結果CSVを保存するか
            model_type: モデルの種類（'best', 'final'など）
            use_timestamp: タイムスタンプをファイル名に含めるかどうか

        Returns:
            保存されたファイルのパスの辞書
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else None

        # 1. 分類レポートを保存
        if save_classification_report:
            all_class_names = sorted(self.class_to_idx.keys())

            report = classification_report(
                true_labels,
                pred_labels,
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
            self.logger.info(f"分類レポートを保存: {report_path}")
            saved_paths['classification_report'] = report_path

        # 2. 混同行列を保存（画像とCSV）
        if save_confusion_matrix or save_confusion_matrix_csv:
            all_class_names = sorted(self.class_to_idx.keys())
            cm = confusion_matrix(
                true_labels,
                pred_labels,
                labels=all_class_names
            )

            # 画像として保存
            if save_confusion_matrix:
                cm_path = self.create_confusion_matrix(
                    all_labels,
                    all_preds,
                    split,
                    model_type=model_type,
                    save_dir=results_dir,
                    use_timestamp=use_timestamp,
                )
                saved_paths['confusion_matrix'] = cm_path

            # CSVとして保存
            if save_confusion_matrix_csv:
                cm_df = pd.DataFrame(
                    cm,
                    index=all_class_names,
                    columns=all_class_names
                )
                filename = f"confusion_matrix_{split}"
                if model_type:
                    filename += f"_{model_type}"
                if timestamp:
                    filename += f"_{timestamp}"
                filename += ".csv"

                cm_csv_path = results_dir / filename
                cm_df.to_csv(cm_csv_path)
                self.logger.info(f"混同行列（CSV）を保存: {cm_csv_path}")
                saved_paths['confusion_matrix_csv'] = cm_csv_path

        # 3. 予測結果を保存（各サンプルの予測確率とラベル）
        if save_predictions_csv:
            results_df = pd.DataFrame({
                'true_label': true_labels,
                'pred_label': pred_labels,
                'correct': [t == p for t, p in zip(true_labels, pred_labels)]
            })

            # 各クラスの確率を追加
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
            self.logger.info(f"予測結果を保存: {results_csv_path}")
            saved_paths['predictions_csv'] = results_csv_path

        # 4. サマリーを表示
        print("\n" + "="*80)
        print(f"推論結果サマリー - {split.upper()}")
        if model_type:
            print(f"モデルタイプ: {model_type.upper()}")
        print("="*80)
        print(f"全体精度: {accuracy:.4f}")
        print(f"\nクラス別精度:")
        for class_name in sorted(self.class_to_idx.keys()):
            class_mask = np.array([l == class_name for l in true_labels])
            if class_mask.sum() > 0:
                class_acc = (np.array(pred_labels)[class_mask] == np.array(true_labels)[class_mask]).mean()
                print(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} サンプル)")
        print("="*80 + "\n")

        return saved_paths
