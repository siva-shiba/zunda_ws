"""データ拡張とサンプリングの共通ユーティリティ."""

from collections import Counter
from typing import Optional

from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(
    dataset,
    use_weighted_sampler: bool = False,
) -> Optional[WeightedRandomSampler]:
    """Weighted Random Samplerを作成.

    Args:
        dataset: データセット（'label'キーを持つ辞書を返す）
        use_weighted_sampler: Weighted samplerを使用するか

    Returns:
        WeightedRandomSampler（使用しない場合はNone）
    """
    if not use_weighted_sampler:
        return None

    # 学習データのラベルを取得
    train_labels = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        train_labels.append(sample['label'])

    # クラスごとのサンプル数をカウント
    class_counts = Counter(train_labels)

    # 各サンプルに重みを割り当て（少数クラスほど大きな重み）
    sample_weights = []
    for label in train_labels:
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)

    # WeightedRandomSamplerを作成
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 重複サンプリングを許可
    )

    return sampler
