"""カスタム損失関数."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss実装.

    Focal Lossは、難しいサンプルに焦点を当てる損失関数です。
    クラス不均衡データセットで特に有効です。

    Paper: https://arxiv.org/abs/1708.02002

    Args:
        alpha: クラス重みのハイパーパラメータ（デフォルト: 0.25）
        gamma: フォーカシングパラメータ（デフォルト: 2.0）
        weight: クラスごとの重みテンソル（オプション）
        reduction: 損失の縮約方法（'mean', 'sum', 'none'）
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal Lossを計算.

        Args:
            inputs: モデルの出力ロジット [batch_size, num_classes]
            targets: 正解ラベル [batch_size]

        Returns:
            計算された損失値
        """
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
