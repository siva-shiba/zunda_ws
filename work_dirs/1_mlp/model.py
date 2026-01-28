"""MLPモデル定義."""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """簡易的なMLPモデル.

    Args:
        input_size: 入力サイズ（画像をFlattenした後のサイズ）
        hidden_size: 隠れ層のサイズ
        num_classes: クラス数
    """
    def __init__(
        self, input_size: int, hidden_size: int = 512, num_classes: int = 10
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
