"""CNNモデル定義（画像分類用）."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """簡易CNNモデル（画像分類用）.

    数層のConv2d + ReLU + MaxPool の後に全結合層で分類します。

    Args:
        in_channels: 入力チャンネル数（例: 3 for RGB）
        image_size: 入力画像の1辺のサイズ（正方形を想定）
        num_classes: クラス数
        fc_hidden: 全結合層の隠れユニット数（デフォルト: 256）
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 256,
        num_classes: int = 10,
        fc_hidden: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.fc_hidden = fc_hidden

        # 3x3 conv, padding=1 でサイズ維持 → MaxPool2d(2) で半分（3回で image_size//8）
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 空間サイズ: image_size // 8
        self.spatial_out = image_size // 8
        self.flat_size = 128 * (self.spatial_out ** 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
