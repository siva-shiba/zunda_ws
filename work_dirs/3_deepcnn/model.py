"""DeepCNNモデル定義（VGG/ResNet/MobileNet/EfficientNet/ConvNeXt）.

torchvision の pre-trained モデルを Config で切り替えて利用します.

注意:
- 各クラス内のコメントアウトされた「手書き構造」は学習用の参照コードです。
- 実際に返すモデルは torchvision 実装です。
- コメントアウトの手書き構造は、可読性のため一部簡略化しています
  （層数・ブロック詳細・分岐の厳密再現はしていません）。
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


# 各バックボーンの利用可能なモデル名
BACKBONE_REGISTRY = {
    "vgg": ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"],
    "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    "mobilenet": ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"],
    "efficientnet": [
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    ],
    "convnext": ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
}


def _get_backbone_family(model_name: str) -> str:
    """モデル名からバックボーン族を取得."""
    for family, names in BACKBONE_REGISTRY.items():
        if model_name in names:
            return family
    raise ValueError(
        f"未知のモデル名: {model_name}. "
        f"利用可能: {list(BACKBONE_REGISTRY.keys())} -> {list(BACKBONE_REGISTRY.values())}"
    )


class DeepCNN(nn.Module):

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = False,
        weights: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.weights = weights
        self.family = _get_backbone_family(model_name)
        self.net = self._build_model(**kwargs)

    def _build_model(self, **kwargs) -> nn.Module:
        # pretrained 時は DEFAULT を利用（明示 weights 優先）
        load_weights = self.weights
        if load_weights is None and self.pretrained:
            load_weights = "DEFAULT"

        if load_weights is not None:
            kwargs_load = {"weights": load_weights}
        else:
            kwargs_load = {"weights": None, "num_classes": self.num_classes}

        model = models.get_model(self.model_name, **kwargs_load)

        if load_weights is not None and self.num_classes != 1000:
            model = _replace_classifier(model, self.num_classes, self.family, self.model_name)
        if self.in_channels != 3:
            model = _replace_first_conv(model, self.in_channels, self.family, self.model_name)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VGGDeepCNN(DeepCNN):
    """VGG バックボーン."""

    def __init__(self, model_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = False, weights: Optional[str] = None, **kwargs):
        # 手書き構造[簡略版]（参考用。実際には torchvision の VGG を返す）
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        #     nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
        #     nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
        #     nn.Linear(4096, num_classes),
        # )
        # def forward(self, x):
        #     x = self.features(x)
        #     x = self.avgpool(x)
        #     return self.classifier(x)
        super().__init__(model_name, num_classes, in_channels, pretrained, weights, **kwargs)


class ResNetDeepCNN(DeepCNN):
    """ResNet バックボーン."""

    def __init__(self, model_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = False, weights: Optional[str] = None, **kwargs):
        # 手書き構造[簡略版]（参考用。実際には torchvision の ResNet を返す）
        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1),
        # )
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
        # )
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(128, num_classes)
        # def forward(self, x):
        #     x = self.stem(x)
        #     x = self.layer1(x) + x  # 簡易残差
        #     x2 = self.layer2(x)
        #     x = x2
        #     x = self.gap(x).flatten(1)
        #     return self.fc(x)
        super().__init__(model_name, num_classes, in_channels, pretrained, weights, **kwargs)


class MobileNetDeepCNN(DeepCNN):
    """MobileNet バックボーン."""

    def __init__(self, model_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = False, weights: Optional[str] = None, **kwargs):
        # 手書き構造[簡略版]（参考用。実際には torchvision の MobileNet を返す）
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
        #     nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16, bias=False),  # depthwise
        #     nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
        #     nn.Conv2d(16, 24, 1, bias=False),  # pointwise
        #     nn.BatchNorm2d(24), nn.ReLU6(inplace=True),
        #     nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24, bias=False),
        #     nn.BatchNorm2d(24), nn.ReLU6(inplace=True),
        #     nn.Conv2d(24, 40, 1, bias=False),
        #     nn.BatchNorm2d(40), nn.ReLU6(inplace=True),
        # )
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(40, num_classes),
        # )
        # def forward(self, x):
        #     x = self.features(x)
        #     x = self.gap(x).flatten(1)
        #     return self.classifier(x)
        super().__init__(model_name, num_classes, in_channels, pretrained, weights, **kwargs)


class EfficientNetDeepCNN(DeepCNN):
    """EfficientNet バックボーン."""

    def __init__(self, model_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = False, weights: Optional[str] = None, **kwargs):
        # 手書き構造[簡略版]（参考用。実際には torchvision の EfficientNet を返す）
        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(32), nn.SiLU(inplace=True),
        # )
        # self.blocks = nn.Sequential(
        #     nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True),
        #     nn.Conv2d(32, 16, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True),
        #     nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True),
        #     nn.Conv2d(16, 24, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(inplace=True),
        # )
        # self.head = nn.Sequential(
        #     nn.Conv2d(24, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.SiLU(inplace=True),
        # )
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        # def forward(self, x):
        #     x = self.stem(x)
        #     x = self.blocks(x)
        #     x = self.head(x)
        #     x = self.gap(x).flatten(1)
        #     return self.classifier(x)
        super().__init__(model_name, num_classes, in_channels, pretrained, weights, **kwargs)


class ConvNeXtDeepCNN(DeepCNN):
    """ConvNeXt バックボーン."""

    def __init__(self, model_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = False, weights: Optional[str] = None, **kwargs):
        # 手書き構造[簡略版]（参考用。実際には torchvision の ConvNeXt を返す）
        # self.stem = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
        # self.stage1 = nn.Sequential(
        #     nn.Conv2d(96, 96, 7, padding=3, groups=96),  # depthwise large kernel
        #     nn.BatchNorm2d(96), nn.GELU(),
        #     nn.Conv2d(96, 96, 1), nn.GELU(),
        # )
        # self.downsample1 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
        # self.stage2 = nn.Sequential(
        #     nn.Conv2d(192, 192, 7, padding=3, groups=192),
        #     nn.BatchNorm2d(192), nn.GELU(),
        #     nn.Conv2d(192, 192, 1), nn.GELU(),
        # )
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(192, num_classes))
        # def forward(self, x):
        #     x = self.stem(x)
        #     x = self.stage1(x)
        #     x = self.downsample1(x)
        #     x = self.stage2(x)
        #     x = self.gap(x)
        #     return self.classifier(x)
        super().__init__(model_name, num_classes, in_channels, pretrained, weights, **kwargs)


def build_deepcnn_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained: bool = False,
    weights: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """DeepCNNモデルを構築.

    pretrained=True の場合、バックボーンは ImageNet 重みで初期化し、
    最終分類層のみ num_classes 用に差し替えます。

    Args:
        model_name: モデル名（例: vgg16, resnet50, mobilenet_v3_small, efficientnet_b0, convnext_tiny）
        num_classes: 分類クラス数
        in_channels: 入力チャンネル数（3推奨、1の場合は最初のConvを差し替えが必要なモデルあり）
        pretrained: ImageNet事前学習済み重みを使うか（weights が None の場合に有効）
        weights: 使用する重みの名前（例: "IMAGENET1K_V1", "DEFAULT"）。None かつ pretrained=True なら "DEFAULT"
        **kwargs: モデル構築時に渡す追加引数

    Returns:
        nn.Module: 分類モデル
    """
    family = _get_backbone_family(model_name)
    model_cls_map = {
        "vgg": VGGDeepCNN,
        "resnet": ResNetDeepCNN,
        "mobilenet": MobileNetDeepCNN,
        "efficientnet": EfficientNetDeepCNN,
        "convnext": ConvNeXtDeepCNN,
    }
    model_cls = model_cls_map[family]
    model = model_cls(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    return model


def _replace_classifier(model: nn.Module, num_classes: int, family: str, model_name: str) -> nn.Module:
    """最終分類層を num_classes 用に差し替え."""
    if family == "vgg":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif family == "resnet":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif family == "mobilenet":
        # mobilenet_v2: classifier[1], mobilenet_v3: classifier[3]
        if "v2" in model_name or model_name == "mobilenet_v2":
            layer = model.classifier[1]
        else:
            layer = model.classifier[3]
        in_features = layer.in_features
        new_fc = nn.Linear(in_features, num_classes)
        if "v2" in model_name or model_name == "mobilenet_v2":
            model.classifier[1] = new_fc
        else:
            model.classifier[3] = new_fc
    elif family == "efficientnet":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif family == "convnext":
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"未知の family: {family}")
    return model


def _replace_first_conv(
    model: nn.Module, in_channels: int, family: str, _model_name: str
) -> nn.Module:
    """最初のConv2dを in_channels 用に差し替え（MNIST等1ch入力用）."""
    if family == "vgg":
        old_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
    elif family == "resnet":
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
    elif family in ("mobilenet", "efficientnet", "convnext"):
        # Conv2dNormActivation の (0) が Conv2d
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=getattr(old_conv, "bias", None) is not None,
        )
    else:
        raise ValueError(f"in_channels != 3 の差し替え未対応: {family}")

    return model
