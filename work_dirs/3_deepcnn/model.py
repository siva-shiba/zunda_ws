"""DeepCNNモデル定義（VGG/ResNet/MobileNet/EfficientNet/ConvNeXt）.

torchvision の pre-trained モデルを Config で切り替えて利用します.
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
    """DeepCNNモデル（torchvision バックボーン切り替え）.

    1_mlp / 2_cnn と同様にクラスとして定義し、内部で torchvision モデルを保持する。
    """

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
    model = DeepCNN(
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
