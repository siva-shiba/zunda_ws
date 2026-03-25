"""ViT分類モデル定義.

torchvision の Vision Transformer（vit_b_16 / vit_b_32 / vit_l_16 等）を
Config で切り替えて利用します。
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torchvision import models


VIT_MODEL_NAMES = [
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
]


def build_vit_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained: bool = False,
    weights: Optional[str] = None,
    image_size: int = 224,
    **kwargs,
) -> nn.Module:
    """ViT モデルを構築.

    - ``pretrained=True`` かつ ``weights=None`` の場合は torchvision の DEFAULT
      重みを使う想定です。
    - 重みを使う場合（ImageNet 事前学習）は分類ヘッドのみ num_classes に差し替えます。
    - 入力 ``in_channels`` が 3 以外の場合は、patch embedding の ``conv_proj`` を差し替えます。
      （本リポジトリの dataset は基本的に RGB=3 想定）
    """

    if model_name not in VIT_MODEL_NAMES:
        raise ValueError(
            f"未知の ViT model_name: {model_name}. "
            f"利用可能: {VIT_MODEL_NAMES}"
        )

    load_weights = weights
    if load_weights is None and pretrained:
        load_weights = "DEFAULT"

    # weights を渡す場合は num_classes が 1000 前提になりやすいので、
    # 一旦作ってから head だけ差し替えます。
    if load_weights is not None:
        model = models.get_model(
            model_name,
            weights=load_weights,
            image_size=image_size,
            **kwargs,
        )
        _replace_vit_head(model, num_classes)
    else:
        model = models.get_model(
            model_name,
            weights=None,
            num_classes=num_classes,
            image_size=image_size,
            **kwargs,
        )

    if in_channels != 3:
        _replace_vit_conv_proj(model, in_channels=in_channels)

    return model


def _replace_vit_head(model: nn.Module, num_classes: int) -> None:
    """torchvision ViT の分類ヘッドを差し替える."""

    # torchvision の ViT は heads.head に Linear が入っている構造が一般的です。
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return

    # 念のためのフォールバック（バージョン差異対策）
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return

    raise ValueError("ViT head の差し替えに失敗しました（heads.head を想定）")


def _replace_vit_conv_proj(model: nn.Module, in_channels: int) -> None:
    """patch embedding の conv_proj を in_channels 用に差し替える."""

    if not hasattr(model, "conv_proj"):
        raise ValueError("ViT conv_proj が見つかりません。入力チャネル差し替え未対応です。")

    old = model.conv_proj
    if not isinstance(old, nn.Conv2d):
        raise ValueError(f"ViT conv_proj の型が想定外です: {type(old)}")

    # conv_proj は（patch_size, patch_size）カーネル・stride（patch_size, patch_size）
    # であることが多いです。
    new_conv = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    model.conv_proj = new_conv
