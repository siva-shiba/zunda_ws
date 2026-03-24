# DeepCNN分類タスク

分類タスク用のDeepCNNモデル（VGG/ResNet/MobileNet/EfficientNet/ConvNeXt）の学習スクリプトです。  
Config でバックボーンを切り替えて利用します。

![DeepCNNアーキテクチャ](3_deepcnn_architecture.png)

## モデル構造

- **入力**: 画像 (224×224×3、image_size で変更可)
- **Backbone**: torchvision の pre-trained モデル（Config で選択）
- **特徴抽出後**: Global Average Pooling
- **分類層**: Linear → num_classes（クラス数に応じて差し替え）

### 各アーキテクチャの構造

#### VGG (VGG16)
層の深い畳み込みブロック（3×3 Conv の積み重ね）

![VGGアーキテクチャ](vgg_architecture.png)

#### ResNet (ResNet50)
残差接続（Skip Connection）を持つブロック

![ResNetアーキテクチャ](resnet_architecture.png)

#### MobileNet (MobileNetV3-Small)
軽量・高速な Depthwise Separable Convolution と Inverted Residual

![MobileNetアーキテクチャ](mobilenet_architecture.png)

#### EfficientNet (EfficientNet-B0)
複合スケーリング（深さ・幅・解像度）による効率的な設計

![EfficientNetアーキテクチャ](efficientnet_architecture.png)

#### ConvNeXt (ConvNeXt-Tiny)
モダンな CNN 設計（LayerNorm、大きなカーネル等）

![ConvNeXtアーキテクチャ](convnext_architecture.png)

## ファイル構成

- `3_deepcnn_architecture.png`: 全体モデル構造図
- `vgg_architecture.png`, `resnet_architecture.png`, `mobilenet_architecture.png`, `efficientnet_architecture.png`, `convnext_architecture.png`: 各アーキテクチャ構造図
- `train.py`: DeepCNNモデルの学習スクリプト
- `model.py`: モデルファクトリ（torchvision の pre-trained モデルをラップ）
- `inference.py`: 推論スクリプト
- `configs/`: 設定ファイル（JSON）ディレクトリ
  - **default はありません**。モデル別に `1_vgg.json`, `2_resnet.json` などを指定してください。

## 使用方法

**config は必須**です。設定ファイル（JSON）と `-o key=value` による上書きで学習します。

### 基本的な使用方法

```bash
python train.py configs/<config名>.json [-o KEY=VALUE ...]
```

### 利用可能な Config

| Config | バックボーン | モデル |
|--------|-------------|--------|
| `configs/1_vgg.json` | VGG | vgg16 |
| `configs/2_resnet.json` | ResNet | resnet50 |
| `configs/3_mobilenet.json` | MobileNet | mobilenet_v3_small |
| `configs/4_efficientnet.json` | EfficientNet | efficientnet_b0 |
| `configs/5_convnext.json` | ConvNeXt | convnext_tiny |

### 例

```bash
# VGGで学習
python train.py configs/1_vgg.json

# ResNetで学習
python train.py configs/2_resnet.json

# MobileNetで学習
python train.py configs/3_mobilenet.json

# 上書き
python train.py configs/4_efficientnet.json \
    -o image_size=224 \
    -o batch_size=16 \
    -o epochs=50

# data_root を上書き
python train.py configs/5_convnext.json -o data_root=data/touhoku_project_images
```

### パス解決

`data_root` など config 内の**相対パス**は、**リポジトリルート**（zunda_ws/）を基準に解決されます。

### 設定ファイル

`configs/1_vgg.json` など JSON で設定を記述します。主な項目:

| 項目 | 説明 | 例 |
|------|------|-----|
| data_root | データセットのルートディレクトリ（必須） | data/touhoku_project_images |
| model_name | モデル名（必須） | vgg16, resnet50, mobilenet_v3_small, efficientnet_b0, convnext_tiny |
| image_size | 画像サイズ | 224（ImageNet系は224推奨） |
| batch_size | バッチサイズ | 32 |
| epochs | エポック数 | 100 |
| lr | 学習率 | 0.001 |
| pretrained | ImageNet事前学習済み重みを使うか | true |
| num_workers | DataLoader ワーカー数 | 4 |
| use_wandb | WANDB を使用する | true |

### 利用可能な model_name

- **VGG**: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **EfficientNet**: efficientnet_b0 ～ efficientnet_b7
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base, convnext_large

`-o model_name=resnet18` などで上書き可能です。

## ログ機能

- **コンソール出力**: 標準出力にログが表示されます
- **ログファイル**: `outputs/YYYYMMDD_HHMMSS/train.log` にタイムスタンプ付きで保存されます
- **WANDB**: 実験結果をWANDBに記録（デフォルトで有効）

## 出力

### チェックポイント

- `outputs/YYYYMMDD_HHMMSS/checkpoints/best_model.pt`: 検証精度が最も高いモデル
- `outputs/YYYYMMDD_HHMMSS/checkpoints/final_model.pt`: 最終エポックのモデル

チェックポイントには `model_config`（model_name, pretrained, in_channels, image_size, num_classes）が含まれ、推論時に自動でモデルを再構築できます。

### 推論

```bash
python inference.py <checkpoint_path> <data_root>

# 例
python inference.py outputs/20240324_120000/checkpoints/best_model.pt data/touhoku_project_images
```

## トラブルシューティング

### 共有メモリ（shm）エラー

Docker環境で `RuntimeError: DataLoader worker exited unexpectedly` が発生する場合:

```bash
python train.py configs/1_vgg.json -o num_workers=0
```

### WANDBを使わない

```bash
python train.py configs/1_vgg.json -o use_wandb=false
```
