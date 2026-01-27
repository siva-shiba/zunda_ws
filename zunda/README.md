# Zunda: 東北ずん子project画像データセット用ライブラリ

東北ずん子projectの画像データセットを扱うためのPyTorchデータローダーライブラリです。

## 機能

- 画像ファイルと対応するテキストファイル（タグ）をペアで読み込み
- PyTorchの`Dataset`と`DataLoader`に対応
- カスタム画像変換・テキスト変換に対応
- 複数の画像形式（PNG, BMP, JPG等）に対応

## インストール

### 1. パッケージのインストール

プロジェクトのルートディレクトリで開発モードでインストール：

```bash
pip install -e .
```

これにより、`zunda`パッケージがインストールされ、どこからでもインポートできるようになります。

### 2. 依存パッケージ

このライブラリを使用するには、PyTorchとPillowが必要です（`setup.py`で自動的にインストールされます）：

```bash
pip install torch torchvision pillow
```

## 基本的な使い方

### 1. データセットを直接使用

```python
from zunda import TouhokuProjectDataset

# データセットを作成
dataset = TouhokuProjectDataset(
    data_root='data/touhoku_project_images',
)

print(f"データセットサイズ: {len(dataset)}")

# サンプルを取得
sample = dataset[0]
image = sample['image']  # PIL Image
text = sample['text']    # タグ文字列
image_path = sample['image_path']
text_path = sample['text_path']
```

### 2. 画像変換を適用

```python
from torchvision import transforms
from zunda import TouhokuProjectDataset

# 画像変換を定義
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# データセットを作成
dataset = TouhokuProjectDataset(
    data_root='data/touhoku_project_images',
    transform=image_transform,
)

sample = dataset[0]
image_tensor = sample['image']  # torch.Tensor
```

### 3. DataLoaderを使用

```python
from torchvision import transforms
from zunda import create_dataloader

# 画像変換を定義
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# DataLoaderを作成
dataloader = create_dataloader(
    data_root='data/touhoku_project_images',
    batch_size=32,
    shuffle=True,
    num_workers=4,
    image_transform=image_transform,
)

# バッチを取得
for batch in dataloader:
    images = batch['image']  # [batch_size, C, H, W]
    texts = batch['text']    # [batch_size] の文字列リスト
    # 学習処理...
```

### 4. train/val/testに分割してDataLoaderを作成

```python
from torchvision import transforms
from zunda import create_train_val_test_dataloaders

# 学習用と検証用で異なる変換を定義
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # データ拡張
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# train/val/testに分割してDataLoaderを作成
train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
    data_root='data/touhoku_project_images',
    train_ratio=0.7,      # 70% を学習用
    val_ratio=0.15,        # 15% を検証用
    test_ratio=0.15,       # 15% をテスト用
    batch_size=32,
    shuffle_train=True,
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=val_transform,
    random_seed=42,        # 再現性のため
)

# 学習ループ
for epoch in range(num_epochs):
    # 学習
    for batch in train_loader:
        images = batch['image']
        texts = batch['text']
        # 学習処理...
    
    # 検証
    for batch in val_loader:
        images = batch['image']
        texts = batch['text']
        # 検証処理...
```

## API リファレンス

### `TouhokuProjectDataset`

画像とテキストのペアを読み込むデータセットクラス。

**パラメータ:**
- `data_root` (str): データセットのルートディレクトリパス
- `transform` (Callable, optional): 画像に適用する変換（PIL Image -> Tensor等）
- `text_transform` (Callable, optional): テキストに適用する変換
- `image_extensions` (List[str], optional): 読み込む画像ファイルの拡張子リスト（デフォルト: `['.png', '.jpg', '.jpeg', '.bmp']`）

**戻り値:**
- `dict`: {
    - `'image'`: 変換後の画像（PIL Image または Tensor）
    - `'text'`: テキスト（タグ）文字列
    - `'image_path'`: 画像ファイルパス
    - `'text_path'`: テキストファイルパス（存在しない場合は空文字列）
  }

### `create_dataloader`

DataLoaderを作成するヘルパー関数。

**パラメータ:**
- `data_root` (str): データセットのルートディレクトリパス
- `batch_size` (int): バッチサイズ（デフォルト: 32）
- `shuffle` (bool): データをシャッフルするか（デフォルト: True）
- `num_workers` (int): データローディングのワーカー数（デフォルト: 4）
- `pin_memory` (bool): GPU転送を高速化するためにメモリをピン留めするか（デフォルト: True）
- `image_transform` (Callable, optional): 画像に適用する変換（Noneの場合はToTensorのみ）
- `text_transform` (Callable, optional): テキストに適用する変換
- `image_extensions` (List[str], optional): 読み込む画像ファイルの拡張子リスト

**戻り値:**
- `DataLoader`: PyTorchのDataLoaderインスタンス

### `create_train_val_test_dataloaders`

データセットをtrain/val/testに分割してDataLoaderを作成する関数。

**パラメータ:**
- `data_root` (str): データセットのルートディレクトリパス
- `train_ratio` (float): 学習データの割合（デフォルト: 0.7）
- `val_ratio` (float): 検証データの割合（デフォルト: 0.15）
- `test_ratio` (float): テストデータの割合（デフォルト: 0.15）
- `batch_size` (int): バッチサイズ（デフォルト: 32）
- `shuffle_train` (bool): 学習データをシャッフルするか（デフォルト: True）
- `num_workers` (int): データローディングのワーカー数（デフォルト: 4）
- `pin_memory` (bool): GPU転送を高速化するためにメモリをピン留めするか（デフォルト: True）
- `train_transform` (Callable, optional): 学習データに適用する画像変換
- `val_transform` (Callable, optional): 検証データに適用する画像変換（Noneの場合はtrain_transformと同じ）
- `test_transform` (Callable, optional): テストデータに適用する画像変換（Noneの場合はval_transformと同じ）
- `text_transform` (Callable, optional): テキストに適用する変換
- `image_extensions` (List[str], optional): 読み込む画像ファイルの拡張子リスト
- `random_seed` (int, optional): ランダムシード（再現性のため）

**戻り値:**
- `Tuple[DataLoader, DataLoader, DataLoader]`: (train_loader, val_loader, test_loader)

**注意:**
- `train_ratio + val_ratio + test_ratio` は 1.0 である必要があります
- 分割はランダムに行われますが、`random_seed`を指定することで再現可能です

## データ構造

データセットは以下の構造を想定しています：

```
data/touhoku_project_images/
├── 01_LoRA学習用データ_A氏提供版_背景白/
│   ├── image1.png
│   ├── image1.txt
│   ├── image2.png
│   ├── image2.txt
│   └── ...
├── 02_LoRA学習用データ_B氏提供版_背景透過/
│   └── ...
└── ...
```

- 画像ファイルと同名の`.txt`ファイルがペアとして扱われます
- `.txt`ファイルにはLoRA学習用のタグ（カンマ区切り）が含まれています
- テキストファイルが存在しない画像も読み込めます（テキストは空文字列になります）

## 使用例

詳細な使用例は `work_dirs/0_example_usage.py` を参照してください。

```bash
python work_dirs/0_example_usage.py
```
