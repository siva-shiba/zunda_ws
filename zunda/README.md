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

### `TouhokuProjectClassificationDataset`

キャラクター名をラベルとして使用する分類タスク用のデータセットクラス。

**パラメータ:**
- `data_root` (str): データセットのルートディレクトリパス
- `transform` (Callable, optional): 画像に適用する変換（PIL Image -> Tensor等）
- `text_transform` (Callable, optional): テキストに適用する変換
- `image_extensions` (List[str], optional): 読み込む画像ファイルの拡張子リスト（デフォルト: `['.png', '.jpg', '.jpeg', '.bmp']`）

**戻り値:**
- `dict`: {
    - `'image'`: 変換後の画像（PIL Image または Tensor）
    - `'label'`: クラスラベル（整数）
    - `'class_name'`: クラス名（文字列）
    - `'text'`: テキスト（タグ）文字列
    - `'image_path'`: 画像ファイルパス
    - `'text_path'`: テキストファイルパス（存在しない場合は空文字列）
  }

**メソッド:**
- `get_class_names()`: すべてのクラス名のリストを取得
- `get_class_to_idx()`: クラス名からインデックスへのマッピングを取得
- `get_idx_to_class()`: インデックスからクラス名へのマッピングを取得

**プロパティ:**
- `num_classes`: クラス数

### `create_classification_dataloader`

分類タスク用のDataLoaderを作成する関数。

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
- `Tuple[DataLoader, Dict[str, int], Dict[int, str]]`: (dataloader, class_to_idx, idx_to_class)

### `create_classification_train_val_test_dataloaders`

分類タスク用にデータセットをtrain/val/testに分割してDataLoaderを作成する関数。

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
- `Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]`: 
  (train_loader, val_loader, test_loader, class_to_idx, idx_to_class)

**注意:**
- `train_ratio + val_ratio + test_ratio` は 1.0 である必要があります
- 分割はランダムに行われますが、`random_seed`を指定することで再現可能です

## 公開データセットを追加する

学習スクリプト（`work_dirs/*/train.py`）と推論スクリプト（`work_dirs/*/inference.py`）は、
`DATASET_REGISTRY` に登録されたデータセットアダプタを config の `dataset` で選んで使う。
新しい公開データセットは **`zunda/dataset_adapters.py` にアダプタを1つ追加するだけ** で、
全 work_dir の学習・推論から使えるようになる。

| 登録名 | アダプタ | 備考 |
|---|---|---|
| `touhoku` | `TouhokuDataset` | ずんだ画像（デフォルト） |
| `mnist` | `MNISTDataset` | 1 チャネル、28px |
| `cifar10` | `CIFAR10Dataset` | 3 チャネル、32px（`image_size` にリサイズ） |

### 手順

1. `zunda/dataset_adapters.py` に `BaseDatasetAdapter` を継承したクラスを書き、`@register_dataset("名前")` を付ける
2. 3つのメソッドを実装する
   - `get_in_channels(cfg)`: 入力チャネル数（グレースケールなら 1、RGB なら 3）
   - `build_default_transforms(cfg)`: `(train_transform, val_transform)`。val 側は推論でも使うので augmentation を入れない
   - `build_dataloaders(cfg, logger, build_transforms_func)`: `(train_loader, val_loader, test_loader, class_to_idx, idx_to_class)`
3. torchvision のデータセットなら `_build_split_dataloaders` を使うと、train を train/val に分割して DataLoader を作るところまでやってくれる。
   バッチは `{"image": ..., "label": ...}` の形にする必要があるので、自前で作る場合は `ImageLabelDictDataset` で包む
4. config に `"dataset": "名前"` と `"data_root"` を書く（または `-o dataset=名前 -o data_root=...`）

```python
@register_dataset("fashion_mnist")
class FashionMNISTDataset(BaseDatasetAdapter):
    name = "fashion_mnist"

    @staticmethod
    def get_in_channels(cfg):
        return 1

    @staticmethod
    def build_default_transforms(cfg):
        tf = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        return tf, tf

    @classmethod
    def build_dataloaders(cls, cfg, logger, build_transforms_func=None):
        from torchvision.datasets import FashionMNIST

        train_tf, val_tf = (build_transforms_func or cls.build_default_transforms)(cfg)
        root = str(cfg.data_root)
        train_full = FashionMNIST(root, train=True, download=True, transform=train_tf)
        val_full = FashionMNIST(root, train=True, download=False, transform=val_tf)
        test_ds = FashionMNIST(root, train=False, download=True, transform=val_tf)
        return _build_split_dataloaders(cfg, logger, train_full, val_full, test_ds, list(train_full.classes))
```

### 推論との関係

学習時の設定はチェックポイントの `train_config` に丸ごと保存される。
`inference.py` はそこから `dataset`・`data_root`・分割比・seed を読み、`build_eval_dataloaders` で学習時と同じ分割を再現する
（transforms は各 work_dir の `build_transforms` の val 側）。そのため推論側の追加作業はない。

```bash
python inference.py outputs/<run_id>/checkpoints/best_model.pt          # data_root は学習時の値
python inference.py outputs/<run_id>/checkpoints/best_model.pt /other/path  # 上書きも可
```

### 注意

- 交差検証（`use_cv=true`）は今のところ `touhoku` のみ対応
- `4_vit` はデータセットによらず ImageNet の mean/std で正規化する（学習済み ViT に合わせるため）。1 チャネルのデータセットはそのままでは使えない

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

## 分類タスク用データセット

キャラクター名をラベルとして使用する分類タスク用のデータセットも提供しています。

### 基本的な使い方

```python
from zunda import TouhokuProjectClassificationDataset

# データセットを作成
dataset = TouhokuProjectClassificationDataset(
    data_root='data/touhoku_project_images',
)

print(f"データセットサイズ: {len(dataset)}")
print(f"クラス数: {dataset.num_classes}")
print(f"クラス一覧: {dataset.get_class_names()}")

# サンプルを取得
sample = dataset[0]
image = sample['image']      # PIL Image
label = sample['label']       # クラスラベル（整数）
class_name = sample['class_name']  # クラス名（文字列）
```

### train/val/testに分割

```python
from zunda import create_classification_train_val_test_dataloaders
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_loader, val_loader, test_loader, class_to_idx, idx_to_class = \
    create_classification_train_val_test_dataloaders(
        data_root='data/touhoku_project_images',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=32,
        train_transform=train_transform,
        val_transform=val_transform,
        random_seed=42,
    )

# 学習ループ
for batch in train_loader:
    images = batch['image']  # [batch_size, C, H, W]
    labels = batch['label']  # [batch_size] の整数テンソル
    # 学習処理...
```

### キャラクター名の正規化ルール

データセットは以下のルールでキャラクター名を正規化します：

1. **`キャラ名_〇〇`形式**: `itako_oc`, `itako_sd`, `zunko_sd` などは、ベース名（`itako`, `zunko`）に丸められます
2. **特殊なマッピング**:
   - `"20_zdm 1boy"` → `"zundamon"`
   - `"png_rgb_txt_deepdan"` → `"未分類"`

## 使用例

詳細な使用例は `work_dirs/0_example_usage.py` を参照してください。

```bash
python work_dirs/0_example_usage.py
```
