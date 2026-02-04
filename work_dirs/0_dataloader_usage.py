"""データローダーの使用例."""

import sys
from pathlib import Path

from torchvision import transforms
from zunda import (
    TouhokuProjectDataset,
    TouhokuProjectClassificationDataset,
)

# データセットのパス（プロジェクトルートからの絶対パス）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
DATA_ROOT = project_root / 'data' / 'touhoku_project_images'


def example_basic_usage():
    """基本的な使用例."""
    # データセットを作成
    dataset = TouhokuProjectDataset(
        data_root=str(DATA_ROOT),
    )

    print(f"データセットサイズ: {len(dataset)}")

    # サンプルを取得
    sample = dataset[0]
    # PIL Imageの場合は.size、Tensorの場合は.shapeを使用
    if hasattr(sample['image'], 'shape'):
        print(f"画像の形状: {sample['image'].shape}")
    else:
        print(f"画像のサイズ: {sample['image'].size}")
    print(f"テキスト: {sample['text'][:100]}...")  # 最初の100文字
    print(f"画像パス: {sample['image_path']}")


def example_with_transforms():
    """変換を適用した使用例."""
    # 画像変換を定義
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # データセットを作成
    dataset = TouhokuProjectDataset(
        data_root=str(DATA_ROOT),
        transform=image_transform,
    )

    sample = dataset[0]
    print(f"変換後の画像形状: {sample['image'].shape}")


def example_dataloader():
    """DataLoaderを使用した例."""
    # 画像変換を定義
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # DataLoaderを作成
    dataloader = TouhokuProjectDataset.create_dataloader(
        data_root=str(DATA_ROOT),
        batch_size=8,
        shuffle=True,
        num_workers=2,
        image_transform=image_transform,
    )

    # バッチを取得
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image']
        texts = batch['text']

        print(f"バッチ {batch_idx}:")
        print(f"  画像形状: {images.shape}")
        print(f"  テキスト数: {len(texts)}")
        print(f"  最初のテキスト: {texts[0][:50]}...")

        if batch_idx >= 2:  # 最初の3バッチのみ表示
            break


def example_train_val_test_split():
    """train/val/testに分割したDataLoaderの使用例."""
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
    train_loader, val_loader, test_loader = TouhokuProjectDataset.create_train_val_test_dataloaders(
        data_root=str(DATA_ROOT),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=8,
        shuffle_train=True,
        num_workers=2,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        random_seed=42,  # 再現性のため
    )

    print(f"学習データ: {len(train_loader.dataset)} サンプル")
    print(f"検証データ: {len(val_loader.dataset)} サンプル")
    print(f"テストデータ: {len(test_loader.dataset)} サンプル")

    # 学習データのバッチを取得
    train_batch = next(iter(train_loader))
    print(f"\n学習バッチ:")
    print(f"  画像形状: {train_batch['image'].shape}")
    print(f"  テキスト数: {len(train_batch['text'])}")

    # 検証データのバッチを取得
    val_batch = next(iter(val_loader))
    print(f"\n検証バッチ:")
    print(f"  画像形状: {val_batch['image'].shape}")
    print(f"  テキスト数: {len(val_batch['text'])}")


def example_classification_dataset():
    """分類タスク用データセットの使用例."""
    # データセットを作成
    dataset = TouhokuProjectClassificationDataset(
        data_root=str(DATA_ROOT),
    )

    print(f"データセットサイズ: {len(dataset)}")
    print(f"クラス数: {dataset.num_classes}")
    print(f"クラス一覧: {dataset.get_class_names()}")
    from collections import Counter
    count = Counter(dataset.get_class_names()[d["label"]] for d in dataset)
    print(f"各クラスデータ数: {count}")

    # サンプルを取得
    sample = dataset[0]
    print(f"\nサンプル:")
    print(f"  画像サイズ: {sample['image'].size}")
    print(f"  ラベル: {sample['label']}")
    print(f"  クラス名: {sample['class_name']}")
    print(f"  テキスト: {sample['text'][:50]}...")


def example_classification_dataloader():
    """分類タスク用DataLoaderの使用例."""
    # 画像変換を定義
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # DataLoaderを作成
    dataloader, class_to_idx, idx_to_class = TouhokuProjectClassificationDataset.create_classification_dataloader(
        data_root=str(DATA_ROOT),
        batch_size=8,
        shuffle=True,
        num_workers=2,
        image_transform=image_transform,
    )

    print(f"クラス数: {len(class_to_idx)}")
    print(f"クラス名 -> インデックス: {class_to_idx}")

    # バッチを取得
    batch = next(iter(dataloader))
    print(f"\nバッチ:")
    print(f"  画像形状: {batch['image'].shape}")
    print(f"  ラベル形状: {batch['label'].shape}")
    print(f"  ラベル: {batch['label'].tolist()}")
    print(f"  クラス名: {[idx_to_class[idx.item()] for idx in batch['label']]}")


def example_classification_train_val_test():
    """分類タスク用train/val/test分割の使用例."""
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
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = \
        TouhokuProjectClassificationDataset.create_classification_train_val_test_dataloaders(
            data_root=str(DATA_ROOT),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=8,
            shuffle_train=True,
            num_workers=2,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=val_transform,
            random_seed=42,
        )

    print(f"クラス数: {len(class_to_idx)}")
    print(f"クラス名 -> インデックス: {class_to_idx}")
    print(f"\n学習データ: {len(train_loader.dataset)} サンプル")
    print(f"検証データ: {len(val_loader.dataset)} サンプル")
    print(f"テストデータ: {len(test_loader.dataset)} サンプル")

    # 学習データのバッチを取得
    train_batch = next(iter(train_loader))
    print(f"\n学習バッチ:")
    print(f"  画像形状: {train_batch['image'].shape}")
    print(f"  ラベル形状: {train_batch['label'].shape}")
    print(f"  ラベル例: {train_batch['label'][:5].tolist()}")
    print(f"  クラス名例: {[idx_to_class[idx.item()] for idx in train_batch['label'][:5]]}")


if __name__ == '__main__':
    print("=== 基本的な使用例 ===")
    example_basic_usage()

    print("\n=== 変換を適用した使用例 ===")
    example_with_transforms()

    print("\n=== DataLoaderを使用した例 ===")
    example_dataloader()

    print("\n=== train/val/testに分割したDataLoaderの使用例 ===")
    example_train_val_test_split()

    print("\n=== 分類タスク用データセットの使用例 ===")
    example_classification_dataset()

    print("\n=== 分類タスク用DataLoaderの使用例 ===")
    example_classification_dataloader()

    print("\n=== 分類タスク用train/val/test分割の使用例 ===")
    example_classification_train_val_test()
