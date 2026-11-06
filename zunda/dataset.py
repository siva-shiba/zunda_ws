"""東北ずん子project画像データセット."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class TouhokuProjectDataset(Dataset):
    """東北ずん子project画像データセット.
    
    画像ファイルと対応するテキストファイル（タグ）をペアで読み込むデータセット.
    
    Args:
        data_root: データセットのルートディレクトリパス
        transform: 画像に適用する変換（PIL Image -> Tensor等）
        text_transform: テキストに適用する変換
        image_extensions: 読み込む画像ファイルの拡張子リスト
    """
    
    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        image_extensions: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.text_transform = text_transform
        
        if image_extensions is None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        self.image_extensions = [ext.lower() for ext in image_extensions]
        
        # データファイルのペアを収集
        self.samples = self._collect_samples()
        
    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        """画像ファイルとテキストファイルのペアを収集."""
        samples = []
        
        # 再帰的に画像ファイルを検索
        # 環境によらず同じ順序になるようにソートする
        for img_path in sorted(self.data_root.rglob('*')):
            if not img_path.is_file():
                continue
                
            # 画像ファイルかチェック
            if img_path.suffix.lower() not in self.image_extensions:
                continue
            
            # 対応するテキストファイルを探す
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                samples.append((img_path, txt_path))
            else:
                # テキストファイルがない場合も画像のみで追加（警告なし）
                samples.append((img_path, None))
        
        return samples
    
    def __len__(self) -> int:
        """データセットのサイズを返す."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """データサンプルを取得.
        
        Returns:
            dict: {
                'image': torch.Tensor,  # 変換後の画像テンソル
                'text': str,            # テキスト（タグ）文字列
                'image_path': str,      # 画像ファイルパス
                'text_path': str        # テキストファイルパス（存在しない場合は空文字列）
            }
        """
        img_path, txt_path = self.samples[idx]
        
        # 画像を読み込み
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"画像の読み込みに失敗しました: {img_path}") from e
        
        # 変換を適用
        if self.transform:
            image = self.transform(image)
        
        # テキストを読み込み
        text = ""
        if txt_path and txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as e:
                print(f"警告: テキストファイルの読み込みに失敗しました: {txt_path}")
        
        # テキスト変換を適用
        if self.text_transform:
            text = self.text_transform(text)
        
        return {
            'image': image,
            'text': text,
            'image_path': str(img_path),
            'text_path': str(txt_path) if txt_path else '',
        }
    
    @classmethod
    def create_dataloader(
        cls,
        data_root: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        image_extensions: Optional[List[str]] = None,
    ) -> DataLoader:
        """DataLoaderを作成するクラスメソッド.
        
        Args:
            data_root: データセットのルートディレクトリパス
            batch_size: バッチサイズ
            shuffle: データをシャッフルするかどうか
            num_workers: データローディングのワーカー数
            pin_memory: GPU転送を高速化するためにメモリをピン留めするか
            image_transform: 画像に適用する変換（Noneの場合はデフォルト変換を使用）
            text_transform: テキストに適用する変換
            image_extensions: 読み込む画像ファイルの拡張子リスト
        
        Returns:
            DataLoader: PyTorchのDataLoaderインスタンス
        """
        # デフォルトの画像変換（PIL Image -> Tensor）
        if image_transform is None:
            image_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        # データセットを作成
        dataset = cls(
            data_root=data_root,
            transform=image_transform,
            text_transform=text_transform,
            image_extensions=image_extensions,
        )
        
        # DataLoaderを作成
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        return dataloader
    
    @classmethod
    def create_train_val_test_dataloaders(
        cls,
        data_root: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        image_extensions: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """データセットをtrain/val/testに分割してDataLoaderを作成するクラスメソッド.
        
        Args:
            data_root: データセットのルートディレクトリパス
            train_ratio: 学習データの割合（デフォルト: 0.7）
            val_ratio: 検証データの割合（デフォルト: 0.15）
            test_ratio: テストデータの割合（デフォルト: 0.15）
            batch_size: バッチサイズ
            shuffle_train: 学習データをシャッフルするかどうか
            num_workers: データローディングのワーカー数
            pin_memory: GPU転送を高速化するためにメモリをピン留めするか
            train_transform: 学習データに適用する画像変換
            val_transform: 検証データに適用する画像変換（Noneの場合はtrain_transformと同じ）
            test_transform: テストデータに適用する画像変換（Noneの場合はval_transformと同じ）
            text_transform: テキストに適用する変換
            image_extensions: 読み込む画像ファイルの拡張子リスト
            random_seed: ランダムシード（再現性のため）
        
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
        
        Raises:
            ValueError: train_ratio + val_ratio + test_ratio が 1.0 でない場合
        """
        # 割合の検証
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio は 1.0 である必要があります。"
                f"現在の値: {total_ratio}"
            )
        
        # ランダムシードの設定
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
        
        # デフォルトの画像変換
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        if val_transform is None:
            val_transform = train_transform
        if test_transform is None:
            test_transform = val_transform
        
        # データセットを作成（変換なしで一度作成してサイズを取得）
        full_dataset = cls(
            data_root=data_root,
            transform=None,  # 後で分割後に適用
            text_transform=text_transform,
            image_extensions=image_extensions,
        )
        
        dataset_size = len(full_dataset)
        
        # データセットを分割
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size  # 端数処理のため
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed if random_seed is not None else 42)
        )
        
        # 各データセットに変換を適用するためのラッパー
        class TransformDataset:
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                sample = self.base_dataset[idx]
                # sampleは辞書なので、コピーを作成してから変換を適用
                result = sample.copy()
                if self.transform and 'image' in result:
                    result['image'] = self.transform(result['image'])
                return result
        
        train_dataset_transformed = TransformDataset(train_dataset, train_transform)
        val_dataset_transformed = TransformDataset(val_dataset, val_transform)
        test_dataset_transformed = TransformDataset(test_dataset, test_transform)
        
        # DataLoaderを作成
        train_loader = DataLoader(
            train_dataset_transformed,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        val_loader = DataLoader(
            val_dataset_transformed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        test_loader = DataLoader(
            test_dataset_transformed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        return train_loader, val_loader, test_loader
