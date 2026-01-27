"""分類タスク用データセット."""

from pathlib import Path
from typing import Optional, List, Callable, Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

from .dataset import TouhokuProjectDataset
from torch.utils.data import DataLoader
from typing import Tuple


def normalize_character_name(folder_name: str) -> str:
    """キャラクター名を正規化.

    特殊なルールを適用してキャラクター名を正規化します。

    Args:
        folder_name: フォルダ名（例: "itako_oc", "zunko_sd"）

    Returns:
        正規化されたキャラクター名
    """
    # 特殊なマッピング
    special_mappings = {
        "20_zdm 1boy": "zundamon",
        "png_rgb_txt_deepdan": "unknown",
    }

    if folder_name in special_mappings:
        return special_mappings[folder_name]

    # キャラ名_〇〇の形式をキャラ名に丸める
    # 例: "itako_oc" -> "itako", "zunko_sd" -> "zunko"
    if "_" in folder_name:
        base_name = folder_name.split("_")[0]
        # 既知のベース名かチェック（オプション: より厳密なチェックが必要な場合）
        return base_name

    return folder_name


class TouhokuProjectClassificationDataset(TouhokuProjectDataset):
    """東北ずん子project画像データセット（分類タスク用）.

    キャラクター名をラベルとして使用する分類タスク用のデータセット.

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

        # データファイルとラベルのペアを収集
        self.samples, self.class_to_idx = self._collect_samples()
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def _collect_samples(self) -> Tuple[List[Tuple[Path, Path, int]], Dict[str, int]]:
        """画像ファイルとラベルのペアを収集.

        Returns:
            (samples, class_to_idx):
                - samples: [(img_path, txt_path, label_idx), ...]
                - class_to_idx: {class_name: class_idx}
        """
        samples = []
        class_to_idx = {}

        # 再帰的に画像ファイルを検索
        for img_path in self.data_root.rglob('*'):
            if not img_path.is_file():
                continue

            # 画像ファイルかチェック
            if img_path.suffix.lower() not in self.image_extensions:
                continue

            # キャラクター名を取得（ROOT/絵師/キャラ名/画像.png の構造から）
            # img_path.parent = キャラ名フォルダ
            # img_path.parent.parent = 絵師フォルダ
            # img_path.parent.parent.parent = ROOT
            try:
                character_folder = img_path.parent.name
            except (IndexError, AttributeError):
                # 構造が想定と異なる場合はスキップ
                continue

            # キャラクター名を正規化
            normalized_name = normalize_character_name(character_folder)

            # クラスIDを取得または作成
            if normalized_name not in class_to_idx:
                class_to_idx[normalized_name] = len(class_to_idx)

            label_idx = class_to_idx[normalized_name]

            # 対応するテキストファイルを探す
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                txt_path = None

            samples.append((img_path, txt_path, label_idx))

        return samples, class_to_idx

    def __len__(self) -> int:
        """データセットのサイズを返す."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """データサンプルを取得.

        Returns:
            dict: {
                'image': torch.Tensor,  # 変換後の画像テンソル
                'label': int,            # クラスラベル（整数）
                'class_name': str,       # クラス名（文字列）
                'text': str,             # テキスト（タグ）文字列
                'image_path': str,       # 画像ファイルパス
                'text_path': str         # テキストファイルパス（存在しない場合は空文字列）
            }
        """
        img_path, txt_path, label_idx = self.samples[idx]

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

        class_name = self.idx_to_class[label_idx]

        return {
            'image': image,
            'label': label_idx,
            'class_name': class_name,
            'text': text,
            'image_path': str(img_path),
            'text_path': str(txt_path) if txt_path else '',
        }

    def get_class_names(self) -> List[str]:
        """すべてのクラス名のリストを取得."""
        return list(self.class_to_idx.keys())

    def get_class_to_idx(self) -> Dict[str, int]:
        """クラス名からインデックスへのマッピングを取得."""
        return self.class_to_idx.copy()

    def get_idx_to_class(self) -> Dict[int, str]:
        """インデックスからクラス名へのマッピングを取得."""
        return self.idx_to_class.copy()
    
    @classmethod
    def create_classification_dataloader(
        cls,
        data_root: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        image_extensions: Optional[List[str]] = None,
    ) -> Tuple[DataLoader, Dict[str, int], Dict[int, str]]:
        """分類タスク用のDataLoaderを作成するクラスメソッド.
        
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
            Tuple[DataLoader, Dict[str, int], Dict[int, str]]: 
                (dataloader, class_to_idx, idx_to_class)
        """
        from torchvision import transforms
        
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
        
        return dataloader, dataset.get_class_to_idx(), dataset.get_idx_to_class()
    
    @classmethod
    def create_classification_train_val_test_dataloaders(
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
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
        """分類タスク用にデータセットをtrain/val/testに分割してDataLoaderを作成するクラスメソッド.
        
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
            Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]: 
                (train_loader, val_loader, test_loader, class_to_idx, idx_to_class)
        
        Raises:
            ValueError: train_ratio + val_ratio + test_ratio が 1.0 でない場合
        """
        # 親クラスのメソッドを呼び出して分割ロジックを再利用
        train_loader, val_loader, test_loader = cls.create_train_val_test_dataloaders(
            data_root=data_root,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            text_transform=text_transform,
            image_extensions=image_extensions,
            random_seed=random_seed,
        )
        
        # クラスマッピングを取得（最初のデータセットから）
        # データセットを作成してクラスマッピングを取得
        from torchvision import transforms
        temp_dataset = cls(
            data_root=data_root,
            transform=None,
            text_transform=text_transform,
            image_extensions=image_extensions,
        )
        class_to_idx = temp_dataset.get_class_to_idx()
        idx_to_class = temp_dataset.get_idx_to_class()
        
        return train_loader, val_loader, test_loader, class_to_idx, idx_to_class
