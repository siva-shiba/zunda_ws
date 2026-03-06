"""分類タスク用データセット."""

from pathlib import Path
from typing import Optional, List, Callable, Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

from .data_augmentation import create_weighted_sampler

from .dataset import TouhokuProjectDataset


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


def extract_base_filename(filename: str) -> List[str]:
    """ファイル名からベース名を抽出（-a, -bなどのサフィックスを考慮）.

    例:
        "a1zunko77-b.png" -> ["a1zunko77-b", "a1zunko77"]
        "a1zunko77.png" -> ["a1zunko77"]
        "ita_3008.png" -> ["ita_3008"]

    Args:
        filename: ファイル名（拡張子ありでもなしでも可）

    Returns:
        ベース名のリスト（完全なファイル名から、サフィックスを除いたものまで）
    """
    # 拡張子を除去
    stem = Path(filename).stem

    # ベース名の候補リスト
    base_names = [stem]

    # -a, -b, -c などのパターンを検出
    # 例: "a1zunko77-b" -> ["a1zunko77-b", "a1zunko77"]
    if '-' in stem:
        # 最後の"-"以降を削除したベース名も追加
        parts = stem.rsplit('-', 1)
        if len(parts) == 2:
            # 最後の部分が1文字のアルファベットまたは数字の場合
            suffix = parts[1]
            if len(suffix) == 1 and (suffix.isalpha() or suffix.isdigit()):
                base_names.append(parts[0])

    return base_names


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

        # ファイル名からクラス名へのマッピングを作成（rgb_zzm_プレフィックスを持つファイルのクラス推測用）
        # 既に分類されているデータから、ファイル名→クラス名のマッピングを作成
        filename_to_class: Dict[str, str] = {}

        # まず、すべての画像をスキャンしてファイル名→クラス名のマッピングを作成
        for img_path in self.data_root.rglob('*'):
            if not img_path.is_file():
                continue

            # 画像ファイルかチェック
            if img_path.suffix.lower() not in self.image_extensions:
                continue

            # キャラクター名を取得（ROOT/絵師/キャラ名/画像.png の構造から）
            try:
                character_folder = img_path.parent.name
            except (IndexError, AttributeError):
                continue

            # キャラクター名を正規化
            normalized_name = normalize_character_name(character_folder)

            # rgb_zzm_プレフィックスでないファイルの場合、ファイル名→クラス名のマッピングに追加
            filename = img_path.name
            if not filename.startswith('rgb_zzm_'):
                # ファイル名のベース名（-a, -bなどのサフィックスを考慮）を取得
                base_names = extract_base_filename(filename)
                for base_name in base_names:
                    # 各ベース名に対してマッピングを作成（既存のものは上書きしない）
                    if base_name not in filename_to_class:
                        filename_to_class[base_name] = normalized_name

        # 次に、すべての画像を処理してサンプルを作成
        for img_path in self.data_root.rglob('*'):
            if not img_path.is_file():
                continue

            # 画像ファイルかチェック
            if img_path.suffix.lower() not in self.image_extensions:
                continue

            # キャラクター名を取得（ROOT/絵師/キャラ名/画像.png の構造から）
            try:
                character_folder = img_path.parent.name
            except (IndexError, AttributeError):
                # 構造が想定と異なる場合はスキップ
                continue

            # キャラクター名を正規化
            normalized_name = normalize_character_name(character_folder)

            # rgb_zzm_プレフィックスを持つファイルの場合、ファイル名から元のファイル名を推測
            filename = img_path.name
            if filename.startswith('rgb_zzm_'):
                # rgb_zzm_プレフィックスを削除して元のファイル名を取得
                original_filename = filename[8:]  # 'rgb_zzm_' の8文字を削除

                # ベース名の候補を取得（-a, -bなどのサフィックスを考慮）
                base_names = extract_base_filename(original_filename)

                # ファイル名→クラス名のマッピングからクラス名を取得
                matched_class = None
                for base_name in base_names:
                    if base_name in filename_to_class:
                        matched_class = filename_to_class[base_name]
                        break

                if matched_class:
                    # 見つかった場合は、そのクラス名を使用
                    normalized_name = matched_class
                    # print(f"ファイル名からクラスを推測: {filename} -> {normalized_name} (マッチ: {base_name})")

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
        exclude_class: Optional[List[str]] = ["unknown"],
        use_stratified_split: bool = True,
        use_weighted_sampler: bool = False,
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
            exclude_class: train/val/testのいずれにも含めず完全に除外するクラス名のリスト
                                  （デフォルト: ["unknown"] - unknownクラスを使用しない）
            use_stratified_split: Stratified Splitを使用するか（クラス比率を保ったまま分割）
            use_weighted_sampler: Weighted Random Samplerを使用するか（少数クラスを多くサンプリング）

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
                (train_loader, val_loader, test_loader, class_to_idx, idx_to_class)

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
        from torchvision import transforms
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

        # クラスマッピングを保存
        class_to_idx = full_dataset.get_class_to_idx()
        idx_to_class = full_dataset.get_idx_to_class()

        # 除外するクラスのインデックスを取得
        exclude_indices = set()
        for class_name in exclude_class:
            if class_name in class_to_idx:
                exclude_indices.add(class_to_idx[class_name])

        # サンプルを除外クラスとそれ以外に分離
        exclude_samples = []
        include_samples = []

        for idx in range(len(full_dataset)):
            img_path, txt_path, label_idx = full_dataset.samples[idx]
            if label_idx in exclude_indices:
                exclude_samples.append(idx)
            else:
                include_samples.append(idx)

        # 除外クラス以外のデータセットを作成
        if len(include_samples) == 0:
            raise ValueError("除外クラス以外のデータが存在しません。")

        # インデックスでサブセットを作成するためのカスタムデータセット
        class FilteredDataset:
            def __init__(self, base_dataset, indices):
                self.base_dataset = base_dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.base_dataset[self.indices[idx]]

        include_dataset = FilteredDataset(full_dataset, include_samples)

        # 除外クラス以外のデータをtrain/val/testに分割
        if use_stratified_split:
            # Stratified Split: クラス比率を保ったまま分割
            # ラベル取得
            labels = np.array([sample['label'] for sample in include_dataset])
            indices = np.arange(len(include_dataset))

            # クラスごとのサンプル数をチェック
            label_counts = Counter(labels)
            min_samples = min(label_counts.values())

            # trainとtemp（val+test）に分割
            use_stratify_first = min_samples >= 2
            train_indices, temp_indices = train_test_split(
                indices,
                test_size=(val_ratio + test_ratio),
                stratify=labels if use_stratify_first else None,
                random_state=random_seed if random_seed is not None else 42
            )

            # tempをvalとtestに分割
            temp_labels = labels[temp_indices]
            temp_label_counts = Counter(temp_labels)
            temp_min_samples = min(temp_label_counts.values()) if temp_label_counts else 0

            # val/test分割でもstratifyを使うには、各クラスに最低2サンプル必要
            use_stratify_second = temp_min_samples >= 2
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=temp_labels if use_stratify_second else None,
                random_state=random_seed if random_seed is not None else 42
            )

            # インデックスからデータセットを作成
            class StratifiedDataset:
                def __init__(self, base_dataset, indices):
                    self.base_dataset = base_dataset
                    self.indices = indices

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, idx):
                    return self.base_dataset[self.indices[idx]]

            train_dataset = StratifiedDataset(include_dataset, train_indices)
            val_dataset = StratifiedDataset(include_dataset, val_indices)
            test_dataset = StratifiedDataset(include_dataset, test_indices)
        else:
            # 通常のランダム分割
            dataset_size = len(include_dataset)
            train_size = int(train_ratio * dataset_size)
            val_size = int(val_ratio * dataset_size)
            test_size = dataset_size - train_size - val_size  # 端数処理のため

            train_dataset, val_dataset, test_dataset = random_split(
                include_dataset,
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

        # Weighted Random Samplerの準備
        sampler = None
        if use_weighted_sampler and shuffle_train:
            sampler = create_weighted_sampler(
                train_dataset_transformed,
                use_weighted_sampler=True,
            )
            shuffle_train = False  # samplerを使用する場合はshuffleはFalseにする

        # DataLoaderを作成
        train_loader = DataLoader(
            train_dataset_transformed,
            batch_size=batch_size,
            shuffle=shuffle_train if sampler is None else False,
            sampler=sampler,
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

        return train_loader, val_loader, test_loader, class_to_idx, idx_to_class
