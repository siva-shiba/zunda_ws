"""東北ずん子project画像データセット."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset


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
        for img_path in self.data_root.rglob('*'):
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
