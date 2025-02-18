"""Description: CIFAR-10 データセットを ImageNet-style に変換するスクリプト.

100枚ずつのデータのみを使用し、各クラスの画像をフォルダに保存します.
"""
import os
from torchvision import datasets
from torchvision.transforms import ToPILImage
import random

# データ保存先
root_dir = "cifar10-image-net-style"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "test")

# クラスラベル
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 訓練データと検証データの割合
train_ratio = 0.8  # 80% を train, 20% を val
num_samples_per_class = 500  # 各クラス100枚のデータのみ使用

# データセットをダウンロード
dataset = datasets.CIFAR10(root=root_dir, train=True, download=True)

# 出力フォルダを作成
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# 訓練データの処理
class_counts = {cls: 0 for cls in classes}  # 各クラスのデータ数カウント
to_pil = ToPILImage()

# 訓練データをフォルダに保存
data = dataset.data
labels = dataset.targets

# 各クラス num_samples_per_class 枚のみ使用
for img, label in zip(data, labels):
    class_name = classes[label]

    # 各クラス num_samples_per_class 枚まで制限
    if class_counts[class_name] >= num_samples_per_class:
        continue

    img = to_pil(img)
    filename = f"{class_counts[class_name]:04d}.png"

    if random.random() < train_ratio:
        img.save(os.path.join(train_dir, class_name, filename))
    else:
        img.save(os.path.join(val_dir, class_name, filename))

    class_counts[class_name] += 1

print(f"✅ CIFAR-10 を ImageNet-style に変換完了！")
print(f"訓練データ: {sum(class_counts.values()) * train_ratio:.0f} 枚")
print(f"検証データ: {sum(class_counts.values()) * (1 - train_ratio):.0f} 枚")
