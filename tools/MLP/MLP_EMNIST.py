"""MLPでEMNISTの分類タスクのコード."""

import os
import argparse
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="MLP for EMNIST Classification")
    parser.add_argument(
        "data_dir", type=str, default="./data", help="Directory to store/load EMNIST data"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=5, help="Number of training epochs, [default=5]"
    )
    parser.add_argument(
        "--batch", "-b", type=int, default=128, help="Batch size for training, [default=128]"
    )
    parser.add_argument(
        "--lr", "-l", type=float, default=1e-3, help="Learning rate, [default=1e-3]"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2, help="Number of DataLoader workers, [default=2]"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility, [default=42]"
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default="./emnist_mlp_best.pt")
    parser.add_argument(
        "--no-samples", action="store_false", dest="show_samples", help="Do not show sample images from the dataset")
    parser.add_argument(
        "--no-curves", action="store_false", dest="show_curves", help="Do not show training curves")
    parser.add_argument(
        "--no-demo", action="store_false", dest="show_demo", help="Do not show demo predictions")
    parser.add_argument("--demo_k", "-i", type=int, default=12, help="Number of demo images to show from validation set, [default=12]")
    return parser.parse_args()


@dataclass
class EMNISTTrainerConfig:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    workers: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "./data"
    ckpt_path: str = "./emnist_mlp_best.pt" 
    show_samples: bool = True
    show_curves: bool = True
    show_demo: bool = True
    demo_k: int = 12


class EMNISTMLP(nn.Module):
    """EMNIST balanced (28x28 grayscale, 47 classes) 用のシンプルMLP."""
    def __init__(self, hidden: int = 256, num_classes: int = 47):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class EMNISTClassifier:
    """EMNIST分類器の実行クラス."""
    def __init__(self, cfg: EMNISTTrainerConfig, hidden: int = 256):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._set_seed(cfg.seed)

        # DataLoaders
        self.train_loader, self. val_loader, self.class_map = self._build_dataloaders(cfg)

        # Model
        self.model = EMNISTMLP(hidden=hidden, num_classes=47).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _set_seed(self, seed: int):
        """乱数シードを設定するメソッド."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_dataloaders(self, cfg: EMNISTTrainerConfig):
        """データローダーを構築するメソッド."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
            transforms.Lambda(lambda x: torch.flip(x, [2])),
        ])

        train_ds = torchvision.datasets.EMNIST(
            cfg.data_dir, split="balanced", train=True, download=True, transform=transform
        )
        val_ds = torchvision.datasets.EMNIST(
            cfg.data_dir, split="balanced", train=False, download=True, transform=transform
        )

        class_map = {i: c for i, c in enumerate(getattr(train_ds, "classes", [str(i) for i in range(47)]))}

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=True,
        )
        return train_loader, val_loader, class_map

    def _acc(self, logits: torch.Tensor, y: torch.Tensor):
        pred = logits.argmax(dim=1)
        return (pred == y).float().mean().item()

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n = 0

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = F.cross_entropy(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += self._acc(logits.detach(), y) * bs
            n += bs

        return total_loss / n, total_acc / n

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n = 0

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = F.cross_entropy(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += self._acc(logits, y) * bs
            n += bs

        return total_loss / n, total_acc / n

    def fit(self):
        """モデルの学習を実行するメソッド."""
        best_val_acc = -1.0

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss, tr_acc = self._train_one_epoch()
            va_loss, va_acc = self._evaluate(self.val_loader)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)

            print(
                f"Epoch {epoch:02d}/{self.cfg.epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val   loss {va_loss:.4f} acc {va_acc:.4f}"
            )

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                self.save(self.cfg.ckpt_path)

        if self.cfg.show_curves:
            self.plot_curves()

        if self.cfg.show_demo:
            self.demo(k=self.cfg.demo_k)

        return self.history

    @torch.no_grad()
    def predict(self, x: torch.Tensor, topk: int = 1):
        """推論メソッド.
        x: (B,1,28,28) float tensor
        returns:
          pred_idx: (B,topk)
          pred_prob: (B,topk)
        """
        self.model.eval()
        x = x.to(self.device)
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)
        p, idx = torch.topk(prob, k=topk, dim=1)
        return idx.cpu(), p.cpu()

    @torch.no_grad()
    def evaluate(self):
        loss, acc = self._evaluate(self.val_loader)
        return {"val_loss": loss, "val_acc": acc}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

    def plot_curves(self):
        h = self.history
        if len(h["train_loss"]) == 0:
            print("[Warn] No history to plot.")
            return

        plt.figure()
        plt.plot(h["train_loss"], label="train_loss")
        plt.plot(h["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(h["train_acc"], label="train_acc")
        plt.plot(h["val_acc"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def demo(self, k: int = 12):
        """valからk枚抜いてGT/予測を表示."""
        self.model.eval()
        x, y = next(iter(self.val_loader))
        x = x[:k]
        y = y[:k]

        pred_idx, _ = self.predict(x, topk=1)
        pred_idx = pred_idx.squeeze(1)

        cols = min(6, k)
        rows = math.ceil(k / cols)

        plt.figure()
        for i in range(k):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(x[i, 0], cmap="gray")

            gt = self.class_map.get(int(y[i]), str(int(y[i])))
            pr = self.class_map.get(int(pred_idx[i]), str(int(pred_idx[i])))

            plt.title(f"GT:{gt}\nPR:{pr}", fontsize=9)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def main():
    args = parse_args()

    cfg = EMNISTTrainerConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        workers=args.workers,
        seed=args.seed,
        show_samples=args.show_samples,
        show_curves=args.show_curves,
        show_demo=args.show_demo,
        demo_k=args.demo_k,
    )

    clf = EMNISTClassifier(cfg=cfg, hidden=args.hidden)
    clf.fit()


if __name__ == "__main__":
    main()
