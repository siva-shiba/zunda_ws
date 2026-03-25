"""ViT分類タスクの学習スクリプト.

torchvision の Vision Transformer（vit_b_16 / vit_b_32 / vit_l_16）で
画像分類（訓練/検証）を行い、ベストモデルを checkpoint に保存します。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import MISSING, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, get_args, get_origin

import torch
import torch.nn as nn
from torchvision import transforms

from zunda import DATASET_REGISTRY, ClassificationPredictor, setup_logging
from zunda.losses import FocalLoss

from model import build_vit_model


# プロジェクトルートをパスに追加（ローカル実行時の import 安定化）
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@dataclass
class TrainerConfig:
    """学習設定."""

    data_root: str
    dataset: str = "touhoku"

    in_channels: Optional[int] = None  # None のとき dataset から推定（本リポジトリは基本 3）
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3

    model_name: str = "vit_b_16"
    pretrained: bool = True

    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: Optional[str] = None  # None のときは save_dir に混在（互換）

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    include_classes: Optional[list] = None

    # クラス不均衡対策
    use_class_weights: bool = False
    use_weighted_sampler: bool = False  # 現状未実装（dataset adapter 側には項目がある）
    use_stratified_split: bool = True
    class_weight_method: str = "balanced"  # balanced / inverse / sqrt

    # Focal Loss
    use_focal_loss: bool = False
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0

    # WANDB（必要なら使う）
    use_wandb: bool = False
    wandb_project: str = "vit"
    wandb_entity: str = "zunda"
    wandb_run_name: str = None
    wandb_group: str = None
    wandb_tags: list = None
    upload_checkpoint: bool = False


def build_transforms(cfg: TrainerConfig):
    """ViT 向けの transforms（ImageNet mean/std を使用）."""

    # ImageNet 正規化 (ViT の事前学習と整合する想定)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, val_transform


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is None:
        # typing.Optional[T] は Union[T, NoneType] なので、ここで扱う
        if (
            getattr(tp, "__origin__", None) is None
            and getattr(tp, "__args__", None)
        ):
            origin = getattr(tp, "__origin__", None)
    if origin is Optional:
        return tp
    args = get_args(tp)
    if len(args) == 2 and type(None) in args:
        return args[0] if args[1] is type(None) else args[1]
    return tp


def load_config(
    path: Optional[Path],
    visited: Optional[set[Path]] = None,
) -> dict[str, Any]:
    """設定ファイル（JSON）を読み込み、_base_ を展開する."""

    if path is None:
        return {}

    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {p}")

    if visited is None:
        visited = set()
    if p in visited:
        raise ValueError(f"_base_ が循環参照しています: {p}")
    visited.add(p)

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    base_cfg_path = raw.get("_base_")
    if not base_cfg_path:
        return raw

    base_path = (p.parent / base_cfg_path).resolve()
    base_cfg = load_config(base_path, visited=visited)
    child_cfg = {k: v for k, v in raw.items() if k != "_base_"}
    return {**base_cfg, **child_cfg}


def parse_overrides(
    overrides: list[str],
    config_types: dict[str, Any],
) -> dict[str, Any]:
    """-o key=value のリストを型変換して辞書にする."""

    def parse_value(raw: str, field_type: Any) -> Any:
        raw = raw.strip()
        if raw.lower() in ("none", "null", ""):
            return None

        field_type_unwrapped = _unwrap_optional(field_type)
        origin = get_origin(field_type_unwrapped)
        args = get_args(field_type_unwrapped)

        if field_type_unwrapped is bool:
            return raw.lower() in ("true", "1", "yes")
        if field_type_unwrapped is int:
            return int(raw)
        if field_type_unwrapped is float:
            return float(raw)
        if field_type_unwrapped is str:
            return raw

        # list[str] / Optional[list]
        if origin is list or field_type_unwrapped is list:
            inner = raw
            if inner.startswith("[") and inner.endswith("]"):
                inner = inner[1:-1].strip()
            parts = [x.strip() for x in inner.split(",") if x.strip()]
            return parts if args == () or args == (str,) else parts

        # その他は文字列のまま
        return raw

    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"上書きは key=value 形式で指定してください: {item}")
        k, _, v = item.partition("=")
        k = k.strip()
        if k not in config_types:
            raise ValueError(f"未知の設定キー: {k}")
        result[k] = parse_value(v, config_types[k])
    return result


def config_dict_to_trainer_config(d: dict[str, Any]) -> TrainerConfig:
    """辞書と dataclass のデフォルトをマージして TrainerConfig を構築する."""

    defaults: dict[str, Any] = {}
    config_types: dict[str, Any] = {}
    for f in fields(TrainerConfig):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        config_types[f.name] = f.type

    merged = {**defaults, **d}
    if "data_root" not in merged or merged["data_root"] in (None, ""):
        raise ValueError("data_root を設定ファイルまたは -o data_root=... で指定してください")
    if "model_name" not in merged or merged["model_name"] in (None, ""):
        raise ValueError("model_name を設定ファイルまたは -o model_name=... で指定してください")

    # bool/int/float の型ブレを抑える（あとは parse_overrides を使う前提）
    return TrainerConfig(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分類タスク用 ViT の学習（設定ファイル + 上書き）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # デフォルト設定で学習（config 省略時は configs/default.json を使用）
  python train.py

  # 設定ファイルを指定
  python train.py configs/default.json

  # 上書き
  python train.py -o data_root=data/touhoku_project_images -o epochs=30
""",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="設定ファイル（JSON）。省略時は configs/default.json を使用",
    )
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="設定の上書き（複数指定可）。例: -o epochs=50 -o use_class_weights=true",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _calculate_class_weights(
    train_loader,
    idx_to_class: dict[int, str],
    method: str,
) -> torch.Tensor:
    """クラス重み（CrossEntropyLoss / FocalLoss 用）を作る."""

    labels: list[int] = []
    dataset = train_loader.dataset
    for i in range(len(dataset)):
        labels.append(dataset[i]["label"])

    class_counts = Counter(labels)
    num_classes = len(idx_to_class)
    total_samples = len(labels)

    weights: list[float] = []
    if method == "balanced":
        # n_samples / (n_classes * count)
        for idx in range(num_classes):
            count = class_counts.get(idx, 1)
            weights.append(total_samples / (num_classes * count))
    elif method == "inverse":
        max_count = max(class_counts.values()) if class_counts else 1
        for idx in range(num_classes):
            count = class_counts.get(idx, 1)
            weights.append(max_count / count)
    elif method == "sqrt":
        max_count = max(class_counts.values()) if class_counts else 1
        for idx in range(num_classes):
            count = class_counts.get(idx, 1)
            weights.append((max_count / count) ** 0.5)
    else:
        raise ValueError(f"未知の class_weight_method: {method}")

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def main() -> None:
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wandb").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    os.environ["WANDB_DIR"] = str(run_dir / "wandb")

    logger = setup_logging(log_dir=str(run_dir), log_level="INFO")
    logger.info(f"出力ルート: {run_dir.resolve()}")

    try:
        config_path = (
            Path(args.config)
            if args.config
            else Path(__file__).parent / "configs" / "default.json"
        )
        base = load_config(config_path)

        field_types = {f.name: f.type for f in fields(TrainerConfig)}
        overrides = parse_overrides(args.overrides or [], field_types)

        cfg = config_dict_to_trainer_config({**base, **overrides})

        # 相対パスはリポジトリルート基準で解決
        data_root_path = Path(cfg.data_root)
        if not data_root_path.is_absolute():
            cfg.data_root = str(project_root / cfg.data_root)

        # 出力先を run_dir 配下に寄せる
        cfg.save_dir = str(run_dir / "checkpoints")
        cfg.results_dir = str(run_dir / "results")

        if cfg.wandb_run_name in (None, ""):
            cfg.wandb_run_name = f"{cfg.model_name}_{run_id}"

        _set_seed(cfg.seed)

        if cfg.use_wandb:
            try:
                import wandb  # type: ignore

                wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    name=cfg.wandb_run_name,
                    group=cfg.wandb_group,
                    tags=cfg.wandb_tags,
                    config=cfg.__dict__,
                )
            except Exception as e:
                logger.warning("wandb の初期化に失敗しました（無効化します）: %s", e)
                cfg.use_wandb = False

        if cfg.dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知の dataset: {cfg.dataset}. "
                "DATASET_REGISTRY に登録してください。"
            )

        dataset_cls = DATASET_REGISTRY[cfg.dataset]
        device = torch.device(cfg.device)

        # データローダー
        (
            train_loader,
            val_loader,
            test_loader,
            class_to_idx,
            idx_to_class,
        ) = dataset_cls.build_dataloaders(
            cfg=cfg,
            logger=logger,
            build_transforms_func=build_transforms,
        )

        in_ch = (
            cfg.in_channels
            if cfg.in_channels is not None
            else dataset_cls.get_in_channels(cfg)
        )
        num_classes = len(class_to_idx)
        logger.info(
            "データセット: %s / num_classes=%s / in_ch=%s",
            cfg.dataset,
            num_classes,
            in_ch,
        )

        model = build_vit_model(
            model_name=cfg.model_name,
            num_classes=num_classes,
            in_channels=in_ch,
            pretrained=cfg.pretrained,
            image_size=cfg.image_size,
        ).to(device)
        logger.info(f"モデル: {cfg.model_name} (pretrained={cfg.pretrained})")

        # 損失関数
        class_weights = None
        if cfg.use_class_weights:
            class_weights = _calculate_class_weights(
                train_loader=train_loader,
                idx_to_class=idx_to_class,
                method=cfg.class_weight_method,
            ).to(device)
            logger.info("クラス重み付き損失関数を使用します")

        if cfg.use_focal_loss:
            criterion = FocalLoss(
                alpha=cfg.focal_loss_alpha,
                gamma=cfg.focal_loss_gamma,
                weight=class_weights,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        save_dir = Path(cfg.save_dir)
        results_dir = Path(cfg.results_dir) if cfg.results_dir else save_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = save_dir / "best_model.pt"
        final_model_path = save_dir / "final_model.pt"

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            logger.info(
                "Epoch %s/%s - train loss %.4f acc %.4f - "
                "val loss %.4f acc %.4f",
                epoch,
                cfg.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if cfg.use_wandb:
                import wandb  # type: ignore

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                    }
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "model_config": {
                        "model_name": cfg.model_name,
                        "pretrained": cfg.pretrained,
                        "in_channels": in_ch,
                        "image_size": cfg.image_size,
                        "num_classes": num_classes,
                    },
                }
                torch.save(ckpt, best_model_path)
                logger.info(f"ベストモデルを保存しました: {best_model_path}")

        # 最終モデル保存
        ckpt = {
            "epoch": cfg.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": best_val_acc,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "model_config": {
                "model_name": cfg.model_name,
                "pretrained": cfg.pretrained,
                "in_channels": in_ch,
                "image_size": cfg.image_size,
                "num_classes": num_classes,
            },
        }
        torch.save(ckpt, final_model_path)
        logger.info(f"最終モデルを保存しました: {final_model_path}")

        # 混同行列（best/final）を出す（推論結果は重いので必要なら後で外してください）
        predictor = ClassificationPredictor(
            model=model,
            device=device,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            logger=logger,
        )

        def dump_confusion(model_type: str, checkpoint_path: Path):
            if not checkpoint_path.exists():
                return
            ck = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            logger.info("confusion matrix を生成: model_type=%s", model_type)
            for split_name, loader in [
                ("train", train_loader),
                ("val", val_loader),
                ("test", test_loader),
            ]:
                _, _, pred_labels, true_labels, _ = predictor.predict(loader)
                predictor.create_confusion_matrix(
                    true_labels=true_labels,
                    pred_labels=pred_labels,
                    split=split_name,
                    model_type=model_type,
                    save_dir=results_dir,
                    use_timestamp=False,
                )

        dump_confusion("best", best_model_path)
        dump_confusion("final", final_model_path)

        logger.info("best_epoch=%s best_val_acc=%.6f", best_epoch, best_val_acc)

        if cfg.use_wandb:
            try:
                import wandb  # type: ignore

                wandb.finish()
            except Exception:
                pass

    except Exception:
        logger.exception("学習中にエラーが発生しました:")
        raise


if __name__ == "__main__":
    main()
