"""重み付きサンプリングを有効にしたとき、1エポックで各クラスが何回使われるか（重複含む）をヒストグラムで可視化する.

WeightedRandomSampler は各サンプルに重み 1/(クラス内サンプル数) を付与するため、
1エポックあたりの抽出回数の期待値は全クラスで同じ（= 総サンプル数/クラス数）になる。
実際のヒストグラムではランダム性で多少ばらつくが、棒の高さはどれも赤線（期待値）付近に並ぶ。
"""

import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語が□にならないようフォントを指定
def _setup_japanese_font():
    # 1) 登録済みフォントから日本語対応を探す
    for font_name in ("IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "Takao", "VL Gothic", "Yu Gothic", "MS Gothic"):
        if any(f.name == font_name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = font_name
            return
    # 2) よくあるパスに IPA フォントがあれば追加
    for path in [
        "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    ]:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
                plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                return
            except Exception:
                pass
    # 見つからなければデフォルトのまま（□になる可能性あり）


_setup_japanese_font()
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の□化を防ぐ

import torch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from train import (
    TrainerConfig,
    load_config,
    parse_overrides,
    config_dict_to_trainer_config,
    build_transforms,
    setup_logging,
    fields,
    project_root,
)
from zunda import DATASET_REGISTRY
import typing


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="重み付きサンプリング時のクラス別使用回数ヒストグラム")
    p.add_argument("config", nargs="?", default=None, help="設定ファイル（JSON）")
    p.add_argument("-o", "--override", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default=None, help="出力画像パス（未指定時は plot_sampler_histogram.png）")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("./logs", "INFO")

    config_path = Path(args.config) if args.config else None
    base = load_config(config_path)
    field_types = {f.name: f.type for f in fields(TrainerConfig)}
    for k, t in list(field_types.items()):
        if getattr(t, "__origin__", None) is typing.Union and hasattr(t, "__args__"):
            non_none = [a for a in t.__args__ if a is not type(None)]
            if len(non_none) == 1:
                field_types[k] = non_none[0]
    overrides = parse_overrides(args.overrides or [], field_types)
    cfg = config_dict_to_trainer_config({**base, **overrides})

    data_root_path = Path(cfg.data_root)
    if not data_root_path.is_absolute():
        tmp = {f.name: getattr(cfg, f.name) for f in fields(TrainerConfig)}
        tmp["data_root"] = str(project_root / cfg.data_root)
        cfg = TrainerConfig(**tmp)

    # ヒストグラム用に重み付きサンプリングを有効化
    # cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(TrainerConfig)}
    # cfg_dict["use_weighted_sampler"] = True
    # cfg = TrainerConfig(**cfg_dict)

    if cfg.dataset not in DATASET_REGISTRY:
        raise ValueError(f"このスクリプトは dataset={cfg.dataset} には未対応です。")

    adapter_cls = DATASET_REGISTRY[cfg.dataset]
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = adapter_cls.build_dataloaders(
        cfg=cfg,
        logger=logger,
        build_transforms_func=build_transforms,
    )

    # 1エポック分を走査し、バッチ内のラベルを重複含めてカウント
    count_per_class = Counter()
    n_samples = 0
    for batch in train_loader:
        labels = batch["label"]
        for i in range(labels.size(0)):
            count_per_class[int(labels[i].item())] += 1
            n_samples += 1

    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    counts = [count_per_class.get(i, 0) for i in range(num_classes)]

    # 理論値: 各クラスとも 約 n_samples / num_classes
    expected = n_samples / num_classes if num_classes else 0

    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.4), 5))
    bars = ax.bar(class_names, counts, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axhline(y=expected, color="red", linestyle="--", linewidth=1.5, label=f"期待値 ≈ {expected:.0f}")

    ax.set_xlabel("クラス")
    ax.set_ylabel("1エポックあたりの使用回数（重複含む）")
    ax.set_title("重み付きサンプリング時のクラス別使用回数")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = Path(args.out) if args.out else Path("plot_sampler_histogram.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"総サンプル数（1エポック）: {n_samples}, クラス数: {num_classes}, 期待値/クラス: {expected:.1f}")
    logger.info(f"ヒストグラムを保存しました: {out_path.resolve()}")
    for i, name in enumerate(class_names):
        logger.info(f"  {name}: {counts[i]} 回")


if __name__ == "__main__":
    main()
