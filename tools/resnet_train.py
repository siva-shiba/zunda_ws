#!/usr/bin/env python
"""resnet学習のテストコード.

DATASET : cifer10
"""

from mmengine.hub import get_config
from mmengine.runner import Runner


PROJECT = "cifar10-test"
PROJECT_NEP = f"siva-shiba/{PROJECT}"
TAGs = ["ResNet18", "CIFAR-10", "MMPretrain"]


def get_token(path=".token"):
    """トークンを取得する関数."""
    f = open(path, "r")
    token = f.read()
    f.close()
    return token


def init_neptune(cfg, project=PROJECT_NEP, tags=TAGs):
    """Neptune.aiの初期化."""
    # 既にNeptune Loggerが存在するかどうか
    neptune_exists = any(backend["type"] == "NeptuneVisBackend" for backend in cfg.visualizer["vis_backends"])
    if neptune_exists:
        print("Neptune Logger already exists.")
        return cfg

    # Neptune Loggerの追加
    token = get_token()
    cfg.vis_backends.append(
        dict(
            type="NeptuneVisBackend",
            init_kwargs=dict(
                project=project,
                api_token=token,
                tags=tags,)))
    cfg.visualizer.vis_backends = cfg.vis_backends
    return cfg


def main():
    """main関数."""
    # 設定ファイルの読み込み
    cfg = get_config('mmpretrain::resnet/resnet18_8xb16_cifar10.py', pretrained=True)

    # 出力ディレクトリを変更
    cfg.work_dir = f'work_dirs/{PROJECT}'
    print(cfg.visualizer)
    init_neptune(cfg)

    # エポック数の短縮（テスト用）
    cfg.train_cfg.max_epochs = 5

    # 評価項目の変更
    cfg.val_evaluator = [
        dict(type='Accuracy', topk=(1, 5)),
        dict(type='SingleLabelMetric',
             items=['precision', 'recall', "f1-score"])]

    # 学習の実行
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
