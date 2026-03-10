# What is this

東北ずん子projectを使って画像関連のaiを勉強していくシリーズ

## 起動方法

### venv で開発する場合

```bash
cd /path/to/zunda_ws
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-mmcv.txt   # mmcv はプレビルド wheel（CUDA_HOME 不要）
pip install -e .   # zunda パッケージをインストール
```

### Docker で開発する場合

```bash
cd /path/to/zunda_ws
sh docker/run-docker.sh
```

- 初回は `docker compose up -d --build` でイメージをビルド（`requirements.txt` で依存関係をインストール）。
- コンテナ起動時に entrypoint が `/ws` を検知し、`pip install -e /ws` で zunda を自動インストールする。
- コンテナ内の作業ディレクトリは `/ws`（ホストのリポジトリがマウントされている）。

## List (作成チェック)

- [x] 画像分類(MLP)
- [x] 画像分類(CNN)
- [ ] 物体検出(R-CNN系)
- [ ] 物体検出(YOLO系)
- [ ] 物体検出(DETR系)
- [ ] 画像セグ(U-Net,DeepLab)
- [ ] 敵対的生成(GAN系)
- [ ] 異常検知(VAE系)
- [ ] 画像生成(Deffusion系)

## ToDo

- [ ] ~~datasetのダウンロードと配置スクリプト作成~~ zipじゃないと自動化できないので保留
- [ ] データローダの作成
