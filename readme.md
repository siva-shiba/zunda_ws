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

東北ずん子PJの公開データ（[公式イラスト・3D](https://zunko.jp/con_illust.html)、[AI画像用学習データ](https://zunko.jp/con_illust.html)、[マルチモーダルDB](https://zunko.jp/multimodal_dev/login.php)等）を前提

- [ ] 画像分類
  - [x] MLP
  - [x] CNN
  - [ ] Deep-CNN
  - [ ] VGG
  - [ ] ResNet
  - [ ] MobileNet
  - [ ] EfficientNet
  - [ ] ConvNeXt
  - [ ] ViT

- [ ] 言語（[シンプルずんだもん](https://huggingface.co/datasets/alfredplpl/simple-zundamon)・[dolly-15k-ja-zundamon](https://huggingface.co/datasets/takaaki-inada/databricks-dolly-15k-ja-zundamon)等で実施。現役のモデル構造に直結するもののみ）
  - [ ] 統計ベースライン（n-gram / TF-IDF）
  - [ ] 埋め込みベクトル（Word2Vec等）
  - [ ] BPE（トークナイザ）
  - [ ] LSTM / GRU
  - [ ] Transformer（自己注意）
  - [ ] BERT系（マスクLM・Encoder）
  - [ ] GPT系（自己回帰LM・Decoder）
  - [ ] 指示チューニング・チャット

- [ ] 画像理解
  - [ ] 物体検出（公式にBBラベルなし → **要自作アノテーション**）
    - [ ] Faster R-CNN
    - [ ] YOLOv1
    - [ ] YOLO11
    - [ ] DETR
  - [ ] セグメンテーション（公式にピクセルラベルなし → **要自作アノテーション**）
    - [ ] U-Net
    - [ ] DeepLabV3+
    - [ ] Mask R-CNN
  - [ ] 埋め込み表現・検索
    - [ ] Siamese Network
    - [ ] Triplet Network
    - [ ] SimCLR
  - [ ] 画像と言語
    - [ ] Show and Tell（キャプション要自作で可能；AI画像用学習データのタグ流用も可）
    - [ ] CLIP（QAペアが公式にない → **要自作**）
    - [ ] BLIP-2

- [ ] 異常検知・再構成
  - [ ] AutoEncoder系
  - [ ] VAE系
  - [ ] Patchベース異常検知

- [ ] 画像生成
  - [ ] GAN系
  - [ ] Diffusion系

- [ ] マルチモーダル
  - [ ] 画像と言語の対応付け(CLIP系)（キャラ名ラベルで画像-テキストペア構築可能）
  - [ ] Vision-Language Model (BLIP / LLaVA系) ※キャプション等は要自作
  - [ ] 読唇・リップシンク（[マルチモーダルDB](https://zunko.jp/multimodal_dev/login.php)：口動き画像＋音声＋ラベル）

## ToDo

- [ ] ~~datasetのダウンロードと配置スクリプト作成~~ zipじゃないと自動化できないので保留
- [x] データローダの作成
- [ ] BBoxアノテーション (顔)
- [ ] Segmアノテーション (未定)
- [ ] キャプションGT
- [ ] QAペア作成
