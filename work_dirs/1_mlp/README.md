# MLP分類タスク

分類タスク用の簡易MLPモデルの学習スクリプトです。

## ファイル構成

- `train.py`: MLPモデルの学習スクリプト

## 使用方法

### 基本的な使用方法

```bash
python train.py <data_root> [オプション]
```

### 例

```bash
# デフォルト設定で学習
python train.py ../data/touhoku_project_images

# カスタム設定で学習
python train.py ../data/touhoku_project_images \
    --image-size 256 \
    --batch-size 32 \
    --epochs 20 \
    --lr 0.001 \
    --hidden-size 512 \
    --save-dir ./checkpoints \
    --log-dir ./logs \
    --log-level INFO
```

## オプション

- `data_root`: データセットのルートディレクトリパス（必須）
- `--image-size`: 画像サイズ（デフォルト: 256）
- `--batch-size`, `-b`: バッチサイズ（デフォルト: 32）
- `--epochs`, `-e`: エポック数（デフォルト: 10）
- `--lr`: 学習率（デフォルト: 1e-3）
- `--hidden-size`: 隠れ層のサイズ（デフォルト: 512）
- `--num-workers`, `-w`: DataLoaderのワーカー数（デフォルト: 2、共有メモリ不足の場合は0を推奨）
- `--device`: デバイス（デフォルト: cuda if available else cpu）
- `--seed`, `-s`: 乱数シード（デフォルト: 42）
- `--save-dir`: チェックポイント保存ディレクトリ（デフォルト: ./checkpoints）
- `--log-dir`: ログファイル保存ディレクトリ（デフォルト: ./logs）
- `--log-level`: ログレベル（デフォルト: INFO、選択肢: DEBUG, INFO, WARNING, ERROR, CRITICAL）
- `--use-wandb`: WANDBを使用する（デフォルト: True）
- `--no-wandb`: WANDBを使用しない
- `--wandb-project`: WANDBプロジェクト名（デフォルト: zunda-mlp-classification）
- `--wandb-entity`: WANDBエンティティ名（デフォルト: None）
- `--wandb-run-name`: WANDBラン名（デフォルト: None、自動生成）
- `--wandb-tags`: WANDBタグ（スペース区切りで複数指定可能）

## ログ機能

学習の進行状況は以下のように記録されます:

- **コンソール出力**: 標準出力にログが表示されます
- **ログファイル**: `logs/train_YYYYMMDD_HHMMSS.log` にタイムスタンプ付きで保存されます
- **WANDB**: 実験結果をWANDBに記録（デフォルトで有効）

ログには以下が含まれます:
- 学習設定（デバイス、バッチサイズ、エポック数など）
- 各エポックの学習/検証損失と精度
- ベストモデルの保存情報
- 学習完了後のサマリー（ベスト精度、テスト精度など）
- エラー発生時の詳細なスタックトレース

## WANDB統合

### セットアップ

1. **WANDBアカウントの作成**
   - [https://wandb.ai](https://wandb.ai) でアカウントを作成

2. **APIキーの取得**
   - WANDBの設定ページからAPIキーを取得

3. **APIキーの設定（推奨: `.wandb`ファイル）**

   **Docker環境とローカル環境の分離**
   
   Docker環境とローカル環境で**別々のWANDBアカウント/プロジェクト**を使用することを推奨します。
   自動的に環境を検出して、適切な設定ファイルを読み込みます。

   **ローカル環境用の設定**
   ```bash
   # プロジェクトルートで実行
   cp .wandb.example .wandb
   # .wandbファイルを編集してローカル環境用のAPIキーを記述
   ```

   **Docker環境用の設定**
   ```bash
   # プロジェクトルートで実行
   cp .wandb.docker.example .wandb.docker
   # .wandb.dockerファイルを編集してDocker環境用のAPIキーを記述
   ```
   
   ファイルの内容例:
   ```
   your_api_key_here
   ```
   
   これらのファイルはプロジェクトルート（`/ws`）に配置してください。
   
   **環境の自動検出**
   - **ローカル環境**: `.wandb`ファイルを読み込み、`~/.config/wandb/`に設定を保存
   - **Docker環境**: `.wandb.docker`ファイルを読み込み、`~/.config/wandb_docker/`に設定を保存
   
   これにより、Docker環境とローカル環境で完全に分離されたWANDB設定が使用されます。

   **その他の方法**
   
   **方法2: 環境変数で設定**
   ```bash
   export WANDB_API_KEY=your_api_key_here
   docker compose up -d --build
   ```

   **方法3: コンテナ内でログイン**
   ```bash
   docker compose exec zunda bash
   wandb login
   ```

   **優先順位**: 
   1. 環境変数 `WANDB_API_KEY_FILE` で指定されたファイル
   2. Docker環境: `.wandb.docker` > 環境変数 `WANDB_API_KEY` > wandb login
   3. ローカル環境: `.wandb` > 環境変数 `WANDB_API_KEY` > wandb login

### 使用方法

```bash
# WANDBを使用して学習（デフォルト）
python train.py ../data/touhoku_project_images

# カスタムプロジェクト名とラン名を指定
python train.py ../data/touhoku_project_images \
    --wandb-project my-project \
    --wandb-run-name experiment-1 \
    --wandb-tags mlp baseline

# WANDBを使用しない
python train.py ../data/touhoku_project_images --no-wandb
```

### WANDBに記録される情報

- **ハイパーパラメータ**: 画像サイズ、バッチサイズ、学習率など
- **メトリクス**: 各エポックの学習/検証/テスト損失と精度
- **モデル構造**: MLPのアーキテクチャ
- **ベストモデル**: 検証精度が最も高いエポックの情報
- **チェックポイント**: ベストモデルのファイル（オプション）

## モデル構造

簡易的なMLPモデル:
- Flatten層
- Linear (入力 -> hidden_size) + ReLU + Dropout(0.2)
- Linear (hidden_size -> hidden_size) + ReLU + Dropout(0.2)
- Linear (hidden_size -> num_classes)

## 出力

### チェックポイント

- `checkpoints/best_model.pt`: 検証精度が最も高いモデル
- `checkpoints/final_model.pt`: 最終エポックのモデル

### ログファイル

- `logs/train_YYYYMMDD_HHMMSS.log`: 学習ログ（タイムスタンプ付き）

チェックポイントには以下が含まれます:
- `model_state_dict`: モデルの重み
- `optimizer_state_dict`: オプティマイザーの状態
- `class_to_idx`: クラス名からインデックスへのマッピング
- `idx_to_class`: インデックスからクラス名へのマッピング
- `val_acc`: 検証精度

## トラブルシューティング

### 共有メモリ（shm）エラー

Docker環境で`RuntimeError: DataLoader worker exited unexpectedly`や`Bus error`が発生する場合:

1. **docker-compose.ymlの共有メモリ設定を確認**
   - `shm_size: 64gb`が設定されています(64GB)。pcのスペックに合わせて調整してください

2. **ワーカー数を減らす**
   ```bash
   python train.py <data_root> --num-workers 0  # シングルプロセス
   # または
   python train.py <data_root> --num-workers 1  # ワーカー1つ
   ```

3. **Dockerコンテナを再起動**
   ```bash
   docker compose down
   docker compose up -d --build
   ```

### WANDBエラー

WANDBの認証エラーが発生する場合:

1. **環境を確認**
   ```bash
   # Docker環境かどうかを確認
   ls -la /.dockerenv  # Docker環境の場合のみ存在
   echo $DOCKER_CONTAINER  # docker-compose.ymlで設定
   ```

2. **適切な`.wandb`ファイルが存在するか確認**
   ```bash
   # ローカル環境用
   ls -la .wandb
   
   # Docker環境用
   ls -la .wandb.docker
   ```

3. **`.wandb`ファイルの内容を確認**
   ```bash
   # ローカル環境用
   cat .wandb
   
   # Docker環境用
   cat .wandb.docker
   # APIキーが正しく記述されているか確認
   ```
   
4. **WANDB設定ディレクトリを確認**
   ```bash
   # ローカル環境
   ls -la ~/.config/wandb/
   
   # Docker環境（コンテナ内で実行）
   ls -la ~/.config/wandb_docker/
   ```

3. **環境変数が設定されているか確認**
   ```bash
   echo $WANDB_API_KEY
   ```

4. **コンテナ内でログイン**
   ```bash
   docker compose exec zunda bash
   wandb login
   ```

5. **オフラインモードで実行**
   ```bash
   export WANDB_MODE=offline
   docker compose up -d --build
   ```
   または
   ```bash
   python train.py <data_root> --no-wandb
   ```
