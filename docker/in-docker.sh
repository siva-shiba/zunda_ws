#!/bin/bash
# コンテナに入るだけの簡易スクリプト

cd "$(dirname "$0")"   # docker/ ディレクトリへ移動

export UID=$(id -u)
export GID=$(id -g)

docker compose exec --user "$UID:$GID" zunda bash