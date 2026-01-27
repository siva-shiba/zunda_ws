#!/bin/bash
# コンテナ起動時にzundaパッケージをインストール

# setup.pyが存在する場合、zundaパッケージをインストール
if [ -f /ws/setup.py ] && [ -d /ws/zunda ]; then
    echo "Installing zunda package..."
    pip install -e /ws --quiet --no-warn-script-location
fi

# 元のENTRYPOINTを実行
exec "$@"
