#!/bin/bash
# マウントされた /ws に zunda がある場合、エディタブルインストール（venv では手動で pip install -e .）
if [ -f /ws/setup.py ] && [ -d /ws/zunda ]; then
    pip install -e /ws --quiet --no-warn-script-location
fi
exec "$@"
