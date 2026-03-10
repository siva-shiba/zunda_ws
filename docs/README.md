# ドキュメントのビルド

## ローカルでビルドする

```bash
# リポジトリルートで
pip install -e .
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
# docs/_build/html/index.html を開く
```

## CI（GitHub Actions）

- `main` または `master` に push するとドキュメントがビルドされ、GitHub Pages にデプロイされます。
- **初回のみ**: リポジトリの **Settings → Pages** で、Source を **GitHub Actions** に設定してください。
