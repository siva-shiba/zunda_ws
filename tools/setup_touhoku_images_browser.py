#!/usr/bin/env python3
"""ブラウザで Google Drive を開き、ユーザーがフォルダ ZIP を取得したあと展開する。

公式フォルダ（公開）: https://drive.google.com/drive/folders/1NIZcBRvr5i8YfPsPYwvVMC7SoH-cWLIk
ガイドライン: https://zunko.jp/con_illust.html

Docker や SSH 先ではデフォルトブラウザが開けないことがある。その場合は表示 URL を手元のブラウザで開き、
取得した ZIP を `scp` 等で持ち込んで `--zip` を指定する。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import webbrowser
import zipfile
from pathlib import Path

DEFAULT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1NIZcBRvr5i8YfPsPYwvVMC7SoH-cWLIk?usp=sharing"
)
DEFAULT_OUTPUT = Path("data") / "touhoku_project_images"


def _is_under_dir(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for member in zf.infolist():
        name = member.filename
        if not name or name.startswith("/"):
            raise ValueError(f"不正な ZIP エントリ: {name!r}")
        parts = Path(name).parts
        if ".." in parts:
            raise ValueError(f"不正な ZIP エントリ: {name!r}")
        target = (dest / name).resolve()
        if not _is_under_dir(target, dest):
            raise ValueError(f"ZIP slip の疑い: {name!r}")
    zf.extractall(dest)


def _count_images(root: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _input_path_with_tab_complete(prompt: str) -> str:
    # GNU readline が使える環境では、Tab でパス補完を有効にする。
    try:
        import readline  # type: ignore
    except Exception:
        return input(prompt)

    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()

    def _complete(text: str, state: int) -> str | None:
        expanded = os.path.expanduser(text)
        matches = glob.glob(expanded + "*")
        if len(matches) == 1 and os.path.isdir(matches[0]):
            matches[0] = matches[0] + "/"
        if text.startswith("~"):
            home = str(Path.home())
            matches = [m.replace(home, "~", 1) if m.startswith(home) else m for m in matches]
        matches.sort()
        return matches[state] if state < len(matches) else None

    readline.set_completer_delims(" \t\n")
    readline.set_completer(_complete)
    readline.parse_and_bind("tab: complete")
    try:
        return input(prompt)
    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


def main() -> None:
    p = argparse.ArgumentParser(
        description="ブラウザで Drive を開き、ZIP を指定して東北ずん子PJ学習画像を配置する",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"展開先（デフォルト: {DEFAULT_OUTPUT}）",
    )
    p.add_argument(
        "--url",
        default=DEFAULT_FOLDER_URL,
        help="開くフォルダの URL",
    )
    p.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=None,
        help="ダウンロード済み ZIP のパス（省略時は対話入力）",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="ブラウザを開かない（URL の表示のみ）",
    )
    args = p.parse_args()
    out = args.output

    if args.zip_path is None:
        if not args.no_browser:
            print("ブラウザでフォルダを開きます…", flush=True)
            opened = webbrowser.open(args.url)
            if not opened:
                print("（自動で開けませんでした。次の URL をブラウザで開いてください）", flush=True)
        print(f"\nURL: {args.url}\n", flush=True)
        print(
            "Drive 上で:\n"
            "  フォルダ名の横の「⋮」→「ダウンロード」\n"
            "  またはフォルダを開いた状態で Ctrl+A → 右クリック →「ダウンロード」\n"
            "（サブフォルダがある場合は ZIP になります。ダウンロード完了まで待ってください。）\n",
            flush=True,
        )
        try:
            line = _input_path_with_tab_complete(
                "ZIP ファイルのパスを入力（Tab 補完可, 例: ~/Downloads/....zip）: "
            ).strip()
        except EOFError:
            print("対話入力がありません。--zip /path/to/file.zip を指定してください。", file=sys.stderr)
            sys.exit(2)
        zip_path = Path(line.strip('"\'')).expanduser()
    else:
        zip_path = args.zip_path.expanduser()

    zip_path = zip_path.resolve()
    if not zip_path.is_file():
        print(f"ファイルがありません: {zip_path}", file=sys.stderr)
        sys.exit(1)
    if zip_path.suffix.lower() != ".zip":
        print("警告: .zip 以外です。Drive のフォルダダウンロードは通常 .zip です。", file=sys.stderr)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extractall(zf, out)
    except zipfile.BadZipFile as e:
        print(f"ZIP を読めませんでした（壊れているか未完了の .crdownload の可能性）: {e}", file=sys.stderr)
        sys.exit(1)

    n = _count_images(out.resolve())
    print(f"展開完了: {zip_path} → {out.resolve()}", flush=True)
    print(f"画像ファイル数（再帰）: {n}", flush=True)


if __name__ == "__main__":
    main()
