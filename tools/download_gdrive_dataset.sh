#!/bin/bash
# Description: Google Driveから画像データセットをダウンロードする手順を案内するスクリプト
# ブラウザで手動ダウンロードする方法を案内します

set +e

# デフォルト値
DEFAULT_URL="https://drive.google.com/drive/folders/1NIZcBRvr5i8YfPsPYwvVMC7SoH-cWLIk"
OUTPUT_DIR="data/touhoku_project_images"

# ヘルプ表示
show_help() {
    cat << EOF
使用方法: $0 [オプション] [URL]

Google Driveから画像データセットをダウンロードする手順を案内します。
ブラウザで手動ダウンロードする方法を表示します。

引数:
  URL                    Google Driveの共有リンクまたはフォルダID
                         指定しない場合はデフォルトURLを使用

オプション:
  -o, --output DIR       ダウンロード先のディレクトリ (デフォルト: data/touhoku_project_images)
  -h, --help            このヘルプを表示

例:
  $0                                    # デフォルトURLからダウンロード
  $0 -o data/lora_dataset              # 出力先を指定
  $0 "https://drive.google.com/..."    # カスタムURLを指定
EOF
}

# フォルダIDを抽出
extract_folder_id() {
    local url="$1"
    local folder_id=""

    # /folders/ の後のIDを抽出
    folder_id=$(echo "$url" | sed -n 's|.*/folders/\([a-zA-Z0-9_-]\+\).*|\1|p')

    # フォルダIDが見つからなかった場合、URLがフォルダIDのみかチェック
    if [ -z "$folder_id" ]; then
        # URLがフォルダIDのみの場合（英数字、ハイフン、アンダースコアのみ）
        if echo "$url" | grep -qE '^[a-zA-Z0-9_-]+$'; then
            folder_id="$url"
        else
            echo "❌ 無効なGoogle Drive URLです: $url" >&2
            exit 1
        fi
    fi

    echo "$folder_id"
}

# ブラウザを開く（可能な場合）
open_browser() {
    local url="$1"
    
    # 利用可能なブラウザコマンドを検索
    if command -v xdg-open > /dev/null 2>&1; then
        xdg-open "$url" 2>/dev/null &
    elif command -v gnome-open > /dev/null 2>&1; then
        gnome-open "$url" 2>/dev/null &
    elif command -v firefox > /dev/null 2>&1; then
        firefox "$url" 2>/dev/null &
    elif command -v google-chrome > /dev/null 2>&1; then
        google-chrome "$url" 2>/dev/null &
    elif command -v chromium-browser > /dev/null 2>&1; then
        chromium-browser "$url" 2>/dev/null &
    else
        return 1
    fi
}

# メイン処理
main() {
    local url="$DEFAULT_URL"

    # 引数の解析
    while [ $# -gt 0 ]; do
        case $1 in
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            http*|*[a-zA-Z0-9_-]*)
                url="$1"
                shift
                ;;
            *)
                echo "❌ 不明なオプション: $1" >&2
                show_help
                exit 1
                ;;
        esac
    done

    # フォルダIDを抽出
    local folder_id=$(extract_folder_id "$url")
    local folder_url="https://drive.google.com/drive/folders/$folder_id"
    
    # 出力ディレクトリのパスを取得
    mkdir -p "$OUTPUT_DIR"
    local output_path=$(realpath "$OUTPUT_DIR")
    
    echo "=========================================="
    echo "📥 Google Drive データセット ダウンロード手順"
    echo "=========================================="
    echo ""
    echo "📁 フォルダID: $folder_id"
    echo "🔗 URL: $folder_url"
    echo "📂 保存先: $output_path"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 ダウンロード手順"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1️⃣  ブラウザでGoogle Driveフォルダを開く"
    echo ""
    
    # ブラウザを開く試み
    if open_browser "$folder_url"; then
        echo "   ✅ ブラウザを開きました: $folder_url"
        echo "   （ブラウザが開かない場合は、上記のURLを手動で開いてください）"
    else
        echo "   ⚠️  ブラウザを自動で開けませんでした。"
        echo "   以下のURLをブラウザで開いてください:"
        echo ""
        echo "   🔗 $folder_url"
    fi
    
    echo ""
    echo "2️⃣  フォルダ全体をダウンロード"
    echo ""
    echo "   ブラウザで:"
    echo "   - フォルダ名の横にある「⋮」（三点メニュー）をクリック"
    echo "   - 「ダウンロード」を選択"
    echo "   - または、フォルダ内のすべてのファイルを選択（Ctrl+A / Cmd+A）"
    echo "   - 右クリック → 「ダウンロード」"
    echo ""
    echo "   ⚠️  注意: フォルダにサブフォルダが含まれている場合、"
    echo "   Google Driveは自動的にZIPファイルとしてダウンロードします。"
    echo ""
    echo "3️⃣  ダウンロードしたファイルを解凍"
    echo ""
    echo "   ダウンロードしたZIPファイルを以下の場所に解凍してください:"
    echo ""
    echo "   📂 $output_path"
    echo ""
    echo "   解凍コマンド例:"
    echo "   unzip ~/Downloads/フォルダ名.zip -d $output_path"
    echo "   または"
    echo "   tar -xzf ~/Downloads/フォルダ名.tar.gz -C $output_path"
    echo ""
    echo "4️⃣  ダウンロード完了の確認"
    echo ""
    echo "   以下のコマンドで確認できます:"
    echo "   ls -la $output_path"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💡 ヒント"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "• 大量のファイルがある場合、ZIPファイルとしてダウンロードされます"
    echo "• ダウンロードが完了するまで時間がかかる場合があります"
    echo "• 解凍後、元のZIPファイルは削除しても構いません"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # 保存先ディレクトリが空かどうか確認
    if [ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
        echo "📂 保存先ディレクトリは現在空です: $output_path"
        echo "   ダウンロードと解凍が完了したら、このディレクトリにファイルが配置されます。"
    else
        echo "📂 保存先ディレクトリには既にファイルがあります: $output_path"
        echo "   既存のファイル:"
        ls -lh "$OUTPUT_DIR" | head -10
        if [ $(ls -1 "$OUTPUT_DIR" | wc -l) -gt 10 ]; then
            echo "   ... (他にもファイルがあります)"
        fi
    fi
    
    echo ""
    echo "✅ 手順の表示が完了しました。"
    echo "   ブラウザで上記のURLを開いて、ダウンロードを開始してください。"
    echo ""
}

# スクリプト実行
main "$@"
