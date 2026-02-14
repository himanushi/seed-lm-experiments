#!/usr/bin/env python3
"""
モデルダウンロード用CLIスクリプト

使用例:
    # 利用可能なモデル一覧を表示
    python scripts/download_model.py --list

    # 特定のモデルをダウンロード
    python scripts/download_model.py bert-japanese

    # 全モデルをダウンロード
    python scripts/download_model.py --all
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import ModelDownloader, MODELS


def main():
    parser = argparse.ArgumentParser(
        description="ローカルLLMモデルをダウンロードする",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python scripts/download_model.py --list          # 利用可能なモデル一覧
  python scripts/download_model.py bert-japanese   # BERT日本語をダウンロード
  python scripts/download_model.py --all           # 全モデルをダウンロード
        """,
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="ダウンロードするモデルのキー",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="利用可能なモデル一覧を表示",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="全てのモデルをダウンロード",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="キャッシュを無視して再ダウンロード",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="カスタムキャッシュディレクトリ",
    )

    args = parser.parse_args()

    downloader = ModelDownloader(cache_dir=args.cache_dir)

    # 一覧表示
    if args.list:
        downloader.list_available()
        return

    # 全モデルダウンロード
    if args.all:
        print("全モデルをダウンロードします...")
        print(f"検出されたデバイス: {downloader.device}")
        print()

        for key in MODELS:
            try:
                downloader.download(key, force=args.force)
                print()
            except Exception as e:
                print(f"✗ エラー: {key} のダウンロードに失敗しました: {e}")
                print()

        print("ダウンロード処理が完了しました。")
        return

    # 単一モデルダウンロード
    if args.model:
        if args.model not in MODELS:
            print(f"エラー: モデル '{args.model}' が見つかりません。")
            print(f"利用可能なモデル: {', '.join(MODELS.keys())}")
            sys.exit(1)

        print(f"検出されたデバイス: {downloader.device}")
        print()

        try:
            downloader.download(args.model, force=args.force)
        except Exception as e:
            print(f"✗ エラー: ダウンロードに失敗しました: {e}")
            sys.exit(1)

        return

    # 引数なしの場合はヘルプを表示
    parser.print_help()


if __name__ == "__main__":
    main()
