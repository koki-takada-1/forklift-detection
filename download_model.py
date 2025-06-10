#!/usr/bin/env python3
"""
Hugging Faceからフォークリフト検出モデルを事前にダウンロードするスクリプト
"""

from huggingface_hub import hf_hub_download
import os


def download_forklift_model():
    """フォークリフト検出モデルをダウンロード"""
    repo_id = "keremberke/yolov8s-forklift-detection"

    print(f"モデルをダウンロードしています: {repo_id}")

    try:
        # モデルファイルをダウンロード
        model_path = hf_hub_download(
            repo_id=repo_id, filename="best.pt", cache_dir="./models"
        )

        print(f"ダウンロード完了: {model_path}")
        print("\nモデルのダウンロードが成功しました！")
        print("forklift_detection.pyを実行できます。")

        return model_path

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("\nモデルのダウンロードに失敗しました。")
        print("インターネット接続を確認してください。")
        return None


if __name__ == "__main__":
    download_forklift_model()
