#!/usr/bin/env python3
"""
フォークリフト検知システム - 動画ファイル処理版（音声付き）
検知時にbeep音を追加した動画を出力
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import sys
import argparse
from huggingface_hub import hf_hub_download
from pathlib import Path
import subprocess
import tempfile
import json

# 定数定義
CONFIDENCE_THRESHOLD = 0.75
IOU_THRESHOLD = 0.45
MIN_DETECTION_SIZE = 10000
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 3.0
FONT_SCALE = 0.8
FONT_THICKNESS = 2  # フォントの太さ
DETECTION_BOX_THICKNESS = 2  # 検出枠の線の太さ
DETECTION_BOX_SCALE = 0.9  # 検出枠のスケール（1.0=100%, 0.9=90%に縮小）
STATUS_BG_COLOR = (50, 50, 50)
STATUS_TEXT_COLOR = (255, 255, 255)
DETECTION_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
WARNING_COLOR = (0, 165, 255)
ALERT_DURATION = 1.0  # アラート音の長さ（秒）
ALERT_COOLDOWN = 3.0  # アラート音のクールダウン時間（秒）


class ForkliftDetectorVideoAudio:
    def __init__(
        self, model_repo="keremberke/yolov8s-forklift-detection", beep_file="beep.mp3"
    ):
        """
        フォークリフト検知器の初期化

        Args:
            model_repo: Hugging FaceのモデルリポジトリID
            beep_file: アラート音声ファイル
        """
        # Hugging Faceからモデルをダウンロード
        print(f"モデルをダウンロードしています: {model_repo}")
        try:
            model_path = hf_hub_download(
                repo_id=model_repo, filename="best.pt", cache_dir="./models"
            )
            print(f"モデルのダウンロード完了: {model_path}")
        except Exception as e:
            print(f"モデルのダウンロードに失敗しました: {e}")
            print("代替として、通常のYOLOv8モデルを使用します")
            model_path = "yolov8s.pt"

        # YOLOv8モデルの読み込み
        print("モデルを読み込んでいます...")
        self.model = YOLO(model_path)
        print(f"検出可能なクラス: {self.model.names}")

        # アラート音声ファイルの確認
        self.beep_file = beep_file
        self.has_beep = os.path.exists(beep_file)
        if not self.has_beep:
            print(f"警告: アラート音声ファイル '{beep_file}' が見つかりません")
            print("音声なしで処理を続行します")

        # FFmpegの確認
        self.has_ffmpeg = self._check_ffmpeg()

        # 検出状態の管理
        self.detection_count = 0
        self.false_positive_count = 0
        self.total_frames = 0
        self.fps = 0
        self.detection_history = []
        self.history_size = 5
        self.detection_timestamps = []  # 検出時刻を記録
        self.last_alert_time = -ALERT_COOLDOWN  # 最後のアラート時刻

    def _check_ffmpeg(self):
        """FFmpegが利用可能か確認"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except:
            print("警告: FFmpegが見つかりません。音声付き動画の生成ができません")
            print("FFmpegをインストールしてください: https://ffmpeg.org/")
            return False

    def _draw_status(self, frame, detections, current_time, total_time):
        """画面左上にステータス情報を表示"""
        height, width = frame.shape[:2]

        # ステータス背景の描画
        status_height = 200
        status_width = 350
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (status_width, status_height), STATUS_BG_COLOR, -1
        )
        frame[:status_height, :status_width] = cv2.addWeighted(
            overlay[:status_height, :status_width],
            0.7,
            frame[:status_height, :status_width],
            0.3,
            0,
        )

        # 進行状況
        progress = current_time / total_time if total_time > 0 else 0

        # ステータステキストの描画
        y_offset = 30
        texts = [
            f"Time: {current_time:.1f}s / {total_time:.1f}s",
            f"Progress: {progress*100:.1f}%",
            f"Confidence: {CONFIDENCE_THRESHOLD:.2f}",
            f"FPS: {self.fps:.1f}",
            f"Detections: {len(detections)}",
            f"Total Count: {self.detection_count}",
            f"Filtered Out: {self.false_positive_count}",
        ]

        # 音声アラートのインジケーター
        if (
            self.has_beep
            and self.last_alert_time >= 0
            and current_time - self.last_alert_time < ALERT_DURATION
        ):
            texts.append("ALERT: Playing beep!")
            cv2.circle(frame, (status_width - 30, 30), 10, (0, 0, 255), -1)
        elif (
            self.has_beep
            and self.last_alert_time >= 0
            and current_time - self.last_alert_time < ALERT_COOLDOWN
        ):
            remaining = ALERT_COOLDOWN - (current_time - self.last_alert_time)
            texts.append(f"Cooldown: {remaining:.1f}s")

        for i, text in enumerate(texts):
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE * 0.7,
                STATUS_TEXT_COLOR,
                FONT_THICKNESS - 1,
            )

    def _scale_box(self, box, scale=DETECTION_BOX_SCALE):
        """
        バウンディングボックスを中心から縮小/拡大

        Args:
            box: 元のバウンディングボックス座標 [x1, y1, x2, y2]
            scale: スケール係数（1.0=100%, 0.9=90%に縮小, 1.1=110%に拡大）

        Returns:
            スケール調整後のボックス座標
        """
        x1, y1, x2, y2 = box

        # ボックスの中心点を計算
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # ボックスの幅と高さ
        width = x2 - x1
        height = y2 - y1

        # スケール調整後の幅と高さ
        new_width = width * scale
        new_height = height * scale

        # 新しい座標を計算（中心点を基準に）
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        return [new_x1, new_y1, new_x2, new_y2]

    def _draw_detection(self, frame, box, conf, cls_name, is_valid=True):
        """
        検出結果の描画

        検出枠のサイズを調整する部分：
        - DETECTION_BOX_SCALE: 枠の大きさ（0.9=90%に縮小, 1.1=110%に拡大）
        - DETECTION_BOX_THICKNESS: 枠線の太さ（2→1などに変更可能）
        - FONT_SCALE: 文字の大きさ（0.6→0.4などに変更可能）
        - FONT_THICKNESS: 文字の太さ（1→1のまま、または変更可能）
        """
        # バウンディングボックスをスケール調整
        scaled_box = self._scale_box(box, DETECTION_BOX_SCALE)
        x1, y1, x2, y2 = map(int, scaled_box)

        height, width = frame.shape[:2]

        # フレーム境界内に収める
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if not is_valid:
            color = WARNING_COLOR
        elif cls_name == "forklift":
            color = DETECTION_COLOR
        else:
            color = ALERT_COLOR

        # 検出枠の描画（DETECTION_BOX_THICKNESSで線の太さを制御）
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, DETECTION_BOX_THICKNESS)

        # ラベルの作成
        label = f"{cls_name}: {conf:.2f}"
        if not is_valid:
            label += " (filtered)"

        # ラベルサイズの計算（FONT_SCALEとFONT_THICKNESSで文字サイズを制御）
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )

        # ラベルが画面外に出ないように調整
        label_y1 = max(label_size[1] + 10, y1)
        label_x2 = min(x1 + label_size[0] + 10, width - 5)

        # ラベルの背景（少し小さめのパディング）
        cv2.rectangle(
            frame, (x1, label_y1 - label_size[1] - 8), (label_x2, label_y1), color, -1
        )

        # ラベルテキストの描画
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )

    def _is_valid_detection(self, box):
        """検出が有効かどうかを判定（元のボックスサイズで判定）"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area < MIN_DETECTION_SIZE:
            return False

        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
            return False

        return True

    def process_frame(self, frame, frame_number, current_time):
        """フレームを処理して検出を実行"""
        # YOLOv8で検出
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

        forklift_detected = False
        detections = []
        valid_forklift_count = 0

        # 検出結果の処理
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)
                    cls_name = self.model.names[cls]
                    box_coords = box.xyxy[0].cpu().numpy()

                    is_valid = self._is_valid_detection(box_coords)

                    if (
                        cls_name == "forklift"
                        and conf >= CONFIDENCE_THRESHOLD
                        and is_valid
                    ):
                        forklift_detected = True
                        valid_forklift_count += 1
                        print(
                            f"フレーム {frame_number}: フォークリフト検出 (信頼度: {conf:.2f}) at {current_time:.2f}s"
                        )
                    elif cls_name == "forklift" and not is_valid:
                        self.false_positive_count += 1

                    detections.append(
                        {
                            "box": box_coords,
                            "conf": conf,
                            "cls_name": cls_name,
                            "is_valid": is_valid,
                        }
                    )

                    self._draw_detection(frame, box_coords, conf, cls_name, is_valid)

        # 連続検出の確認
        self.detection_history.append(valid_forklift_count > 0)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)

        if len(self.detection_history) >= 3:
            detection_ratio = sum(self.detection_history) / len(self.detection_history)
            if detection_ratio < 0.6:
                forklift_detected = False

        # 検出時刻の記録（音声追加用）
        if forklift_detected:
            self.detection_count += 1
            # クールダウン期間をチェック（ALERT_COOLDOWN秒間は音を鳴らさない）
            if current_time - self.last_alert_time >= ALERT_COOLDOWN:
                self.detection_timestamps.append(current_time)
                self.last_alert_time = current_time
                print(f"  → アラート音を追加: {current_time:.2f}秒")
            else:
                remaining = ALERT_COOLDOWN - (current_time - self.last_alert_time)
                print(f"  → クールダウン中（残り{remaining:.1f}秒）")

        return frame, forklift_detected, [d for d in detections if d["is_valid"]]

    def _add_audio_to_video(self, video_path, output_path):
        """FFmpegを使用して動画に音声を追加"""
        if (
            not self.has_ffmpeg
            or not self.has_beep
            or len(self.detection_timestamps) == 0
        ):
            # 音声追加不可の場合は、元の動画をコピー
            if video_path != output_path:
                import shutil

                shutil.copy2(video_path, output_path)
            return

        print(f"\n音声を追加しています... ({len(self.detection_timestamps)}箇所)")

        # 一時ファイルで音声トラックを作成
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            # FFmpeg concat用のリストファイルを作成
            for timestamp in self.detection_timestamps:
                f.write(f"file '{os.path.abspath(self.beep_file)}'\n")
            concat_file = f.name

        try:
            # 複雑なFFmpegコマンドを構築
            filter_complex = []
            audio_inputs = []

            # 各検出タイムスタンプに対してビープ音を配置
            for i, timestamp in enumerate(self.detection_timestamps):
                delay_ms = int(timestamp * 1000)
                filter_complex.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[beep{i}]")
                audio_inputs.extend(["-i", self.beep_file])

            # すべての音声を混合
            if len(self.detection_timestamps) == 1:
                amix_inputs = "[beep0]"
            else:
                amix_inputs = "".join(
                    [f"[beep{i}]" for i in range(len(self.detection_timestamps))]
                )

            filter_complex.append(
                f"{amix_inputs}amix=inputs={len(self.detection_timestamps)}:duration=longest[mixed]"
            )

            # FFmpegコマンド
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,  # 元の動画
                *audio_inputs,  # ビープ音（複数）
                "-filter_complex",
                ";".join(filter_complex),
                "-map",
                "0:v",  # 元の動画の映像
                "-map",
                "[mixed]",  # 混合された音声
                "-c:v",
                "copy",  # 映像はコピー
                "-c:a",
                "aac",  # 音声はAAC
                output_path,
            ]

            # コマンド実行
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"FFmpegエラー: {result.stderr}")
                # エラー時は元の動画をコピー
                import shutil

                shutil.copy2(video_path, output_path)
            else:
                print("音声の追加が完了しました")

        finally:
            # 一時ファイルを削除
            if os.path.exists(concat_file):
                os.unlink(concat_file)

    def process_video(self, input_path, output_path=None):
        """動画ファイルを処理"""
        if not os.path.exists(input_path):
            print(f"エラー: 入力ファイル '{input_path}' が見つかりません")
            return False

        # 出力ファイル名の生成
        if output_path is None:
            input_name = Path(input_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{input_name}_detected_{timestamp}.mp4"

        # 一時ファイル（映像のみ）
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        # 動画の読み込み
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{input_path}' を開けません")
            return False

        # 動画のプロパティ取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_time = total_frames / fps if fps > 0 else 0

        print(f"\n動画情報:")
        print(f"- 入力: {input_path}")
        print(f"- 解像度: {width}x{height}")
        print(f"- FPS: {fps}")
        print(f"- 総フレーム数: {total_frames}")
        print(f"- 長さ: {total_time:.1f}秒")

        # 動画書き込みの準備（一時ファイルへ）
        # H.264コーデックを使用して互換性を向上
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        if not out.isOpened():
            # 別のコーデックで再試行
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                temp_video.replace(".mp4", ".avi"), fourcc, fps, (width, height)
            )
            temp_video = temp_video.replace(".mp4", ".avi")

        if not out.isOpened():
            print(f"エラー: 一時ファイルを作成できません")
            cap.release()
            return False

        print(f"\n処理を開始します...")

        # 元の動画と同じFPSとサイズを確実に使用
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"出力設定: {actual_width}x{actual_height} @ {actual_fps}fps")

        # 処理開始
        start_time = datetime.now()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 現在の時間計算
                current_time = frame_count / fps if fps > 0 else 0

                # FPS計算
                if frame_count > 0 and frame_count % 30 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    self.fps = frame_count / elapsed

                # フレーム処理
                processed_frame, detected, detections = self.process_frame(
                    frame, frame_count, current_time
                )

                # ステータス描画
                self._draw_status(processed_frame, detections, current_time, total_time)

                # フレーム書き込み
                out.write(processed_frame)

                # 進捗表示
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"進捗: {progress:.1f}% ({frame_count}/{total_frames}フレーム)"
                    )

                frame_count += 1
                self.total_frames = frame_count

        except KeyboardInterrupt:
            print("\n処理を中断しました")
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
        finally:
            # リソースの解放
            cap.release()
            out.release()

            # 音声を追加
            self._add_audio_to_video(temp_video, output_path)

            # 一時ファイルを削除
            if os.path.exists(temp_video):
                os.unlink(temp_video)

            # 処理時間の表示
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"\n処理完了:")
            print(f"- 処理時間: {processing_time:.1f}秒")
            print(f"- 処理フレーム数: {frame_count}")
            print(f"- 平均FPS: {frame_count/processing_time:.1f}")
            print(f"- フォークリフト検出回数: {self.detection_count}")
            print(f"- アラート音追加箇所: {len(self.detection_timestamps)}")
            print(f"- フィルタされた誤検知: {self.false_positive_count}")
            print(f"- 出力ファイル: {output_path}")

        return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="フォークリフト検知システム - 動画ファイル処理版（音声付き）"
    )
    parser.add_argument("input", type=str, help="入力動画ファイルのパス")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="出力動画ファイルのパス（省略時は自動生成）",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.85,
        help="検出の信頼度閾値（0.0-1.0、デフォルト: 0.85）",
    )
    parser.add_argument(
        "-s",
        "--min-size",
        type=int,
        default=10000,
        help="最小検出サイズ（ピクセル数、デフォルト: 10000）",
    )
    parser.add_argument(
        "-b",
        "--beep",
        type=str,
        default="beep.mp3",
        help="アラート音声ファイル（デフォルト: beep.mp3）",
    )

    args = parser.parse_args()

    # グローバル設定の更新
    global CONFIDENCE_THRESHOLD, MIN_DETECTION_SIZE
    CONFIDENCE_THRESHOLD = args.confidence
    MIN_DETECTION_SIZE = args.min_size

    # 検知器の初期化と実行
    detector = ForkliftDetectorVideoAudio(beep_file=args.beep)
    success = detector.process_video(args.input, args.output)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
