import cv2
import numpy as np
from ultralytics import YOLO
import pygame
from datetime import datetime
import threading
import queue
import os
from huggingface_hub import hf_hub_download
import platform

# 定数定義
CONFIDENCE_THRESHOLD = 0.75  # 信頼度の閾値（カーテン誤検知対策で高めに設定）
IOU_THRESHOLD = 0.45  # IoU閾値（重複検出の抑制）
DETECTION_COOLDOWN = 2.0  # アラート再生のクールダウン時間（秒）
MIN_DETECTION_SIZE = 10000  # 最小検出サイズ（ピクセル数）
ASPECT_RATIO_MIN = 0.5  # 最小アスペクト比（幅/高さ）
ASPECT_RATIO_MAX = 3.0  # 最大アスペクト比（幅/高さ）
FONT_SCALE = 0.8  # フォントサイズ
FONT_THICKNESS = 2  # フォントの太さ
STATUS_BG_COLOR = (50, 50, 50)  # ステータス背景色
STATUS_TEXT_COLOR = (255, 255, 255)  # ステータステキスト色
DETECTION_COLOR = (0, 255, 0)  # 検出枠の色（緑）
ALERT_COLOR = (0, 0, 255)  # アラート時の色（赤）
WARNING_COLOR = (0, 165, 255)  # 警告色（オレンジ）


# 日本語フォントの設定
def get_japanese_font():
    """OSに応じた日本語フォントのパスを取得"""
    system = platform.system()
    if system == "Darwin":  # macOS
        return "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
    elif system == "Windows":
        return "C:/Windows/Fonts/meiryo.ttc"
    else:  # Linux
        return "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


class ForkliftDetector:
    def __init__(
        self, model_repo="keremberke/yolov8s-forklift-detection", sound_file="beep.mp3"
    ):
        """
        フォークリフト検知器の初期化

        Args:
            model_repo: Hugging FaceのモデルリポジトリID
            sound_file: アラート音声ファイルのパス
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

        # モデルのクラス名を確認
        print(f"検出可能なクラス: {self.model.names}")

        # Pygameでサウンドシステムを初期化
        pygame.mixer.init()
        try:
            self.alert_sound = pygame.mixer.Sound(sound_file)
        except:
            print(f"警告: {sound_file}が見つかりません。アラート音は無効です。")
            self.alert_sound = None

        # 検出状態の管理
        self.last_alert_time = 0
        self.detection_count = 0
        self.total_frames = 0
        self.fps = 0
        self.false_positive_count = 0  # 誤検知カウント

        # 連続検出の追跡
        self.detection_history = []  # 最近の検出履歴
        self.history_size = 5  # 履歴サイズ

        # マルチスレッド用のキュー
        self.alert_queue = queue.Queue()

        # アラート再生用のスレッドを開始
        self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
        self.alert_thread.start()

    def _alert_worker(self):
        """アラート音を再生するワーカースレッド"""
        while True:
            try:
                self.alert_queue.get(timeout=1)
                if self.alert_sound:
                    self.alert_sound.play()
            except queue.Empty:
                continue

    def _draw_status(self, frame, detections):
        """
        画面左上にステータス情報を表示

        Args:
            frame: 画像フレーム
            detections: 検出結果
        """
        height, width = frame.shape[:2]

        # ステータス背景の描画
        status_height = 180
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

        # ステータステキストの描画（英語で表示）
        y_offset = 30
        texts = [
            f"Confidence: {CONFIDENCE_THRESHOLD:.2f}",
            f"Min Size: {MIN_DETECTION_SIZE} px",
            f"FPS: {self.fps:.1f}",
            f"Detections: {len(detections)}",
            f"Total Count: {self.detection_count}",
            f"Filtered Out: {self.false_positive_count}",
        ]

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

    def _draw_detection(self, frame, box, conf, cls_name, is_valid=True):
        """
        検出結果の描画

        Args:
            frame: 画像フレーム
            box: バウンディングボックス座標
            conf: 信頼度
            cls_name: クラス名
            is_valid: 有効な検出かどうか
        """
        x1, y1, x2, y2 = map(int, box)

        # 検出枠の色を決定
        if not is_valid:
            color = WARNING_COLOR  # フィルタされた検出はオレンジ
        elif cls_name == "forklift":
            color = DETECTION_COLOR  # 有効なフォークリフトは緑
        else:
            color = ALERT_COLOR  # その他は赤

        # 検出枠の描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ラベルの背景
        label = f"{cls_name}: {conf:.2f}"
        if not is_valid:
            label += " (filtered)"
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        cv2.rectangle(
            frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1
        )

        # ラベルテキスト
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )

    def _is_valid_detection(self, box):
        """
        検出が有効かどうかを判定（誤検知フィルタリング）

        Args:
            box: バウンディングボックス座標

        Returns:
            bool: 有効な検出かどうか
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # サイズチェック
        if area < MIN_DETECTION_SIZE:
            return False

        # アスペクト比チェック（極端に細長い物体を除外）
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
            return False

        return True

    def process_frame(self, frame):
        """
        フレームを処理して検出を実行

        Args:
            frame: 入力フレーム

        Returns:
            処理済みフレーム, フォークリフト検出フラグ
        """
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

                    # 検出の有効性をチェック
                    is_valid = self._is_valid_detection(box_coords)

                    # デバッグ用：検出情報を出力
                    if cls_name == "forklift":
                        x1, y1, x2, y2 = box_coords
                        area = (x2 - x1) * (y2 - y1)
                        aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                        status = "VALID" if is_valid else "FILTERED"
                        print(
                            f"Forklift detected: conf={conf:.2f}, area={area:.0f}, aspect={aspect_ratio:.2f} - {status}"
                        )

                    # フォークリフトの検出確認
                    if (
                        cls_name == "forklift"
                        and conf >= CONFIDENCE_THRESHOLD
                        and is_valid
                    ):
                        forklift_detected = True
                        valid_forklift_count += 1
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

                    # 検出結果の描画
                    self._draw_detection(frame, box_coords, conf, cls_name, is_valid)

        # 連続検出の確認（ノイズ除去）
        self.detection_history.append(valid_forklift_count > 0)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)

        # 最近の検出履歴で過半数が検出している場合のみ有効
        if len(self.detection_history) >= 3:
            detection_ratio = sum(self.detection_history) / len(self.detection_history)
            if detection_ratio < 0.6:  # 60%未満の場合は無視
                forklift_detected = False

        # 有効な検出のみカウント
        if forklift_detected:
            self.detection_count += 1

        # ステータス情報の描画
        self._draw_status(frame, [d for d in detections if d["is_valid"]])

        # アラート処理
        if forklift_detected:
            current_time = datetime.now().timestamp()
            if current_time - self.last_alert_time > DETECTION_COOLDOWN:
                self.alert_queue.put(True)
                self.last_alert_time = current_time

        return frame, forklift_detected

    def run(self):
        """メインループ：カメラからの映像を処理"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("カメラを開けませんでした")
            return

        print("フォークリフト検知システムを開始します...")
        print("'q'キーで終了します")
        print(f"\n設定:")
        print(f"- 信頼度閾値: {CONFIDENCE_THRESHOLD}")
        print(f"- IoU閾値: {IOU_THRESHOLD}")
        print(f"- アラートクールダウン: {DETECTION_COOLDOWN}秒\n")

        prev_time = datetime.now().timestamp()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("フレームの読み込みに失敗しました")
                    break

                # FPS計算
                current_time = datetime.now().timestamp()
                self.fps = 1.0 / (current_time - prev_time)
                prev_time = current_time
                self.total_frames += 1

                # フレーム処理
                processed_frame, detected = self.process_frame(frame)

                # 結果の表示
                cv2.imshow("Forklift Detection System", processed_frame)

                # 'q'キーで終了
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            print(f"エラーが発生しました: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n総フレーム数: {self.total_frames}")
            print(f"総検出回数: {self.detection_count}")
            print(f"フィルタされた誤検知: {self.false_positive_count}")


def main():
    """メイン関数"""
    detector = ForkliftDetector()
    detector.run()


if __name__ == "__main__":
    main()
