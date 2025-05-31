
import cv2
import mediapipe as mp
import json
import os
from datetime import datetime
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDataCollector:
    def __init__(self, label, display_label: QLabel, status_label: QLabel, save_dir='data', frame_count=30):
        self.label = label
        self.display_label = display_label
        self.status_label = status_label
        self.save_dir = save_dir
        self.frame_count = frame_count

        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.collected_data = []
        self.count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def start(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.collected_data.clear()
        self.count = 0
        self.timer.start(30)
        self.status_label.setText("状态：采集中...")

    def _update_frame(self):
        if not self.cap.isOpened() or self.count >= self.frame_count:
            self.timer.stop()
            self.cap.release()
            self.hands.close()
            self._save_data()
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            self.collected_data.append(keypoints)
            self.count += 1

            # 绘制手部骨架
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            cv2.putText(frame, f"Recording {self.count}/{self.frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "请将手放在摄像头前...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示图像到 QLabel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(self.display_label.width(), self.display_label.height())
        self.display_label.setPixmap(pixmap)

    def _save_data(self):
        if self.collected_data:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.save_dir, f"{self.label}_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump({
                    "label": self.label,
                    "frames": self.collected_data
                }, f, indent=4)
            print(f"[✅] 数据保存至：{filename}")
            self.status_label.setText(f"✅ 采集完成\n文件：{filename}")
        else:
            print("[⚠️] 未采集到手部关键点")
            self.status_label.setText("⚠️ 采集失败")
