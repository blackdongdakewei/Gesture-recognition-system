import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

class RealTimeRecognizer:
    def __init__(self, model_path='models/gesture_lstm_model.h5', encoder_path='models/label_encoder.pkl'):
        if not os.path.exists(model_path):
            print(f"❌ 模型文件未找到: {model_path}")
            self.model = None
            return

        if not os.path.exists(encoder_path):
            print(f"❌ 标签编码器文件未找到: {encoder_path}")
            self.model = None
            return

        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.sequence = []
        self.max_length = 30
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

    def process_frame(self, frame):
        if self.model is None:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            self.sequence.append(keypoints)

            if len(self.sequence) > self.max_length:
                self.sequence.pop(0)

            if len(self.sequence) == self.max_length:
                input_data = np.array(self.sequence)[np.newaxis, ...]
                prediction = self.model.predict(input_data)[0]
                predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
                confidence = float(np.max(prediction))
                return predicted_label, confidence

        return None, None
