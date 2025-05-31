import sys
from data_collector import HandDataCollector
from trainers import train_lstm_model, plot_training_history_to_pixmap,  plot_confusion_matrix_to_pixmap
from recognizer import RealTimeRecognizer
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QTabWidget, QLineEdit, QTextEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt



class DataCollectionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("输入动作名称:")
        self.input_name = QLineEdit()
        self.start_button = QPushButton("开始录制")
        self.status = QLabel("状态：等待录制")
        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 360)

        layout.addWidget(self.label)
        layout.addWidget(self.input_name)
        layout.addWidget(self.start_button)
        layout.addWidget(self.status)
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.start_button.clicked.connect(self.start_recording)
        self.collector = None

    def start_recording(self):
        name = self.input_name.text().strip()
        if not name:
            QMessageBox.warning(self, "错误", "请输入动作名称")
            return

        self.collector = HandDataCollector(
            label=name,
            display_label=self.video_label,
            status_label=self.status
        )
        self.collector.start()


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.train_button = QPushButton("开始训练")
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.train_button)
        layout.addWidget(self.output)
        self.setLayout(layout)

        self.train_button.clicked.connect(self.start_training)

        self.history_plot = QLabel()
        self.confusion_plot = QLabel()
        layout.addWidget(self.history_plot)
        layout.addWidget(self.confusion_plot)

    def start_training(self):
        self.output.append("加载数据中...")
        QApplication.processEvents()

        try:
            history, cm, classes, report = train_lstm_model()
            self.output.append("训练完成 ✅\n")
            self.output.append(report)

            pix_history = plot_training_history_to_pixmap(history)
            pix_confusion = plot_confusion_matrix_to_pixmap(cm, classes)

            self.history_plot.setPixmap(pix_history)
            self.confusion_plot.setPixmap(pix_confusion)


        except Exception as e:
            self.output.append(f"❌ 训练失败: {str(e)}")
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class RecognitionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.start_button = QPushButton("启动实时识别")
        self.info_label = QLabel("当前识别结果：未开始")
        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 360)

        layout.addWidget(self.start_button)
        layout.addWidget(self.info_label)
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.start_button.clicked.connect(self.start_recognition)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.recognizer = None

        self.stop_button = QPushButton("停止识别")
        self.stop_button.setEnabled(False)  # 启动前不可点
        layout.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.stop_recognition)

    def start_recognition(self):
        self.info_label.setText("🔍 启动摄像头识别中...")
        QApplication.processEvents()

        self.cap = cv2.VideoCapture(0)
        self.recognizer = RealTimeRecognizer()

        if self.recognizer.model is None:
            QMessageBox.critical(self, "模型错误", "未找到模型文件或标签编码器，无法启动识别。")
            self.cap.release()
            self.cap = None
            self.info_label.setText("❌ 无法启动识别（缺失模型）")
            return

        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recognition(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.video_label.clear()
        self.info_label.setText("识别已停止 ⛔️")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if self.recognizer:
            self.recognizer.hands.close()
            self.recognizer = None

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        label, conf = self.recognizer.process_frame(frame)
        if label:
            text = f"识别: {label} ({conf:.2f})"
            self.info_label.setText(text)
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            self.info_label.setText("识别中...")
            cv2.putText(frame, "识别中...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2)

        # OpenCV 图像转 QImage 显示在 QLabel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.recognizer:
            self.recognizer.hands.close()
        event.accept()



class GestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动态手势识别系统")
        self.setGeometry(300, 100, 600, 400)

        self.tabs = QTabWidget()
        self.tabs.addTab(DataCollectionTab(), "数据采集")
        self.tabs.addTab(TrainingTab(), "模型训练")
        self.tabs.addTab(RecognitionTab(), "实时识别")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureApp()
    window.show()
    sys.exit(app.exec_())
