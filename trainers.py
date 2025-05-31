import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_dataset(data_dir='data'):
    X, y = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                frames = data['frames']
                if len(frames) == 30:  # 仅处理标准长度序列
                    X.append(frames)
                    y.append(data['label'])
    return np.array(X), np.array(y)

def train_lstm_model(data_dir='data', model_path='models/gesture_lstm_model.h5', encoder_path='models/label_encoder.pkl'):
    X, y = load_dataset(data_dir)
    if len(X) == 0:
        raise ValueError("没有找到有效的数据。请先采集手势。")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # 保存标签编码器
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    joblib.dump(label_encoder, encoder_path)

    # 归一化
    X = X.astype('float32')

    # 切分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 构建模型
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30, 63)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练
    history = model.fit(X_train, y_train, epochs=30, batch_size=16,
                        validation_data=(X_test, y_test), verbose=1)

    # 保存模型
    model.save(model_path)

    # 混淆矩阵
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=False)

    return history, cm, label_encoder.classes_, report
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PyQt5.QtGui import QImage, QPixmap

def plot_training_history_to_pixmap(history):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(history.history['accuracy'], label='训练准确率')
    axs[0].plot(history.history['val_accuracy'], label='验证准确率')
    axs[0].set_title("准确率")
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='训练损失')
    axs[1].plot(history.history['val_loss'], label='验证损失')
    axs[1].set_title("损失")
    axs[1].legend()

    fig.tight_layout()

    buffer = BytesIO()
    FigureCanvas(fig).print_png(buffer)
    plt.close(fig)

    qimg = QImage.fromData(buffer.getvalue())
    return QPixmap.fromImage(qimg)

def plot_confusion_matrix_to_pixmap(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵")

    buffer = BytesIO()
    FigureCanvas(fig).print_png(buffer)
    plt.close(fig)

    qimg = QImage.fromData(buffer.getvalue())
    return QPixmap.fromImage(qimg)
