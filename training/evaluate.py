import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import mediapipe as mp
import tensorflow as tf
from setting.setting import LABELS, PROCESSING_SIZE

# Thiết bị chạy mô hình (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Định nghĩa các lớp model PyTorch
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (PROCESSING_SIZE // 8) * (PROCESSING_SIZE // 8), 256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 26)  # 26 lớp A-Z

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=26):
        super(MobileNetV2, self).__init__()
        self.model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=26):
        super(VGG16, self).__init__()
        self.model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=False)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


# Hàm tiền xử lý ảnh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    height, width, _ = image.shape
    hand_mask = np.zeros((height, width), dtype=np.uint8)
    for hand_landmarks in results.multi_hand_landmarks:
        points = []
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            points.append((x, y))
        cv2.fillPoly(hand_mask, [np.array(points, dtype=np.int32)], 255)
    kernel = np.ones((5, 5), np.uint8)
    hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
    hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21, 21), 0)
    hand_on_black_background = np.where(hand_mask_blurred[..., None] > 0, image, np.zeros_like(image))
    contours, _ = cv2.findContours(hand_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x_min, x_max, y_min, y_max = width, 0, height, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)
    hand_area = hand_on_black_background[y_min:y_max, x_min:x_max]
    hand_width, hand_height = x_max - x_min, y_max - y_min
    scale_factor = min(PROCESSING_SIZE / hand_width, PROCESSING_SIZE / hand_height) * 0.8
    scaled_w, scaled_h = int(hand_width * scale_factor), int(hand_height * scale_factor)
    resized_hand_area = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    centered_image = np.zeros((PROCESSING_SIZE, PROCESSING_SIZE, 3), dtype=np.uint8)
    x_offset = (PROCESSING_SIZE - scaled_w) // 2
    y_offset = (PROCESSING_SIZE - scaled_h) // 2
    centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand_area
    grayscale_frame = cv2.cvtColor(centered_image, cv2.COLOR_BGR2GRAY)
    expanded_frame = np.stack((grayscale_frame,) * 3, axis=-1)
    return expanded_frame


# Hàm tải và dự đoán với model Keras
def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_keras(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    preprocessed = preprocess_image(image)
    if preprocessed is None:
        return None
    prediction_frame = np.expand_dims(np.float32(preprocessed) / 255.0, axis=0)
    prediction = model.predict(prediction_frame)
    predicted_index = np.argmax(prediction)
    return predicted_index


# Hàm tải và dự đoán với model PyTorch
def load_pytorch_model(model_path, model_class):
    model = model_class().to(device)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Chỉ thêm tiền tố "model." cho các key liên quan đến features và classifier
        if key.startswith("features") or key.startswith("classifier"):
            new_key = "model." + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value  # Giữ nguyên các key khác (nếu có)
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


transform = transforms.Compose([transforms.ToTensor()])


def predict_pytorch(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    preprocessed = preprocess_image(image)
    if preprocessed is None:
        return None
    preprocessed = transform(preprocessed).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(preprocessed)
    predicted_index = torch.argmax(outputs, dim=1).item()
    return predicted_index


# Hàm lấy danh sách ảnh từ tập train
def get_all_images(train_dir):
    all_images = []
    for label in LABELS:
        label_dir = os.path.join(train_dir, label)
        images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith('.png')]
        all_images.extend([(img, label) for img in images])
    return all_images


# Hàm đánh giá model
def evaluate_model(model_type, model_path, eval_images):
    if model_type == 'keras':
        model = load_keras_model(model_path)
        predict_func = lambda img: predict_keras(model, img)
    elif model_type == 'pytorch':
        if 'cnn' in model_path.lower():
            model_class = CNNModel
        elif 'mobilenetv2' in model_path.lower():
            model_class = MobileNetV2
        elif 'vgg16' in model_path.lower():
            model_class = VGG16
        model = load_pytorch_model(model_path, model_class)
        predict_func = lambda img: predict_pytorch(model, img)
    else:
        raise ValueError("Invalid model_type. Choose 'keras' or 'pytorch'.")

    correct = 0
    total_time = 0
    total_images = len(eval_images)
    for img_path, label in eval_images:
        start_time = time.time()
        predicted_index = predict_func(img_path)
        end_time = time.time()
        if predicted_index is not None:
            predicted_label = LABELS[predicted_index]
            if predicted_label == label:
                correct += 1
            total_time += (end_time - start_time)

    accuracy = correct / total_images
    avg_time = total_time / total_images
    return accuracy, avg_time


# Main execution
if __name__ == "__main__":
    # Lấy danh sách ảnh và chọn ngẫu nhiên 1000 ảnh
    train_dir = '../dataset/raw/Train_Alphabet/'
    all_images = get_all_images(train_dir)
    random.seed(42)  # Để tái tạo kết quả
    eval_images = random.sample(all_images, 10000)

    # Danh sách model cần đánh giá
    models_info = [
        ('../models/CNN/cnn_best_model_v1.keras', 'keras'),
        ('../models/CNN/cnn_best_model_v1.pth', 'pytorch'),
        ('../models/MobileNetV2/mobilenetv2_best_model_v1.keras', 'keras'),
        ('../models/MobileNetV2/mobilenetv2_best_model_v1.pth', 'pytorch'),
        ('../models/VGG16/vgg16_best_model_v1.keras', 'keras'),
        ('../models/VGG16/vgg16_best_model_v1.pth', 'pytorch'),
    ]

    # Đánh giá từng model
    results = []
    for model_path, model_type in models_info:
        model_name = os.path.basename(model_path)
        print(f"Đang đánh giá {model_name}...")
        accuracy, avg_time = evaluate_model(model_type, model_path, eval_images)
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Average Time (s)': avg_time
        })

    # Lưu kết quả vào bảng CSV
    df = pd.DataFrame(results)
    df.to_csv('model_evaluation.csv', index=False)
    print("\nKết quả đã được lưu vào 'model_evaluation.csv':")
    print(df)

    # Vẽ đồ thị so sánh
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(df['Model']))

    plt.bar(index, df['Accuracy'], bar_width, color='blue', alpha=0.7, label='Accuracy')
    plt.bar(index + bar_width, df['Average Time (s)'], bar_width, color='red', alpha=0.7, label='Average Time (s)')

    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('So sánh độ chính xác và thời gian tính toán của các model')
    plt.xticks(index + bar_width / 2, df['Model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.show()
    print("Đồ thị đã được lưu vào 'model_evaluation.png'")