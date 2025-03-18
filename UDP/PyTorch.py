import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
from setting.setting import LABELS, PROCESSING_SIZE  # Giả định file setting chứa LABELS và PROCESSING_SIZE
import time


# Định nghĩa mô hình CNN trong PyTorch
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Cấu trúc mô hình CNN với 3 tầng tích chập và 2 tầng fully connected
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (PROCESSING_SIZE // 8) * (PROCESSING_SIZE // 8), 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 26)  # 26 lớp tương ứng với A-Z

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Lớp nhận diện ngôn ngữ ký hiệu
class SignLanguageRecognizer:
    def __init__(self, model_path):
        # Khởi tạo mô hình và tải trọng số
        self.model = CNNModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        # Khởi tạo Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.class_names = LABELS  # Danh sách nhãn từ A-Z
        self.processing_size = PROCESSING_SIZE  # Kích thước xử lý ảnh
        # Chuẩn hóa dữ liệu chỉ dùng ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Chuyển pixel từ [0, 255] về [0, 1]
        ])

    def _preprocess(self, image):
        """Tiền xử lý ảnh: phát hiện tay bằng Mediapipe và chuẩn bị tensor đầu vào"""
        # Chuyển đổi ảnh sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        # Nếu không phát hiện tay, trả về None
        if not results.multi_hand_landmarks:
            return None

        # Tạo mask tay
        height, width, _ = image.shape
        hand_mask = np.zeros((height, width), dtype=np.uint8)
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * width), int(lm.y * height)
                points.append((x, y))
            cv2.fillPoly(hand_mask, [np.array(points, dtype=np.int32)], 255)

        # Làm mịn và mở rộng mask
        kernel = np.ones((5, 5), np.uint8)
        hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
        hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21, 21), 0)

        # Tách vùng tay ra khỏi nền
        hand_on_black_background = np.where(hand_mask_blurred[..., None] > 0, image, np.zeros_like(image))

        # Tìm contour để xác định bounding box của tay
        contours, _ = cv2.findContours(hand_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Xác định vùng bao quanh tay
        x_min, x_max, y_min, y_max = width, 0, height, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)

        # Cắt vùng tay
        hand_area = hand_on_black_background[y_min:y_max, x_min:x_max]
        hand_width, hand_height = x_max - x_min, y_max - y_min

        # Điều chỉnh kích thước vùng tay
        scale_factor = min(self.processing_size / hand_width, self.processing_size / hand_height) * 0.8
        scaled_w, scaled_h = int(hand_width * scale_factor), int(hand_height * scale_factor)
        resized_hand_area = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # Đặt vùng tay vào giữa ảnh kích thước PROCESSING_SIZE
        centered_image = np.zeros((self.processing_size, self.processing_size, 3), dtype=np.uint8)
        x_offset = (self.processing_size - scaled_w) // 2
        y_offset = (self.processing_size - scaled_h) // 2
        centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand_area

        # Chuyển sang ảnh grayscale và mở rộng thành 3 kênh
        grayscale_frame = cv2.cvtColor(centered_image, cv2.COLOR_BGR2GRAY)
        expanded_frame = np.stack((grayscale_frame,) * 3, axis=-1)

        # Chuyển thành tensor
        image_pil = Image.fromarray(expanded_frame)
        image_tensor = self.transform(image_pil)
        return image_tensor

    def preprocess_image(self, path):
        """Đọc ảnh từ đường dẫn và tiền xử lý"""
        image = cv2.imread(path)
        if image is None:
            return None
        return self._preprocess(image)

    def predict(self, path):
        """Dự đoán nhãn từ ảnh đầu vào"""
        preprocessed = self.preprocess_image(path)
        if preprocessed is None:
            return None

        # Thêm batch dimension và dự đoán
        preprocessed = preprocessed.unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(preprocessed)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return self.class_names[predicted_index]


if __name__ == "__main__":
    # Khởi tạo đối tượng nhận diện và kiểm tra hiệu suất
    recognizer = SignLanguageRecognizer('D:\\AAAA\\models\\CNN\\cnn_best_model.pth')
    t = time.time()
    count = 0
    label = "B"
    for i in range(1, 900, 1):
        result = recognizer.predict(f'D:\\met_moi_qua_di\\dataset\\raw\\Train_Alphabet\\{label}\\{label}_{i}.png')
        if result == label:
            count += 1
    print(f"Số ảnh dự đoán đúng là {label}: {count}")
    print(f"Thời gian chạy: {time.time() - t} giây")