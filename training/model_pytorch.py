import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
from enum import Enum, auto
from setting.setting import LABELS, PROCESSING_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelType(Enum):
    MOBILENETV2 = auto()
    VGG16 = auto()
    CNN = auto()


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
        self.fc2 = nn.Linear(256, len(LABELS))

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


class SignLanguageRecognizer:
    def __init__(self, model_type: ModelType, model_path: str):
        self.model_type = model_type
        self._init_model()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self._init_mediapipe()
        self.class_names = LABELS

    def _init_model(self):
        if self.model_type == ModelType.MOBILENETV2:
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(LABELS))
            self.processing_size = 224
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.model_type == ModelType.VGG16:
            self.model = models.vgg16(pretrained=False)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, len(LABELS))
            self.processing_size = 224
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.model_type == ModelType.CNN:
            self.model = CNNModel()
            self.processing_size = PROCESSING_SIZE
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Invalid model type")

        self.model = self.model.to(device)

    def _init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def _preprocess(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None

        height, width = image.shape[:2]
        hand_mask = np.zeros((height, width), dtype=np.uint8)

        for hand_landmarks in results.multi_hand_landmarks:
            points = [(int(lm.x * width), int(lm.y * height))
                      for lm in hand_landmarks.landmark]
            cv2.fillPoly(hand_mask, [np.array(points, dtype=np.int32)], 255)

        # Tối ưu hóa các phép toán xử lý ảnh
        kernel = np.ones((5, 5), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
        hand_mask = cv2.GaussianBlur(hand_mask, (21, 21), 0)

        hand_image = np.where(hand_mask[..., None] > 0, image, 0)
        hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)

        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        hand_roi = hand_image[y:y + h, x:x + w]

        scale = min(self.processing_size / w, self.processing_size / h) * 0.8
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(hand_roi, new_size, interpolation=cv2.INTER_AREA)

        processed = np.zeros((self.processing_size, self.processing_size, 3), dtype=np.uint8)
        dx = (self.processing_size - new_size[0]) // 2
        dy = (self.processing_size - new_size[1]) // 2
        processed[dy:dy + new_size[1], dx:dx + new_size[0]] = resized

        return self.transform(Image.fromarray(processed))

    def predict(self, input_data):
        image = self._load_image(input_data)
        if image is None:
            return None

        tensor = self._preprocess(image)
        if tensor is None:
            return None

        with torch.no_grad():
            outputs = self.model(tensor.unsqueeze(0).to(device))
            probabilities = F.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_idx].item()
            return self.class_names[predicted_idx], confidence

    def _load_image(self, input_data):
        if isinstance(input_data, str):
            image = cv2.imread(input_data)
        elif isinstance(input_data, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(input_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            raise ValueError("Invalid input type")
        return image

    def detect_hand(self, input_data):
        try:
            image = self._load_image(input_data)
            if image is None:
                return False

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            return results.multi_hand_landmarks is not None
        except Exception as e:
            print(f"Error in hand detection: {str(e)}")
            return False

    def process_hand_image(self, input_data):
        try:
            image = self._load_image(input_data)
            if image is None:
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
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
            
            # Tìm contour của bàn tay
            kernel = np.ones((5, 5), np.uint8)
            hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=2)  # Tăng số lần dilation
            
            contours, _ = cv2.findContours(hand_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            # Tìm vùng bao quanh bàn tay và thêm padding
            x_min, x_max, y_min, y_max = width, 0, height, 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x + w), max(y_max, y + h)
            
            # Thêm padding cho vùng bàn tay (30% mỗi bên)
            padding_x = int((x_max - x_min) * 0.3)
            padding_y = int((y_max - y_min) * 0.3)
            
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(width, x_max + padding_x)
            y_max = min(height, y_max + padding_y)
            
            # Cắt vùng bàn tay từ ảnh gốc (không sử dụng mask)
            hand_roi = image[y_min:y_max, x_min:x_max]
            
            # Điều chỉnh tỷ lệ để vừa với kích thước 320x240 mà không biến dạng
            target_size = (320, 240)
            hand_h, hand_w = hand_roi.shape[:2]
            
            # Tính toán tỷ lệ thu nhỏ để giữ nguyên tỷ lệ khung hình
            scale = min(target_size[0] / hand_w, target_size[1] / hand_h)
            new_w, new_h = int(hand_w * scale), int(hand_h * scale)
            
            # Resize ảnh giữ nguyên tỷ lệ
            resized_hand = cv2.resize(hand_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Mã hóa ảnh thành bytes
            _, processed_image_bytes = cv2.imencode('.png', resized_hand)
            return processed_image_bytes.tobytes()
        except Exception as e:
            print(f"Error processing hand image: {str(e)}")
            return None
