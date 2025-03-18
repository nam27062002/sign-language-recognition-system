import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from setting.setting import LABELS, PROCESSING_SIZE
import time

class SignLanguageRecognizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.class_names = LABELS
        self.processing_size = PROCESSING_SIZE

    def _preprocess(self, image):
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
        scale_factor = min(self.processing_size / hand_width, self.processing_size / hand_height) * 0.8
        scaled_w, scaled_h = int(hand_width * scale_factor), int(hand_height * scale_factor)
        resized_hand_area = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        centered_image = np.zeros((self.processing_size, self.processing_size, 3), dtype=np.uint8)
        x_offset = (self.processing_size - scaled_w) // 2
        y_offset = (self.processing_size - scaled_h) // 2
        centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand_area
        grayscale_frame = cv2.cvtColor(centered_image, cv2.COLOR_BGR2GRAY)
        expanded_frame = np.stack((grayscale_frame,) * 3, axis=-1)
        return expanded_frame

    def preprocess_image(self, path):
        image = cv2.imread(path)
        if image is None:
            return None
        return self._preprocess(image)

    def predict(self, input_data):
        if isinstance(input_data, str):
            preprocessed = self.preprocess_image(input_data)
        elif isinstance(input_data, bytes):
            image = cv2.imdecode(np.frombuffer(input_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return None
            preprocessed = self._preprocess(image)
        else:
            raise ValueError("Đầu vào phải là đường dẫn tệp (str) hoặc mảng byte hình ảnh (bytes)")

        if preprocessed is None:
            return None

        prediction_frame = np.expand_dims(np.float32(preprocessed) / 255.0, axis=0)
        prediction = self.model.predict(prediction_frame)
        predicted_index = np.argmax(prediction)
        return self.class_names[predicted_index], 1

#
# if __name__ == "__main__":
#     recognizer = SignLanguageRecognizer('../models/CNN/cnn_best_model.keras')
#     label = "B"
#     i = 10
#     result_from_path = recognizer.predict(f'../dataset/raw/Train_Alphabet/{label}/{label}_{i}.png')
#     print("Dự đoán từ đường dẫn:", result_from_path)
#
#
#     with open(f'../dataset/raw/Train_Alphabet/{label}/{label}_{i}.png', 'rb') as file:
#         image_bytes = file.read()
#     result_from_bytes = recognizer.predict(image_bytes)
#     print("Dự đoán từ byte:", result_from_bytes)