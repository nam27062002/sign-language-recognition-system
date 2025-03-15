import cv2
import mediapipe as mp
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from setting.setting import *

mp_hands = mp.solutions.hands
def process_image(image_path, output_path):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.05,
        min_tracking_confidence=0.5,
    ) as hands:
        image = cv2.imread(image_path)
        if image is None:
            return False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return False
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
            return False
        x_min, x_max, y_min, y_max = width, 0, height, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)
        hand_area = hand_on_black_background[y_min:y_max, x_min:x_max]
        hand_width, hand_height = x_max - x_min, y_max - y_min
        target_size = (224, 224)
        occupy_percent = 0.8
        scale_factor = min(target_size[0] / hand_width, target_size[1] / hand_height) * occupy_percent
        scaled_w, scaled_h = int(hand_width * scale_factor), int(hand_height * scale_factor)
        resized_hand_area = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        centered_image = np.zeros(target_size + (3,), dtype=np.uint8)
        x_offset = (target_size[0] - scaled_w) // 2
        y_offset = (target_size[1] - scaled_h) // 2
        centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand_area
        grayscale_image = cv2.cvtColor(centered_image, cv2.COLOR_RGB2GRAY)
        if cv2.imwrite(output_path, grayscale_image):
            return True
        else:
            return False


def process_directory(input_dir, output_dir):
    for label in LABELS:
        input_label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        if os.path.exists(output_label_dir):
            for filename in os.listdir(output_label_dir):
                file_path = os.path.join(output_label_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  #
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Không thể xóa {file_path}. Lý do: {e}')
        else:
            os.makedirs(output_label_dir, exist_ok=True)

        if not os.path.exists(input_label_dir):
            continue
        files = [f for f in os.listdir(input_label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        index = 1
        file_paths = []

        for file in files:
            extension = os.path.splitext(file)[1]
            new_filename = f"{label}_{index}{extension}"
            output_file_path = os.path.join(output_label_dir, new_filename)
            file_paths.append((os.path.join(input_label_dir, file), output_file_path))
            index += 1

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            futures = [
                executor.submit(process_image, inp, outp)
                for inp, outp in file_paths
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {label}"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Lỗi xử lý ảnh: {e}")

process_directory("../dataset/raw/Train_Alphabet", "../dataset/processed/v1/training")
process_directory("../dataset/raw/Test_Alphabet", "../dataset/processed/v1/validation")