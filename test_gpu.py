from setting.setting import *
import os
input_train_dir = "dataset/raw/Train_Alphabet"
for label in LABELS:
    dir_path = os.path.join(input_train_dir, label)
    if os.path.exists(dir_path):
        num_files = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{label}: {num_files} files")