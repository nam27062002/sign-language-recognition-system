import os
from tqdm import tqdm
from setting.setting import *
test_folder = '../../dataset/raw/Test_Alphabet'
train_folder = '../../dataset/raw/Train_Alphabet'
labels = LABELS

def rename_files_in_folder(folder_path, label):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()
    for index, filename in tqdm(enumerate(files), total=len(files), desc=f"Đổi tên trong {label}"):
        file_ext = os.path.splitext(filename)[1]
        new_name = f"{label}_{index}{file_ext}"
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)


for label in tqdm(labels, desc="Xử lý thư mục Test"):
    folder_path = os.path.join(test_folder, label)
    if os.path.exists(folder_path):
        rename_files_in_folder(folder_path, label)
    else:
        print(f"Thư mục {folder_path} không tồn tại!")

for label in tqdm(labels, desc="Xử lý thư mục Train"):
    folder_path = os.path.join(train_folder, label)
    if os.path.exists(folder_path):
        rename_files_in_folder(folder_path, label)
    else:
        print(f"Thư mục {folder_path} không tồn tại!")