import kagglehub
import shutil
import os
from tqdm import tqdm

dest = './raw'
os.makedirs(dest, exist_ok=True)
path = kagglehub.dataset_download("lexset/synthetic-asl-alphabet")
print(f"Đường dẫn nguồn: {path}")
print("Nội dung nguồn:", os.listdir(path))

def copy_with_progress(src, dst):
    total_files = sum(len(files) for _, _, files in os.walk(src))
    with tqdm(total=total_files, unit="file", desc="Đang sao chép") as pbar:
        for root, dirs, files in os.walk(src):
            rel_path = os.path.relpath(root, src)
            dest_path = os.path.join(dst, rel_path)
            os.makedirs(dest_path, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)
                shutil.copy2(src_file, dest_file)
                pbar.update(1)

copy_with_progress(path, dest)
print("Đã sao chép xong! Nội dung đích:", os.listdir(dest))