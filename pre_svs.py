import os
import shutil

source_root = '/home/zyc/data/'
target_dir = '/home/zyc/data/DATA_DIRECTORY'
os.makedirs(target_dir, exist_ok=True)

for root, dirs, files in os.walk(source_root):
    # 跳过目标文件夹，避免死循环复制
    if os.path.abspath(root) == os.path.abspath(target_dir):
        continue

    for file in files:
        if file.lower().endswith('.svs') and not file.lower().endswith('.svs.parcel'):
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)

            if os.path.exists(target_file):
                base, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(os.path.join(target_dir, f"{base}_{i}{ext}")):
                    i += 1
                target_file = os.path.join(target_dir, f"{base}_{i}{ext}")

            shutil.copy2(source_file, target_file)
            print(f"Copied: {source_file} -> {target_file}")
