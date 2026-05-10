import os
from PIL import Image

root_folder = r"D:\Masters\Deep.Learning\Projects\HW2\Output"

target_size = (420, 350)

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file.lower().endswith(extensions):
            full_path = os.path.join(subdir, file)
            try:
                img = Image.open(full_path)
                img = img.resize(target_size, Image.LANCZOS)
                img.save(full_path)
                print(f"Resized: {full_path}")
            except Exception as e:
                print(f"Error processing {full_path}: {e}")

