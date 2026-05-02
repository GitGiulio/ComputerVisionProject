import os
import shutil
import kagglehub
# 70, 15, 15 - Train, eval, test

DATA_DIR = "./DATA/Cat_dog"
os.makedirs(DATA_DIR, exist_ok=True)

# Download dataset (cached by kagglehub)
download_path = kagglehub.dataset_download(
    "bhavikjikadara/dog-and-cat-classification-dataset"
)

print("Downloaded to cache:", download_path)

# Copy dataset into target directory
for item in os.listdir(download_path):
    src = os.path.join(download_path, item)
    dst = os.path.join(DATA_DIR, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset copied to:", DATA_DIR)