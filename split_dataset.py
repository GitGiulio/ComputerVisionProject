import os
import random
import shutil
from pathlib import Path

SRC_DIR = "/mnt/scratch/Stable_diffusion/Cat_dog/PetImages"
DST_DIR = "/mnt/scratch/Cat_dog/PetImages_split"

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

CLASSES = ["Cat", "Dog"]

random.seed(42)

for cls in CLASSES:
    src_class_dir = Path(SRC_DIR) / cls
    images = [p for p in src_class_dir.iterdir() if p.is_file()]

    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in split_map.items():
        dst_class_dir = Path(DST_DIR) / split / cls
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(f, dst_class_dir / f.name)

print("Done.")