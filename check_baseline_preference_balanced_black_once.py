import os
import re
import csv
from typing import Optional

import numpy as np
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF

from shared_code import (
    get_tensor_transform,
    I_HAVE_A_THEORY,
)


# ============================================================
# EDIT ONLY THESE VALUES
# ============================================================

DATA_DIR = "DATA/Cat_dog_splitted/test"

MODEL_PATH = "models/model_kernel=[5,9,19]_32_3_512_3_wd0.0001_do0.0.pth"

OUTPUT_DIR = "./baseline_preference_testset_balanced"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "baseline_preference_per_image.csv")
OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "baseline_preference_summary.csv")

BLUR_KERNEL_SIZE = 61
BLUR_SIGMA = 20.0

CLASS_FOLDERS = {
    0: "Cat",
    1: "Dog",
}

CLASS_NAMES = {
    0: "Cat",
    1: "Dog",
}

# Black is handled separately and evaluated only once.
IMAGE_DEPENDENT_BASELINES = ["blur", "mean"]


# ============================================================
# HELPERS
# ============================================================

def parse_model_filename(model_path: str) -> Optional[dict]:
    fname = os.path.basename(model_path)

    patterns = [
        r"model_kernel=\[5,9,(\d+)\]_(\d+)_(\d+)_(\d+)_(\d+)_wd([0-9eE+\-\.]+)_do([0-9eE+\-\.]+)(?:\.pth)?$",
        r"finetuned_fidelity_k=\[5,9,(\d+)\]_(\d+)_(\d+)_(\d+)_(\d+)_wd([0-9eE+\-\.]+)_do([0-9eE+\-\.]+)(?:\.pth)?$",
    ]

    for pattern in patterns:
        m = re.search(pattern, fname)
        if m is not None:
            return {
                "kernel_size": int(m.group(1)),
                "conv_filter": int(m.group(2)),
                "conv_layer": int(m.group(3)),
                "dense_neuron": int(m.group(4)),
                "dense_layer": int(m.group(5)),
                "weight_decay": float(m.group(6)),
                "dropout": float(m.group(7)),
            }

    return None


def load_model(device: torch.device):
    hp = parse_model_filename(MODEL_PATH)

    if hp is None:
        raise ValueError(
            f"Could not parse model hyperparameters from:\n{MODEL_PATH}\n\n"
            "Expected format like:\n"
            "model_kernel=[5,9,11]_64_3_64_1_wd0.0_do0.0.pth"
        )

    print("Parsed model hyperparameters:")
    for key, value in hp.items():
        print(f"  {key}: {value}")

    model = I_HAVE_A_THEORY(
        kernel_size=hp["kernel_size"],
        conv_filters=hp["conv_filter"],
        conv_layers=hp["conv_layer"],
        dense_neurons=hp["dense_neuron"],
        dense_layers=hp["dense_layer"],
        dropout_rate=hp["dropout"],
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, hp


def numeric_sort_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"\d+", stem)

    if match is not None:
        return (0, int(match.group(0)), stem)

    return (1, stem)


def collect_balanced_image_paths():
    """
    Collects the exact same number of Cat and Dog images.

    The number used per class is the smaller class count. Example:
      Cat = 1867, Dog = 2000 -> use 1867 Cat and 1867 Dog.

    Returns:
        List of (image_path, label) tuples.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    paths_by_label = {}

    for label, folder in CLASS_FOLDERS.items():
        class_dir = os.path.join(DATA_DIR, folder)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        class_paths = []
        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                class_paths.append(os.path.join(class_dir, fname))

        class_paths = sorted(class_paths, key=numeric_sort_key)
        paths_by_label[label] = class_paths

        print(f"Found {len(class_paths)} image(s) for {folder}.")

    if len(paths_by_label) == 0:
        raise RuntimeError(f"No class folders/images found under {DATA_DIR}")

    min_count = min(len(paths) for paths in paths_by_label.values())

    if min_count == 0:
        raise RuntimeError(
            "At least one class has zero images. Cannot build balanced sample."
        )

    print(f"\nBalanced sampling: using {min_count} image(s) per class.")

    items = []
    for label in sorted(CLASS_FOLDERS.keys()):
        selected_paths = paths_by_label[label][:min_count]
        for path in selected_paths:
            items.append((path, label))

        print(
            f"Using {len(selected_paths)} image(s) for "
            f"{CLASS_NAMES.get(label, str(label))}."
        )

    print(f"Total balanced image count: {len(items)}")

    return items


def load_image(image_path: str, device: torch.device):
    transform = get_tensor_transform()

    img = read_image(image_path)

    if img.ndim != 3:
        raise ValueError(f"Invalid image shape: {img.shape}")

    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    elif img.shape[0] > 3:
        img = img[:3, ...]

    img = transform(img)
    return img.float().to(device)


def make_baseline(image: torch.Tensor, method: str):
    method = method.lower()

    if method == "black":
        return torch.zeros_like(image)

    if method == "blur":
        kernel_size = BLUR_KERNEL_SIZE
        if kernel_size % 2 == 0:
            kernel_size += 1

        return TF.gaussian_blur(
            image,
            kernel_size=[kernel_size, kernel_size],
            sigma=[BLUR_SIGMA, BLUR_SIGMA],
        )

    if method == "mean":
        mean_rgb = image.mean(dim=(1, 2), keepdim=True)
        return mean_rgb.expand_as(image)

    raise ValueError(f"Unknown baseline: {method}")


def get_prediction_values(model: torch.nn.Module, image: torch.Tensor):
    with torch.no_grad():
        logit = model(image.unsqueeze(0)).squeeze()
        p_dog = torch.sigmoid(logit).item()
        p_cat = 1.0 - p_dog
        pred = int(p_dog >= 0.5)

    return {
        "logit": float(logit.item()),
        "p_cat": float(p_cat),
        "p_dog": float(p_dog),
        "pred_label": int(pred),
    }


def summarize_rows(rows, method: str, true_label: Optional[int] = None):
    subset = [r for r in rows if r["baseline"] == method]

    if true_label is not None:
        subset = [r for r in subset if int(r["true_label"]) == true_label]

    if len(subset) == 0:
        return None

    logits = np.array([float(r["logit"]) for r in subset], dtype=np.float32)
    p_cat = np.array([float(r["p_cat"]) for r in subset], dtype=np.float32)
    p_dog = np.array([float(r["p_dog"]) for r in subset], dtype=np.float32)
    preds = np.array([int(r["pred_label"]) for r in subset], dtype=np.int64)
    labels = np.array([int(r["true_label"]) for r in subset], dtype=np.int64)

    return {
        "baseline": method,
        "class_subset": "all" if true_label is None else CLASS_NAMES.get(true_label, str(true_label)),
        "n_images": len(subset),
        "mean_logit": float(logits.mean()),
        "std_logit": float(logits.std()),
        "mean_p_cat": float(p_cat.mean()),
        "std_p_cat": float(p_cat.std()),
        "mean_p_dog": float(p_dog.mean()),
        "std_p_dog": float(p_dog.std()),
        "cat_prediction_rate": float((preds == 0).mean()),
        "dog_prediction_rate": float((preds == 1).mean()),
        "accuracy_against_true_label": float((preds == labels).mean()),
    }


def print_summary(summary_rows):
    print("\n" + "=" * 90)
    print("BASELINE PREFERENCE SUMMARY")
    print("=" * 90)

    for row in summary_rows:
        print(
            f"{row['baseline'].upper():<8} | "
            f"subset={row['class_subset']:<4} | "
            f"n={row['n_images']:<5} | "
            f"mean P(Dog)={row['mean_p_dog']:.4f} | "
            f"mean P(Cat)={row['mean_p_cat']:.4f} | "
            f"dog pred rate={row['dog_prediction_rate']:.4f} | "
            f"cat pred rate={row['cat_prediction_rate']:.4f}"
        )

    print("\nInterpretation:")
    print("  Mean P(Dog) > 0.5 means the baseline leans Dog.")
    print("  Mean P(Dog) < 0.5 means the baseline leans Cat.")
    print("  The most neutral baseline is usually the one with mean P(Dog) closest to 0.5.")
    print("  BLACK is evaluated once and then reused for every image, because it is identical for all transformed images.")
    print("  Cat/Dog counts are balanced by using the smaller class count for both classes.")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading model...")
    model, _hp = load_model(device)

    image_items = collect_balanced_image_paths()

    if len(image_items) == 0:
        raise RuntimeError(f"No images found under {DATA_DIR}")

    rows = []

    print(f"\nProcessing {len(image_items)} balanced test image(s)...")

    black_values = None

    for idx, (image_path, true_label) in enumerate(image_items, start=1):
        try:
            image = load_image(image_path, device)

            original = get_prediction_values(model, image)
            rows.append({
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "true_label": true_label,
                "true_class": CLASS_NAMES.get(true_label, str(true_label)),
                "baseline": "original",
                **original,
            })

            # Black baseline is identical for every transformed image, so run once.
            if black_values is None:
                black_img = torch.zeros_like(image)
                black_values = get_prediction_values(model, black_img)
                print(
                    "\nBlack baseline evaluated once:"
                    f" logit={black_values['logit']:.6f},"
                    f" P(Cat)={black_values['p_cat']:.6f},"
                    f" P(Dog)={black_values['p_dog']:.6f}"
                )

            rows.append({
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "true_label": true_label,
                "true_class": CLASS_NAMES.get(true_label, str(true_label)),
                "baseline": "black",
                **black_values,
            })

            # Blur and mean depend on the image, so evaluate them per image.
            for method in IMAGE_DEPENDENT_BASELINES:
                baseline_img = make_baseline(image, method)
                values = get_prediction_values(model, baseline_img)

                rows.append({
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "true_label": true_label,
                    "true_class": CLASS_NAMES.get(true_label, str(true_label)),
                    "baseline": method,
                    **values,
                })

            if idx % 100 == 0 or idx == len(image_items):
                print(f"  [{idx}/{len(image_items)}] processed")

        except Exception as exc:
            print(f"  [SKIP] {image_path}: {exc}")

    fieldnames = [
        "image_path",
        "image_name",
        "true_label",
        "true_class",
        "baseline",
        "logit",
        "p_cat",
        "p_dog",
        "pred_label",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-image results to: {OUTPUT_CSV}")

    summary_rows = []

    for method in ["original", "black"] + IMAGE_DEPENDENT_BASELINES:
        summary_rows.append(summarize_rows(rows, method, true_label=None))

        for true_label in sorted(CLASS_FOLDERS.keys()):
            summary_rows.append(summarize_rows(rows, method, true_label=true_label))

    summary_rows = [r for r in summary_rows if r is not None]

    summary_fieldnames = [
        "baseline",
        "class_subset",
        "n_images",
        "mean_logit",
        "std_logit",
        "mean_p_cat",
        "std_p_cat",
        "mean_p_dog",
        "std_p_dog",
        "cat_prediction_rate",
        "dog_prediction_rate",
        "accuracy_against_true_label",
    ]

    with open(OUTPUT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary results to: {OUTPUT_SUMMARY_CSV}")

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
