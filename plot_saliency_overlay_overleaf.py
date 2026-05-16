import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision.io import read_image
import torchvision.transforms.functional as TF

from shared_code import (
    get_tensor_transform,
    I_HAVE_A_THEORY,
    GradCAM,
    SafeImageFolder,
)

from full_metrics_pipeline import (
    EvalConfig,
    Explainer,
    collect_shap_background,
)


# ============================================================
# EDIT ONLY THESE VALUES
# ============================================================

cat_nums = [10704, 5389, 7375]
dog_nums = [4475, 8370, 11759]


MODEL_PATH = "models/model_kernel=[5,9,19]_32_3_512_1_wd0.001_do0.0.pth"

DATA_DIR = "./DATA/Cat_dog_splitted/"
OUTPUT_DIR = "./saliency_comparison_plots_overleaf/"

CLASS_NAMES = {
    0: "Cat",
    1: "Dog",
}

GRADCAM_CMAP = "jet"
SHAP_CMAP = "bwr"

SHAP_ALPHA = 5
SHAP_THRESHOLD = 0.001

SHAP_BACKGROUND_SIZE = 50
SHAP_BACKGROUND_BATCH_SIZE = 25
SHAP_EXPLAIN_PROBABILITY = False


# ============================================================
# PARSE MODEL HYPERPARAMETERS
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
                "kernel_size":  int(m.group(1)),
                "conv_filter":  int(m.group(2)),
                "conv_layer":   int(m.group(3)),
                "dense_neuron": int(m.group(4)),
                "dense_layer":  int(m.group(5)),
                "weight_decay": float(m.group(6)),
                "dropout":      float(m.group(7)),
            }

    return None


# ============================================================
# MODEL
# ============================================================

def load_model(device):

    hp = parse_model_filename(MODEL_PATH)

    if hp is None:
        raise ValueError(
            f"Could not parse hyperparameters from:\n{MODEL_PATH}"
        )

    model = I_HAVE_A_THEORY(
        kernel_size=hp["kernel_size"],
        conv_filters=hp["conv_filter"],
        conv_layers=hp["conv_layer"],
        dense_neurons=hp["dense_neuron"],
        dense_layers=hp["dense_layer"],
        dropout_rate=hp["dropout"],
    ).to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model, hp


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(image_path: str, device):

    transform = get_tensor_transform()

    try:
        img = read_image(image_path)

    except RuntimeError:

        from PIL import Image
        import torchvision.transforms.functional as TVF

        pil_img = Image.open(image_path).convert("RGB")

        img = TVF.to_tensor(pil_img)

        img = (img * 255).to(torch.uint8)

    if img.ndim != 3:
        raise ValueError(f"Invalid image shape: {img.shape}")

    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)

    elif img.shape[0] > 3:
        img = img[:3, ...]

    img = transform(img)

    return img.float().to(device)


def tensor_to_numpy_image(img: torch.Tensor):

    return (
        img.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(model, img):

    model.eval()

    with torch.no_grad():

        logit = model(img.unsqueeze(0)).squeeze()

        prob_class_1 = torch.sigmoid(logit).item()

    pred_label = int(prob_class_1 >= 0.5)

    pred_conf = (
        prob_class_1
        if pred_label == 1
        else 1.0 - prob_class_1
    )

    return pred_label, pred_conf, prob_class_1


# ============================================================
# GRADCAM
# ============================================================

def find_last_conv_layer(model: nn.Module):

    last_conv = None

    for module in model.modules():

        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise RuntimeError("No Conv2d layer found.")

    return last_conv


def normalise_heatmap(cam_map: np.ndarray):

    hm = np.array(cam_map, dtype=np.float32)

    hm = np.nan_to_num(hm)

    hm = np.clip(hm, 0, None)

    max_val = hm.max()

    return hm / max_val if max_val > 0 else np.zeros_like(hm)


def generate_gradcam(model, img, device):

    target_layer = find_last_conv_layer(model)

    cam = GradCAM(model, target_layer)

    try:

        input_batch = img.unsqueeze(0).detach().clone().to(device)

        input_batch.requires_grad_(True)

        with torch.enable_grad():

            cam_map = (
                cam.generate(
                    input_batch,
                    use_logits=True,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

    finally:

        cam.remove_hooks()

    return normalise_heatmap(cam_map)


# ============================================================
# SHAP
# ============================================================

def build_shap_config(device):

    train_dataset = SafeImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=get_tensor_transform(),
    )

    shap_background = collect_shap_background(
        train_dataset,
        n_background=SHAP_BACKGROUND_SIZE,
        batch_size=SHAP_BACKGROUND_BATCH_SIZE,
    )

    config = EvalConfig(
        target_layer=None,
        n_samples=1,
        batch_size=1,
        device=str(device),
        output_dir=OUTPUT_DIR,
        fidelity_features_per_step=100,
        stability_n_perturbations=5,
        stability_noise_std=0.05,
        separability_n_pairs=20,
        separability_eps=1e-3,
        shap_background=shap_background,
        shap_explain_probability=SHAP_EXPLAIN_PROBABILITY,
    )

    return config


def generate_shap(model, img, config, target_label):

    explainer = Explainer(
        "shap",
        model,
        config,
    )

    try:

        saliency = explainer.explain(
            img,
            target_label,
        )

        shap_heatmap = saliency.sum(axis=0)

        max_abs = np.max(np.abs(shap_heatmap)) + 1e-8

        shap_heatmap = shap_heatmap / max_abs

    finally:

        explainer.remove_hooks()

    return shap_heatmap.astype(np.float32)


# ============================================================
# OVERLAY
# ============================================================

def overlay_heatmap_on_image(
    image: torch.Tensor,
    heatmap: np.ndarray,
    cmap: str,
    alpha: float = 0.45,
):

    img_np = tensor_to_numpy_image(image)

    cmap_fn = plt.get_cmap(cmap)

    heatmap_rgba = cmap_fn(heatmap)

    heatmap_rgb = heatmap_rgba[..., :3]

    overlay = (
        (1 - alpha) * img_np
        + alpha * heatmap_rgb
    )

    overlay = np.clip(overlay, 0, 1)

    return overlay


def overlay_shap_on_image(
    image: torch.Tensor,
    shap_heatmap: np.ndarray,
    alpha_scale: float = 0.8,
    threshold: float = 0.1,
):
    img_np = tensor_to_numpy_image(image)
    cmap_fn = plt.get_cmap("bwr")
    heatmap_norm = (shap_heatmap + 1.0) / 2.0
    heatmap_rgba = cmap_fn(heatmap_norm)
    abs_val = np.abs(shap_heatmap)
    alpha_map = np.clip((abs_val - threshold) / (1.0 - threshold), 0, 1)
    alpha_map = alpha_map * alpha_scale
    shap_rgb = heatmap_rgba[..., :3]
    alpha_map_3 = alpha_map[..., np.newaxis]
    overlay = img_np * (1.0 - alpha_map_3) + shap_rgb * alpha_map_3
    return np.clip(overlay, 0, 1)

# ============================================================
# PLOT
# ============================================================

def save_plot(
    images,
    gradcam_heatmaps,
    shap_heatmaps,
    img_nums,
    class_names,
    pred_labels,
    pred_confs,
    prob_dogs,
    out_path,
):

    n_cols = len(images)

    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(5 * n_cols, 14),
    )

    # Handle single-image case
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col in range(n_cols):

        image = images[col]

        gradcam_heatmap = gradcam_heatmaps[col]

        shap_heatmap = shap_heatmaps[col]

        img_num = img_nums[col]

        class_name = class_names[col]

        pred_label = pred_labels[col]

        pred_conf = pred_confs[col]

        prob_dog = prob_dogs[col]

        pred_name = CLASS_NAMES.get(pred_label)

        # ====================================================
        # ROW 1 — ORIGINAL IMAGE
        # ====================================================

        axes[0, col].imshow(
            tensor_to_numpy_image(image)
        )

        axes[0, col].set_title(
            f"{class_name} #{img_num}\n"
            f"Pred: {pred_name} ({pred_conf:.3f})\n",
            fontsize=20,
            fontweight="bold",
        )


        # ====================================================
        # ROW 2 — SHAP
        # ====================================================

        shap_overlay = overlay_shap_on_image(
            image=image,
            shap_heatmap=shap_heatmap,
            alpha_scale=SHAP_ALPHA,
            threshold=SHAP_THRESHOLD,
        )

        axes[1, col].imshow(shap_overlay)

        # ====================================================
        # ROW 3 — GRADCAM
        # ====================================================

        gradcam_overlay = overlay_heatmap_on_image(
            image=image,
            heatmap=gradcam_heatmap,
            cmap=GRADCAM_CMAP,
            alpha=0.45,
        )

        axes[2, col].imshow(gradcam_overlay)

    # Row labels
    
    row_labels = ["Original", "SHAP", "GradCAM"]
    for row, label in enumerate(row_labels):
        fig.text(
            0.01,                          # x position (left margin)
            1 - (row + 0.5) / 3,          # y position, centered in each row
            label,
            va="center",
            ha="left",
            fontsize=30,
            fontweight="bold",
            rotation=90,
        )

    plt.tight_layout()
    fig.subplots_adjust(left=0.05)   # make room for the row labels

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True,
    )

    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    model,
    shap_config,
    img_num,
    target_label,
    device,
):

    class_name = CLASS_NAMES[target_label]

    image_path = os.path.join(
        DATA_DIR,
        "test",
        class_name,
        f"{img_num}.jpg",
    )

    print(f"\n[{class_name} #{img_num}]")

    if not os.path.isfile(image_path):

        print("  File not found.")

        return

    image = load_image(
        image_path,
        device,
    )

    pred_label, pred_conf, prob_dog = predict(
        model,
        image,
    )

    print("  Generating GradCAM...")

    gradcam_heatmap = generate_gradcam(
        model,
        image,
        device,
    )

    print("  Generating SHAP...")

    shap_heatmap = generate_shap(
        model,
        image,
        shap_config,
        target_label,
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        class_name.lower(),
        f"{class_name.lower()}_{img_num}.png",
    )

    save_plot(
        image=image,
        gradcam_heatmap=gradcam_heatmap,
        shap_heatmap=shap_heatmap,
        img_num=img_num,
        class_name=class_name,
        pred_label=pred_label,
        pred_conf=pred_conf,
        prob_dog=prob_dog,
        out_path=out_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    print("\nLoading model...")

    model, hp = load_model(device)

    print("\nBuilding SHAP background...")

    shap_config = build_shap_config(device)

    work = (
        [(n, 0) for n in cat_nums]
        + [(n, 1) for n in dog_nums]
    )

    print(f"\nTotal images: {len(work)}")

    images = []
    gradcam_heatmaps = []
    shap_heatmaps = []
    img_nums = []
    class_names = []
    pred_labels = []
    pred_confs = []
    prob_dogs = []

    for i, (img_num, target_label) in enumerate(work, 1):

        class_name = CLASS_NAMES[target_label]

        image_path = os.path.join(
            DATA_DIR,
            "test",
            class_name,
            f"{img_num}.jpg",
        )

        print(f"\n--- [{i}/{len(work)}] ---")

        if not os.path.isfile(image_path):

            print("File not found.")

            continue

        image = load_image(
            image_path,
            device,
        )

        pred_label, pred_conf, prob_dog = predict(
            model,
            image,
        )

        print("Generating GradCAM...")

        gradcam_heatmap = generate_gradcam(
            model,
            image,
            device,
        )

        print("Generating SHAP...")

        shap_heatmap = generate_shap(
            model,
            image,
            shap_config,
            target_label,
        )

        images.append(image)
        gradcam_heatmaps.append(gradcam_heatmap)
        shap_heatmaps.append(shap_heatmap)

        img_nums.append(img_num)
        class_names.append(class_name)

        pred_labels.append(pred_label)
        pred_confs.append(pred_conf)
        prob_dogs.append(prob_dog)

    out_path = os.path.join(
        OUTPUT_DIR,
        "combined_saliency_plot.png",
    )

    save_plot(
        images=images,
        gradcam_heatmaps=gradcam_heatmaps,
        shap_heatmaps=shap_heatmaps,
        img_nums=img_nums,
        class_names=class_names,
        pred_labels=pred_labels,
        pred_confs=pred_confs,
        prob_dogs=prob_dogs,
        out_path=out_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
