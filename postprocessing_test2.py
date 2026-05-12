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
    SafeImageFolder,
)

from ComputerVisionProject.full_metrics_pipeline import (
    EvalConfig,
    Explainer,
    _fidelity_curve,
    collect_shap_background,
)


# ============================================================
# EDIT ONLY THESE VALUES
# ============================================================

DELETION_BASELINE = "black"   # "black", "blur", or "mean"

BLUR_KERNEL_SIZE = 61
BLUR_SIGMA = 20.0

IMG_NUM = 2

MODEL_PATH = "models/cat_dog/model_kernel=[5,9,15]_32_3_512_1_wd0.0001_do0.0.pth"
IMAGE_PATH = f"DATA/Cat_dog_splitted/test/Cat/{IMG_NUM}.jpg"

DATA_DIR = "./DATA/Cat_dog_splitted/"

OUTPUT_DIR = "deletion_debug"
OUTPUT_PATH_DELETION = os.path.join(
    OUTPUT_DIR,
    f"paper_style_deletion_SHAP_{IMG_NUM}_{DELETION_BASELINE}.png",
)
OUTPUT_PATH_OVERLAY = os.path.join(
    OUTPUT_DIR,
    f"shap_overlay_{IMG_NUM}_{DELETION_BASELINE}.png",
)

# Cat = 0, Dog = 1
TARGET_LABEL = 0

CLASS_NAMES = {
    0: "Cat",
    1: "Dog",
}

DELETION_STEPS = 300
SNAPSHOT_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]

# SHAP settings from your pipeline
SHAP_BACKGROUND_SIZE = 50
SHAP_BACKGROUND_BATCH_SIZE = 25
SHAP_EXPLAIN_PROBABILITY = False

SHAP_CMAP = "bwr"
OVERLAY_ALPHA = 0.50


# ============================================================
# PARSE MODEL HYPERPARAMETERS FROM FILE NAME
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


# ============================================================
# MODEL / IMAGE HELPERS
# ============================================================

def load_model(device: torch.device):
    hp = parse_model_filename(MODEL_PATH)

    if hp is None:
        raise ValueError(
            f"Could not parse model hyperparameters from:\n{MODEL_PATH}\n\n"
            "Expected format like:\n"
            "model_kernel=[5,9,15]_32_3_512_1_wd0.0001_do0.0.pth"
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


def predict(model: torch.nn.Module, img: torch.Tensor):
    model.eval()

    with torch.no_grad():
        logit = model(img.unsqueeze(0)).squeeze()
        prob_class_1 = torch.sigmoid(logit).item()

    pred_label = int(prob_class_1 >= 0.5)
    pred_conf = prob_class_1 if pred_label == 1 else 1.0 - prob_class_1

    return pred_label, pred_conf, prob_class_1


def model_prob_for_label(model: torch.nn.Module, img: torch.Tensor, label: int):
    model.eval()

    with torch.no_grad():
        logit = model(img.unsqueeze(0)).squeeze()
        prob_class_1 = torch.sigmoid(logit)

        if label == 1:
            score = prob_class_1
        else:
            score = 1.0 - prob_class_1

    return float(score.item())


def tensor_to_numpy_image(img: torch.Tensor):
    return img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


# ============================================================
# SHAP USING YOUR PIPELINE IMPLEMENTATION
# ============================================================

def build_shap_config(model, device):
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
        fidelity_features_per_step=DELETION_STEPS,
        stability_n_perturbations=5,
        stability_noise_std=0.05,
        separability_n_pairs=20,
        separability_eps=1e-3,
        shap_background=shap_background,
        shap_explain_probability=SHAP_EXPLAIN_PROBABILITY,
    )

    return config


def generate_shap_using_project_code(model, img, config):
    """
    Uses your full_metrics_pipeline.Explainer("shap", ...).

    Returns:
        saliency: normalized SHAP map, shape (C,H,W), used for metric ranking.
        shap_heatmap: 2D visualization heatmap, shape (H,W), signed and normalized to [-1,1].
    """
    explainer = Explainer("shap", model, config)

    try:
        saliency = explainer.explain(img, TARGET_LABEL)  # (C,H,W), your pipeline format

        # For visualization, collapse channels.
        # This follows the idea in your SaliencyVisualiser.save_shap:
        # sum across channels, then symmetric normalization.
        shap_heatmap = saliency.sum(axis=0)  # (H,W)

        max_abs = np.max(np.abs(shap_heatmap)) + 1e-8
        shap_heatmap = shap_heatmap / max_abs

    finally:
        explainer.remove_hooks()

    return saliency.astype(np.float32), shap_heatmap.astype(np.float32)


# ============================================================
# DELETION BASELINE OPTIONS FOR VISUALIZATION
# ============================================================

def make_deletion_baseline(
    image: torch.Tensor,
    method: str = DELETION_BASELINE,
) -> torch.Tensor:
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

    raise ValueError(
        f"Unknown DELETION_BASELINE='{method}'. "
        "Choose from: 'black', 'blur', or 'mean'."
    )


def make_masked_image_like_project_metric(image, ranked, fraction_deleted, device):
    """
    Visualization of deletion snapshots.

    This uses the same flattened C*H*W feature deletion order as your current
    metric pipeline, but it displays the selected baseline: black, blur, or mean.

    For the curve to match visually, full_metrics_pipeline._fidelity_curve must
    use the same DELETION_BASELINE internally.
    """
    image = image.to(device)

    C, H, W = image.shape
    n_features = C * H * W

    ranked_t = torch.from_numpy(ranked.copy()).long().to(device)
    ranked_t = ranked_t.clamp(0, n_features - 1)

    n_top = int(fraction_deleted * n_features)

    mask = torch.ones(n_features, device=device)

    if n_top > 0:
        mask[ranked_t[:n_top]] = 0.0

    mask = mask.view(C, H, W)

    baseline = make_deletion_baseline(image).to(device)

    masked = image * mask + baseline * (1.0 - mask)

    return masked


# ============================================================
# PROJECT METRIC COMPATIBILITY
# ============================================================

def compute_project_deletion_curve(model, image, saliency, label, device):
    """
    Uses your metric implementation:
        full_metrics_pipeline._fidelity_curve
    """
    ranked = np.argsort(saliency.flatten())[::-1]

    deletion_scores_t = _fidelity_curve(
        model=model,
        image=image.to(device),
        ranked=ranked,
        label=label,
        steps=DELETION_STEPS,
        device=device,
        mode="deletion",
    )

    xs_t = torch.linspace(0.0, 1.0, DELETION_STEPS + 1, device=device)
    deletion_auc = float(torch.trapezoid(deletion_scores_t, xs_t).item())

    return (
        xs_t.detach().cpu().numpy(),
        deletion_scores_t.detach().cpu().numpy(),
        deletion_auc,
        ranked,
    )


# ============================================================
# PLOTS
# ============================================================

def save_shap_overlay_plot(img, shap_heatmap, pred_label, pred_conf, tracked_label, hp):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_np = tensor_to_numpy_image(img)

    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
    tracked_name = CLASS_NAMES.get(tracked_label, str(tracked_label))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title(
        f"Original image\nPred: {pred_name} ({pred_conf:.3f})\nTracked: {tracked_name}"
    )
    axes[0].axis("off")

    im = axes[1].imshow(shap_heatmap, cmap=SHAP_CMAP, vmin=-1, vmax=1)
    axes[1].set_title("SHAP heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_np)
    axes[2].imshow(
        shap_heatmap,
        cmap=SHAP_CMAP,
        alpha=OVERLAY_ALPHA,
        vmin=-1,
        vmax=1,
    )
    axes[2].set_title("SHAP overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"k={hp['kernel_size']}, cf={hp['conv_filter']}, cl={hp['conv_layer']}, "
        f"dn={hp['dense_neuron']}, dl={hp['dense_layer']}, do={hp['dropout']}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH_OVERLAY, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SHAP overlay plot to: {OUTPUT_PATH_OVERLAY}")


def save_paper_style_deletion_plot(
    model,
    image,
    shap_heatmap,
    ranked,
    xs,
    scores,
    deletion_auc,
    device,
    hp,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_cols = 1 + len(SNAPSHOT_FRACTIONS) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols, 4))

    # Explanation panel
    axes[0].imshow(shap_heatmap, cmap=SHAP_CMAP, vmin=-1, vmax=1)
    axes[0].set_title("Explanation\nSHAP")
    axes[0].axis("off")

    # Deletion steps
    for ax, frac in zip(axes[1:1 + len(SNAPSHOT_FRACTIONS)], SNAPSHOT_FRACTIONS):
        masked_img = make_masked_image_like_project_metric(
            image=image,
            ranked=ranked,
            fraction_deleted=frac,
            device=device,
        )

        prob = model_prob_for_label(model, masked_img, TARGET_LABEL)

        ax.imshow(tensor_to_numpy_image(masked_img))
        ax.set_title(
            f"Deleted {frac * 100:.0f}%\nP={prob:.4f}",
            fontsize=8,
        )
        ax.axis("off")

    # Deletion curve
    curve_ax = axes[-1]
    curve_ax.plot(xs, scores)
    curve_ax.fill_between(xs, scores, alpha=0.25)

    snapshot_scores = [
        scores[int(round(frac * DELETION_STEPS))]
        for frac in SNAPSHOT_FRACTIONS
    ]

    curve_ax.scatter(
        SNAPSHOT_FRACTIONS,
        snapshot_scores,
        s=25,
        zorder=3,
    )

    curve_ax.set_title(f"dAUC: {deletion_auc:.3f}")
    curve_ax.set_xlabel("Pixels deleted")
    curve_ax.set_ylabel("Prediction score")
    curve_ax.set_ylim(0, 1.05)
    curve_ax.grid(alpha=0.25)

    tracked_name = CLASS_NAMES.get(TARGET_LABEL, str(TARGET_LABEL))
    fig.suptitle(
        f"Project SHAP + project deletion metric | Tracked class: {tracked_name}\n"
        f"k={hp['kernel_size']}, cf={hp['conv_filter']}, cl={hp['conv_layer']}, "
        f"dn={hp['dense_neuron']}, dl={hp['dense_layer']}, do={hp['dropout']}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH_DELETION, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SHAP deletion plot to: {OUTPUT_PATH_DELETION}")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading model...")
    model, hp = load_model(device)

    print("Loading image...")
    image = load_image(IMAGE_PATH, device)

    pred_label, pred_conf, prob_class_1 = predict(model, image)

    print("Prediction info:")
    print(f"  Image path: {IMAGE_PATH}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Predicted label: {pred_label} ({CLASS_NAMES.get(pred_label, pred_label)})")
    print(f"  Predicted confidence: {pred_conf:.4f}")
    print(f"  P(class 1): {prob_class_1:.4f}")
    print(f"  Tracked label: {TARGET_LABEL} ({CLASS_NAMES.get(TARGET_LABEL, TARGET_LABEL)})")
    print(f"  Initial tracked-class probability: {model_prob_for_label(model, image, TARGET_LABEL):.4f}")

    print("Building SHAP config and background using your pipeline...")
    config = build_shap_config(model, device)

    print("Generating SHAP using full_metrics_pipeline.Explainer...")
    saliency, shap_heatmap = generate_shap_using_project_code(
        model=model,
        img=image,
        config=config,
    )

    print("Saving SHAP overlay plot...")
    save_shap_overlay_plot(
        img=image,
        shap_heatmap=shap_heatmap,
        pred_label=pred_label,
        pred_conf=pred_conf,
        tracked_label=TARGET_LABEL,
        hp=hp,
    )

    print("Computing deletion curve using full_metrics_pipeline._fidelity_curve...")
    xs, scores, deletion_auc, ranked = compute_project_deletion_curve(
        model=model,
        image=image,
        saliency=saliency,
        label=TARGET_LABEL,
        device=device,
    )

    print(f"Deletion AUC: {deletion_auc:.4f}")

    print("Saving paper-style SHAP deletion plot...")
    save_paper_style_deletion_plot(
        model=model,
        image=image,
        shap_heatmap=shap_heatmap,
        ranked=ranked,
        xs=xs,
        scores=scores,
        deletion_auc=deletion_auc,
        device=device,
        hp=hp,
    )

    print("Done.")


if __name__ == "__main__":
    main()