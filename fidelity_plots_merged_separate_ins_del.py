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
    _fidelity_curve,
    collect_shap_background,
)


# ============================================================
# EDIT ONLY THESE VALUES
# ============================================================

cat_nums = [10291]
# dog_nums = [2253, 2005, 7350, 400, 408, 8233, 10906, 9724, 3653, 343, 10657, 7621, 8487, 1535, 3134, 12165, 3316, 6195, 7809, 8334, 9895, 9126, 8546, 1334, 5154, 12189, 9737, 4208, 1694, 9722, 9956, 10379]
# dog_nums = [1535]
dog_nums = []

MODEL_PATH = "models/model_kernel=[5,9,19]_32_3_512_1_wd0.001_do0.0.pth"

DATA_DIR = "./DATA/Cat_dog_splitted/"
OUTPUT_DIR = "./gradcam_dogs_only_plots/merged_individual_overleaf/"

CLASS_NAMES = {0: "Cat", 1: "Dog"}

DELETION_STEPS = 300
SNAPSHOT_FRACTIONS = [0.0, 0.25, 0.75, 1.0]

BLUR_KERNEL_SIZE = 61
BLUR_SIGMA = 20.0

SHAP_BACKGROUND_SIZE = 50
SHAP_BACKGROUND_BATCH_SIZE = 25
SHAP_EXPLAIN_PROBABILITY = False

GRADCAM_CMAP = "jet"
SHAP_CMAP = "bwr"

# Each entry produces TWO rows: one deletion row, one insertion row
ROW_ORDER = [
    # ("shap", "black"),
    ("shap", "blur"),
    # ("shap", "mean"),
]

# For printing
METHOD = "SHAP"


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
# MODEL / IMAGE HELPERS
# ============================================================

def load_model(device: torch.device):
    hp = parse_model_filename(MODEL_PATH)
    if hp is None:
        raise ValueError(f"Could not parse model hyperparameters from:\n{MODEL_PATH}")

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


def predict(model, img):
    model.eval()
    with torch.no_grad():
        logit = model(img.unsqueeze(0)).squeeze()
        prob_class_1 = torch.sigmoid(logit).item()
    pred_label = int(prob_class_1 >= 0.5)
    pred_conf = prob_class_1 if pred_label == 1 else 1.0 - prob_class_1
    return pred_label, pred_conf, prob_class_1


def model_prob_for_label(model, img, label):
    model.eval()
    with torch.no_grad():
        logit = model(img.unsqueeze(0)).squeeze()
        prob_class_1 = torch.sigmoid(logit)
        score = prob_class_1 if label == 1 else (1.0 - prob_class_1)
    return float(score.item())


def tensor_to_numpy_image(img: torch.Tensor):
    return img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_heatmap_on_image(
    image: torch.Tensor,
    heatmap: np.ndarray,
    cmap: str = "Reds",
    alpha: float = 0.45,
):
    """
    Returns an RGB image with GradCAM heatmap overlayed.
    """

    img_np = tensor_to_numpy_image(image)

    cmap_fn = plt.get_cmap(cmap)

    # Convert heatmap -> RGBA
    heatmap_rgba = cmap_fn(heatmap)

    # Remove alpha channel
    heatmap_rgb = heatmap_rgba[..., :3]

    # Blend
    overlay = (
        (1 - alpha) * img_np
        + alpha * heatmap_rgb
    )

    overlay = np.clip(overlay, 0, 1)

    return overlay

def overlay_shap_on_image(
    image: torch.Tensor,
    shap_heatmap: np.ndarray,   # values in [-1, 1]
    alpha_scale: float = 0.8,   # max opacity of the SHAP colour
    threshold: float = 0.1,     # values below this (abs) are invisible
):
    img_np = tensor_to_numpy_image(image)
    cmap_fn = plt.get_cmap("bwr")

    # Map [-1,1] → [0,1] for bwr  (−1=blue, 0=white, +1=red)
    heatmap_norm = (shap_heatmap + 1.0) / 2.0
    heatmap_rgba = cmap_fn(heatmap_norm)          # (H, W, 4)

    # Build per-pixel alpha: zero near zero, full at ±1
    abs_val = np.abs(shap_heatmap)
    alpha_map = np.clip((abs_val - threshold) / (1.0 - threshold), 0, 1)
    alpha_map = alpha_map * alpha_scale            # (H, W)

    # Composite: result = img * (1 - alpha) + shap_rgb * alpha
    shap_rgb = heatmap_rgba[..., :3]
    alpha_map_3 = alpha_map[..., np.newaxis]
    overlay = img_np * (1.0 - alpha_map_3) + shap_rgb * alpha_map_3
    return np.clip(overlay, 0, 1)

# ============================================================
# DELETION / INSERTION BASELINE + MASKING
# ============================================================

def make_deletion_baseline(image: torch.Tensor, method: str) -> torch.Tensor:
    method = method.lower()
    if method == "black":
        return torch.zeros_like(image)
    if method == "blur":
        kernel_size = BLUR_KERNEL_SIZE + (1 - BLUR_KERNEL_SIZE % 2)
        return TF.gaussian_blur(
            image,
            kernel_size=[kernel_size, kernel_size],
            sigma=[BLUR_SIGMA, BLUR_SIGMA],
        )
    if method == "mean":
        return image.mean(dim=(1, 2), keepdim=True).expand_as(image)
    raise ValueError(f"Unknown baseline method '{method}'.")


def make_masked_image(image, ranked, fraction_deleted, device, baseline_method):
    """Deletion mask: top-fraction pixels replaced by baseline."""
    image = image.to(device)
    C, H, W = image.shape
    n_features = C * H * W

    ranked_t = torch.from_numpy(ranked.copy()).long().to(device).clamp(0, n_features - 1)
    n_top = int(fraction_deleted * n_features)
    mask = torch.ones(n_features, device=device)
    if n_top > 0:
        mask[ranked_t[:n_top]] = 0.0
    mask = mask.view(C, H, W)

    baseline = make_deletion_baseline(image, baseline_method).to(device)
    return image * mask + baseline * (1.0 - mask)


def make_insertion_image(image, ranked, fraction_inserted, device, baseline_method):
    """
    Insertion mask: start from the baseline and reveal the top-fraction
    most salient pixels from the original image.

    fraction_inserted=0.0  → pure baseline (nothing revealed)
    fraction_inserted=1.0  → original image fully revealed
    """
    image = image.to(device)
    C, H, W = image.shape
    n_features = C * H * W

    ranked_t = torch.from_numpy(ranked.copy()).long().to(device).clamp(0, n_features - 1)
    n_top = int(fraction_inserted * n_features)

    # mask=1 where we show the *original*, mask=0 where we show the *baseline*
    mask = torch.zeros(n_features, device=device)
    if n_top > 0:
        mask[ranked_t[:n_top]] = 1.0
    mask = mask.view(C, H, W)

    baseline = make_deletion_baseline(image, baseline_method).to(device)
    return image * mask + baseline * (1.0 - mask)


# ============================================================
# GRADCAM
# ============================================================

def find_last_conv_layer(model: nn.Module):
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for GradCAM.")
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
            cam_map = cam.generate(input_batch, use_logits=True)[0].detach().cpu().numpy()
    finally:
        cam.remove_hooks()
    return normalise_heatmap(cam_map)


def build_pipeline_saliency_from_gradcam(heatmap: np.ndarray, image: torch.Tensor):
    C = image.shape[0]
    return np.stack([heatmap] * C, axis=0).astype(np.float32)


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
        fidelity_features_per_step=DELETION_STEPS,
        stability_n_perturbations=5,
        stability_noise_std=0.05,
        separability_n_pairs=20,
        separability_eps=1e-3,
        shap_background=shap_background,
        shap_explain_probability=SHAP_EXPLAIN_PROBABILITY,
    )
    return config


def generate_shap(model, img, config, target_label):
    explainer = Explainer("shap", model, config)
    try:
        saliency = explainer.explain(img, target_label)
        shap_heatmap = saliency.sum(axis=0)
        max_abs = np.max(np.abs(shap_heatmap)) + 1e-8
        shap_heatmap = shap_heatmap / max_abs
    finally:
        explainer.remove_hooks()
    return saliency.astype(np.float32), shap_heatmap.astype(np.float32)


# ============================================================
# FIDELITY CURVES
# ============================================================

def compute_curves(model, image, saliency, label, device, baseline_method):
    ranked = np.argsort(saliency.flatten())[::-1]
    xs_t = torch.linspace(0.0, 1.0, DELETION_STEPS + 1, device=device)

    del_scores_t = _fidelity_curve(
        model=model, image=image.to(device), ranked=ranked,
        label=label, steps=DELETION_STEPS, device=device, mode="deletion",
        deletion_baseline=baseline_method,
    )
    ins_scores_t = _fidelity_curve(
        model=model, image=image.to(device), ranked=ranked,
        label=label, steps=DELETION_STEPS, device=device, mode="insertion",
        deletion_baseline=baseline_method,
    )

    del_auc = float(torch.trapezoid(del_scores_t, xs_t).item())
    ins_auc = float(torch.trapezoid(ins_scores_t, xs_t).item())

    return (
        xs_t.detach().cpu().numpy(),
        del_scores_t.detach().cpu().numpy(),
        ins_scores_t.detach().cpu().numpy(),
        del_auc,
        ins_auc,
        ranked,
    )


# ============================================================
# PLOT — two draw_row variants
# ============================================================

def draw_deletion_row(axes_row, model, image, heatmap, ranked,
                      xs, del_scores, del_auc,
                      target_label, explainer_name, baseline_method, device):
    """
    Deletion row layout:
      col 0          : saliency heatmap
      cols 1..N      : deletion snapshots at SNAPSHOT_FRACTIONS
      col N+1        : deletion curve only
    """
    cmap = GRADCAM_CMAP if explainer_name == "gradcam" else SHAP_CMAP
    vmin, vmax = (0, 1) if explainer_name == "gradcam" else (-1, 1)
    method_label = "GradCAM" if explainer_name == "gradcam" else "SHAP"

    # Col 0: heatmap
    # axes_row[0].imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    # axes_row[0].set_title(f"{method_label}\n{baseline_method}\n[DELETION]", fontsize=8)
    # axes_row[0].axis("off")
    
    if explainer_name == "shap":
        overlay = overlay_shap_on_image(image, heatmap, alpha_scale=5, threshold=0.001)
    else:
        overlay = overlay_heatmap_on_image(
            image=image,
            heatmap=heatmap,
            cmap=cmap,
            alpha=0.45,
        )

    axes_row[0].imshow(overlay)

    axes_row[0].set_title(
        f"To mask: {baseline_method}",
        fontsize=16
    )

    axes_row[0].axis("off")


    # Cols 1..N: deletion snapshots
    for ax, frac in zip(axes_row[1:1 + len(SNAPSHOT_FRACTIONS)], SNAPSHOT_FRACTIONS):
        masked = make_masked_image(image, ranked, frac, device, baseline_method)
        prob = model_prob_for_label(model, masked, target_label)
        ax.imshow(tensor_to_numpy_image(masked))
        ax.set_title(f"Deleted {frac*100:.0f}%\nP={prob:.3f}", fontsize=18)
        ax.axis("off")

    # Last col: deletion curve
    ax_c = axes_row[-1]
    ax_c.plot(xs, del_scores, color="tab:red", label=f"Del (dAUC={del_auc:.3f})")
    ax_c.fill_between(xs, del_scores, alpha=0.12, color="tab:red")
    snap_scores = [del_scores[int(round(f * DELETION_STEPS))] for f in SNAPSHOT_FRACTIONS]
    ax_c.scatter(SNAPSHOT_FRACTIONS, snap_scores, s=20, color="tab:red", zorder=3)
    ax_c.set_xlabel("Pixel fraction deleted", fontsize=15)
    ax_c.set_ylabel("Pred. score", fontsize=15)
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=12)
    ax_c.tick_params(labelsize=6)
    ax_c.grid(alpha=0.25)
    ax_c.set_title(f"dAUC={del_auc:.3f}", fontsize=15)


def draw_insertion_row(axes_row, model, image, heatmap, ranked,
                       xs, ins_scores, ins_auc,
                       target_label, explainer_name, baseline_method, device):
    """
    Insertion row layout:
      col 0          : saliency heatmap (same as deletion row — for reference)
      cols 1..N      : insertion snapshots at SNAPSHOT_FRACTIONS
      col N+1        : insertion curve only
    """
    cmap = GRADCAM_CMAP if explainer_name == "gradcam" else SHAP_CMAP
    vmin, vmax = (0, 1) if explainer_name == "gradcam" else (-1, 1)
    method_label = "GradCAM" if explainer_name == "gradcam" else "SHAP"

    # Col 0: heatmap (repeated for easy visual comparison with its deletion partner)
    # axes_row[0].imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    # axes_row[0].set_title(f"{method_label}\n{baseline_method}\n[INSERTION]", fontsize=8)
    # axes_row[0].axis("off")

    if explainer_name == "shap":
        overlay = overlay_shap_on_image(image, heatmap, alpha_scale=5, threshold=0.001)
    else:
        overlay = overlay_heatmap_on_image(
            image=image,
            heatmap=heatmap,
            cmap=cmap,
            alpha=0.45,
        )

    axes_row[0].imshow(overlay)

    # axes_row[0].set_title(
    #     f"Mask: {baseline_method}\n[INSERTION]",
    #     fontsize=15
    # )

    axes_row[0].axis("off")

    # Cols 1..N: insertion snapshots
    # fraction_inserted=0 → baseline only; fraction_inserted=1 → full image
    for ax, frac in zip(axes_row[1:1 + len(SNAPSHOT_FRACTIONS)], SNAPSHOT_FRACTIONS):
        revealed = make_insertion_image(image, ranked, frac, device, baseline_method)
        prob = model_prob_for_label(model, revealed, target_label)
        ax.imshow(tensor_to_numpy_image(revealed))
        ax.set_title(f"Inserted {frac*100:.0f}%\nP={prob:.3f}", fontsize=18)
        ax.axis("off")

    # Last col: insertion curve
    ax_c = axes_row[-1]
    ax_c.plot(xs, ins_scores, color="tab:blue", label=f"Ins (iAUC={ins_auc:.3f})")
    ax_c.fill_between(xs, ins_scores, alpha=0.12, color="tab:blue")
    snap_scores = [ins_scores[int(round(f * DELETION_STEPS))] for f in SNAPSHOT_FRACTIONS]
    ax_c.scatter(SNAPSHOT_FRACTIONS, snap_scores, s=20, color="tab:blue", zorder=3)
    ax_c.set_xlabel("Pixel fraction inserted", fontsize=15)
    ax_c.set_ylabel("Pred. score", fontsize=15)
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=12)
    ax_c.tick_params(labelsize=6)
    ax_c.grid(alpha=0.25)
    ax_c.set_title(f"iAUC={ins_auc:.3f}", fontsize=15)


# ============================================================
# MERGED PLOT — 2 rows per (explainer, baseline) entry
# ============================================================

def save_merged_plot(
    model, image, device, hp,
    gradcam_heatmap, gradcam_saliency,
    shap_heatmap, shap_saliency,
    pred_label, pred_conf, target_label,
    img_num, class_name, out_path,
):
    # Each entry in ROW_ORDER produces 2 rows (deletion + insertion)
    n_rows = len(ROW_ORDER) * 2
    n_cols = 1 + len(SNAPSHOT_FRACTIONS) + 1   # heatmap + snapshots + curve

    # fig, axes = plt.subplots(
    #     n_rows, n_cols,
    #     figsize=(3.2 * n_cols, 3.0 * n_rows),
    # )
    
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # Pre-build axes array so draw_deletion_row / draw_insertion_row still work
    axes = np.empty((n_rows, n_cols), dtype=object)
    for entry_idx in range(len(ROW_ORDER)):
        del_row = entry_idx * 2
        ins_row = entry_idx * 2 + 1
        # Col 0: shared saliency spanning both rows
        shared_ax = fig.add_subplot(gs[del_row:ins_row + 1, 0])
        axes[del_row, 0] = shared_ax
        axes[ins_row, 0] = shared_ax          # same axis for both rows
        # Remaining cols: individual axes per row
        for col in range(1, n_cols):
            axes[del_row, col] = fig.add_subplot(gs[del_row, col])
            axes[ins_row, col] = fig.add_subplot(gs[ins_row, col])

    # Pre-compute curves (one pass per unique (explainer, baseline))
    curve_cache = {}
    for explainer_name, baseline_method in ROW_ORDER:
        key = (explainer_name, baseline_method)
        if key in curve_cache:
            continue
        saliency = gradcam_saliency if explainer_name == "gradcam" else shap_saliency
        print(f"  Computing curves: {explainer_name} / {baseline_method} ...")
        xs, del_scores, ins_scores, del_auc, ins_auc, ranked = compute_curves(
            model, image, saliency, target_label, device, baseline_method,
        )
        curve_cache[key] = (xs, del_scores, ins_scores, del_auc, ins_auc, ranked)

    # Draw rows: deletion row then insertion row for each entry
    for entry_idx, (explainer_name, baseline_method) in enumerate(ROW_ORDER):
        key = (explainer_name, baseline_method)
        xs, del_scores, ins_scores, del_auc, ins_auc, ranked = curve_cache[key]
        heatmap = gradcam_heatmap if explainer_name == "gradcam" else shap_heatmap

        del_row_idx = entry_idx * 2        # e.g. 0, 2, 4
        ins_row_idx = entry_idx * 2 + 1    # e.g. 1, 3, 5

        draw_deletion_row(
            axes_row=axes[del_row_idx],
            model=model, image=image, heatmap=heatmap, ranked=ranked,
            xs=xs, del_scores=del_scores, del_auc=del_auc,
            target_label=target_label,
            explainer_name=explainer_name, baseline_method=baseline_method,
            device=device,
        )
        draw_insertion_row(
            axes_row=axes[ins_row_idx],
            model=model, image=image, heatmap=heatmap, ranked=ranked,
            xs=xs, ins_scores=ins_scores, ins_auc=ins_auc,
            target_label=target_label,
            explainer_name=explainer_name, baseline_method=baseline_method,
            device=device,
        )

    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
    # fig.suptitle(
    #     f"{class_name} #{img_num}  |  Pred: {pred_name} ({pred_conf:.3f})  |  "
    #     f"True Label: {CLASS_NAMES.get(target_label, target_label)}\n",
    #     # f"k={hp['kernel_size']}, cf={hp['conv_filter']}, cl={hp['conv_layer']}, "
    #     # f"dn={hp['dense_neuron']}, dl={hp['dense_layer']}, do={hp['dropout']}",
    #     fontsize=30,
    # )
    
    fig.suptitle(
        f"{METHOD} Fidelity | True Label: {CLASS_NAMES.get(target_label, target_label)} #{img_num}\n",
        # f"k={hp['kernel_size']}, cf={hp['conv_filter']}, cl={hp['conv_layer']}, "
        # f"dn={hp['dense_neuron']}, dl={hp['dense_layer']}, do={hp['dropout']}",
        fontsize=30,
    )

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05) 
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# PER-IMAGE PIPELINE
# ============================================================

def process_image(model, shap_config, hp, img_num, target_label, device):
    class_name = CLASS_NAMES[target_label]
    image_path = os.path.join(DATA_DIR, "test", class_name, f"{img_num}.jpg")

    print(f"\n[{class_name} #{img_num}]  path: {image_path}")
    if not os.path.isfile(image_path):
        print(f"  WARNING: file not found, skipping.")
        return

    image = load_image(image_path, device)
    pred_label, pred_conf, _ = predict(model, image)
    print(f"  Pred: {CLASS_NAMES.get(pred_label)} ({pred_conf:.4f})  |  Tracked: {class_name}")

    print("  Generating GradCAM...")
    gradcam_heatmap = generate_gradcam(model, image, device)
    gradcam_saliency = build_pipeline_saliency_from_gradcam(gradcam_heatmap, image)

    print("  Generating SHAP...")
    shap_saliency, shap_heatmap = generate_shap(model, image, shap_config, target_label)

    out_path = os.path.join(
        OUTPUT_DIR,
        class_name.lower(),
        f"{METHOD}_{class_name.lower()}_{img_num}_merged.png",
    )

    save_merged_plot(
        model=model, image=image, device=device, hp=hp,
        gradcam_heatmap=gradcam_heatmap, gradcam_saliency=gradcam_saliency,
        shap_heatmap=shap_heatmap, shap_saliency=shap_saliency,
        pred_label=pred_label, pred_conf=pred_conf, target_label=target_label,
        img_num=img_num, class_name=class_name, out_path=out_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("\nLoading model...")
    model, hp = load_model(device)

    print("\nBuilding SHAP background (done once, reused for all images)...")
    shap_config = build_shap_config(device)

    work = [(n, 0) for n in cat_nums] + [(n, 1) for n in dog_nums]
    print(f"\nTotal images to process: {len(work)} ({len(cat_nums)} cats, {len(dog_nums)} dogs)")

    for i, (img_num, target_label) in enumerate(work, 1):
        print(f"\n--- [{i}/{len(work)}] ---")
        process_image(model, shap_config, hp, img_num, target_label, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()