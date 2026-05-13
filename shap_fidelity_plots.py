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

from full_metrics_pipeline import (
    EvalConfig,
    Explainer,
    _fidelity_curve,
    collect_shap_background,
)


# ============================================================
# EDIT ONLY THESE VALUES
# ============================================================

DELETION_BASELINE = "mean"   # "black", "blur", or "mean"

BLUR_KERNEL_SIZE = 61
BLUR_SIGMA = 20.0

# --- Paste your numbers from load_correct_samples here ---
cat_nums = []   # replace with your actual list
dog_nums = [2073]  # replace with your actual list

MODEL_PATH = "models/model_kernel=[5,9,11]_32_3_512_1_wd0.001_do0.0.pth"

DATA_DIR = "./DATA/Cat_dog_splitted/"

OUTPUT_DIR = "./fidelity_plots/SHAP/"

# Cat = 0, Dog = 1
CLASS_NAMES = {
    0: "Cat",
    1: "Dog",
}

DELETION_STEPS = 300
SNAPSHOT_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]

# SHAP settings
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
                "conv_layer":  int(m.group(3)),
                "dense_neuron": int(m.group(4)),
                "dense_layer": int(m.group(5)),
                "weight_decay": float(m.group(6)),
                "dropout":     float(m.group(7)),
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

    try:
        img = read_image(image_path)
    except RuntimeError:
        # Fallback for corrupted/mislabelled files (e.g. AVIF/HEIC with .jpg extension)
        from PIL import Image
        import torchvision.transforms.functional as TVF
        pil_img = Image.open(image_path).convert("RGB")
        img = TVF.to_tensor(pil_img)  # (C, H, W), float32 in [0, 1]
        img = (img * 255).to(torch.uint8)  # match read_image output format

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
    pred_conf  = prob_class_1 if pred_label == 1 else 1.0 - prob_class_1

    return pred_label, pred_conf, prob_class_1


def model_prob_for_label(model: torch.nn.Module, img: torch.Tensor, label: int):
    model.eval()

    with torch.no_grad():
        logit = model(img.unsqueeze(0)).squeeze()
        prob_class_1 = torch.sigmoid(logit)
        score = prob_class_1 if label == 1 else (1.0 - prob_class_1)

    return float(score.item())


def tensor_to_numpy_image(img: torch.Tensor):
    return img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


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
        saliency = explainer.explain(img, target_label)   # (C, H, W)

        shap_heatmap = saliency.sum(axis=0)               # (H, W)
        max_abs = np.max(np.abs(shap_heatmap)) + 1e-8
        shap_heatmap = shap_heatmap / max_abs             # [-1, 1]

    finally:
        explainer.remove_hooks()

    return saliency.astype(np.float32), shap_heatmap.astype(np.float32)


# ============================================================
# DELETION / INSERTION BASELINE + MASKING
# ============================================================

def make_deletion_baseline(image: torch.Tensor) -> torch.Tensor:
    method = DELETION_BASELINE.lower()

    if method == "black":
        return torch.zeros_like(image)

    if method == "blur":
        kernel_size = BLUR_KERNEL_SIZE + (1 - BLUR_KERNEL_SIZE % 2)  # ensure odd
        return TF.gaussian_blur(
            image,
            kernel_size=[kernel_size, kernel_size],
            sigma=[BLUR_SIGMA, BLUR_SIGMA],
        )

    if method == "mean":
        return image.mean(dim=(1, 2), keepdim=True).expand_as(image)

    raise ValueError(f"Unknown DELETION_BASELINE='{DELETION_BASELINE}'.")


def make_masked_image(image, ranked, fraction_deleted, device):
    image = image.to(device)
    C, H, W = image.shape
    n_features = C * H * W

    ranked_t = torch.from_numpy(ranked.copy()).long().to(device).clamp(0, n_features - 1)

    n_top = int(fraction_deleted * n_features)
    mask = torch.ones(n_features, device=device)
    if n_top > 0:
        mask[ranked_t[:n_top]] = 0.0
    mask = mask.view(C, H, W)

    baseline = make_deletion_baseline(image).to(device)
    return image * mask + baseline * (1.0 - mask)


# ============================================================
# FIDELITY CURVES
# ============================================================

def compute_curves(model, image, saliency, label, device):
    """Compute both deletion and insertion AUC using the pipeline's _fidelity_curve."""
    ranked = np.argsort(saliency.flatten())[::-1]

    xs_t = torch.linspace(0.0, 1.0, DELETION_STEPS + 1, device=device)

    del_scores_t = _fidelity_curve(
        model=model, image=image.to(device), ranked=ranked,
        label=label, steps=DELETION_STEPS, device=device, mode="deletion",
    )
    ins_scores_t = _fidelity_curve(
        model=model, image=image.to(device), ranked=ranked,
        label=label, steps=DELETION_STEPS, device=device, mode="insertion",
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
# PLOT
# ============================================================

def save_plot(
    model, image, shap_heatmap, ranked,
    xs, del_scores, ins_scores,
    del_auc, ins_auc,
    pred_label, pred_conf, target_label,
    img_num, class_name, device, hp, out_path,
):
    """
    Save a single figure with:
      Col 0   : SHAP heatmap
      Cols 1-5: deletion snapshots at SNAPSHOT_FRACTIONS
      Col 6   : deletion + insertion curves with dAUC / iAUC
    """
    n_cols = 1 + len(SNAPSHOT_FRACTIONS) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols, 4))

    # --- SHAP heatmap ---
    axes[0].imshow(shap_heatmap, cmap=SHAP_CMAP, vmin=-1, vmax=1)
    axes[0].set_title("SHAP\nexplanation")
    axes[0].axis("off")

    # --- Deletion snapshots ---
    for ax, frac in zip(axes[1:1 + len(SNAPSHOT_FRACTIONS)], SNAPSHOT_FRACTIONS):
        masked = make_masked_image(image, ranked, frac, device)
        prob   = model_prob_for_label(model, masked, target_label)
        ax.imshow(tensor_to_numpy_image(masked))
        ax.set_title(f"Del {frac*100:.0f}%\nP={prob:.3f}", fontsize=8)
        ax.axis("off")

    # --- Curves ---
    ax_c = axes[-1]
    ax_c.plot(xs, del_scores, label=f"Deletion (dAUC={del_auc:.3f})", color="tab:red")
    ax_c.plot(xs, ins_scores, label=f"Insertion (iAUC={ins_auc:.3f})", color="tab:blue")
    ax_c.fill_between(xs, del_scores, alpha=0.10, color="tab:red")
    ax_c.fill_between(xs, ins_scores, alpha=0.10, color="tab:blue")

    # Mark snapshot positions on deletion curve
    snap_scores = [del_scores[int(round(f * DELETION_STEPS))] for f in SNAPSHOT_FRACTIONS]
    ax_c.scatter(SNAPSHOT_FRACTIONS, snap_scores, s=25, color="tab:red", zorder=3)

    ax_c.set_xlabel("Pixel fraction")
    ax_c.set_ylabel("Prediction score")
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=7)
    ax_c.grid(alpha=0.25)
    ax_c.set_title(f"dAUC={del_auc:.3f} | iAUC={ins_auc:.3f}")

    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
    fig.suptitle(
        f"{class_name} #{img_num}  |  Pred: {pred_name} ({pred_conf:.3f})  |  "
        f"Tracked: {CLASS_NAMES.get(target_label, target_label)}\n"
        f"k={hp['kernel_size']}, cf={hp['conv_filter']}, cl={hp['conv_layer']}, "
        f"dn={hp['dense_neuron']}, dl={hp['dense_layer']}, do={hp['dropout']}",
        fontsize=10,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# PER-IMAGE PIPELINE
# ============================================================

def process_image(model, config, hp, img_num, target_label, device):
    class_name = CLASS_NAMES[target_label]
    image_path = f"{DATA_DIR}/test/{class_name}/{img_num}.jpg"

    print(f"\n[{class_name} #{img_num}]  path: {image_path}")

    if not os.path.isfile(image_path):
        print(f"  WARNING: file not found, skipping.")
        return

    image = load_image(image_path, device)
    pred_label, pred_conf, _ = predict(model, image)

    print(f"  Pred: {CLASS_NAMES.get(pred_label)} ({pred_conf:.4f})  |  "
          f"Tracked: {class_name}")

    saliency, shap_heatmap = generate_shap(model, image, config, target_label)

    xs, del_scores, ins_scores, del_auc, ins_auc, ranked = compute_curves(
        model, image, saliency, target_label, device
    )

    print(f"  dAUC={del_auc:.4f}  iAUC={ins_auc:.4f}")

    out_path = os.path.join(
        OUTPUT_DIR,
        class_name.lower(),
        f"SHAP_{class_name.lower()}_{img_num}_{DELETION_BASELINE}.png",
    )

    save_plot(
        model=model, image=image, shap_heatmap=shap_heatmap, ranked=ranked,
        xs=xs, del_scores=del_scores, ins_scores=ins_scores,
        del_auc=del_auc, ins_auc=ins_auc,
        pred_label=pred_label, pred_conf=pred_conf, target_label=target_label,
        img_num=img_num, class_name=class_name, device=device, hp=hp,
        out_path=out_path,
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
    config = build_shap_config(device)

    # Build the full work list: (img_num, target_label)
    work = (
        [(n, 0) for n in cat_nums] +
        [(n, 1) for n in dog_nums]
    )

    print(f"\nTotal images to process: {len(work)} "
          f"({len(cat_nums)} cats, {len(dog_nums)} dogs)")

    for i, (img_num, target_label) in enumerate(work, 1):
        print(f"\n--- [{i}/{len(work)}] ---")
        process_image(model, config, hp, img_num, target_label, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()