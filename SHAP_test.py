import os
import re
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
import shap
from shared_code import SafeImageFolder, collate_skip_none, data_loaders, CIFAKE_CNN, I_HAVE_A_THEORY, parse_hparams_from_model_path, evaluate

DEVICE = f"cuda:1" if torch.cuda.is_available() else "cpu"

DATA_DIR = "/mnt/scratch/Stable_diffusion/Stable_diffusion_ready"

MODEL_PATH = "/home/cv04f26/ComputerVisionProject/models/model_kernel=[5,9,17]_32_3_64_1.pth"

BATCH_SIZE = 64
IMAGE_SIZE = 256

NUM_BACKGROUND = 50
NUM_EXPLAIN = 10
OUT_DIR = "/home/cv04f26/ComputerVisionProject/interpretability/shap_outputs/kernel=[5,9,17]"

EXPLAIN_PROBABILITY = True

USE_FILENAME_HPARAMS = False
MANUAL_CONV_FILTER = 32
MANUAL_CONV_LAYER = 3
MANUAL_DENSE_NEURON = 64
MANUAL_DENSE_LAYER = 1


class ShapBinaryWrapper(nn.Module):
    """
    Wraps your model for SHAP.
    """
    def __init__(self, base_model, explain_probability=True):
        super().__init__()
        self.base_model = base_model
        self.explain_probability = explain_probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.base_model(x)

        if self.explain_probability:
            out = self.sigmoid(out)

        return out



def collect_n_samples(loader, n_samples):
    """
    Collects exactly n_samples (or as many as available) from a DataLoader.
    Returns:
      images: Tensor [N,C,H,W]
      labels: Tensor [N]
      paths:  list[str]
    """
    xs, ys, paths_all = [], [], []
    total = 0
    skip = 0 #int(len(loader)//2) + 1

    print(skip)
    for batch in loader:
        if skip:
            skip -= 1
            continue   
        if batch is None:
            continue

        images, labels, paths = batch
        xs.append(images)
        ys.append(labels)
        paths_all.extend(paths)
        total += images.size(0)

        if total >= n_samples:
            break

    if total == 0:
        raise RuntimeError("No valid images could be collected from the loader.")

    X = torch.cat(xs, dim=0)[:n_samples]
    y = torch.cat(ys, dim=0)[:n_samples]
    paths_all = paths_all[:n_samples]
    return X, y, paths_all


def to_numpy_image(x):
    """
    x: torch tensor [C,H,W] in [0,1]
    returns np.uint8 [H,W,C]
    """
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


def save_shap_visualizations(images_tensor, shap_values, probs, labels, paths, idx_to_class, out_dir,explainer):
    """
    Saves one figure per explained image:
    - original image
    - SHAP heatmap
    - overlay
    """
    print(probs)
    probs = np.array(probs).reshape(-1)
    print(probs)
    os.makedirs(out_dir, exist_ok=True)

    images_np = images_tensor.detach().cpu().numpy()
    images_np = np.transpose(images_np, (0, 2, 3, 1))

    shap_values = np.array(shap_values)
    
    if shap_values.ndim == 5 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    if shap_values.ndim == 4 and shap_values.shape[1] in [1, 3]:
        shap_values = np.transpose(shap_values, (0, 2, 3, 1))

    if shap_values.ndim != 4:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    
    # heatmaps = shap_values.mean(axis=-1) # this does the mean across the RGB channes
    heatmaps = shap_values.sum(axis=-1)
    
    for i in range(len(images_np)):
        img = images_np[i]
        hm = heatmaps[i]
        
        shap_total = shap_values[i].sum()
        base_val = np.array(explainer.expected_value).reshape(-1)[0]

        reconstructed_pred = base_val + shap_total

        # Symmetric normalization for display
        max_abs = np.max(np.abs(hm)) + 1e-8
        hm_norm = hm / max_abs  # roughly in [-1,1]

        pred_prob = float(probs[i])
        pred_idx = 1 if pred_prob >= 0.5 else 0
        true_idx = int(labels[i].item())

        pred_name = idx_to_class.get(pred_idx, str(pred_idx))
        true_name = idx_to_class.get(true_idx, str(true_idx))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title(f"Input\nTrue: {true_name}\nPred: {pred_name} ({pred_prob:.3f})")
        axes[0].axis("off")

        im = axes[1].imshow(hm_norm, cmap="bwr", vmin=-1, vmax=1)
        axes[1].set_title(f"SHAP heatmap \n SHAP total val: {shap_total:.3f} \n SHAP base val: {base_val:.3f} \n SHAP pred: {reconstructed_pred:.3f}")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img)
        axes[2].imshow(hm_norm, cmap="bwr", alpha=0.50, vmin=-1, vmax=1)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        basename = os.path.basename(paths[i])
        
        out_path_ai = os.path.join(os.path.join(out_dir,"ai"), f"shap_{i:03d}_{basename}")
        out_path_nature = os.path.join(os.path.join(out_dir,"nature"), f"shap_{i:03d}_{basename}")
        plt.tight_layout()
        if true_name=="ai":
            plt.savefig(out_path_ai, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(out_path_nature, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[SHAP] Saved {len(images_np)} visualizations to: {out_dir}")


if __name__ == "__main__":
    print("DEVICE:", DEVICE)

    train_loader, val_loader, idx_to_class = data_loaders(DEVICE,BATCH_SIZE)
    print("Class mapping:", idx_to_class)

    if USE_FILENAME_HPARAMS:
        conv_filter, conv_layer, dense_neuron, dense_layer = parse_hparams_from_model_path(MODEL_PATH)
    else:
        conv_filter = MANUAL_CONV_FILTER
        conv_layer = MANUAL_CONV_LAYER
        dense_neuron = MANUAL_DENSE_NEURON
        dense_layer = MANUAL_DENSE_LAYER

    print("Loading model with:")
    print(f"  conv_filter = {conv_filter}")
    print(f"  conv_layer  = {conv_layer}")
    print(f"  dense_neuron= {dense_neuron}")
    print(f"  dense_layer = {dense_layer}")

    model = I_HAVE_A_THEORY( # CIFAKE_CNN | I_HAVE_A_THEORY
        kernel_size=17,
        conv_filters=conv_filter,
        conv_layers=conv_layer,
        dense_neurons=dense_neuron,
        dense_layers=dense_layer
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    evaluate(model,val_loader,DEVICE)

    shap_model = ShapBinaryWrapper(
        base_model=model,
        explain_probability=EXPLAIN_PROBABILITY
    ).to(DEVICE)
    shap_model.eval()

    background_images, _, _ = collect_n_samples(train_loader, NUM_BACKGROUND)
    background_images = background_images.to(DEVICE, non_blocking=True)

    print("Background shape:", tuple(background_images.shape))

    explain_images, explain_labels, explain_paths = collect_n_samples(val_loader, NUM_EXPLAIN)
    explain_images = explain_images.to(DEVICE, non_blocking=True)

    print("Explain shape:", tuple(explain_images.shape))

    with torch.no_grad():
        probs = shap_model(explain_images).detach().cpu().numpy()

    explainer = shap.DeepExplainer(shap_model, background_images)
    shap_values = explainer.shap_values(explain_images)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    print("SHAP values shape:", np.array(shap_values).shape)

    explain_images_cpu = explain_images.detach().cpu()
    save_shap_visualizations(
        images_tensor=explain_images_cpu,
        shap_values=shap_values,
        probs=probs,
        labels=explain_labels,
        paths=explain_paths,
        idx_to_class=idx_to_class,
        out_dir=OUT_DIR,
        explainer=explainer
    )

    print("Done.")