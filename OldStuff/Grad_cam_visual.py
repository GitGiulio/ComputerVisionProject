import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import torch.nn.functional as F
from datetime import datetime
from torch.amp import autocast, GradScaler

from shared_code import SafeImageFolder, collate_skip_none, data_loaders, CIFAKE_CNN, parse_hparams_from_model_path, GradCAM, I_HAVE_A_THEORY, evaluate


torch.backends.cudnn.benchmark = True

# Grad-CAM / output config
OUT_DIR = "/home/cv04f26/ComputerVisionProject/interpretability/gradcam_outputs_new_2_kernel=[5,9,17]"
NUM_CAM_SAMPLES = 10
IMAGE_SIZE = 256
DEVICE = f"cuda:1" if torch.cuda.is_available() else "cpu"

DATA_DIR = "./DATA/Cat_dog_splitted/"

MODEL_PATH = "/home/cv04f26/ComputerVisionProject/models/model_kernel=[5,9,17]_32_3_64_1.pth"

BATCH_SIZE = 64
IMAGE_SIZE = 256

EXPLAIN_PROBABILITY = True

USE_FILENAME_HPARAMS = False
MANUAL_CONV_FILTER = 32
MANUAL_CONV_LAYER = 3
MANUAL_DENSE_NEURON = 64
MANUAL_DENSE_LAYER = 1

def to_numpy_img(t: torch.Tensor):
    """
    t: torch tensor [C,H,W] in [0,1]
    returns np array [H,W,C] in [0,1]
    """
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

def save_gradcam_visualization(
    image_tensor: torch.Tensor,
    cam_map: np.ndarray,
    pred_prob: float,
    true_label: int,
    pred_label: int,
    idx_to_class: dict,
    out_path: str,
    cmap: str = "YlOrRd",       # better than bwr for Grad-CAM
    overlay_alpha: float = 0.40,
):
    """
    Save a single Grad-CAM visualization in a 3-panel layout:
      1) input image
      2) heatmap
      3) overlay
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = to_numpy_img(image_tensor)  # [H,W,C] in [0,1]

    hm = np.array(cam_map, dtype=np.float32)
    hm = np.nan_to_num(hm)

    hm = np.clip(hm, 0, None)

    max_val = hm.max()
    if max_val > 0:
        hm = hm / max_val
    else:
        hm = np.zeros_like(hm)

    true_name = idx_to_class.get(int(true_label), str(true_label))
    pred_name = idx_to_class.get(int(pred_label), str(pred_label))

    # Optional "overall intensity" number (NOT additive like SHAP, just descriptive)
    #cam_sum = float(hm.sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title(
        f"Input\nTrue: {true_name}\nPred: {pred_name} ({pred_prob:.3f})"
    )
    axes[0].axis("off")

    im = axes[1].imshow(hm, cmap=cmap, vmin=0, vmax=1)
    #axes[1].set_title(f"Grad-CAM heatmap\nCAM sum: {cam_sum:.2f}")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img)
    axes[2].imshow(hm, cmap=cmap, alpha=overlay_alpha, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def find_last_conv_layer(module: nn.Module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last_conv

def generate_gradcam_samples(model,test_loader,idx_to_class,n_samples):
    model.eval()
    last_conv = find_last_conv_layer(model)
    cam = GradCAM(model, last_conv)

    saved = 0
    base_out = OUT_DIR
    skip = 0 #int(len(test_loader)//2) + 1
    try:
        for batch_idx, batch in enumerate(test_loader):
            if skip > 0:
                skip -= 1
                continue
            if saved >= n_samples:
                break
            if batch is None:
                continue

            images, labels, _path = batch
            images = images.to(DEVICE, non_blocking=True)

            # forward pass for predictions
            with torch.no_grad():
                with autocast(
                    device_type=DEVICE,
                    dtype=torch.float16,
                    enabled=(DEVICE == "cuda")
                ):
                    logits = model(images)
                    probs = torch.sigmoid(logits)

            probs_np = probs.detach().cpu().numpy().reshape(-1)
            preds_np = (probs_np > 0.5).astype(int)

            for i in range(images.size(0)):
                if saved >= n_samples:
                    break

                true_label = int(labels[i].item())
                pred_label = int(preds_np[i])
                pred_prob = float(probs_np[i])

                # Grad-CAM needs gradients, so do NOT use torch.no_grad() here
                img = images[i:i+1].detach().clone().to(DEVICE)
                img.requires_grad_(True)

                with torch.enable_grad():
                    with autocast(
                        device_type=DEVICE,
                        dtype=torch.float16,
                        enabled=(DEVICE == "cuda")
                    ):
                        cam_map = cam.generate(img, use_logits=True)[0].detach().cpu().numpy()

                out_name = (
                    f"gradcam_idx{batch_idx:03d}_{i:02d}"
                    f"_true-{idx_to_class[true_label]}"
                    f"_pred-{idx_to_class[pred_label]}.png"
                )
                out_path_ai = os.path.join(os.path.join(base_out,"ai"), out_name)
                out_path_nature = os.path.join(os.path.join(base_out,"nature"), out_name)

                if true_label:
                    out_path = out_path_nature
                else:
                    out_path = out_path_ai

                save_gradcam_visualization(
                    image_tensor=images[i].detach().cpu(),
                    cam_map=cam_map,
                    pred_prob=pred_prob,
                    true_label=true_label,
                    pred_label=pred_label,
                    idx_to_class=idx_to_class,
                    out_path=out_path,
                    cmap="Reds",          # change to "bwr" if you want exact SHAP-like colors
                    overlay_alpha=0.50,
                )

                saved += 1

    finally:
        cam.remove_hooks()

    print(f"\n[Grad-CAM] Saved {saved} visualizations to: {base_out}")


if __name__ == "__main__":
    print("DEVICE:", DEVICE)

    train_loader, val_loader,test_loaedr, idx_to_class = data_loaders(DEVICE,BATCH_SIZE)
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

    model = I_HAVE_A_THEORY(
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


    generate_gradcam_samples(model,val_loader,idx_to_class,NUM_CAM_SAMPLES)

    print("Done.")