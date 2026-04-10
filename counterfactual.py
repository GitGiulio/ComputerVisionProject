
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision.utils import save_image
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from shared_code import SafeImageFolder, collate_skip_none, data_loaders, CIFAKE_CNN, I_HAVE_A_THEORY, parse_hparams_from_model_path, evaluate

DEVICE = f"cuda:1" if torch.cuda.is_available() else "cpu"

DATA_DIR = "/mnt/scratch/Stable_diffusion/Stable_diffusion_ready"

MODEL_PATH = "/home/cv04f26/ComputerVisionProject/models/model_32_2_64_1.pth"

OUT_DIR = "/home/cv04f26/ComputerVisionProject/interpretability/counterfactual_outputs"

BATCH_SIZE = 64
IMAGE_SIZE = 256

USE_FILENAME_HPARAMS = True
MANUAL_CONV_FILTER = 32
MANUAL_CONV_LAYER = 3
MANUAL_DENSE_NEURON = 64
MANUAL_DENSE_LAYER = 1

def generate_counterfactual(model, x, target_class, steps=200, lr=0.01, lam=0.01): 

    x_cf = x.clone().detach().requires_grad_(True)  # clone image

    optimizer = torch.optim.Adam([x_cf], lr=lr) # Test image as parameter to optimize (pixel values)

    target = torch.tensor([[target_class]], dtype=torch.float32, device=x.device)   # target label tensor [[0]] or [[1]]
    
    for _ in range(steps): 

        optimizer.zero_grad()   # Clear previous grad

        logits = model(x_cf)    # Find classifier output from modified image

        loss_pred = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)  # Measure how far prediction is from desired class

        loss_l2 = torch.norm(x_cf - x)  # finds pixel change between orignal image and new image (image change measure) - prevents completly changing image
       
        loss = loss_pred + lam * loss_l2   # Combined loss: match target class while staying close to original image

        loss.backward() 
        
        optimizer.step() 
        
        x_cf.data.clamp_(0, 1)   

    return x_cf.detach() 



if __name__ == "__main__":
    print("DEVICE:", DEVICE)

    train_loader, val_loader, idx_to_class = data_loaders(DEVICE,BATCH_SIZE)
    print(idx_to_class) # {0: 'ai', 1: 'nature'}

    if USE_FILENAME_HPARAMS:
        conv_filter, conv_layer, dense_neuron, dense_layer = parse_hparams_from_model_path(MODEL_PATH)
    else:
        conv_filter = MANUAL_CONV_FILTER
        conv_layer = MANUAL_CONV_LAYER
        dense_neuron = MANUAL_DENSE_NEURON
        dense_layer = MANUAL_DENSE_LAYER 
    
    
    model = CIFAKE_CNN( # CIFAKE_CNN | I_HAVE_A_THEORY
        conv_filters=conv_filter,
        conv_layers=conv_layer,
        dense_neurons=dense_neuron,
        dense_layers=dense_layer
    ).to(DEVICE) 
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    # evaluate(model,val_loader,DEVICE)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # select dataset image
    img_num = 15
    dataset = val_loader.dataset
    
    for img_i in range(img_num):
        img_i*=100
        x, y, _ = dataset[img_i]
        x = x.unsqueeze(0).to(DEVICE)

        # will generate a counterfactual toward 0 ("ai") instead of 1 ("nature")
        # and predictions will move toward 0 rather than 1. just a test
        if y == 1:
            continue
        
        target_class = 1 - y

        # generate counterfactual
        x_cf = generate_counterfactual(model, x, target_class, steps=1000)

        with torch.no_grad():
            print("original:", torch.sigmoid(model(x)))
            print("counterfactual:", torch.sigmoid(model(x_cf)))
            
        # build heatmap
        diff = (x_cf - x).abs()
        diff = diff.mean(dim=1, keepdim=True)
        diff = diff / diff.max()

        heatmap_np = diff.squeeze().cpu().numpy()
        heatmap_color = cm.hot(heatmap_np)[..., :3]
        heatmap_color = (
            torch.tensor(heatmap_color)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        # concatenate horizontally
        combined = torch.cat([x, x_cf, heatmap_color], dim=3)
        
        # Find label prediction from model. For original and counterfactual
        orig_prob = torch.sigmoid(model(x)).item()
        cf_prob = torch.sigmoid(model(x_cf)).item()

        # Turn into binary
        pred_label = int(orig_prob >= 0.5)

        gt_label_name = idx_to_class[y]                 # Ground truth label
        pred_label_name = idx_to_class[pred_label]      # Model prediction label
        target_label_name = idx_to_class[target_class]  # Label counterfactual strives for

        correct = (pred_label == y)     # Just a check to see if the original model is even correct with ground truth (not counterfactaul)

        combined_np = combined.squeeze().permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(figsize=(12,4))
        ax.imshow(combined_np)
        ax.axis("off")

        W = combined_np.shape[1] // 3
        H = combined_np.shape[0]

        ax.text(W*0.5, H + 15,
                f"GT: {gt_label_name} | pred: {pred_label_name} ({orig_prob:.3f}) | correct: {correct}",
                ha="center")

        ax.text(W*1.5, H + 15,
                f"counterfactual → {target_label_name} ({cf_prob:.3f})",
                ha="center")

        ax.text(W*2.5, H + 15,
                "difference heatmap",
                ha="center")
        
        # create real scale colorbar
        norm = plt.Normalize(vmin=0, vmax=diff.max().item())
        sm = plt.cm.ScalarMappable(cmap="hot", norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("pixel change magnitude")
        

        plt.savefig(
            f"{OUT_DIR}/comparison_hot_{img_i}.png",
            bbox_inches="tight",
            pad_inches=0.3
        )

        plt.close()

        print(f"Image {img_i} saved")
        
        # save single combined image
        # save_image(combined, f"{OUT_DIR}/comparison_hot_{img_i}.png")
       
    
    #  Saves 3 seperate images
    # # Counterfactual
    # img_num = 100
    # dataset = val_loader.dataset
    # x, y, _ = dataset[img_num]
    # x = x.unsqueeze(0).to(DEVICE)
    
    # # x_batch, y_batch, _ = next(iter(val_loader))
    # # x = x_batch[img_num].unsqueeze(0).to(DEVICE)
    # # y = y_batch[img_num].item()

    # target_class = 1 - y

    # x_cf = generate_counterfactual(model, x, target_class)
        
    # with torch.no_grad():
    #     print("original:", torch.sigmoid(model(x)))
    #     print("counterfactual:", torch.sigmoid(model(x_cf)))
    
    # diff = (x_cf - x).abs()
    # diff = diff.mean(dim=1, keepdim=True)  # collapse RGB
    # diff = diff / diff.max()

    # heatmap = diff.squeeze().cpu().numpy()

    # plt.imshow(heatmap, cmap="hot")
    # plt.axis("off")
    # plt.savefig(f"{OUT_DIR}/difference_heatmap_hot{img_num}.png", bbox_inches="tight", pad_inches=0)
    # plt.close()

   
    
    # save_image(x, f"{OUT_DIR}/original{img_num}.png")
    # save_image(x_cf, f"{OUT_DIR}/counterfactual{img_num}.png")
    # # save_image((x_cf - x).abs(), f"{OUT_DIR}/difference{img_num}.png")