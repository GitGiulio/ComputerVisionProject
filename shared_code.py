"""
@author: Giulio Lo Cigno

This file contains implementation of classes and functions that are used across multiple of the other files
"""

import os
import torch
import re
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from torch.amp import autocast
#from tqdm import tqdm
DATA_DIR = "./DATA/Cat_dog_splitted/"

IMAGE_SIZE = 224

def get_tensor_transform():
    _use_v2 = False
    tensor_transform = None

    try:
        import torchvision.transforms.v2 as T
        from torchvision.transforms import InterpolationMode
        _use_v2 = True
        tensor_transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.ToDtype(torch.float32, scale=True),  # uint8 -> float32 in [0,1]
        ])
    except Exception:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        def tensor_transform(img: torch.Tensor) -> torch.Tensor:
            img = TF.resize(
                img,
                [IMAGE_SIZE, IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            )
            img = img.float() / 255.0
            return img
    return tensor_transform


class SafeImageFolder(datasets.ImageFolder):
    """
    Same as your training script:
    - uses torchvision.io.read_image
    - forces 3 channels
    - skips bad/corrupted images
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            devnull = open(os.devnull, 'w')
            old_stderr = os.dup(2)
            os.dup2(devnull.fileno(), 2)
            img = read_image(path)  # uint8 tensor [C,H,W]
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            devnull.close()
            if img.ndim != 3:
                return None

            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif img.shape[0] > 3:
                img = img[:3, ...]

            if self.transform is not None:
                img = self.transform(img)
            else:
                img = img.float() / 255.0

            return img, target, path

        except (RuntimeError, OSError, ValueError):
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, labels, paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(labels, dtype=torch.long),
        list(paths)
    )


def data_loaders(DEVICE, BATCH_SIZE):
    tensor_transform = get_tensor_transform()
    train_data = SafeImageFolder(os.path.join(DATA_DIR, "train"), transform=tensor_transform)
    val_data   = SafeImageFolder(os.path.join(DATA_DIR, "val"),   transform=tensor_transform)
    test_data  = SafeImageFolder(os.path.join(DATA_DIR, "test"),  transform=tensor_transform)

    pin_mem = DEVICE.startswith("cuda")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=pin_mem,
        collate_fn=collate_skip_none
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=8,
        pin_memory=pin_mem,
        collate_fn=collate_skip_none
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=8,
        pin_memory=pin_mem,
        collate_fn=collate_skip_none
    )

    idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}
    return train_loader, val_loader, test_loader, idx_to_class


class CIFAKE_CNN(nn.Module):
    def __init__(self, conv_filters, conv_layers, dense_neurons, dense_layers):
        super().__init__()

        conv_blocks = []
        in_channels = 3

        for _ in range(conv_layers):
            conv_blocks.append(nn.Conv2d(in_channels, conv_filters, kernel_size=3, stride=1, padding=1))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = conv_filters

        self.conv = nn.Sequential(*conv_blocks)

        reduction = 2 ** conv_layers
        flattened_size = (IMAGE_SIZE // reduction) * (IMAGE_SIZE // reduction) * conv_filters

        dense_blocks = []
        in_features = flattened_size

        for _ in range(dense_layers):
            dense_blocks.append(nn.Linear(in_features, dense_neurons))
            dense_blocks.append(nn.ReLU())
            in_features = dense_neurons

        dense_blocks.append(nn.Linear(in_features, 1))  # binary logit
        self.fc_logits = nn.Sequential(*dense_blocks)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        logit = self.fc_logits(x)
        return logit  # (B,1) logits

    def forward_logits(self, x):
        return self.forward(x)


class I_HAVE_A_THEORY(nn.Module):
    """Multi-kernel CNN with optional dropout regularization.

    The first three conv layers use kernel sizes 5, 9, and `kernel_size`
    respectively. Dropout is applied before each dense (fully-connected) layer.

    Args:
        kernel_size: Kernel size for conv layers beyond the first two.
        conv_filters: Number of output channels for every conv layer.
        conv_layers: Total number of conv+pool blocks.
        dense_neurons: Width of each hidden dense layer.
        dense_layers: Number of hidden dense layers.
        dropout_rate: Dropout probability applied before each dense layer.
                      0.0 disables dropout entirely.
    """

    def __init__(
        self,
        kernel_size: int,
        conv_filters: int,
        conv_layers: int,
        dense_neurons: int,
        dense_layers: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        conv_blocks = []
        in_channels = 3

        for i in range(conv_layers):
            if i == 0:
                k = 5  # NOTE: this implementaiton fixes the kernel size of the first two layers, and the one passed to the constructor is used only from the 3rd layer onward.
            elif i == 1:
                k = 9
            else:
                k = kernel_size
            conv_blocks.append(
                nn.Conv2d(in_channels, conv_filters, kernel_size=k,
                          stride=1, padding=(k - 1) // 2)
            )
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = conv_filters

        self.conv = nn.Sequential(*conv_blocks)

        reduction = 2 ** conv_layers
        flattened_size = (IMAGE_SIZE // reduction) * (IMAGE_SIZE // reduction) * conv_filters

        dense_blocks = []
        in_features = flattened_size

        for _ in range(dense_layers):
            if dropout_rate > 0.0:
                dense_blocks.append(nn.Dropout(p=dropout_rate))
            dense_blocks.append(nn.Linear(in_features, dense_neurons))
            dense_blocks.append(nn.ReLU())
            in_features = dense_neurons

        dense_blocks.append(nn.Linear(in_features, 1))
        self.fc_logits = nn.Sequential(*dense_blocks)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        logit = self.fc_logits(x)
        return logit  # (B, 1) logits

    def forward_logits(self, x):
        return self.forward(x)


def parse_hparams_from_model_path(model_path):
    """
    Expects filename like:
    model_32_2_64_1.pth
    => conv_filter=32, conv_layer=2, dense_neuron=64, dense_layer=1
    """
    fname = os.path.basename(model_path)
    match = re.search(r"model_(\d+)_(\d+)_(\d+)_(\d+)\.pth", fname)
    if match is None:
        raise ValueError(
            f"Could not parse hyperparameters from filename: {fname}\n"
            f"Expected format like: model_32_2_64_1.pth"
        )

    conv_filter = int(match.group(1))
    conv_layer = int(match.group(2))
    dense_neuron = int(match.group(3))
    dense_layer = int(match.group(4))
    return conv_filter, conv_layer, dense_neuron, dense_layer


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def full_backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(full_backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    @torch.no_grad()
    def _normalize_cam(self, cam):
        B, H, W = cam.shape
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam.view(B, H, W)

    def generate(self, input_tensor: torch.Tensor, use_logits: bool = True):
        self.model.zero_grad()
        if use_logits and hasattr(self.model, "forward_logits"):
            output = self.model.forward_logits(input_tensor)
            target = output.squeeze(1)
        else:
            output = self.model(input_tensor)
            target = output.squeeze(1)
        target.sum().backward()
        grads  = self.gradients
        activs = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activs).sum(dim=1)
        cam = torch.relu(cam)
        cam = self._normalize_cam(cam)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False).squeeze(1)
        return cam  # (B, IMAGE_SIZE, IMAGE_SIZE) in [0, 1]


def evaluate_2(model, test_loader, DEVICE):
    """
    An outdated versuib if the evaluate() function, still correct and used in Grad_cam_visual.py, but with a different signature
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        with autocast(device_type=DEVICE, dtype=torch.float16):
            for batch in test_loader:
                if batch is None:
                    continue
                images, labels, _path = batch
                images = images.to(DEVICE, non_blocking=True)
                logits = model(images)
                probs  = torch.sigmoid(logits)

                predictions = (probs.float().cpu().numpy() > 0.5).astype(int)
                preds.extend(predictions.flatten())
                trues.extend(labels.numpy().flatten())

    if len(preds) == 0:
        print("\n--- TEST RESULTS ---")
        print("No valid samples to evaluate.")
        return 0, 0, 0, 0

    accuracy  = np.mean(np.array(preds) == np.array(trues))
    precision = precision_score(trues, preds, zero_division=0)
    recall    = recall_score(trues, preds, zero_division=0)
    f1        = f1_score(trues, preds, zero_division=0)

    print("\n--- TEST RESULTS ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return accuracy, precision, recall, f1