import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import os
import torchvision
from torchvision import transforms
from shared_code import CIFAKE_CNN,I_HAVE_A_THEORY, data_loaders,get_tensor_transform,SafeImageFolder,collate_skip_none
from train import evaluate
from Grad_cam_visual import find_last_conv_layer
from captum.metrics import infidelity
import shap
import re
from torchvision.transforms import functional as TF

DELETION_BASELINE = "blur"   # "black", "blur", or "mean"

BLUR_KERNEL_SIZE = 61
BLUR_SIGMA = 20.0

class ShapBinaryWrapper(torch.nn.Module):
    """Wraps a model for use with shap.DeepExplainer.

    Optionally applies sigmoid so that SHAP explains probabilities rather
    than raw logit, controlled by the explain_probability flag.

    Args:
        base_model: The underlying PyTorch model.
        explain_probability: If True, sigmoid is applied to the output.
    """

    def __init__(self, base_model: torch.nn.Module, explain_probability: bool = True):
        super().__init__()
        self.base_model = base_model
        self.explain_probability = explain_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional sigmoid.

        Args:
            x: Input tensor.

        Returns:
            Model output, optionally passed through sigmoid.
        """
        out = self.base_model(x)
        if self.explain_probability:
            out = torch.sigmoid(out)
        return out


@dataclass
class EvalConfig:
    """Central configuration for one evaluation run.

    Args:
        interpretability_method: One of 'lime', 'shap', 'gradcam', 'gradcam++',
                    'intgrad', 'smoothgrad'.
        target_layer: The nn.Module layer used by gradient-based methods
                      (GradCAM, GradCAM++). Not needed for LIME/SHAP/IntGrad.
        n_samples: Number of images to evaluate on.
        batch_size: DataLoader batch size.da
        device: 'cuda' or 'cpu'.
        output_dir: Folder where plots and CSV are saved.
        # Metric-specific knobs
        fidelity_features_per_step: Exact number of features added (insertion) or removed (deletion) at each curve step.
            Using a fixed count, rather than a fixed number of steps, guarantees that insertion and deletion are perfect duals with
            identical step sizes. Total steps = ceil(C*H*W / fidelity_features_per_step).
        stability_n_perturbations: Number of noisy copies per image.
        stability_noise_std: Std of Gaussian noise added for stability test.
        separability_n_pairs: Number of cross-class image pairs to check.
        separability_eps: Minimum L2 distance to consider two maps different.
        smoothgrad_n_samples: Noise samples averaged in SmoothGrad.
        smoothgrad_noise_std: Noise std for SmoothGrad.
        lime_n_segments: Number of superpixel segments for LIME.
        lime_n_samples: Number of perturbed samples for LIME surrogate.
    """
    #interpretability_method: str = "gradcam"
    target_layer: Optional[torch.nn.Module] = None
    n_samples: int = 50
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./interpretability_results"
    fidelity_features_per_step: int = 100
    stability_n_perturbations: int = 10
    stability_noise_std: float = 0.05
    separability_n_pairs: int = 30
    separability_eps: float = 1e-3
    shap_background: Optional[torch.Tensor] = None
    shap_explain_probability: bool = True


@dataclass
class MetricResults:
    """Stores aggregated metric scores for one method.

    Args:
        deletion_auc: Lower is better, AUC of prediction score as top pixels
                      are progressively deleted (masked to zero).
        insertion_auc: Higher is better, AUC as masked pixels are revealed.
        stability: Lower is better, Lipschitz estimate over perturbed inputs.
        identity: 1.0 = perfectly deterministic, 0.0 = random.
        separability: Fraction of image pairs with sufficiently different maps.
        avg_time_sec: Mean wall-clock seconds per explanation.
        raw: Per-sample raw values for each metric (for plotting distributions).
    """
    model_tot_param: int = 0
    model_val_acc: float = 0.0
    model_val_f1: float = 0.0
    model_test_acc: float = 0.0
    model_test_f1: float = 0.0
    deletion_auc: float = 0.0
    insertion_auc: float = 0.0
    stability: float = 0.0
    identity: float = 0.0
    separability: float = 0.0
    avg_time_sec: float = 0.0
    raw: dict = field(default_factory=dict)


class Explainer:
    """Wraps multiple Interpretability backends behind a single `.explain()` interface.

    All methods return a saliency map as a numpy array of shape (C, H, W),
    preserving per-channel attribution information. Callers that need a 2-D
    summary (e.g. visualisation) should collapse the channel axis themselves
    (e.g. ``sal.mean(axis=0)``).

    Args:
        method: Interpretability method name.
        model: PyTorch model in eval mode.
        config: EvalConfig instance.
    """

    SUPPORTED = {"shap", "gradcam"}

    def __init__(self, method: str, model: torch.nn.Module, config: EvalConfig):
        self.method = method.lower()
        if self.method not in self.SUPPORTED:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.SUPPORTED}")
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Gradient-based methods register forward/backward hooks on target_layer
        self._hooks = []
        self._gradients = None
        self._activations = None
        if self.method == "gradcam":
            if config.target_layer is None:
                raise ValueError("target_layer must be set for GradCAM")
            self._register_gradcam_hooks(config.target_layer)

    def _register_gradcam_hooks(self, layer: torch.nn.Module):
        """Register forward and backward hooks on the target conv layer."""
        def fwd_hook(_, __, output):
            self._activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(fwd_hook))
        self._hooks.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        """Clean up all registered hooks, call after evaluation is done."""
        for h in self._hooks:
            h.remove()

    def explain(self, image_tensor: torch.Tensor, label: int) -> np.ndarray:
        """Generate a per-channel saliency map for a single image.

        Args:
            image_tensor: Float tensor of shape (C, H, W), already normalised.
            label: Ground-truth (or predicted) class index.

        Returns:
            3-D numpy array of shape (C, H, W) with per-channel attributions.
            GradCAM broadcasts its single-channel map across all C channels.
            SHAP returns per-channel values normalised by their max absolute value.
        """
        dispatch = {
            "shap":      self._explain_shap,
            "gradcam":   self._explain_gradcam,
        }
        return dispatch[self.method](image_tensor, label)


    def _gradcam_raw(self, img: torch.Tensor,label: int) -> np.ndarray:
        """Shared GradCAM forward/backward pass, returns raw upsampled CAM.

        Backpropagates on the raw model output (scalar logit for binary models,
        or the max-score class for multi-class). This matches the original
        implementation which called cam.generate(img, use_logits=True).

        Args:
            img: Image tensor (C, H, W), already on the correct device.
            label: Ground-truth (or predicted) class index.

        Returns:
            Raw float32 CAM of shape (H, W), ReLU-ed but NOT normalised.
        """
        x = img.unsqueeze(0).requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)

        if out.numel() == 1 or out.shape[-1] == 1:
            score = out.squeeze()
        else:
            score = out[0].max()
        
        prob = torch.sigmoid(score)
        score = prob if label == 1 else (1.0 - prob)
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()   # (H', W')
        cam = F.relu(cam)

        # Upsample to input resolution without normalising
        cam_np = cam.detach().cpu().numpy()
        from scipy.ndimage import zoom
        zh = img.shape[1] / cam_np.shape[0]
        zw = img.shape[2] / cam_np.shape[1]
        return zoom(cam_np, (zh, zw), order=1).astype(np.float32)

    def gradcam_raw_map(self, img: torch.Tensor) -> np.ndarray:
        """Return the raw (non-normalised) GradCAM map for visualisation.

        The map is ReLU-ed and upsampled to input resolution but NOT divided
        by its maximum, so the absolute values are preserved.

        Args:
            img: Image tensor (C, H, W).

        Returns:
            Raw float32 array of shape (H, W).
        """
        if self.method not in {"gradcam", "gradcam++"}:
            raise ValueError("gradcam_raw_map is only available for gradcam / gradcam++")
        return self._gradcam_raw(img.to(self.device))

    def _explain_gradcam(self, img: torch.Tensor, label: int) -> np.ndarray:
        """GradCAM: gradient-weighted average of conv feature maps.

        Returns a normalised map broadcast across all input channels so the
        output shape is (C, H, W), matching the other explain methods.
        Use gradcam_raw_map() directly if you need the unnormalised (H, W) map
        for visualisation.

        Args:
            img: Image tensor (C, H, W).
            label: Unused for GradCAM (kept for API consistency); the score
                   is taken from the model's own highest-confidence output,
                   matching the original use_logits=True behaviour.

        Returns:
            Saliency map (C, H, W), values in [0, 1], identical across channels.
        """
        raw = self._gradcam_raw(img.to(self.device),label)          # (H, W)
        normalised_hw = _normalise(raw)                        # (H, W)
        C = img.shape[0]
        return np.stack([normalised_hw] * C, axis=0)           # (C, H, W)
    
    def _explain_shap(self, img: torch.Tensor, label: int) -> np.ndarray:
        """SHAP via shap.DeepExplainer with a background reference distribution.
            Wraps the model in ShapBinaryWrapper when shap_explain_probability=True 
            so SHAP sees sigmoid probabilities rather than raw logits.
            Background images come from config.shap_background; falls back to a zero tensor if not provided.

        Args:
            img: Image tensor (C, H, W).
            label: Unused (binary single-output model assumed).

        Returns:
            Per-channel saliency map (C, H, W) with values in [-1, 1],
            symmetrically normalised by the global max absolute value across
            all channels.
        """
        wrapped = ShapBinaryWrapper(
            self.model,
            explain_probability=self.config.shap_explain_probability
        ).to(self.device).eval()

        if self.config.shap_background is not None:
            background = self.config.shap_background.to(self.device)
        else:
            background = torch.zeros(1, *img.shape, device=self.device)

        explainer = shap.DeepExplainer(wrapped, background)
        x = img.unsqueeze(0).to(self.device)
        shap_vals = explainer.shap_values(x)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_arr = np.array(shap_vals)

        if shap_arr.ndim == 5 and shap_arr.shape[-1] == 1:
            shap_arr = shap_arr[..., 0]
        if shap_arr.ndim == 4 and shap_arr.shape[-1] in [1, 3]:
            shap_arr = np.transpose(shap_arr, (0, 3, 1, 2))

        chw = shap_arr[0].astype(np.float32)

        # For label 0 (cat), the model output is the dog score.
        # Negative SHAP values = evidence for cat, so flip the sign
        # so that "most important for the true class" always sorts correctly.
        if label == 0:
            chw = -chw

        max_abs = np.max(np.abs(chw)) + 1e-8
        return chw / max_abs  # (C, H, W)
    
    def shap_raw_values(self, img: torch.Tensor) -> np.ndarray:
        """Return raw per-channel SHAP values without normalisation.

        Useful for computing SHAP base value diagnostics or custom visualisation.

        Args:
            img: Image tensor (C, H, W).

        Returns:
            Array of shape (H, W, C) with raw signed SHAP values.
        """
        wrapped = ShapBinaryWrapper(
            self.model,
            explain_probability=self.config.shap_explain_probability
        ).to(self.device).eval()

        background = (self.config.shap_background.to(self.device)
                      if self.config.shap_background is not None
                      else torch.zeros(1, *img.shape, device=self.device))

        explainer = shap.DeepExplainer(wrapped, background)
        x = img.unsqueeze(0).to(self.device)
        shap_vals = explainer.shap_values(x)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        arr = np.array(shap_vals)
        if arr.ndim == 5 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 4 and arr.shape[1] in [1, 3]:
            arr = np.transpose(arr, (0, 2, 3, 1))
        return arr[0]  # (H, W, C)
    
def make_deletion_baseline(
    image: torch.Tensor,
    method: str = DELETION_BASELINE,
) -> torch.Tensor:
    """
    Creates the baseline used to replace deleted pixels.

    Options:
        "black": deleted pixels become 0.
        "blur": deleted pixels become Gaussian-blurred pixels.
        "mean": deleted pixels become the image's mean RGB value.
    """
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
        # Per-channel mean: one average value for R, G, B separately.
        # Shape: (C, 1, 1), broadcast to (C, H, W)
        mean_rgb = image.mean(dim=(1, 2), keepdim=True)
        return mean_rgb.expand_as(image)

    raise ValueError(
        f"Unknown DELETION_BASELINE='{method}'. "
        "Choose from: 'black', 'blur', 'mean'."
    )
    
def _fidelity_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    ranked: np.ndarray,
    label: int,
    steps: int,
    device: torch.device,
    mode: str,          # "insertion" or "deletion"
) -> torch.Tensor:
    assert mode in ("insertion", "deletion"), f"Unknown mode: {mode}"

    C, H, W = image.shape
    n_features = C * H * W

    if len(ranked) != n_features:
        raise ValueError(
            f"ranked has {len(ranked)} entries but image has {n_features} pixels "
            f"({H}x{W}). Saliency map spatial size must match the image."
        )

    image = image.to(device)

    # New baseline: blurred image instead of black/zero image
    baseline = gaussian_blur_baseline(
        image,
        kernel_size=31,
        sigma=10.0,
    ).to(device)

    ranked_t = torch.from_numpy(ranked.copy()).long().to(device)
    ranked_t = ranked_t.clamp(0, n_features - 1)

    image = image.to(device)
    baseline = make_deletion_baseline(image).to(device)

    scores = []
    model.eval()
    with torch.no_grad():
        for step in range(steps + 1):
            n_top = int(step / steps * n_features)

            if mode == "insertion":
                # Start from blurred image, reveal important original pixels
                mask = torch.zeros(n_features, device=device)
                if n_top > 0:
                    mask[ranked_t[:n_top]] = 1.0
            else:
                # Start from original image, replace important pixels with blur
                mask = torch.ones(n_features, device=device)
                if n_top > 0:
                    mask[ranked_t[:n_top]] = 0.0

            mask = mask.view(C, H, W)

            masked = image * mask + baseline * (1.0 - mask)

            logit = model(masked.unsqueeze(0))
            prob  = torch.sigmoid(logit).squeeze()
            score = prob if label == 1 else (1.0 - prob)
            scores.append(score)

    return torch.stack(scores)

# def _fidelity_curve(
#     model: torch.nn.Module,
#     image: torch.Tensor,
#     ranked: np.ndarray,
#     label: int,
#     steps: int,
#     device: torch.device,
#     mode: str,          # "insertion" or "deletion"
# ) -> torch.Tensor:
#     """Core pixel-masking loop shared by insertion and deletion AUC.
 
#     Runs the model in eval() + no_grad() mode and returns a (steps+1,) tensor
#     of class scores, one per masking step.
 
#     For *insertion*: starts from an all-zero image and progressively reveals
#     the most important pixels.  Score should rise quickly → high AUC is good.
 
#     For *deletion*: starts from the full image and progressively removes the
#     most important pixels.  Score should fall quickly → low AUC is good.
 
#     Args:
#         model:   PyTorch model already on `device`, will be put in eval mode.
#         image:   Single image tensor (C, H, W) on `device`.
#         ranked:  1-D int array of pixel indices sorted most-to-least important,
#                  length H*W.  Comes from argsort(saliency.flatten())[::-1].
#         label:   Ground-truth class index (0 or 1).
#         steps:   Number of masking steps (curve resolution).
#         device:  Torch device.
#         mode:    "insertion" or "deletion".
 
#     Returns:
#         1-D float tensor of shape (steps+1,) with values in [0, 1].
#     """
#     assert mode in ("insertion", "deletion"), f"Unknown mode: {mode}"

#     C, H, W = image.shape
#     n_features = C * H * W

#     # Guard: saliency must have been computed for the same spatial size
#     if len(ranked) != n_features:
#         raise ValueError(
#             f"ranked has {len(ranked)} entries but image has {n_features} pixels "
#             f"({H}x{W}). Saliency map spatial size must match the image."
#         )

#     ranked_t = torch.from_numpy(ranked.copy()).long().to(device)

#     # Safety clamp: catches any remaining off-by-one issues
#     ranked_t = ranked_t.clamp(0, n_features - 1)

#     scores = []
#     model.eval()
#     with torch.no_grad():
#         for step in range(steps + 1):
#             n_top = int(step / steps * n_features)

#             if mode == "insertion":
#                 mask = torch.zeros(n_features, device=device)
#                 if n_top > 0:
#                     mask[ranked_t[:n_top]] = 1.0
#             else:  # deletion
#                 mask = torch.ones(n_features, device=device)
#                 if n_top > 0:
#                     mask[ranked_t[:n_top]] = 0.0

#             mask   = mask.view(C, H, W)
#             # masked = image * mask
#             masked = image * mask + blurred_image * (1.0 - mask)

#             logit = model(masked.unsqueeze(0))
#             prob  = torch.sigmoid(logit).squeeze()
#             score = prob if label == 1 else (1.0 - prob)
#             scores.append(score)

#     return torch.stack(scores)

def gaussian_blur_baseline(
    image: torch.Tensor,
    kernel_size: int = 61,
    sigma: float = 20.0,
) -> torch.Tensor:
    """
    Creates a blurred version of the image to use as deletion/insertion baseline.

    Args:
        image: Tensor of shape (C, H, W), values in [0, 1].
        kernel_size: Gaussian blur kernel size. Must be odd.
        sigma: Blur strength.

    Returns:
        Blurred image tensor of shape (C, H, W).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    return TF.gaussian_blur(
        image,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )

def compute_fidelity(
    model: torch.nn.Module,
    image: torch.Tensor,
    saliency: np.ndarray,
    label: int,
    steps: int,
    device: torch.device,
    path: str
) -> tuple[float, float]:
    """Compute Deletion AUC and Insertion AUC for one image.

    Deletion: progressively replace the most important features with the baseline
              value -> the model score for the true class should drop quickly.
              Lower AUC is better.

    Insertion: start from the baseline and progressively reveal the most important
               features from the real image -> score should rise quickly.
               Higher AUC is better.

    Args:
        model: PyTorch model in eval mode, outputting a single scalar logit.
        image: Image tensor (C, H, W).
        saliency: Per-channel saliency map (C, H, W) from Explainer.explain().
        label: Target class index (0 or 1 for binary classification).
        steps:    Number of masking steps.
        device: Torch device.
        path: File path to save the fidelity curve plot.

    Returns:
        Tuple of (deletion_auc, insertion_auc), both in [0, 1].
    """
    image = image.to(device)
 
    # Pixel ranking: most important first (used by both curves)
    ranked = np.argsort(saliency.flatten())[::-1]  # (H*W,) descending
 
    del_scores = _fidelity_curve(model, image, ranked, label, steps, device, mode="deletion")
    ins_scores = _fidelity_curve(model, image, ranked, label, steps, device, mode="insertion")
 
    xs = torch.linspace(0.0, 1.0, steps + 1, device=device)
    del_auc = float(torch.trapezoid(del_scores, xs).item())
    ins_auc = float(torch.trapezoid(ins_scores, xs).item())
 
    # DEBUG plot
    xs_np  = xs.cpu().numpy()
    del_np = del_scores.cpu().numpy()
    ins_np = ins_scores.cpu().numpy()
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.plot(xs_np, del_np)
    ax1.set_title(f"Deletion AUC = {del_auc:.4f}  (lower = better)")
    ax1.set_xlabel("Fraction of pixels removed")
    ax1.set_ylabel("Model score")
    ax2.plot(xs_np, ins_np)
    ax2.set_title(f"Insertion AUC = {ins_auc:.4f}  (higher = better)")
    ax2.set_xlabel("Fraction of pixels revealed")
    ax2.set_ylabel("Model score")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    # DEBUG plot

    return del_auc, ins_auc


def compute_stability(
    explainer: Explainer,
    image: torch.Tensor,
    label: int,
    n_perturbations: int,
    noise_std: float,
) -> float:
    """Estimate explanation stability via a Lipschitz ratio.

    Adds Gaussian noise to the input multiple times and measures how much
    the explanation changes relative to the input change.
    Lower values indicate more stable (robust) explanations.

    The Lipschitz ratio is computed over the flattened (C×H×W) attribution
    vectors, so channel-level differences are fully captured.

    Args:
        explainer: Explainer instance.
        image: Original image tensor (C, H, W).
        label: Target class index.
        n_perturbations: Number of noisy copies to generate.
        noise_std: Standard deviation of the Gaussian noise.

    Returns:
        Mean Lipschitz estimate (float, lower = more stable).
    """
    base_sal = explainer.explain(image, label).flatten()
    ratios = []

    for _ in range(n_perturbations):
        noise = torch.randn_like(image) * noise_std
        perturbed = (image + noise).clamp(0, 1)

        perturbed_sal = explainer.explain(perturbed, label).flatten()

        input_dist = np.linalg.norm(noise.cpu().numpy().flatten()) + 1e-8
        sal_dist   = np.linalg.norm(perturbed_sal - base_sal)
        ratios.append(sal_dist / input_dist)

    return float(np.mean(ratios))


def compute_identity(
    explainer: Explainer,
    image: torch.Tensor,
    label: int,
    tol: float = 1e-5,
) -> float:
    """Check if the same input always produces the same explanation.

    A value of 1.0 means the method is deterministic; 0.0 means it is not.
    Comparison is element-wise over the full (C, H, W) attribution array.

    Args:
        explainer: Explainer instance.
        image: Image tensor (C, H, W).
        label: Target class index.
        tol: Maximum allowed element-wise difference to be considered identical.

    Returns:
        1.0 if maps are identical within tolerance, else 0.0.
    """
    sal1 = explainer.explain(image, label)
    sal2 = explainer.explain(image, label)

    
    # diff = np.max(np.abs(sal1 - sal2))
    # print(f"  [{explainer.method}] identity max diff: {diff:.2e}")  # e.g. 1.23e-06
    
    return 1.0 if np.allclose(sal1, sal2, atol=tol) else 0.0


def compute_separability(
    explainer: Explainer,
    images: list[torch.Tensor],
    labels: list[int],
    n_pairs: int,
    eps: float,
) -> float:
    """Fraction of cross-class image pairs that have sufficiently different maps.

    A good explainer should produce distinct explanations for distinct inputs.
    Pairs are sampled from images belonging to different classes.

    The L2 distance is computed over the flattened (C×H×W) attribution vectors,
    capturing both spatial and channel-level differences.

    Args:
        explainer: Explainer instance.
        images: List of image tensors.
        labels: Corresponding class labels.
        n_pairs: Number of pairs to evaluate.
        eps: Minimum L2 distance between flattened (C, H, W) maps to count as
             "separable".

    Returns:
        Fraction of pairs that are separable (float in [0, 1]).
    """
    # Group image indices by class
    class_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_to_idxs[lbl].append(i)

    classes = list(class_to_idxs.keys())
    if len(classes) < 2:
        return 1.0  # trivially separable if only one class present

    rng = np.random.default_rng(42)
    separable = 0

    for _ in range(n_pairs):
        # Pick two different classes, then one image from each
        c1, c2 = rng.choice(classes, size=2, replace=False)
        i1 = rng.choice(class_to_idxs[c1])
        i2 = rng.choice(class_to_idxs[c2])

        sal1 = explainer.explain(images[i1], labels[i1]).flatten()
        sal2 = explainer.explain(images[i2], labels[i2]).flatten()

        if np.linalg.norm(sal1 - sal2) > eps:
            separable += 1

    return separable / n_pairs


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1].

    Args:
        arr: Input numpy array.

    Returns:
        Normalised array with same shape.
    """
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)



class SaliencyVisualiser:
    """3-panel visualisation for GradCAM and SHAP saliency maps.

    Makes:
      Panel 1: input image with true / predicted label and probability
      Panel 2: heatmap with colorbar (and SHAP diagnostic values if SHAP)
      Panel 3: image + heatmap overlay

    Usage:
        vis = SaliencyVisualiser(out_dir="./interpretability_results", idx_to_class={0:"real", 1:"ai"})
        vis.save_gradcam(image, cam_raw, pred_prob, true_label, pred_label, filename)
        vis.save_shap(image, shap_hwc, shap_explainer, pred_prob, true_label, pred_label, filename)

    Args:
        out_dir: Root directory; sub-folders per class are created automatically.
        idx_to_class: Dict mapping int label → class name string.
        gradcam_cmap: Matplotlib colormap name for GradCAM heatmaps.
        shap_cmap: Matplotlib colormap name for SHAP heatmaps (should be diverging).
        overlay_alpha: Transparency of the heatmap layer in the overlay panel.
        dpi: Output image resolution.
    """

    def __init__(
        self,
        out_dir: str,
        idx_to_class: dict[int, str],
        gradcam_cmap: str = "Reds",
        shap_cmap: str = "bwr",
        overlay_alpha: float = 0.50,
        dpi: int = 150,
    ):
        self.out_dir = out_dir
        self.idx_to_class = idx_to_class
        self.gradcam_cmap = gradcam_cmap
        self.shap_cmap = shap_cmap
        self.overlay_alpha = overlay_alpha
        self.dpi = dpi

    @staticmethod
    def _to_hwc(image_tensor: torch.Tensor) -> np.ndarray:
        """Convert a (C, H, W) float tensor in [0,1] to (H, W, C) float array.

        Args:
            image_tensor: Tensor of shape (C, H, W).

        Returns:
            numpy array of shape (H, W, C), clipped to [0, 1].
        """
        return image_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    def _resolve_path(self, true_label: int, filename: str) -> str:
        """Build output path under a per-class sub-folder.

        Args:
            true_label: Ground-truth class index.
            filename: Base filename (no directory).

        Returns:
            Full output path string.
        """
        class_name = self.idx_to_class.get(true_label, str(true_label))
        folder = os.path.join(self.out_dir, class_name)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    def save_gradcam(
        self,
        image_tensor: torch.Tensor,
        cam_raw: np.ndarray,
        pred_prob: float,
        true_label: int,
        pred_label: int,
        filename: str,
    ):
        """Save a 3-panel GradCAM visualisation.

        The raw CAM is normalised to [0, 1] here at plot time, exactly as in
        the original save_gradcam_visualization(): NaN → 0, clamp to ≥ 0,
        divide by max. Nothing is modified before this point.

        Args:
            image_tensor: Input image tensor (C, H, W) in [0, 1].
            cam_raw: Raw (non-normalised) GradCAM map (H, W).
            pred_prob: Sigmoid/softmax probability for the predicted class.
            true_label: Ground-truth class index.
            pred_label: Predicted class index.
            filename: Output filename (e.g. 'gradcam_001.png').
        """
        img = self._to_hwc(image_tensor)
        true_name = self.idx_to_class.get(true_label, str(true_label))
        pred_name = self.idx_to_class.get(pred_label, str(pred_label))

        # Normalise at plot time, matching original behaviour exactly
        hm = np.array(cam_raw, dtype=np.float32)
        hm = np.nan_to_num(hm)
        hm = np.clip(hm, 0, None)
        max_val = hm.max()
        hm = hm / max_val if max_val > 0 else np.zeros_like(hm)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title(f"Input\nTrue: {true_name}\nPred: {pred_name} ({pred_prob:.3f})")
        axes[0].axis("off")

        im = axes[1].imshow(hm, cmap=self.gradcam_cmap, vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img)
        axes[2].imshow(hm, cmap=self.gradcam_cmap, alpha=self.overlay_alpha,
                       vmin=0, vmax=1)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = self._resolve_path(true_label, filename)
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_shap(
        self,
        image_tensor: torch.Tensor,
        shap_hwc: np.ndarray,
        pred_prob: float,
        true_label: int,
        pred_label: int,
        filename: str,
        base_value: Optional[float] = None,
    ):
        """Save a 3-panel SHAP visualisation.

        The heatmap uses symmetric normalisation (divide by max absolute value)
        and a diverging colormap so positive/negative contributions are visible,
        matching save_shap_visualizations() in the original SHAP_test.py.

        Args:
            image_tensor: Input image tensor (C, H, W) in [0, 1].
            shap_hwc: Raw per-channel SHAP values of shape (H, W, C).
            pred_prob: Sigmoid probability for the predicted class.
            true_label: Ground-truth class index.
            pred_label: Predicted class index.
            filename: Output filename (e.g. 'shap_001.png').
            base_value: SHAP expected value (explainer.expected_value), used in
                        the diagnostic title. Pass None to omit.
        """
        img = self._to_hwc(image_tensor)
        true_name = self.idx_to_class.get(true_label, str(true_label))
        pred_name = self.idx_to_class.get(pred_label, str(pred_label))

        # Sum across channels → (H, W), then symmetrically normalise
        hm = shap_hwc.sum(axis=-1)
        shap_total = float(hm.sum())
        max_abs = np.max(np.abs(hm)) + 1e-8
        hm_norm = hm / max_abs  # in [-1, 1]

        # Build heatmap title with optional diagnostics
        hm_title = f"SHAP heatmap\nSHAP total: {shap_total:.3f}"
        if base_value is not None:
            hm_title += f"\nBase val: {base_value:.3f}"
            hm_title += f"\nSHAP pred: {base_value + shap_total:.3f}"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title(f"Input\nTrue: {true_name}\nPred: {pred_name} ({pred_prob:.3f})")
        axes[0].axis("off")

        im = axes[1].imshow(hm_norm, cmap=self.shap_cmap, vmin=-1, vmax=1)
        axes[1].set_title(hm_title)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img)
        axes[2].imshow(hm_norm, cmap=self.shap_cmap, alpha=self.overlay_alpha,
                       vmin=-1, vmax=1)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = self._resolve_path(true_label, filename)
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_generic(
        self,
        image_tensor: torch.Tensor,
        saliency: np.ndarray,
        method_name: str,
        pred_prob: float,
        true_label: int,
        pred_label: int,
        filename: str,
    ):
        """Save a generic 3-panel saliency visualisation for non-GradCAM/SHAP methods.

        Uses a hot colormap with [0, 1] range. Suitable for IntGrad, SmoothGrad,
        LIME, etc.

        Accepts saliency of shape (C, H, W) or (H, W). When (C, H, W) is given,
        the mean absolute value across channels is taken to produce a (H, W) map
        for display.

        Args:
            image_tensor: Input image tensor (C, H, W).
            saliency: Saliency map of shape (C, H, W) or (H, W).
            method_name: Method name for the plot title.
            pred_prob: Probability for the predicted class.
            true_label: Ground-truth class index.
            pred_label: Predicted class index.
            filename: Output filename.
        """
        img = self._to_hwc(image_tensor)
        true_name = self.idx_to_class.get(true_label, str(true_label))
        pred_name = self.idx_to_class.get(pred_label, str(pred_label))

        # Collapse (C, H, W) → (H, W) by mean absolute value across channels
        hm = np.abs(saliency).mean(axis=0) if saliency.ndim == 3 else saliency
        hm = _normalise(hm)  # re-normalise to [0, 1] after abs+mean

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title(f"Input\nTrue: {true_name}\nPred: {pred_name} ({pred_prob:.3f})")
        axes[0].axis("off")

        im = axes[1].imshow(hm, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title(f"{method_name.upper()} heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img)
        axes[2].imshow(hm, cmap="hot", alpha=self.overlay_alpha, vmin=0, vmax=1)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = self._resolve_path(true_label, filename)
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_grid(
        self,
        images: list[torch.Tensor],
        saliency_maps: list[np.ndarray],
        method_name: str,
        n_show: int = 8,
        filename: str = "saliency_grid.png",
    ):
        """Save a compact image/saliency grid (2 rows × N cols) to out_dir root.

        Args:
            images: List of image tensors (C, H, W).
            saliency_maps: Corresponding saliency maps of shape (C, H, W) or (H, W).
                           (C, H, W) maps are collapsed to (H, W) by mean absolute
                           value across channels before display.
            method_name: Label for the plot title.
            n_show: Number of pairs to display.
            filename: Output filename saved directly under out_dir.
        """
        n = min(n_show, len(images))
        fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
        fig.suptitle(f"Saliency maps — {method_name.upper()}", fontsize=13, y=1.01)

        for i in range(n):
            axes[0, i].imshow(self._to_hwc(images[i]))
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Image", fontsize=9)

            # Collapse (C, H, W) → (H, W) if needed
            sal = saliency_maps[i]
            hm = _normalise(np.abs(sal).mean(axis=0)) if sal.ndim == 3 else sal

            axes[1, i].imshow(hm, cmap="hot")
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Saliency", fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(self.out_dir, filename)
        os.makedirs(self.out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

def plot_saliency_grid(
    images: list[torch.Tensor],
    saliency_maps: list[np.ndarray],
    method_name: str,
    n_show: int = 8,
    save_path: str = "saliency_grid.png",
):
    """Save a grid of images alongside their saliency maps.

    Args:
        images: List of image tensors (C, H, W).
        saliency_maps: Corresponding saliency maps of shape (C, H, W) or (H, W).
                       (C, H, W) maps are collapsed to (H, W) by mean absolute
                       value across channels before display.
        method_name: Label shown in the plot title.
        n_show: Number of image/saliency pairs to show.
        save_path: Output file path.
    """
    n = min(n_show, len(images))
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
    fig.suptitle(f"Saliency maps: {method_name.upper()}", fontsize=13, y=1.01)

    for i in range(n):
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        axes[0, i].imshow(img_np)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Image", fontsize=9)

        # Collapse (C, H, W) → (H, W) if needed
        sal = saliency_maps[i]
        hm = _normalise(np.abs(sal).mean(axis=0)) if sal.ndim == 3 else sal

        axes[1, i].imshow(hm, cmap="hot")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Saliency", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_radar_chart(results: dict[str, MetricResults], save_path: str = "radar.png"):
    """Radar (spider) chart comparing methods across all 5 metrics.

    Metrics are normalised so that a higher value always means "better" on the chart:
      - deletion_auc is inverted (lower raw = better)
      - stability is inverted (lower raw = better)
      - avg_time_sec is inverted and capped

    Args:
        results: Dict mapping method name -> MetricResults.
        save_path: Output file path.
    """
    categories = ["Fidelity\n(Insertion)", "Fidelity\n(1-Deletion)", "Stability\n(inv.)",
                  "Identity", "Separability"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10.colors

    for idx, (method, res) in enumerate(results.items()):
        # Collect and normalise values so all are "higher = better"
        raw_vals = [
            res.insertion_auc,
            1 - res.deletion_auc,
            1 / (1 + res.stability),    # invert: lower stability score = better
            res.identity,
            res.separability,
        ]
        vals = raw_vals + raw_vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=method.upper(),
                color=colors[idx % len(colors)])
        ax.fill(angles, vals, alpha=0.08, color=colors[idx % len(colors)])

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Interpretability Method Comparison", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bar_charts(results: dict[str, MetricResults], save_path: str = "bars.png"):
    """Side-by-side bar charts for each metric across all methods.

    Args:
        results: Dict mapping method name -> MetricResults.
        save_path: Output file path.
    """
    methods = list(results.keys())
    metrics = {
        "Deletion AUC\n(lower = better)":  [r.deletion_auc   for r in results.values()],
        "Insertion AUC\n(higher = better)": [r.insertion_auc  for r in results.values()],
        "Stability\n(lower = better)":      [r.stability      for r in results.values()],
        "Identity\n(higher = better)":      [r.identity       for r in results.values()],
        "Separability\n(higher = better)":  [r.separability   for r in results.values()],
        "Avg Time (s)\n(lower = better)":   [r.avg_time_sec   for r in results.values()],
    }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors[:len(methods)]

    for ax, (title, vals) in zip(axes, metrics.items()):
        bars = ax.bar(methods, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.upper() for m in methods], rotation=20, ha="right", fontsize=8)
        # Annotate bar heights
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Metric Breakdown by Interpretability Method", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary_table(results: dict[str, MetricResults]):
    """Print a formatted summary table to stdout.

    Args:
        results: Dict mapping method name -> MetricResults.
    """
    header = f"{'Method':<12} {'Del AUC':>10} {'Ins AUC':>10} {'Stability':>12} {'Identity':>10} {'Separability':>10} {'Time(s)':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for method, res in results.items():
        print(
            f"{method.upper():<12} "
            f"{res.deletion_auc:>10.4f} "
            f"{res.insertion_auc:>10.4f} "
            f"{res.stability:>12.4f} "
            f"{res.identity:>10.4f} "
            f"{res.separability:>10.4f} "
            f"{res.avg_time_sec:>10.4f}"
        )
    print(sep + "\n")


def save_csv(results: dict[str, MetricResults], save_path: str = "metrics.csv"):
    """Save all metric results to a CSV file.

    Args:
        results: Dict mapping method name -> MetricResults.
        save_path: Output file path.
    """
    import csv
    fields = ["method", "deletion_auc", "insertion_auc", "stability",
              "identity", "separability", "avg_time_sec","model_tot_param"
              ,"model_val_acc","model_val_f1","model_test_acc","model_test_f1"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, res in results.items():
            writer.writerow({
                "method":       method,
                "deletion_auc": round(res.deletion_auc, 6),
                "insertion_auc": round(res.insertion_auc, 6),
                "stability":    round(res.stability, 6),
                "identity":     round(res.identity, 6),
                "separability": round(res.separability, 6),
                "avg_time_sec": round(res.avg_time_sec, 6),
                "model_tot_param": res.model_tot_param,
                "model_val_acc": round(res.model_val_acc, 6),
                "model_val_f1": round(res.model_val_f1, 6),
                "model_test_acc": round(res.model_test_acc, 6),
                "model_test_f1": round(res.model_test_f1, 6),
            })
    print(f"  Saved: {save_path}")
def load_correct_samples(
    dataset: Dataset,
    model: torch.nn.Module,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[int]]:
    """Draw n_samples from a Dataset where the model predicts correctly,
    with equal class balance (n_samples // 2 per class).

    Args:
        dataset: PyTorch Dataset with a .targets attribute, returning (image_tensor, label, path).
        model: PyTorch model in eval mode.
        n_samples: Total number of correctly classified samples. Must be even.
        batch_size: DataLoader batch size.
        device: Torch device.

    Returns:
        Tuple of (list of image tensors, list of integer labels), interleaved by class.
    """
    if not hasattr(dataset, "targets"):
        raise AttributeError("Dataset must have a .targets attribute.")

    n_per_class = n_samples // 2

    # Separate shuffled indices by class upfront
    class_to_idxs: dict[int, list[int]] = defaultdict(list)
    shuffled = torch.randperm(len(dataset)).tolist()
    for idx in shuffled:
        class_to_idxs[int(dataset.targets[idx])].append(idx)

    classes = sorted(class_to_idxs.keys())[:2]  # only class 0 and 1

    images: dict[int, list[torch.Tensor]] = {c: [] for c in classes}
    labels: dict[int, list[int]]          = {c: [] for c in classes}

    model.eval()

    # Process each class independently so we hit n_per_class for both
    for cls in classes:
        idxs = class_to_idxs[cls]
        loader = DataLoader(
            Subset(dataset, idxs),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_skip_none,
        )

        with torch.no_grad():
            for imgs, lbls, _ in loader:
                imgs = imgs.float().to(device)

                logits = model(imgs)
                probs  = torch.sigmoid(logits).squeeze(-1)
                preds  = (probs >= 0.5).long()

                for img, lbl, pred in zip(imgs, lbls, preds):
                    if pred.item() == lbl.item():
                        images[cls].append(img.cpu())
                        labels[cls].append(int(lbl.item()))

                    if len(images[cls]) >= n_per_class:
                        break

                if len(images[cls]) >= n_per_class:
                    break

        found = len(images[cls])
        if found < n_per_class:
            print(f"  Warning: class {cls} only yielded {found} correct samples "
                  f"(requested {n_per_class}).")
        else:
            print(f"  Class {cls}: collected {found} correctly classified samples.")

    # Interleave classes so the sample list isn't class-sorted
    out_images, out_labels = [], []
    for img0, lbl0, img1, lbl1 in zip(
        images[classes[0]], labels[classes[0]],
        images[classes[1]], labels[classes[1]],
    ):
        out_images += [img0, img1]
        out_labels += [lbl0, lbl1]
    
    # Extract numeric IDs from filenames for reproducibility reporting
    nums_by_class: dict[str, list[int]] = {}
    for cls in classes:
        class_name = dataset.classes[cls]

        # Walk the subset loader indices back to original dataset paths
        nums = []
        collected = 0
        for idx in class_to_idxs[cls]:
            if collected >= len(images[cls]):
                break
            path, _ = dataset.samples[idx]
            match = re.search(r"(\d+)", os.path.basename(path))
            if match:
                nums.append(int(match.group(1)))
            collected += 1

        nums_by_class[class_name] = nums
        print(f"  {class_name}_nums = {nums}")

    return out_images, out_labels, nums_by_class


def load_samples(
    dataset: Dataset,
    n_samples: int,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """Draw n_samples from a Dataset and return individual tensors + labels.

    Args:
        dataset: Any PyTorch Dataset returning (image_tensor, label).
        n_samples: How many samples to use.
        batch_size: DataLoader batch size (affects speed, not correctness).

    Returns:
        Tuple of (list of image tensors, list of integer labels).
    """
    n_samples = min(n_samples, len(dataset))
    
    selected_indices = torch.randperm(len(dataset))[:n_samples].tolist()

    subset = Subset(dataset, selected_indices)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,collate_fn=collate_skip_none)

    images, labels = [], []
    for imgs, lbls, _ in loader:
        for img, lbl in zip(imgs, lbls):
            images.append(img.float())
            labels.append(int(lbl))

    return images, labels


def evaluate_method(
    method_name: str,
    model: torch.nn.Module,
    images: list[torch.Tensor],
    labels: list[int],
    config: EvalConfig,
) -> tuple[MetricResults, list[np.ndarray]]:
    """Run the full evaluation pipeline for one Interpretability method.

    Args:
        method_name: Interpretability method key (e.g. 'gradcam').
        model: PyTorch model in eval mode.
        images: List of image tensors to evaluate on.
        labels: Corresponding class labels.
        config: EvalConfig with all hyperparameters.

    Returns:
        Tuple of (MetricResults, list of per-channel saliency maps of shape (C, H, W)).
    """
    device = torch.device(config.device)
    model = model.to(device).eval()

    explainer = Explainer(method_name, model, config)

    del_aucs, ins_aucs, del_aucs_cap, stab_scores, id_scores, times = [], [], [], [], [], []
    saliency_maps = []

    print(f"\n  Evaluating: {method_name.upper()} on {len(images)} samples")
    for i, (img, lbl) in enumerate(zip(images, labels)):
        print(f"    [{i+1}/{len(images)}]", end="\r")

        # --- Computational Time + generate explanation ---
        t0 = time.perf_counter()
        sal = explainer.explain(img, lbl)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        saliency_maps.append(sal)

        # --- Fidelity ---
        d_auc, i_auc = compute_fidelity(model, img, sal, lbl,
                                         config.fidelity_features_per_step, device,f"{config.output_dir}/{method_name}/fidelity_{i}.png")
        del_aucs.append(d_auc)
        ins_aucs.append(i_auc)

        #d_auc_cap, i_auc_cap = get_captum_fidelity(model, img, sal, lbl,device,500)
        #del_aucs_cap.append(d_auc_cap)

        # --- Stability ---
        stab = compute_stability(explainer, img, lbl,
                                  config.stability_n_perturbations,
                                  config.stability_noise_std)
        stab_scores.append(stab)

        # --- Identity ---
        ident = compute_identity(explainer, img, lbl)
        id_scores.append(ident)

    # --- Separability (needs the full image pool) ---
    sep = compute_separability(explainer, images, labels,
                                config.separability_n_pairs,
                                config.separability_eps)

    explainer.remove_hooks()

    results = MetricResults(
        deletion_auc  = float(np.mean(del_aucs)),
        insertion_auc = float(np.mean(ins_aucs)),
        stability     = float(np.mean(stab_scores)),
        identity      = float(np.mean(id_scores)),
        separability  = sep,
        avg_time_sec  = float(np.mean(times)),
        raw = {
            "deletion_auc":  del_aucs,
            "insertion_auc": ins_aucs,
            "stability":     stab_scores,
            "identity":      id_scores,
            "time":          times,
        },
    )
    return results, saliency_maps


def run_pipeline(
    model: torch.nn.Module,
    dataset: Dataset,
    methods: list[str],
    config: EvalConfig,
    idx_to_class: Optional[dict] = None,
) -> dict[str, MetricResults]:
    """Run the full evaluation pipeline for one or more interpretability methods.

    Saves all plots and a CSV to config.output_dir.

    Args:
        model: PyTorch model in eval mode.
        dataset: PyTorch Dataset returning (image_tensor, label).
        methods: List of Interpretability method names to evaluate.
        config: EvalConfig with all hyperparameters.
        idx_to_class: Optional dict {int: str} for label names in plots.
                      Falls back to numeric strings if not provided.

    Returns:
        Dict mapping method name -> MetricResults.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    if idx_to_class is None:
        idx_to_class = {}

    print(f"\n{'='*60}")
    print(f"  Interpretability Evaluation Pipeline")
    print(f"  Device : {config.device}")
    print(f"  Samples: {config.n_samples}")
    print(f"  Methods: {', '.join(m.upper() for m in methods)}")
    print(f"{'='*60}")


    # images, labels = load_samples(dataset, config.n_samples, config.batch_size)
    
    # Loading correct samples
    device = torch.device(config.device)
    images, labels, sample_nums = load_correct_samples(
        dataset, model, config.n_samples, config.batch_size, device
    )

    # Save the sampled image numbers for reproducibility
    nums_path = os.path.join(config.output_dir, "sampled_image_nums.txt")
    with open(nums_path, "w") as _f:
        for class_name, nums in sample_nums.items():
            _f.write(f"{class_name}_nums = {nums}\n")
    print(f"  Saved: {nums_path}")
    
    all_results: dict[str, MetricResults] = {}
    all_saliency: dict[str, list[np.ndarray]] = {}

    for lbl in labels:
        if lbl not in idx_to_class:
            idx_to_class[lbl] = str(lbl)

    all_results: dict[str, MetricResults] = {}

    model_total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"model_total_param: {model_total_param}")
    _, val_loader, test_loader, idx_to_class = data_loaders(DEVICE, BATCH_SIZE)

    val_acc,_,_,val_f1 = evaluate(model,val_loader,"Val")
    test_acc,_,_,test_f1 = evaluate(model,test_loader,"Test")


    for method in methods:
        
        method_dir = os.path.join(config.output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        results, sal_maps = evaluate_method(method, model, images, labels, config)
        all_results[method] = results

        all_results[method].model_tot_param = model_total_param
        all_results[method].model_val_acc = val_acc
        all_results[method].model_val_f1 = val_f1
        all_results[method].model_test_acc = test_acc
        all_results[method].model_test_f1 = test_f1

        #[VISUALIZATION]
        """
        # Build per-method visualiser, rooted under a method sub-folder
        method_dir = os.path.join(config.output_dir, method)
        vis = SaliencyVisualiser(out_dir=method_dir, idx_to_class=idx_to_class)

        # Get model predictions for visualisation labels/probs
        device = torch.device(config.device)
        model.to(device).eval()

        for i, (img, lbl, sal) in enumerate(zip(images, labels, sal_maps)):
            with torch.no_grad():
                out = model(img.unsqueeze(0).to(device))
                # Support binary (sigmoid) and multi-class (softmax) outputs
                if out.shape[-1] == 1 or out.numel() == 1:
                    prob = float(torch.sigmoid(out).item())
                    pred = int(prob >= 0.5)
                else:
                    probs = torch.softmax(out, dim=1)[0]
                    pred = int(probs.argmax().item())
                    prob = float(probs[pred].item())

            fname = f"{method}_{i:03d}_true{lbl}_pred{pred}.png"

            if method == "gradcam":
                # Re-run explainer to get the raw map for visualisation
                exp = Explainer(method, model, config)
                raw = exp.gradcam_raw_map(img.to(device))
                exp.remove_hooks()
                vis.save_gradcam(img, raw, prob, lbl, pred, fname)

            elif method == "shap":
                exp = Explainer(method, model, config)
                raw_hwc = exp.shap_raw_values(img)
                exp.remove_hooks()
                vis.save_shap(img, raw_hwc, prob, lbl, pred, fname)

            else:
                vis.save_generic(img, sal, method, prob, lbl, pred, fname)

        # Compact grid overview for this method
        vis.save_grid(images, sal_maps, method,
                      filename=f"grid_{method}.png")
        """

    # Aggregate comparison plots — only meaningful with >1 method
    if len(all_results) > 1:
        plot_radar_chart(all_results,
                         save_path=f"{config.output_dir}/radar_chart.png")
        plot_bar_charts(all_results,
                        save_path=f"{config.output_dir}/bar_charts.png")

    save_csv(all_results, save_path=f"{config.output_dir}/metrics_summary.csv")
    print_summary_table(all_results)

    return all_results

def collect_shap_background(
    dataset: Dataset,
    n_background: int = 50,
    batch_size: int = 8,
) -> torch.Tensor:
    """Collect a class-balanced background tensor for shap.DeepExplainer.
    Samples exactly n_background // 2 images from each class (class 0 and
    class 1), giving a balanced reference distribution. This is important
    because a skewed background biases SHAP attributions toward the
    over-represented class.

    The dataset must expose a .targets attribute (list of int labels), which
    is standard for torchvision ImageFolder and its subclasses like
    SafeImageFolder.

    Args:
        dataset: Training Dataset with a .targets attribute.
        n_background: Total number of background images. Must be even.
                      Half will be class 0, half class 1.
        batch_size: DataLoader batch size used internally.

    Returns:
        Float tensor of shape (n_background, C, H, W), kept on CPU.
    """
    if not hasattr(dataset, "targets"):
        raise AttributeError(
            "Dataset must have a .targets attribute (list of int labels). "
            "torchvision ImageFolder and SafeImageFolder both provide this."
        )

    n_per_class = n_background // 2

    # Collect indices for each class from dataset.targets
    class_to_idxs: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_to_idxs[int(label)].append(idx)

    classes = sorted(class_to_idxs.keys())
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 classes, found: {classes}")

    # Take the first n_per_class indices from each class
    selected_idxs = []
    for cls in classes[:2]:  # only use class 0 and class 1
        available = class_to_idxs[cls]
        if len(available) < n_per_class:
            raise ValueError(
                f"Class {cls} only has {len(available)} samples, "
                f"but {n_per_class} were requested."
            )
        selected_idxs.extend(available[:n_per_class])

    loader = DataLoader(
        Subset(dataset, selected_idxs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_skip_none,
    )

    imgs_list = []
    for batch in loader:
        imgs = batch[0]  # works for both (img, lbl) and (img, lbl, path)
        imgs_list.append(imgs.float())

    background = torch.cat(imgs_list, dim=0)

    # Count per class for the confirmation message
    labels_list = [dataset.targets[i] for i in selected_idxs]
    counts = {cls: labels_list.count(cls) for cls in classes[:2]}
    print(f"  SHAP background: {tuple(background.shape)}, "
          f"class counts: {counts}, "
          f"value range [{background.min():.2f}, {background.max():.2f}]")

    return background
def parse_model_filename(fname: str) -> Optional[dict]:
    """Parse hyperparameters from a model filename.
 
    Expected pattern:
        model_kernel=[5,9,<K>]_<CF>_<CL>_<DN>_<DL>_wd<WD>_do<DO>
        (with an optional .pth extension)
 
    Example:
        model_kernel=[5,9,23]_32_3_64_1_wd0.0001_do0.0.pth
        → {kernel_size: 23, conv_filter: 32, conv_layer: 3,
           dense_neuron: 64, dense_layer: 1,
           weight_decay: 0.0001, dropout: 0.0}
 
    Returns:
        Dict of parsed values, or None if the filename does not match.
    """
    pattern = (
        r"finetuned_fidelity_k=\[5,9,(\d+)\]"   # kernel_size  (third element of [5,9,K])
        r"_(\d+)"                         # conv_filter
        r"_(\d+)"                         # conv_layer
        r"_(\d+)"                         # dense_neuron
        r"_(\d+)"                         # dense_layer
        r"_wd([0-9eE+\-\.]+)"            # weight_decay  (float, e.g. 1e-4 or 0.0001)
        r"_do([0-9eE+\-\.]+)"            # dropout       (float, e.g. 0.0)
        r"(?:\.pth)?$"                    # optional .pth extension
    )
    m = re.search(pattern, fname)
    if m is None:
        return None
    return {
        "kernel_size":   int(m.group(1)),
        "conv_filter":   int(m.group(2)),
        "conv_layer":    int(m.group(3)),
        "dense_neuron":  int(m.group(4)),
        "dense_layer":   int(m.group(5)),
        "weight_decay":  float(m.group(6)),
        "dropout":       float(m.group(7)),
    }
if __name__ == "__main__":
    DATA_DIR = "./DATA/Cat_dog_splitted/"
    DEVICE =  f"cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 100
    #train_loader, val_dataset, test_loader, idx_to_class = data_loaders(DEVICE,BATCH_SIZE)
    train_dataset = SafeImageFolder(os.path.join(DATA_DIR, "train"),   transform=get_tensor_transform())
    #val_dataset = SafeImageFolder(os.path.join(DATA_DIR, "val"),   transform=get_tensor_transform())
    test_dataset = SafeImageFolder(os.path.join(DATA_DIR, "test"),   transform=get_tensor_transform())

    idx_to_class = {i: name for i, name in enumerate(test_dataset.classes)}

    methods_to_evaluate = ["gradcam", "shap"]

    shap_background = collect_shap_background(train_dataset,50,25)
    
    MODELS_DIR = "models/to_test"

    all_pth_files = sorted(
        f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")
    )
 
    matched_models = []
    for fname in all_pth_files:
        stem  = fname[:-4]          # strip ".pth"
        hparams = parse_model_filename(stem)
        if hparams is None:
            print(f"  [SKIP] Could not parse filename: {fname}")
            continue
        matched_models.append((fname, hparams))
    
    print(f"\nFound {len(matched_models)} matching model(s) in '{MODELS_DIR}':")
    for fname, hp in matched_models:
        print(f"  {fname}  ->  {hp}")

    if not matched_models:
        raise SystemExit("No models matched the expected filename pattern. Nothing to do.")

    for fname, hp in matched_models:
        model_path = os.path.join(MODELS_DIR, fname)
 
        output_dir = f"./finetuning/interpretability_results_dogs_k=[5,9,{hp["kernel_size"]}]_{hp["conv_filter"]}_{hp["conv_layer"]}_{hp["dense_neuron"]}_{hp["dense_layer"]}_wd{hp["weight_decay"]}_do{hp["dropout"]}"
 
        # Skip already-completed runs (CSV written as the last step)
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        if os.path.isfile(csv_path):
            print(f"\n[SKIP] Already evaluated: {fname}  (found {csv_path})")
            continue
 
        print(f"\n{'='*70}")
        print(f"  Model : {fname}")
        print(f"  Params: {hp}")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}")
 
        # Build model from parsed hyperparameters
        #try:
        model = I_HAVE_A_THEORY(
            kernel_size   = hp["kernel_size"],
            conv_filters  = hp["conv_filter"],
            conv_layers   = hp["conv_layer"],
            dense_neurons = hp["dense_neuron"],
            dense_layers  = hp["dense_layer"],
            dropout_rate  = hp["dropout"],
        ).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        #except Exception as exc:
        #    print(f"  [ERROR] Could not load model {fname}: {exc}")
        #    continue
 
        target_layer = find_last_conv_layer(model)
        
        config = EvalConfig(
            target_layer           = target_layer,
            n_samples              = 32,           # keep low for a quick test run
            batch_size             = 32,
            device                 = "cuda" if torch.cuda.is_available() else "cpu",
            output_dir             = output_dir,
            fidelity_features_per_step = 300,      # for a 3×224×224 image: 150528 features → ~500 features per step
            stability_n_perturbations = 5,
            stability_noise_std    = 0.05,
            separability_n_pairs   = 20,
            separability_eps       = 1e-3,      # TODO In report we should argue for why this amount. 
            shap_background           = shap_background,
            shap_explain_probability  = False,         # True = explain sigmoid probs
        )

        try:
            results = run_pipeline(model, test_dataset, methods_to_evaluate, config, idx_to_class)
        except Exception as exc:
            print(f"  [ERROR] Pipeline failed for {fname}: {exc}")
            import traceback; traceback.print_exc()
            continue

        # Free GPU memory before loading the next model
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()