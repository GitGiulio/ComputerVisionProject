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
from torchvision import transforms
from counterfactual import generate_counterfactual


def _try_import(module_name: str, pip_name: str):
    """Lazily import an optional dependency with a helpful error if missing."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Required package '{pip_name}' is not installed. "
            f"Run: pip install {pip_name}"
        )

@dataclass
class EvalConfig:
    """Central configuration for one evaluation run.

    Args:
        interpretability_method: One of 'lime', 'shap', 'gradcam', 'gradcam++',
                    'intgrad', 'smoothgrad'.
        target_layer: The nn.Module layer used by gradient-based methods
                      (GradCAM, GradCAM++). Not needed for LIME/SHAP/IntGrad.
        n_samples: Number of images to evaluate on.
        batch_size: DataLoader batch size.
        device: 'cuda' or 'cpu'.
        output_dir: Folder where plots and CSV are saved.
        # Metric-specific knobs
        fidelity_steps: Number of masking steps for deletion/insertion AUC.
        stability_n_perturbations: Number of noisy copies per image.
        stability_noise_std: Std of Gaussian noise added for stability test.
        separability_n_pairs: Number of cross-class image pairs to check.
        separability_eps: Minimum L2 distance to consider two maps different.
        smoothgrad_n_samples: Noise samples averaged in SmoothGrad.
        smoothgrad_noise_std: Noise std for SmoothGrad.
        lime_n_segments: Number of superpixel segments for LIME.
        lime_n_samples: Number of perturbed samples for LIME surrogate.
    """
    interpretability_method: str = "gradcam"
    target_layer: Optional[torch.nn.Module] = None
    n_samples: int = 50
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./interpretability_results"
    # Fidelity
    fidelity_steps: int = 20
    # Stability
    stability_n_perturbations: int = 10
    stability_noise_std: float = 0.05
    # Separability
    separability_n_pairs: int = 30
    separability_eps: float = 1e-3
    # SmoothGrad
    smoothgrad_n_samples: int = 50
    smoothgrad_noise_std: float = 0.15
    
    # Counterfactual
    counterfactual_steps: int = 200
    counterfactual_lam: float = 0.5


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
    deletion_auc: float = 0.0
    insertion_auc: float = 0.0
    stability: float = 0.0
    identity: float = 0.0
    separability: float = 0.0
    avg_time_sec: float = 0.0
    raw: dict = field(default_factory=dict)



class Explainer:
    """Wraps multiple Interpretability backends behind a single `.explain()` interface.

    All methods return a saliency map as a numpy array of shape (H, W), normalised to [0, 1].

    Args:
        method: Interpretability method name.
        model: PyTorch model in eval mode.
        config: EvalConfig instance.
    """

    SUPPORTED = {"shap", "gradcam","counterfactuals"} #OTHERS:  "lime", "gradcam++", "intgrad", "smoothgrad"}

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
        """Generate a saliency map for a single image.

        Args:
            image_tensor: Float tensor of shape (C, H, W), already normalised.
            label: Ground-truth (or predicted) class index.

        Returns:
            2-D numpy array of shape (H, W) with values in [0, 1].
        """
        dispatch = {
            "shap":      self._explain_shap,
            "gradcam":   self._explain_gradcam,
            "counterfactuals":   self._explain_counterfactuals,
        }
        return dispatch[self.method](image_tensor, label)


    def _explain_gradcam(self, img: torch.Tensor, label: int) -> np.ndarray:
        """GradCAM: gradient-weighted average of conv feature maps.

        Args:
            img: Image tensor (C, H, W).
            label: Target class index.

        Returns:
            Normalised saliency map (H, W).
        """
        x = img.unsqueeze(0).to(self.device).requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        out[0, label].backward()

        # Pool gradients over spatial dims -> channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()   # (H', W')
        cam = F.relu(cam)
        return _resize_and_normalise(cam, img.shape[1:])

    def _explain_shap(self, img: torch.Tensor, label: int) -> np.ndarray:
        """GradientSHAP: SHAP values via expected gradients w.r.t. baselines.

        Args:
            img: Image tensor (C, H, W).
            label: Target class index.

        Returns:
            Normalised saliency map (H, W).
        """
        captum = _try_import("captum.attr", "captum")
        gs = captum.GradientShap(self.model)
        x = img.unsqueeze(0).to(self.device)
        # Baseline: zero tensor (black image)
        baseline = torch.zeros_like(x)
        attrs = gs.attribute(x, baselines=baseline, target=label)
        sal = attrs.squeeze().abs().sum(dim=0).cpu().numpy()
        return _normalise(sal)
    
    def _explain_counterfactuals(self, img: torch.Tensor, label: int) -> np.ndarray:
        """Counterfactuals: we can use it to evaluate contribution of each pixel to the final decision

        Args:
            img: Image tensor (C, H, W).
            label: Target class index.

        Returns:
            Normalised saliency map (H, W).
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        target_class = 1 - label
        x = img.unsqueeze(0).to(self.device)
        
        x_cf = generate_counterfactual(
            self.model, x, target_class,
            steps=self.config.counterfactual_steps,
            lam=self.config.counterfactual_lam,
        )
        
        diff = (x_cf - x).abs().mean(dim=1).squeeze()
        
        # Does it flip the class check
        # with torch.no_grad():
        #     orig_prob = torch.sigmoid(self.model(x)).item()
        #     cf_prob = torch.sigmoid(self.model(x_cf)).item()
        #     print(f"Original: {orig_prob:.3f} → Counterfactual: {cf_prob:.3f}")
            
        return _normalise(diff.cpu().numpy())
    
        


def compute_fidelity(
    model: torch.nn.Module,
    image: torch.Tensor,
    saliency: np.ndarray,
    label: int,
    steps: int,
    device: torch.device,
) -> tuple[float, float]:
    """Compute Deletion AUC and Insertion AUC for one image.

    Deletion: progressively mask the most important pixels -> score should drop.
              Lower AUC is better (the explanation correctly identifies key pixels).

    Insertion: start from a fully masked image and reveal pixels in importance
               order -> score should rise quickly. Higher AUC is better.

    Args:
        model: PyTorch model in eval mode.
        image: Image tensor (C, H, W).
        saliency: Saliency map (H, W), values in [0, 1].
        label: Target class index.
        steps: Number of masking steps.
        device: Torch device.

    Returns:
        Tuple of (deletion_auc, insertion_auc), both in [0, 1].
    """
    C, H, W = image.shape
    n_pixels = H * W

    # Flatten saliency and sort pixel indices from most to least important
    flat_sal = saliency.flatten()
    ranked = np.argsort(flat_sal)[::-1]  # descending importance

    deletion_scores = []
    insertion_scores = []

    for step in range(steps + 1):
        n_masked = int(step / steps * n_pixels)
        mask = np.ones(n_pixels, dtype=np.float32)
        mask[ranked[:n_masked]] = 0.0          # deletion: blank top pixels
        mask_tensor = torch.tensor(mask.reshape(1, H, W)).to(device)

        # --- Deletion ---
        deleted = image.to(device) * mask_tensor
        with torch.no_grad():
            # score_del = F.softmax(model(deleted.unsqueeze(0)), dim=1)[0, label].item()
            score_del = torch.sigmoid(model(deleted.unsqueeze(0))).item() # Binary output
            
        deletion_scores.append(score_del)

        # --- Insertion: reveal top pixels on a blurred/black baseline ---
        inserted = image.to(device) * (1 - mask_tensor)  # only top pixels visible
        with torch.no_grad():
            # score_ins = F.softmax(model(inserted.unsqueeze(0)), dim=1)[0, label].item()
            score_ins = torch.sigmoid(model(inserted.unsqueeze(0))).item()  # Binary output
            
        insertion_scores.append(score_ins)

    # AUC via trapezoidal integration over evenly-spaced steps
    xs = np.linspace(0, 1, steps + 1)
    del_auc = float(np.trapezoid(deletion_scores, xs))
    ins_auc = float(np.trapezoid(insertion_scores, xs))
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

    Args:
        explainer: Explainer instance.
        images: List of image tensors.
        labels: Corresponding class labels.
        n_pairs: Number of pairs to evaluate.
        eps: Minimum L2 distance between maps to count as "separable".

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


# ===========================================================================
# VISUALISATION
# ===========================================================================

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


def _resize_and_normalise(cam: torch.Tensor, target_hw: tuple) -> np.ndarray:
    """Upsample a CAM tensor to target spatial size and normalise it.

    Args:
        cam: 2-D tensor (H', W') from a conv layer.
        target_hw: Desired (H, W) output size.

    Returns:
        Normalised numpy array of shape target_hw.
    """
    cam_np = cam.cpu().numpy()
    # Upsample using PIL-style zoom
    from scipy.ndimage import zoom
    zoom_h = target_hw[0] / cam_np.shape[0]
    zoom_w = target_hw[1] / cam_np.shape[1]
    resized = zoom(cam_np, (zoom_h, zoom_w), order=1)
    return _normalise(resized)


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
        saliency_maps: Corresponding saliency maps (H, W).
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

        axes[1, i].imshow(saliency_maps[i], cmap="hot")
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
              "identity", "separability", "avg_time_sec"]
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
            })
    print(f"  Saved: {save_path}")


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
    subset = Subset(dataset, list(range(n_samples)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

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
        Tuple of (MetricResults, list of saliency maps for visualisation).
    """
    device = torch.device(config.device)
    model = model.to(device).eval()

    explainer = Explainer(method_name, model, config)

    del_aucs, ins_aucs, stab_scores, id_scores, times = [], [], [], [], []
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
                                         config.fidelity_steps, device)
        del_aucs.append(d_auc)
        ins_aucs.append(i_auc)

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
) -> dict[str, MetricResults]:
    """Run the full evaluation pipeline for one or more interpretability methods.

    Saves all plots and a CSV to config.output_dir.

    Args:
        model: PyTorch model in eval mode.
        dataset: PyTorch Dataset returning (image_tensor, label).
        methods: List of Interpretability method names to evaluate.
        config: EvalConfig with all hyperparameters.

    Returns:
        Dict mapping method name -> MetricResults.
    """
    import os
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Interpretability Evaluation Pipeline")
    print(f"  Device : {config.device}")
    print(f"  Samples: {config.n_samples}")
    print(f"  Methods: {', '.join(m.upper() for m in methods)}")
    print(f"{'='*60}")

    images, labels = load_samples(dataset, config.n_samples, config.batch_size)
    all_results: dict[str, MetricResults] = {}
    all_saliency: dict[str, list[np.ndarray]] = {}

    for method in methods:
        results, sal_maps = evaluate_method(method, model, images, labels, config)
        all_results[method] = results
        all_saliency[method] = sal_maps

        # Per-method saliency grid
        plot_saliency_grid(
            images, sal_maps, method,
            save_path=f"{config.output_dir}/saliency_{method}.png"
        )

    # --- Aggregate plots (only meaningful with >1 method) ---
    if len(all_results) > 1:
        plot_radar_chart(all_results,
                         save_path=f"{config.output_dir}/radar_chart.png")
        plot_bar_charts(all_results,
                        save_path=f"{config.output_dir}/bar_charts.png")

    save_csv(all_results, save_path=f"{config.output_dir}/metrics_summary.csv")
    print_summary_table(all_results)

    return all_results


if __name__ == "__main__":
    import os
    import torchvision
    from shared_code import CIFAKE_CNN,I_HAVE_A_THEORY, data_loaders,get_tensor_transform,SafeImageFolder
    from Grad_cam_visual import find_last_conv_layer

    DATA_DIR = "/mnt/scratch/Stable_diffusion/Stable_diffusion_ready"
    DEVICE =  f"cuda:1" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    KERNEL_SIZE,CONV_FILTER, CONV_LAYER, DENSE_NEURON, DENSE_LAYER = 0,32,2,64,1

    model = CIFAKE_CNN(CONV_FILTER, CONV_LAYER, DENSE_NEURON, DENSE_LAYER).to(DEVICE)
    MODEL_PATH = "/home/cv04f26/ComputerVisionProject/models/model_32_2_64_1.pth"
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict) 
    model.eval()

    target_layer = find_last_conv_layer(model)

   # train_loader, test_loader, idx_to_class = data_loaders(DEVICE,BATCH_SIZE)
    val_dataset = SafeImageFolder(os.path.join(DATA_DIR, "val"),   transform=get_tensor_transform())

    methods_to_evaluate = ["gradcam", "shap", "counterfactuals"]#, "counterfactuals(TODO)"]

    config = EvalConfig(
        interpretability_method= "shap",
        target_layer           = target_layer,
        n_samples              = 100,           # keep low for a quick test run
        batch_size             = 8,
        device                 = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir             = "./interpretability_results",
        fidelity_steps         = 20,
        stability_n_perturbations = 5,
        stability_noise_std    = 0.05,
        separability_n_pairs   = 20,
        separability_eps       = 1e-3,      # TODO In report we should argue for why this amount. 
        smoothgrad_n_samples   = 30,
        smoothgrad_noise_std   = 0.15,
        counterfactual_steps   = 100,
        counterfactual_lam     = 0.5
    )

    results = run_pipeline(model, val_dataset, methods_to_evaluate, config)