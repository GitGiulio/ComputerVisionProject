"""
Fine-tunes a pre-trained I_HAVE_A_THEORY model with a combined loss:

    L = (1 - alpha) * BCE_loss  +  alpha * (1 - insertion_AUC)

The insertion AUC term is computed on a small fixed fidelity batch once per
epoch, using SHAP attributions as a fixed pixel-ranking signal.
"""

import os
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import shap 

from shared_code import (
    SafeImageFolder,
    collate_skip_none,
    data_loaders,
    I_HAVE_A_THEORY,
    get_tensor_transform,
)
from full_metrics_pipeline import (
    EvalConfig,
    Explainer,
    ShapBinaryWrapper,
    collect_shap_background,
)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = "./DATA/Cat_dog_splitted/"
MODEL_PATH  = "./models/model_kernel=[5,9,15]_32_3_512_1_wd0.0001_do0.0.pth"
OUT_PATH    = "./models/finetuned_fidelity_k=[5,9,15]_32_3_512_1_wd0.0001_do0.0.pth"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Fine-tuning schedule
LR              = 1e-4      # lower than normal training — we're fine-tuning
EPOCHS          = 20
BATCH_SIZE      = 64
EARLY_STOP_PAT  = 5

# Fidelity loss settings
ALPHA               = 0.2   # weight of the fidelity term  (0 = pure BCE, 1 = pure fidelity)
FIDELITY_BATCH_SIZE = 32     # number of images used for the insertion AUC gradient each epoch
INSERTION_STEPS     = 30    # number of masking steps in the differentiable insertion curve

# ── Helpers ───────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = -1.0
        self.best_state = None
        self.counter    = 0

    def step(self, val_acc: float, model: nn.Module) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc   = val_acc
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def evaluate(model: nn.Module, loader: DataLoader, split_name: str = "Val"):
    """Standard accuracy/precision/recall/F1 evaluation. Returns accuracy."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        with autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            for batch in loader:
                if batch is None:
                    continue
                images, labels, _ = batch
                images = images.to(DEVICE, non_blocking=True)
                probs  = torch.sigmoid(model(images))
                predictions = (probs.float().cpu().numpy() > 0.5).astype(int)
                preds.extend(predictions.flatten())
                trues.extend(labels.numpy().flatten())

    if not preds:
        print(f"  [{split_name}] No samples.")
        return 0.0

    acc  = float(np.mean(np.array(preds) == np.array(trues)))
    prec = precision_score(trues, preds, zero_division=0)
    rec  = recall_score(trues, preds, zero_division=0)
    f1   = f1_score(trues, preds, zero_division=0)
    print(f"  [{split_name}] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    return acc


def compute_shap_saliency(
    model: nn.Module,
    image: torch.Tensor,
    background: torch.Tensor,
    explain_probability: bool = False,
) -> np.ndarray:
    """Run SHAP DeepExplainer and return a (H, W) saliency map.

    The map is the absolute-value sum across colour channels, normalised to
    [0, 1].  We take the absolute value because the insertion metric only
    needs a pixel *ranking*, not signed attribution values — we want to reveal
    the pixels that the model cares about most, regardless of direction.

    Args:
        model: The model to explain, in eval mode.
        image: Single input image tensor (C, H, W) on CPU.
        background: Background reference tensor (N, C, H, W) on CPU.
        explain_probability: If True, wrap model with sigmoid for SHAP.

    Returns:
        Normalised saliency map (H, W) with values in [0, 1].
    """
    wrapped = ShapBinaryWrapper(
            model.cpu(),
            explain_probability=explain_probability
        ).eval()

    explainer = shap.DeepExplainer(wrapped, background.cpu())

    x = img.unsqueeze(0)
    shap_vals = explainer.shap_values(x)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    shap_arr = np.array(shap_vals)  # (1, C, H, W) or (1, H, W, C)

    # Normalise axis order to (1, C, H, W)
    if shap_arr.ndim == 5 and shap_arr.shape[-1] == 1:
        shap_arr = shap_arr[..., 0]
    if shap_arr.ndim == 4 and shap_arr.shape[-1] in [1, 3]:
        # (1, H, W, C) → (1, C, H, W)
        shap_arr = np.transpose(shap_arr, (0, 3, 1, 2))

    # shap_arr[0] is now (C, H, W)
    chw = shap_arr[0].astype(np.float32)

    # Symmetric normalisation by global max-abs so range is [-1, 1] (whe rank based on abs were needed)
    max_abs = np.max(np.abs(chw)) + 1e-8
    return (chw / max_abs).astype(np.float32)  # (C, H, W)

def differentiable_insertion_auc(
    model: nn.Module,
    image: torch.Tensor,       # (C, H, W) on DEVICE, no batch dim
    saliency: np.ndarray,      # (C,H, W), values in [0,1], used only for ranking
    label: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute insertion AUC in a way that allows gradients to flow to `model`.

    The saliency map is used purely as a fixed pixel ranking (detached from
    the computation graph).  At each step we reveal the top-k pixels from a
    zero baseline and run a differentiable forward pass.  The scores are
    collected as a differentiable tensor and integrated with torch.trapezoid.

    The returned value is in [0, 1].  Maximising it means the model assigns
    high confidence to the target class as soon as the most-important pixels
    are revealed, which is exactly what good fidelity means.

    Args:
        model: Model in train mode (gradients will flow through it).
        image: Single image (C, H, W) on `device`.
        saliency: Numpy array (H, W) — pixel ranking, detached from graph.
        label: Ground-truth class index (0 or 1).
        steps: Number of masking steps (more = smoother AUC, slower).
        device: Torch device.

    Returns:
        Scalar tensor: the insertion AUC, differentiable w.r.t. model params.
    """
    C, H, W = image.shape
    n_features = C * H * W

    ranked = np.argsort(saliency.flatten())[::-1].copy()
    ranked_t = torch.from_numpy(ranked).long()           

    ranked_t = ranked_t.clamp(0, n_features - 1)

    scores = []
    for step in range(steps + 1):
        n_revealed = int(step / steps * n_features)

        # Build a binary mask: 1 where pixels are revealed, 0 elsewhere
        mask = torch.zeros(n_features, device=device)
        if n_revealed > 0:
            mask[ranked_t[:n_revealed].to(device)] = 1.0
        mask = mask.view(C, H, W)                         # (1, H, W) for broadcast

        # Revealed image: top pixels shown, rest stay at zero baseline
        revealed = image * mask                            # (C, H, W) differentiable

        logit = model(revealed.unsqueeze(0))               # (1, 1)
        prob  = torch.sigmoid(logit).squeeze()             # scalar

        # Score for the target class
        score = prob if label == 1 else (1.0 - prob)
        scores.append(score)

    # Stack into (steps+1,) tensor and integrate — differentiable all the way
    score_tensor = torch.stack(scores)                     # (steps+1,)
    xs = torch.linspace(0, 1, steps + 1, device=device)
    auc = torch.trapezoid(score_tensor, xs)                # scalar tensor

    return auc


def compute_fidelity_loss(
    model: nn.Module,
    fidelity_images: list[torch.Tensor],   # list of (C,H,W) CPU tensors
    fidelity_labels: list[int],
    background: torch.Tensor,              # (N, C, H, W) CPU
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute the mean (1 - insertion_AUC) over the fidelity batch.

    SHAP saliency is recomputed each epoch so it tracks the evolving model.
    The model is temporarily moved to CPU for SHAP (DeepExplainer requires
    both model and data on the same device, and SHAP works on CPU).
    After SHAP, the model is moved back to `device` for the differentiable
    insertion pass.

    Args:
        model: The model being fine-tuned.
        fidelity_images: Small fixed list of images (CPU tensors, (C,H,W)).
        fidelity_labels: Corresponding integer labels.
        background: SHAP background tensor (CPU).
        steps: Insertion curve resolution.
        device: Training device.

    Returns:
        Scalar tensor = mean(1 - insertion_AUC) over the fidelity batch.
        Minimising this is equivalent to maximising insertion AUC.
    """
    fidelity_losses = []

    for img_cpu, label in zip(fidelity_images, fidelity_labels):

        model.eval()
        saliency = compute_shap_saliency(model, img_cpu, background,explain_probability=False)

        model.to(device).train()
        img_dev = img_cpu.to(device)

        ins_auc = differentiable_insertion_auc(
            model, img_dev, saliency, label, steps, device
        )

        fidelity_losses.append(1.0 - ins_auc)   # minimise → maximise AUC

    return torch.stack(fidelity_losses).mean()


# ── Fine-tuning epoch ─────────────────────────────────────────────────────────

def finetune_one_epoch(
    epoch: int,
    epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    fidelity_images: list[torch.Tensor],
    fidelity_labels: list[int],
    background: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    alpha: float,
    insertion_steps: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """One fine-tuning epoch.  Returns (avg_total_loss, avg_bce, avg_fid, train_acc).

    The epoch has two phases:

    Phase A — normal mini-batch SGD over the full training set (BCE loss).
    Phase B — one gradient step using only the fidelity loss on the fixed
              fidelity batch.  We do a *separate* backward pass so the
              expensive SHAP computation happens only once per epoch, not
              once per mini-batch.

    Both phases use the same optimizer, so the parameter updates from both
    loss terms accumulate within the same epoch.  Phase A runs first so the
    model is already in a reasonable state when we compute SHAP for Phase B.
    """
    running_bce  = 0.0
    correct = 0
    total   = 0

    # ── Phase A: standard mini-batch training ─────────────────────────────────
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [BCE]", leave=False)
    for batch in loop:
        if batch is None:
            continue
        images, labels, _ = batch
        images   = images.to(device, non_blocking=True)
        labels_f = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            logits   = model(images)
            bce_loss = criterion(logits, labels_f)

            # Scale the BCE contribution by (1 - alpha)
            total_loss = (1.0 - alpha) * bce_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_bce += bce_loss.item()
        with torch.no_grad():
            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).long().cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_bce   = running_bce / max(total // max(len(train_loader), 1), 1)
    train_acc = correct / max(total, 1)

    # ── Phase B: fidelity loss gradient step ─────────────────────────────────
    # We do a single backward on the mean fidelity loss across the fidelity batch.
    t0 = time.perf_counter()
    print(f"  Computing SHAP + insertion AUC for {len(fidelity_images)} images …", end=" ", flush=True)

    optimizer.zero_grad(set_to_none=True)
    fid_loss = compute_fidelity_loss(
        model, fidelity_images, fidelity_labels,
        background, insertion_steps, device,
    )
    fid_term = alpha * fid_loss
    scaler.scale(fid_term).backward()
    scaler.step(optimizer)
    scaler.update()

    elapsed = time.perf_counter() - t0
    avg_fid = fid_loss.item()
    print(f"done in {elapsed:.1f}s  |  fidelity_loss={avg_fid:.4f}  ins_AUC≈{1-avg_fid:.4f}")

    avg_total = (1.0 - alpha) * avg_bce + alpha * avg_fid
    return avg_total, avg_bce, avg_fid, train_acc


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"Device : {DEVICE}")
    print(f"Alpha  : {ALPHA}  (fidelity weight)")
    print(f"Fidelity batch size : {FIDELITY_BATCH_SIZE}")
    print(f"Insertion steps     : {INSERTION_STEPS}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_dataset, idx_to_class = data_loaders(DEVICE, BATCH_SIZE)

    train_dataset = SafeImageFolder(
        os.path.join(DATA_DIR, "train"), transform=get_tensor_transform()
    )

    print("\nCollecting SHAP background …")
    background = collect_shap_background(train_dataset, n_background=50,
                                         batch_size=25) 

    # We draw a small random subset of the validation set so the fidelity signal
    # is independent of the training data, this avoids overfitting the fidelity
    # term to training images the model has already memorized.
    print(f"\nSampling {FIDELITY_BATCH_SIZE} validation images for fidelity batch …")
    val_dataset = SafeImageFolder(
        os.path.join(DATA_DIR, "val"), transform=get_tensor_transform()
    )
    

    KERNEL_SIZE,CONV_FILTER, CONV_LAYER, DENSE_NEURON, DENSE_LAYER = 15, 32, 3, 512, 1
    model = I_HAVE_A_THEORY(KERNEL_SIZE, CONV_FILTER, CONV_LAYER, DENSE_NEURON, DENSE_LAYER).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"\nLoaded checkpoint: {MODEL_PATH}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler(enabled=(DEVICE == "cuda"))
    early_stop = EarlyStopping(patience=EARLY_STOP_PAT)

    print(f"\n{'='*65}")
    print(f"  Starting fidelity-aware fine-tuning")
    print(f"{'='*65}\n")

    for epoch in range(EPOCHS):
        rng_idxs = torch.randperm(len(val_dataset))[:FIDELITY_BATCH_SIZE * 4].tolist()
        fidelity_images: list[torch.Tensor] = []
        fidelity_labels: list[int] = []
        for idx in rng_idxs:
            sample = val_dataset[idx]
            if sample is None:
                continue
            img, lbl, _ = sample
            fidelity_images.append(img.float()) 
            fidelity_labels.append(int(lbl))
            if len(fidelity_images) == FIDELITY_BATCH_SIZE:
                break
        print(f"  Fidelity batch ready: {len(fidelity_images)} images, "
            f"labels={fidelity_labels}")
        avg_total, avg_bce, avg_fid, train_acc = finetune_one_epoch(
            epoch=epoch,
            epochs=EPOCHS,
            model=model,
            train_loader=train_loader,
            fidelity_images=fidelity_images,
            fidelity_labels=fidelity_labels,
            background=background,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            alpha=ALPHA,
            insertion_steps=INSERTION_STEPS,
            device=torch.device(DEVICE),
        )

        val_acc = evaluate(model, val_loader, split_name="Val")
        gap = train_acc - val_acc
        overfit_flag = " overfit!" if gap > 0.10 else ""

        print(
            f"  Epoch {epoch+1:>3}/{EPOCHS} | "
            f"total={avg_total:.4f}  bce={avg_bce:.4f}  fid={avg_fid:.4f} | "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  gap={gap:+.4f}{overfit_flag}"
        )

        torch.save(model.state_dict(), OUT_PATH)

        if early_stop.step(val_acc, model):
            print(f"\n  Early stopping, best val_acc={early_stop.best_acc:.4f}")
            break

    early_stop.restore_best(model)
    torch.save(model.state_dict(), OUT_PATH)
    print(f"\n  Best model saved -> {OUT_PATH}")
    evaluate(model, val_loader, split_name="Final Val")