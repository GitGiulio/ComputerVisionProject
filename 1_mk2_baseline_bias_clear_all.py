"""
Simple baseline-bias + original-performance check for Cat/Dog binary classifiers.

What this tests:
  1. Normal classification performance on original test images.
  2. Whether uninformative baseline inputs make the model systematically prefer Cat or Dog.

For a binary model:
  raw logit < 0  -> class 0
  raw logit = 0  -> neutral
  raw logit > 0  -> class 1

  sigmoid(logit) = probability-like score for class 1.

For ImageFolder, classes are alphabetical, so usually:
  class 0 = Cat
  class 1 = Dog

Baselines:
  black:
    one all-black image, checked once because every black baseline is identical.

  blur:
    each test image is fully blurred, then predicted.

  mean:
    each test image is replaced by its own per-channel mean RGB image, then predicted.

Outputs:
  baseline_bias_summary.csv
  baseline_bias_per_image.csv
  model_original_metrics.csv
  sampled_test_images.txt
"""

from __future__ import annotations

import csv
import inspect
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF

from shared_code import (
    SafeImageFolder,
    collate_skip_none,
    get_tensor_transform,
    I_HAVE_A_THEORY,
    CIFAKE_CNN,
)


# =============================================================================
# Settings
# =============================================================================

# DATA_DIR = "./Cat_dog_splitted"
DATA_DIR = "./Cat_dog_splitted_Mathijs_original"
TEST_DIR = os.path.join(DATA_DIR, "test")
# MODELS_DIR = "models_giulio"
MODELS_DIR = "models"
OUTPUT_DIR = "./baseline_bias_check"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# None = use all balanced test images.
# Example: 100 means use at most 100 Cat and 100 Dog images.
N_PER_CLASS_CAP: Optional[int] = None

BLUR_KERNEL_SIZE = 61
BLUR_SIGMA = 20.0

# If True, only original test images that the model classifies correctly are used.
# IMPORTANT:
#   Keep this False when you want real accuracy/confusion-matrix values.
#   If True, accuracy becomes artificially perfect/near-perfect.
ONLY_CORRECT_ORIGINALS = False

# If True, skip models already present in baseline_bias_summary.csv.
# If you changed this script and want fresh metrics, set False or delete old CSVs.
SKIP_ALREADY_DONE = True

# Toggle for evaluating only one model.
# Set to None to evaluate every parsable .pth model in MODELS_DIR.
# Example:
# ONLY_MODEL = "model_kernel=[5,9,23]_32_3_64_1_wd0.0001_do0.0.pth"
ONLY_MODEL: Optional[str] = None


@dataclass
class ModelSpec:
    fname: str
    path: str
    architecture: str
    kernel_size: Optional[int] = None
    conv_filter: Optional[int] = None
    conv_layer: Optional[int] = None
    dense_neuron: Optional[int] = None
    dense_layer: Optional[int] = None
    weight_decay: Optional[float] = None
    dropout: float = 0.0


def parse_model_filename(fname: str) -> Optional[ModelSpec]:
    """Parse project model filenames.

    Supports:
      model_kernel=[5,9,K]_CF_CL_DN_DL_wdWD_doDO.pth
      model_CF_CL_DN_DL.pth
    """
    stem = fname[:-4] if fname.endswith(".pth") else fname

    pattern_theory = (
        r"model_kernel=\[5,9,(\d+)\]"
        r"_(\d+)_(\d+)_(\d+)_(\d+)"
        r"(?:_wd([0-9eE+\-.]+))?"
        r"(?:_do([0-9eE+\-.]+))?$"
    )
    m = re.search(pattern_theory, stem)
    if m is not None:
        return ModelSpec(
            fname=fname,
            path=os.path.join(MODELS_DIR, fname),
            architecture="I_HAVE_A_THEORY",
            kernel_size=int(m.group(1)),
            conv_filter=int(m.group(2)),
            conv_layer=int(m.group(3)),
            dense_neuron=int(m.group(4)),
            dense_layer=int(m.group(5)),
            weight_decay=float(m.group(6)) if m.group(6) is not None else None,
            dropout=float(m.group(7)) if m.group(7) is not None else 0.0,
        )

    pattern_plain = (
        r"model_(\d+)_(\d+)_(\d+)_(\d+)"
        r"(?:_wd([0-9eE+\-.]+))?"
        r"(?:_do([0-9eE+\-.]+))?$"
    )
    m = re.search(pattern_plain, stem)
    if m is not None:
        return ModelSpec(
            fname=fname,
            path=os.path.join(MODELS_DIR, fname),
            architecture="CIFAKE_CNN",
            conv_filter=int(m.group(1)),
            conv_layer=int(m.group(2)),
            dense_neuron=int(m.group(3)),
            dense_layer=int(m.group(4)),
            weight_decay=float(m.group(5)) if m.group(5) is not None else None,
            dropout=float(m.group(6)) if m.group(6) is not None else 0.0,
        )

    return None


def call_model_constructor(cls, kwargs: dict) -> torch.nn.Module:
    """Call model constructor while ignoring unsupported keyword arguments.

    This makes the script robust if your local shared_code.py has or does not have
    arguments like dropout_rate.
    """
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


def build_model(spec: ModelSpec) -> torch.nn.Module:
    if spec.architecture == "I_HAVE_A_THEORY":
        model = call_model_constructor(
            I_HAVE_A_THEORY,
            {
                "kernel_size": spec.kernel_size,
                "conv_filters": spec.conv_filter,
                "conv_layers": spec.conv_layer,
                "dense_neurons": spec.dense_neuron,
                "dense_layers": spec.dense_layer,
                "dropout_rate": spec.dropout,
            },
        )
    elif spec.architecture == "CIFAKE_CNN":
        model = call_model_constructor(
            CIFAKE_CNN,
            {
                "conv_filters": spec.conv_filter,
                "conv_layers": spec.conv_layer,
                "dense_neurons": spec.dense_neuron,
                "dense_layers": spec.dense_layer,
                "dropout_rate": spec.dropout,
            },
        )
    else:
        raise ValueError(f"Unknown architecture: {spec.architecture}")

    state_dict = torch.load(spec.path, map_location=DEVICE)

    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


def class_balanced_indices(dataset: Dataset, n_per_class_cap: Optional[int]) -> dict[int, list[int]]:
    if not hasattr(dataset, "targets"):
        raise AttributeError("Dataset must expose .targets, as ImageFolder/SafeImageFolder does.")

    class_to_idxs: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_to_idxs[int(label)].append(idx)

    classes = sorted(class_to_idxs.keys())
    if len(classes) != 2:
        raise ValueError(f"Expected exactly 2 classes, found {classes}")

    min_count = min(len(class_to_idxs[c]) for c in classes)
    n_per_class = min_count if n_per_class_cap is None else min(min_count, n_per_class_cap)

    selected = {c: class_to_idxs[c][:n_per_class] for c in classes}
    print(f"Balanced sample size: {n_per_class} per class, {2 * n_per_class} total")
    return selected


def make_blur_batch(images: torch.Tensor) -> torch.Tensor:
    kernel_size = BLUR_KERNEL_SIZE + (BLUR_KERNEL_SIZE % 2 == 0)
    blurred = [
        TF.gaussian_blur(
            img,
            kernel_size=[kernel_size, kernel_size],
            sigma=[BLUR_SIGMA, BLUR_SIGMA],
        )
        for img in images
    ]
    return torch.stack(blurred, dim=0)


def make_mean_batch(images: torch.Tensor) -> torch.Tensor:
    means = images.mean(dim=(2, 3), keepdim=True)
    return means.expand_as(images)


def model_scores(model: torch.nn.Module, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return raw logits and class-1 probabilities."""
    images = images.to(DEVICE, non_blocking=True).float()
    with torch.no_grad():
        logits = model(images).reshape(-1)
        probs_class_1 = torch.sigmoid(logits)
    return (
        logits.detach().cpu().numpy(),
        probs_class_1.detach().cpu().numpy(),
    )


def evaluate_original_images(
    model: torch.nn.Module,
    dataset: Dataset,
    selected_by_class: dict[int, list[int]],
) -> dict:
    """
    Evaluate model on original, unmodified test images.

    For binary classes:
      class 0 = dataset.classes[0]
      class 1 = dataset.classes[1]

    Confusion matrix layout:
      [[tn, fp],
       [fn, tp]]
    """
    all_indices = []
    for cls in sorted(selected_by_class.keys()):
        all_indices.extend(selected_by_class[cls])

    loader = DataLoader(
        Subset(dataset, all_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_skip_none,
        pin_memory=DEVICE.startswith("cuda"),
    )

    y_true = []
    y_pred = []
    y_prob = []
    y_logit = []

    for batch in loader:
        if batch is None:
            continue

        images, labels, _paths = batch
        logits, probs = model_scores(model, images)
        preds = (probs >= 0.5).astype(int)

        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.astype(int).tolist())
        y_prob.extend(probs.astype(float).tolist())
        y_logit.extend(logits.astype(float).tolist())

    if len(y_true) == 0:
        raise RuntimeError("No valid original images available for evaluation.")

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)

    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    class_0_total = int((y_true_np == 0).sum())
    class_1_total = int((y_true_np == 1).sum())

    class_0_correct = int(((y_true_np == 0) & (y_pred_np == 0)).sum())
    class_1_correct = int(((y_true_np == 1) & (y_pred_np == 1)).sum())

    class_0_accuracy = class_0_correct / class_0_total if class_0_total > 0 else float("nan")
    class_1_accuracy = class_1_correct / class_1_total if class_1_total > 0 else float("nan")

    return {
        "original_n": len(y_true),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "precision_class_1": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "recall_class_1": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "f1_class_1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "class_0_total": class_0_total,
        "class_1_total": class_1_total,
        "class_0_correct": class_0_correct,
        "class_1_correct": class_1_correct,
        "class_0_accuracy": float(class_0_accuracy),
        "class_1_accuracy": float(class_1_accuracy),
        "mean_original_logit": float(np.mean(y_logit)),
        "mean_original_p_class_1": float(np.mean(y_prob)),
    }


def maybe_filter_correct_indices(
    dataset: Dataset,
    model: torch.nn.Module,
    selected_by_class: dict[int, list[int]],
) -> dict[int, list[int]]:
    if not ONLY_CORRECT_ORIGINALS:
        return selected_by_class

    filtered: dict[int, list[int]] = {}

    for cls, idxs in selected_by_class.items():
        correct = []
        loader = DataLoader(
            Subset(dataset, idxs),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_skip_none,
            pin_memory=DEVICE.startswith("cuda"),
        )

        running_offset = 0
        for batch in loader:
            if batch is None:
                continue

            images, labels, _paths = batch
            _logits, probs = model_scores(model, images)
            preds = (probs >= 0.5).astype(int)

            for j, (pred, label) in enumerate(zip(preds, labels.numpy())):
                if int(pred) == int(label):
                    correct.append(idxs[running_offset + j])

            running_offset += len(labels)

        filtered[cls] = correct

    min_correct = min(len(v) for v in filtered.values())
    if min_correct == 0:
        raise RuntimeError("No balanced correctly classified samples available.")

    if N_PER_CLASS_CAP is not None:
        min_correct = min(min_correct, N_PER_CLASS_CAP)

    filtered = {c: filtered[c][:min_correct] for c in filtered}
    print(f"Using correctly classified originals only: {min_correct} per class")
    return filtered


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }


def prediction_count_stats(p1_values: list[float]) -> dict:
    arr = np.array(p1_values, dtype=np.float64)
    pred_1 = arr >= 0.5

    n = int(arr.size)
    n_class_1 = int(pred_1.sum())
    n_class_0 = int(n - n_class_1)

    return {
        "baseline_pred_class_0_count": n_class_0,
        "baseline_pred_class_1_count": n_class_1,
        "baseline_pred_class_0_frac": n_class_0 / n if n > 0 else float("nan"),
        "baseline_pred_class_1_frac": n_class_1 / n if n > 0 else float("nan"),
    }


def predicted_side(mean_p_class_1: float, class_0_name: str, class_1_name: str) -> str:
    if mean_p_class_1 < 0.5:
        return f"{class_0_name} / class 0"
    if mean_p_class_1 > 0.5:
        return f"{class_1_name} / class 1"
    return "neutral"


def bias_strength(mean_p_class_1: float) -> str:
    distance = abs(mean_p_class_1 - 0.5)
    if distance >= 0.40:
        return "very strong"
    if distance >= 0.25:
        return "strong"
    if distance >= 0.10:
        return "moderate"
    if distance >= 0.05:
        return "weak"
    return "near neutral"


def append_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def already_done_models(summary_path: str) -> set[str]:
    if not os.path.isfile(summary_path):
        return set()

    done = set()
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if "model" in row:
                done.add(row["model"])
    return done


def print_result_line(
    baseline: str,
    original_class_name: str,
    stats_prob: dict[str, float],
    stats_logit: dict[str, float],
    pred_counts: dict,
    class_0_name: str,
    class_1_name: str,
) -> None:
    p1 = stats_prob["mean"]
    p0 = 1.0 - p1
    side = predicted_side(p1, class_0_name, class_1_name)
    strength = bias_strength(p1)

    print(
        f"  {baseline.upper():5s} on {original_class_name:>4s} images | "
        f"n={stats_prob['n']:>4d} | "
        f"avg p({class_0_name})={p0:.3f}, avg p({class_1_name})={p1:.3f} | "
        f"avg logit={stats_logit['mean']:+.3f} | "
        f"pred {class_0_name}={pred_counts['baseline_pred_class_0_count']}, "
        f"pred {class_1_name}={pred_counts['baseline_pred_class_1_count']} | "
        f"leans {side} ({strength})"
    )


def make_summary_row(
    spec: ModelSpec,
    original_metrics: dict,
    baseline: str,
    original_class_id,
    original_class_name: str,
    stats_logit: dict,
    stats_prob: dict,
    pred_counts: dict,
    class_0_name: str,
    class_1_name: str,
) -> dict:
    mean_p1 = stats_prob["mean"]
    mean_p0 = 1.0 - mean_p1

    return {
        "model": spec.fname,
        "architecture": spec.architecture,
        "kernel_size": spec.kernel_size,
        "conv_filter": spec.conv_filter,
        "conv_layer": spec.conv_layer,
        "dense_neuron": spec.dense_neuron,
        "dense_layer": spec.dense_layer,
        "weight_decay": spec.weight_decay,
        "dropout": spec.dropout,

        **original_metrics,

        "baseline": baseline,
        "original_class_id": original_class_id,
        "original_class_name": original_class_name,
        "n": stats_prob["n"],
        "mean_logit": stats_logit["mean"],
        "std_logit": stats_logit["std"],
        "min_logit": stats_logit["min"],
        "max_logit": stats_logit["max"],
        "mean_p_class_0": mean_p0,
        "mean_p_class_1": mean_p1,
        "std_p_class_1": stats_prob["std"],
        "min_p_class_1": stats_prob["min"],
        "max_p_class_1": stats_prob["max"],

        **pred_counts,

        "neutral_distance": abs(mean_p1 - 0.5),
        "cat_bias_if_class_0_cat": 0.5 - mean_p1,
        "dog_bias_if_class_1_dog": mean_p1 - 0.5,
        "predicted_baseline_side": predicted_side(mean_p1, class_0_name, class_1_name),
        "bias_strength": bias_strength(mean_p1),
    }


def run_for_model(spec: ModelSpec, dataset: Dataset, base_selected: dict[int, list[int]]) -> None:
    print("\n" + "=" * 88)
    print(f"Model: {spec.fname}")
    print("=" * 88)

    class_0_name = dataset.classes[0]
    class_1_name = dataset.classes[1]

    try:
        model = build_model(spec)
        print(f"[OK] Loaded model: {spec.fname}")

    except Exception as e:
        print(f"[CORRUPT/SKIPPED] Could not load model: {spec.fname}")
        print(f"  Path: {spec.path}")
        print(f"  Reason: {type(e).__name__}: {e}")
        return
    
    selected_by_class = maybe_filter_correct_indices(dataset, model, base_selected)
    classes = sorted(selected_by_class.keys())

    original_metrics = evaluate_original_images(
        model=model,
        dataset=dataset,
        selected_by_class=selected_by_class,
    )

    print("\nOriginal test-image performance:")
    print(f"  accuracy           = {original_metrics['accuracy']:.4f}")
    print(f"  balanced accuracy  = {original_metrics['balanced_accuracy']:.4f}")
    print(f"  precision class 1  = {original_metrics['precision_class_1']:.4f}")
    print(f"  recall class 1     = {original_metrics['recall_class_1']:.4f}")
    print(f"  f1 class 1         = {original_metrics['f1_class_1']:.4f}")
    print(
        "  confusion matrix   = "
        f"tn={original_metrics['tn']}, "
        f"fp={original_metrics['fp']}, "
        f"fn={original_metrics['fn']}, "
        f"tp={original_metrics['tp']}"
    )
    print(f"  {class_0_name} accuracy = {original_metrics['class_0_accuracy']:.4f}")
    print(f"  {class_1_name} accuracy = {original_metrics['class_1_accuracy']:.4f}")

    summary_rows = []
    per_image_rows = []

    print("\nBaseline prediction meaning:")
    print(f"  p({class_1_name}) = sigmoid(logit). p({class_0_name}) = 1 - p({class_1_name}).")
    print(f"  logit < 0 leans {class_0_name}; logit > 0 leans {class_1_name}; logit = 0 is neutral.\n")

    # -------------------------------------------------------------------------
    # Black baseline: one all-black image is enough.
    # -------------------------------------------------------------------------
    sample_img, _, _ = dataset[0]
    black = torch.zeros_like(sample_img).unsqueeze(0)
    black_logit_arr, black_p1_arr = model_scores(model, black)

    black_logit = float(black_logit_arr[0])
    black_p1 = float(black_p1_arr[0])
    black_p0 = 1.0 - black_p1

    black_stats_logit = {
        "n": 1,
        "mean": black_logit,
        "std": 0.0,
        "min": black_logit,
        "max": black_logit,
    }
    black_stats_prob = {
        "n": 1,
        "mean": black_p1,
        "std": 0.0,
        "min": black_p1,
        "max": black_p1,
    }
    black_pred_counts = prediction_count_stats([black_p1])

    print(
        f"  BLACK once           | "
        f"avg p({class_0_name})={black_p0:.3f}, avg p({class_1_name})={black_p1:.3f} | "
        f"logit={black_logit:+.3f} | "
        f"leans {predicted_side(black_p1, class_0_name, class_1_name)} ({bias_strength(black_p1)})"
    )

    summary_rows.append(
        make_summary_row(
            spec=spec,
            original_metrics=original_metrics,
            baseline="black",
            original_class_id="all",
            original_class_name="ALL",
            stats_logit=black_stats_logit,
            stats_prob=black_stats_prob,
            pred_counts=black_pred_counts,
            class_0_name=class_0_name,
            class_1_name=class_1_name,
        )
    )

    # -------------------------------------------------------------------------
    # Blur and mean baselines: per original class + pooled ALL.
    # -------------------------------------------------------------------------
    all_values = {
        "blur": {"logit": [], "p1": []},
        "mean": {"logit": [], "p1": []},
    }

    for cls in classes:
        class_name = dataset.classes[cls]
        idxs = selected_by_class[cls]

        loader = DataLoader(
            Subset(dataset, idxs),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_skip_none,
            pin_memory=DEVICE.startswith("cuda"),
        )

        values = {
            "blur": {"logit": [], "p1": [], "path": []},
            "mean": {"logit": [], "p1": [], "path": []},
        }

        for batch in loader:
            if batch is None:
                continue

            images, labels, paths = batch

            blur_logits, blur_probs = model_scores(model, make_blur_batch(images))
            mean_logits, mean_probs = model_scores(model, make_mean_batch(images))

            for path, logit, p1 in zip(paths, blur_logits, blur_probs):
                values["blur"]["logit"].append(float(logit))
                values["blur"]["p1"].append(float(p1))
                values["blur"]["path"].append(path)

            for path, logit, p1 in zip(paths, mean_logits, mean_probs):
                values["mean"]["logit"].append(float(logit))
                values["mean"]["p1"].append(float(p1))
                values["mean"]["path"].append(path)

        for baseline in ["blur", "mean"]:
            all_values[baseline]["logit"].extend(values[baseline]["logit"])
            all_values[baseline]["p1"].extend(values[baseline]["p1"])

            stats_logit = summarize(values[baseline]["logit"])
            stats_prob = summarize(values[baseline]["p1"])
            pred_counts = prediction_count_stats(values[baseline]["p1"])

            print_result_line(
                baseline=baseline,
                original_class_name=class_name,
                stats_prob=stats_prob,
                stats_logit=stats_logit,
                pred_counts=pred_counts,
                class_0_name=class_0_name,
                class_1_name=class_1_name,
            )

            summary_rows.append(
                make_summary_row(
                    spec=spec,
                    original_metrics=original_metrics,
                    baseline=baseline,
                    original_class_id=cls,
                    original_class_name=class_name,
                    stats_logit=stats_logit,
                    stats_prob=stats_prob,
                    pred_counts=pred_counts,
                    class_0_name=class_0_name,
                    class_1_name=class_1_name,
                )
            )

            for path, logit, p1 in zip(
                values[baseline]["path"],
                values[baseline]["logit"],
                values[baseline]["p1"],
            ):
                per_image_rows.append({
                    "model": spec.fname,
                    "architecture": spec.architecture,
                    "kernel_size": spec.kernel_size,
                    "conv_filter": spec.conv_filter,
                    "conv_layer": spec.conv_layer,
                    "dense_neuron": spec.dense_neuron,
                    "dense_layer": spec.dense_layer,
                    "weight_decay": spec.weight_decay,
                    "dropout": spec.dropout,
                    "baseline": baseline,
                    "original_class_id": cls,
                    "original_class_name": class_name,
                    "logit": logit,
                    "p_class_0": 1.0 - p1,
                    "p_class_1": p1,
                    "predicted_class_id": int(p1 >= 0.5),
                    "predicted_class_name": class_1_name if p1 >= 0.5 else class_0_name,
                    "path": path,
                })

    print("\nCombined baseline results, using Cat + Dog images together:")
    for baseline in ["blur", "mean"]:
        stats_logit = summarize(all_values[baseline]["logit"])
        stats_prob = summarize(all_values[baseline]["p1"])
        pred_counts = prediction_count_stats(all_values[baseline]["p1"])

        print_result_line(
            baseline=baseline,
            original_class_name="ALL",
            stats_prob=stats_prob,
            stats_logit=stats_logit,
            pred_counts=pred_counts,
            class_0_name=class_0_name,
            class_1_name=class_1_name,
        )

        summary_rows.append(
            make_summary_row(
                spec=spec,
                original_metrics=original_metrics,
                baseline=baseline,
                original_class_id="all",
                original_class_name="ALL",
                stats_logit=stats_logit,
                stats_prob=stats_prob,
                pred_counts=pred_counts,
                class_0_name=class_0_name,
                class_1_name=class_1_name,
            )
        )

    # -------------------------------------------------------------------------
    # Save CSVs.
    # -------------------------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "baseline_bias_summary.csv")
    per_image_path = os.path.join(OUTPUT_DIR, "baseline_bias_per_image.csv")
    model_metrics_path = os.path.join(OUTPUT_DIR, "model_original_metrics.csv")

    summary_fields = [
        "model",
        "architecture",
        "kernel_size",
        "conv_filter",
        "conv_layer",
        "dense_neuron",
        "dense_layer",
        "weight_decay",
        "dropout",

        "original_n",
        "accuracy",
        "balanced_accuracy",
        "precision_class_1",
        "recall_class_1",
        "f1_class_1",
        "tn",
        "fp",
        "fn",
        "tp",
        "class_0_total",
        "class_1_total",
        "class_0_correct",
        "class_1_correct",
        "class_0_accuracy",
        "class_1_accuracy",
        "mean_original_logit",
        "mean_original_p_class_1",

        "baseline",
        "original_class_id",
        "original_class_name",
        "n",
        "mean_logit",
        "std_logit",
        "min_logit",
        "max_logit",
        "mean_p_class_0",
        "mean_p_class_1",
        "std_p_class_1",
        "min_p_class_1",
        "max_p_class_1",

        "baseline_pred_class_0_count",
        "baseline_pred_class_1_count",
        "baseline_pred_class_0_frac",
        "baseline_pred_class_1_frac",

        "neutral_distance",
        "cat_bias_if_class_0_cat",
        "dog_bias_if_class_1_dog",
        "predicted_baseline_side",
        "bias_strength",
    ]

    per_image_fields = [
        "model",
        "architecture",
        "kernel_size",
        "conv_filter",
        "conv_layer",
        "dense_neuron",
        "dense_layer",
        "weight_decay",
        "dropout",
        "baseline",
        "original_class_id",
        "original_class_name",
        "logit",
        "p_class_0",
        "p_class_1",
        "predicted_class_id",
        "predicted_class_name",
        "path",
    ]

    model_metrics_row = {
        "model": spec.fname,
        "architecture": spec.architecture,
        "kernel_size": spec.kernel_size,
        "conv_filter": spec.conv_filter,
        "conv_layer": spec.conv_layer,
        "dense_neuron": spec.dense_neuron,
        "dense_layer": spec.dense_layer,
        "weight_decay": spec.weight_decay,
        "dropout": spec.dropout,
        **original_metrics,
    }

    model_metrics_fields = [
        "model",
        "architecture",
        "kernel_size",
        "conv_filter",
        "conv_layer",
        "dense_neuron",
        "dense_layer",
        "weight_decay",
        "dropout",
        "original_n",
        "accuracy",
        "balanced_accuracy",
        "precision_class_1",
        "recall_class_1",
        "f1_class_1",
        "tn",
        "fp",
        "fn",
        "tp",
        "class_0_total",
        "class_1_total",
        "class_0_correct",
        "class_1_correct",
        "class_0_accuracy",
        "class_1_accuracy",
        "mean_original_logit",
        "mean_original_p_class_1",
    ]

    append_csv(summary_path, summary_rows, summary_fields)
    append_csv(per_image_path, per_image_rows, per_image_fields)
    append_csv(model_metrics_path, [model_metrics_row], model_metrics_fields)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {per_image_path}")
    print(f"  {model_metrics_path}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Test dir: {TEST_DIR}")

    dataset = SafeImageFolder(TEST_DIR, transform=get_tensor_transform())

    print(f"Classes: {dataset.classes}")
    print(f"Class 0 = {dataset.classes[0]}")
    print(f"Class 1 = {dataset.classes[1]}")
    print("The model output is interpreted as sigmoid(logit) = p(class 1).\n")

    if ONLY_CORRECT_ORIGINALS:
        print("WARNING: ONLY_CORRECT_ORIGINALS=True.")
        print("Accuracy/confusion matrix will be computed on the filtered correct-only subset.")
        print("Use ONLY_CORRECT_ORIGINALS=False for real model accuracy.\n")

    base_selected = class_balanced_indices(dataset, N_PER_CLASS_CAP)

    sample_path = os.path.join(OUTPUT_DIR, "sampled_test_images.txt")
    with open(sample_path, "w") as f:
        for cls, idxs in base_selected.items():
            name = dataset.classes[cls]
            for idx in idxs:
                path, _ = dataset.samples[idx]
                f.write(f"{cls},{name},{path}\n")
    print(f"Saved sampled image list: {sample_path}")

    pth_files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".pth"))

    if ONLY_MODEL is not None:
        pth_files = [f for f in pth_files if f == ONLY_MODEL]
        if not pth_files:
            raise SystemExit(f"ONLY_MODEL was set, but this file was not found: {ONLY_MODEL}")

    specs = []
    for fname in pth_files:
        spec = parse_model_filename(fname)
        if spec is None:
            print(f"[SKIP] Could not parse model filename: {fname}")
            continue
        specs.append(spec)

    if not specs:
        raise SystemExit(f"No parsable .pth files found in {MODELS_DIR}")

    summary_path = os.path.join(OUTPUT_DIR, "baseline_bias_summary.csv")
    done = already_done_models(summary_path) if SKIP_ALREADY_DONE and ONLY_MODEL is None else set()

    print(f"Found {len(specs)} model(s) to check.")
    if ONLY_MODEL is not None:
        print(f"ONLY_MODEL enabled: {ONLY_MODEL}")

    for spec in specs:
        if spec.fname in done:
            print(f"[SKIP] Already done: {spec.fname}")
            continue

        try:
            run_for_model(spec, dataset, base_selected)
        except Exception as exc:
            print(f"[ERROR] Failed on {spec.fname}: {exc}")
            continue

    print("\nDone.")
    print(f"Summary CSV:      {os.path.join(OUTPUT_DIR, 'baseline_bias_summary.csv')}")
    print(f"Per-image CSV:    {os.path.join(OUTPUT_DIR, 'baseline_bias_per_image.csv')}")
    print(f"Model metrics CSV:{os.path.join(OUTPUT_DIR, 'model_original_metrics.csv')}")


if __name__ == "__main__":
    main()