
"""
Confusion matrix check for Cat/Dog models.

Purpose:
  This checks whether the classifier itself is biased toward Cat or Dog on
  normal, unmodified test images.

It does NOT use baselines.
It simply runs each model on the test set and computes sklearn's confusion matrix.

Expected class mapping with ImageFolder:
  class 0 = Cat
  class 1 = Dog

Confusion matrix layout:
                 Pred Cat     Pred Dog
  Actual Cat      cm[0,0]      cm[0,1]
  Actual Dog      cm[1,0]      cm[1,1]

Useful interpretation:
  actual_dog_pred_cat high  -> model sends many Dogs to Cat -> Cat bias
  actual_cat_pred_dog high  -> model sends many Cats to Dog -> Dog bias

Outputs:
  ./baseline_bias_check/confusion_matrix_summary.csv
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from shared_code import (
    SafeImageFolder,
    collate_skip_none,
    get_tensor_transform,
    I_HAVE_A_THEORY,
    CIFAKE_CNN,
)


DATA_DIR = "./Cat_dog_splitted"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = "models"
OUTPUT_DIR = "./baseline_bias_check"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "confusion_matrix_summary.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# Set to None to evaluate all parsable .pth files in MODELS_DIR.
# Or set to one exact filename.
ONLY_MODEL: Optional[str] = None #"model_kernel=[5,9,19]_32_3_512_1_wd0.001_do0.0.pth"

# If True, overwrite existing CSV.
OVERWRITE_CSV = True


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
    dropout: float = 0.0


def parse_model_filename(fname: str) -> Optional[ModelSpec]:
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
            dropout=float(m.group(7)) if m.group(7) is not None else 0.0,
        )

    pattern_plain = r"model_(\d+)_(\d+)_(\d+)_(\d+)$"
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
        )

    return None


def build_model(spec: ModelSpec) -> torch.nn.Module:
    if spec.architecture == "I_HAVE_A_THEORY":
        model = I_HAVE_A_THEORY(
            kernel_size=spec.kernel_size,
            conv_filters=spec.conv_filter,
            conv_layers=spec.conv_layer,
            dense_neurons=spec.dense_neuron,
            dense_layers=spec.dense_layer,
            dropout_rate=spec.dropout,
        )
    elif spec.architecture == "CIFAKE_CNN":
        model = CIFAKE_CNN(
            conv_filters=spec.conv_filter,
            conv_layers=spec.conv_layer,
            dense_neurons=spec.dense_neuron,
            dense_layers=spec.dense_layer,
        )
    else:
        raise ValueError(f"Unknown architecture: {spec.architecture}")

    state_dict = torch.load(spec.path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


def compute_confusion_matrix(model: torch.nn.Module, dataset: SafeImageFolder) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_skip_none,
        pin_memory=DEVICE.startswith("cuda"),
    )

    y_true = []
    y_pred = []
    p_dog_all = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            images, labels, _paths = batch
            images = images.to(DEVICE).float()

            logits = model(images).reshape(-1)
            p_dog = torch.sigmoid(logits)
            preds = (p_dog >= 0.5).long().cpu().numpy()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            p_dog_all.extend(p_dog.detach().cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p_dog_all = np.array(p_dog_all, dtype=np.float64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    print(f"Confusion matrix:===================\n{cm}")

    actual_cat_pred_cat = int(cm[0, 0])
    actual_cat_pred_dog = int(cm[0, 1])
    actual_dog_pred_cat = int(cm[1, 0])
    actual_dog_pred_dog = int(cm[1, 1])

    total = int(cm.sum())
    actual_cat_total = actual_cat_pred_cat + actual_cat_pred_dog
    actual_dog_total = actual_dog_pred_cat + actual_dog_pred_dog
    pred_cat_total = actual_cat_pred_cat + actual_dog_pred_cat
    pred_dog_total = actual_cat_pred_dog + actual_dog_pred_dog

    cat_accuracy = actual_cat_pred_cat / actual_cat_total if actual_cat_total else float("nan")
    dog_accuracy = actual_dog_pred_dog / actual_dog_total if actual_dog_total else float("nan")
    pred_cat_rate = pred_cat_total / total if total else float("nan")
    pred_dog_rate = pred_dog_total / total if total else float("nan")

    if pred_cat_rate > pred_dog_rate:
        predicted_class_bias = "Cat"
    elif pred_dog_rate > pred_cat_rate:
        predicted_class_bias = "Dog"
    else:
        predicted_class_bias = "neutral"

    return {
        "actual_cat_pred_cat": actual_cat_pred_cat,
        "actual_cat_pred_dog": actual_cat_pred_dog,
        "actual_dog_pred_cat": actual_dog_pred_cat,
        "actual_dog_pred_dog": actual_dog_pred_dog,
        "actual_cat_total": actual_cat_total,
        "actual_dog_total": actual_dog_total,
        "pred_cat_total": pred_cat_total,
        "pred_dog_total": pred_dog_total,
        "total": total,
        "cat_accuracy": cat_accuracy,
        "dog_accuracy": dog_accuracy,
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "precision_dog": precision_score(y_true, y_pred, zero_division=0),
        "recall_dog": recall_score(y_true, y_pred, zero_division=0),
        "f1_dog": f1_score(y_true, y_pred, zero_division=0),
        "pred_cat_rate": pred_cat_rate,
        "pred_dog_rate": pred_dog_rate,
        "predicted_class_bias": predicted_class_bias,
        "mean_p_dog": float(p_dog_all.mean()),
        "mean_p_cat": float(1.0 - p_dog_all.mean()),
    }


def print_confusion_result(model_name: str, result: dict) -> None:
    print()
    print(f"Model: {model_name}")
    print("Confusion matrix on normal test images")
    print("Rows = actual, columns = predicted")
    print()
    print("                 Pred Cat   Pred Dog")
    print(f"Actual Cat       {result['actual_cat_pred_cat']:8d}   {result['actual_cat_pred_dog']:8d}")
    print(f"Actual Dog       {result['actual_dog_pred_cat']:8d}   {result['actual_dog_pred_dog']:8d}")
    print()
    print(
        f"Cat acc={result['cat_accuracy']:.4f} | "
        f"Dog acc={result['dog_accuracy']:.4f} | "
        f"Overall acc={result['overall_accuracy']:.4f}"
    )
    print(
        f"Pred Cat rate={result['pred_cat_rate']:.4f} | "
        f"Pred Dog rate={result['pred_dog_rate']:.4f} | "
        f"Predicted-class bias={result['predicted_class_bias']}"
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Test dir: {TEST_DIR}")
    print(f"Models dir: {MODELS_DIR}")
    print()

    dataset = SafeImageFolder(TEST_DIR, transform=get_tensor_transform())

    print(f"Classes: {dataset.classes}")
    print("Expected mapping: class 0 = Cat, class 1 = Dog")
    print()

    model_files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".pth"))

    if ONLY_MODEL is not None:
        model_files = [f for f in model_files if f == ONLY_MODEL]
        if not model_files:
            raise SystemExit(f"ONLY_MODEL was set but not found: {ONLY_MODEL}")

    specs = []
    for fname in model_files:
        spec = parse_model_filename(fname)
        if spec is None:
            print(f"[SKIP] Could not parse model filename: {fname}")
            continue
        specs.append(spec)

    if not specs:
        raise SystemExit("No parsable models found.")

    print(f"Found {len(model_files)} .pth file(s).")
    print(f"Found {len(specs)} parsable model(s).")
    print()

    fieldnames = [
        "model",
        "architecture",
        "actual_cat_pred_cat",
        "actual_cat_pred_dog",
        "actual_dog_pred_cat",
        "actual_dog_pred_dog",
        "actual_cat_total",
        "actual_dog_total",
        "pred_cat_total",
        "pred_dog_total",
        "total",
        "cat_accuracy",
        "dog_accuracy",
        "overall_accuracy",
        "precision_dog",
        "recall_dog",
        "f1_dog",
        "pred_cat_rate",
        "pred_dog_rate",
        "predicted_class_bias",
        "mean_p_cat",
        "mean_p_dog",
    ]

    if OVERWRITE_CSV and os.path.isfile(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, spec in enumerate(specs, start=1):
            print(f"[{i}/{len(specs)}] Evaluating {spec.fname}")

            try:
                model = build_model(spec)
                result = compute_confusion_matrix(model, dataset)
            except Exception as exc:
                print(f"[ERROR] Failed on {spec.fname}: {exc}")
                continue

            print_confusion_result(spec.fname, result)

            row = {
                "model": spec.fname,
                "architecture": spec.architecture,
            }
            row.update(result)

            writer.writerow(row)
            f.flush()

            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print()
    print(f"Done. Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
