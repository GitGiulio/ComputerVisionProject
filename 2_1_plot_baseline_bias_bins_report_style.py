"""
Plot baseline-bias probability histograms from baseline_bias_per_image.csv.

This reads the CSV created by baseline_bias_clear_all.py and makes histogram plots
with bins of width 0.01 on the x-axis.

Expected CSV columns:
  model, baseline, original_class_name, p_class_1

For your Cat/Dog setup:
  p_class_1 is renamed in the plots as p(Dog)
  p(Cat) = 1 - p(Dog)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Settings
# =============================================================================

CSV_PATH = "./baseline_bias_check/baseline_bias_per_image.csv"
OUTPUT_DIR = "./baseline_bias_check/plots"

# Class names used for plot labels.
CLASS_0_NAME = "Cat"
CLASS_1_NAME = "Dog"

# Set to None to plot every model in the CSV.
# Or set to one exact model filename, e.g.
# ONLY_MODEL = "model_kernel=[5,9,23]_32_3_64_1_wd0.0001_do0.0.pth"
ONLY_MODEL: Optional[str] = None

# Use 0.01 probability bins: 0.00, 0.01, 0.02, ..., 1.00
BIN_WIDTH = 0.01

# If True, make separate plots for Cat and Dog original images.
MAKE_CLASS_PLOTS = True

# If True, make combined ALL plots where Cat and Dog original images are pooled.
MAKE_ALL_PLOTS = True

# If True, plot blur and mean on the same figure for easier comparison.
MAKE_OVERLAY_PLOTS = True

# Larger report-friendly fonts
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 15

# Short titles only. Model details can go in the report caption.
SINGLE_HIST_TITLE = "Baseline predictions are concentrated toward one class."
OVERLAY_TITLE = "Blur and mean baselines produce similar prediction distributions."


def safe_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "-")
        .replace("=", "")
        .replace(" ", "_")
    )


def describe_bias(mean_p_dog: float) -> str:
    """Return a human-readable bias description."""
    mean_p_cat = 1.0 - mean_p_dog

    if mean_p_dog > 0.5:
        direction = f"leans {CLASS_1_NAME}"
        confidence = mean_p_dog
    elif mean_p_dog < 0.5:
        direction = f"leans {CLASS_0_NAME}"
        confidence = mean_p_cat
    else:
        return "neutral"

    if confidence >= 0.90:
        strength = "very strong"
    elif confidence >= 0.75:
        strength = "strong"
    elif confidence >= 0.60:
        strength = "moderate"
    else:
        strength = "weak"

    return f"{direction} ({strength})"


def plot_histogram(
    values: np.ndarray,
    out_path: str,
    xlabel: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bins = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)

    plt.figure(figsize=(11, 6))
    plt.hist(values, bins=bins, edgecolor="black")

    mean_p_dog = float(np.mean(values))
    mean_p_cat = 1.0 - mean_p_dog

    plt.axvline(
        mean_p_dog,
        linestyle="--",
        linewidth=2.5,
        label=f"Mean p({CLASS_1_NAME}) = {mean_p_dog:.3f}",
    )
    plt.axvline(
        0.5,
        linestyle=":",
        linewidth=2.5,
        label="Neutral = 0.5",
    )

    # Short one-sentence title only.
    plt.title(SINGLE_HIST_TITLE, fontsize=TITLE_FONTSIZE)

    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Number of images", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xlim(0, 1)

    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45, fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_overlay(
    df: pd.DataFrame,
    out_path: str,
    xlabel: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bins = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)

    blur = df[df["baseline"] == "blur"]["p_class_1"].to_numpy(dtype=float)
    mean = df[df["baseline"] == "mean"]["p_class_1"].to_numpy(dtype=float)

    plt.figure(figsize=(11, 6))

    if len(blur) > 0:
        blur_mean = float(blur.mean())
        plt.hist(
            blur,
            bins=bins,
            alpha=0.55,
            edgecolor="black",
            label=f"Blur: mean p({CLASS_1_NAME}) = {blur_mean:.3f}",
        )
        plt.axvline(blur_mean, linestyle="--", linewidth=2.5)

    if len(mean) > 0:
        mean_mean = float(mean.mean())
        plt.hist(
            mean,
            bins=bins,
            alpha=0.55,
            edgecolor="black",
            label=f"Mean: mean p({CLASS_1_NAME}) = {mean_mean:.3f}",
        )
        plt.axvline(mean_mean, linestyle="--", linewidth=2.5)

    plt.axvline(0.5, linestyle=":", linewidth=2.5, label="Neutral = 0.5")

    # Short one-sentence title only.
    plt.title(OVERLAY_TITLE, fontsize=TITLE_FONTSIZE)

    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Number of images", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xlim(0, 1)

    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45, fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Could not find CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {"model", "baseline", "original_class_name", "p_class_1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Rename for readability inside this plotting script.
    df = df.rename(columns={"p_class_1": f"p_{CLASS_1_NAME.lower()}"})

    # Keep only blur/mean rows. The per-image CSV does not contain black because black is one global value.
    df = df[df["baseline"].isin(["blur", "mean"])].copy()

    if ONLY_MODEL is not None:
        df = df[df["model"] == ONLY_MODEL].copy()
        if df.empty:
            raise ValueError(f"No rows found for ONLY_MODEL={ONLY_MODEL}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loaded {len(df)} per-image rows from: {CSV_PATH}")
    print(f"Saving plots to: {OUTPUT_DIR}")
    print(f"X-axis is p({CLASS_1_NAME}). Left side means {CLASS_0_NAME}; right side means {CLASS_1_NAME}.")

    prob_col = f"p_{CLASS_1_NAME.lower()}"
    xlabel = f"Predicted probability of {CLASS_1_NAME}, p({CLASS_1_NAME})"

    for model, model_df in df.groupby("model"):
        model_safe = safe_name(model)

        # Class-specific plots: Cat blur, Cat mean, Dog blur, Dog mean
        if MAKE_CLASS_PLOTS:
            for class_name, class_df in model_df.groupby("original_class_name"):
                class_safe = safe_name(str(class_name))

                for baseline, sub in class_df.groupby("baseline"):
                    values = sub[prob_col].to_numpy(dtype=float)
                    out_path = os.path.join(
                        OUTPUT_DIR,
                        f"{model_safe}__{class_safe}__{baseline}__hist_bins_{BIN_WIDTH}.png",
                    )
                    plot_histogram(values, out_path, xlabel=xlabel)
                    print(f"Saved: {out_path}")

                if MAKE_OVERLAY_PLOTS:
                    out_path = os.path.join(
                        OUTPUT_DIR,
                        f"{model_safe}__{class_safe}__blur_vs_mean_overlay_bins_{BIN_WIDTH}.png",
                    )
                    # temporarily rename back for function simplicity
                    overlay_df = class_df.rename(columns={prob_col: "p_class_1"})
                    plot_overlay(overlay_df, out_path, xlabel=xlabel)
                    print(f"Saved: {out_path}")

        # ALL plots: Cat + Dog together
        if MAKE_ALL_PLOTS:
            for baseline, sub in model_df.groupby("baseline"):
                values = sub[prob_col].to_numpy(dtype=float)
                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"{model_safe}__ALL__{baseline}__hist_bins_{BIN_WIDTH}.png",
                )
                plot_histogram(values, out_path, xlabel=xlabel)
                print(f"Saved: {out_path}")

            if MAKE_OVERLAY_PLOTS:
                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"{model_safe}__ALL__blur_vs_mean_overlay_bins_{BIN_WIDTH}.png",
                )
                overlay_df = model_df.rename(columns={prob_col: "p_class_1"})
                plot_overlay(overlay_df, out_path, xlabel=xlabel)
                print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
