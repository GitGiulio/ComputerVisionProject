"""
Plot ALL models from baseline_bias_summary.csv.

This script reads:
    ./baseline_bias_check/baseline_bias_summary_34models.csv

and creates:
    ./baseline_bias_check/all_models_from_summary_cat_probability.csv
    ./baseline_bias_check/all_models_from_summary_cat_probability.png

It uses:
    - black row: baseline == black, original_class_name == all
    - blur row:  baseline == blur,  original_class_name == ALL
    - mean row:  baseline == mean,  original_class_name == ALL

The y-axis is p(Cat), taken from mean_p_class_0.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 20
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 17

SUMMARY_CSV = "./baseline_bias_check/baseline_bias_summary_34models.csv"
OUTPUT_DIR = "./baseline_bias_check"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "all_models_from_summary_cat_probability.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "all_models_from_summary_cat_probability.png")

EXPECTED_MODELS = 36


def short_model_name(model: str) -> str:
    name = model.replace("model_kernel=", "k=")
    name = name.replace(".pth", "")
    name = name.replace("_wd", "\nwd")
    name = name.replace("_do", " do")
    return name


def get_value(model_df: pd.DataFrame, baseline: str) -> float:
    if baseline == "black":
        sub = model_df[
            (model_df["baseline"] == "black")
            & (model_df["original_class_name"].astype(str).str.lower() == "all")
        ]
    else:
        sub = model_df[
            (model_df["baseline"] == baseline)
            & (model_df["original_class_name"].astype(str) == "ALL")
        ]

    if sub.empty:
        return np.nan

    # If the same model appears multiple times because the baseline script was rerun,
    # use the last occurrence.
    return float(sub.iloc[-1]["mean_p_class_0"])


def main() -> None:
    if not os.path.isfile(SUMMARY_CSV):
        raise FileNotFoundError(f"Could not find: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    required = {"model", "baseline", "original_class_name", "mean_p_class_0"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    models = sorted(df["model"].dropna().unique())

    print(f"Found {len(models)} unique models in {SUMMARY_CSV}")
    if len(models) != EXPECTED_MODELS:
        print(f"WARNING: expected {EXPECTED_MODELS} models, but found {len(models)}.")

    rows = []
    for model in models:
        model_df = df[df["model"] == model]

        black_p_cat = get_value(model_df, "black")
        blur_p_cat = get_value(model_df, "blur")
        mean_p_cat = get_value(model_df, "mean")

        rows.append({
            "model": model,
            "black_p_cat": black_p_cat,
            "blur_all_p_cat": blur_p_cat,
            "mean_all_p_cat": mean_p_cat,
            "average_p_cat": np.nanmean([black_p_cat, blur_p_cat, mean_p_cat]),
        })

    out = pd.DataFrame(rows)

    # Sort by average Cat probability so the strongest Cat-biased models are on the left/top.
    out = out.sort_values("average_p_cat", ascending=False).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved table: {OUTPUT_CSV}")

    # Plot
    labels = [short_model_name(m) for m in out["model"]]
    x = np.arange(len(out))
    width = 0.25

    fig_width = max(18, len(out) * 0.6)
    plt.figure(figsize=(fig_width, 8))

    plt.bar(x - width, out["black_p_cat"], width, label="Black baseline")
    plt.bar(x, out["blur_all_p_cat"], width, label="Blur baseline")
    plt.bar(x + width, out["mean_all_p_cat"], width, label="Mean baseline")

    plt.axhline(0.5, linestyle=":", linewidth=2.5, label="Neutral = 0.5")

    plt.ylabel("Mean probability of Cat, p(Cat)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xlabel("Model", fontsize=AXIS_LABEL_FONTSIZE)
    plt.title("Baseline bias of 34 models", fontsize=TITLE_FONTSIZE)

    plt.xticks(x, labels, rotation=75, ha="right", fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.ylim(0, 1)
    plt.legend(fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220)
    plt.close()

    print(f"Saved plot: {OUTPUT_PNG}")

    print("\nPreview:")
    print(out[["model", "black_p_cat", "blur_all_p_cat", "mean_all_p_cat", "average_p_cat"]].to_string(index=False))


if __name__ == "__main__":
    main()
