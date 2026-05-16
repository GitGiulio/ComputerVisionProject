"""
summarize_identity_metrics.py

Scans interpretability result folders, loads CSV files, and:
1. Prints average values of:
      - identity
      - separability
      - avg_time_sec

2. Prints averages:
      - overall
      - SHAP only
      - GradCAM only

3. Generates TWO histograms for avg_time_sec:
      - SHAP
      - GradCAM

Usage:
  python summarize_identity_metrics.py \
      --root_dir ./fixed_fidelity_interpretability \
      --output_dir ./plots
"""

import os
import re
import argparse

import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# Regex for parsing folder names
# ──────────────────────────────────────────────────────────────

FOLDER_RE = re.compile(
    r"interpretability_results_dogs_k=(?:\[5,9,(?P<kernel_size_a>[^\]]+)\]|(?P<kernel_size_b>[^_]+))"
    r"_(?P<conv_filter>[^_]+)"
    r"_(?P<conv_layer>[^_]+)"
    r"_(?P<dense_neuron>[^_]+)"
    r"_(?P<dense_layer>[^_]+)"
    r"_wd(?P<weight_decay>[^_]+)"
    r"_do(?P<dropout>.+)$"
)

STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#111111",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "text.color":       "#111111",
    "grid.color":       "#dddddd",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
}

METHOD_COLORS = {
    "shap": "#7c9ef7",
    "gradcam": "#f7a07c",
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def parse_folder_name(folder_name: str):
    base = os.path.basename(folder_name.rstrip("/\\"))
    return FOLDER_RE.match(base)


def load_csv(folder_path: str):

    csv_path = os.path.join(folder_path, "metrics_summary.csv")

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)

        df.columns = df.columns.str.strip()

        if "method" in df.columns:
            df["method"] = (
                df["method"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        return df

    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}")
        return None


def collect_data(root_dir: str):

    records = []

    for entry in os.scandir(root_dir):

        if not entry.is_dir():
            continue

        if parse_folder_name(entry.name) is None:
            continue

        df = load_csv(entry.path)

        if df is None:
            continue

        for _, row in df.iterrows():
            records.append({
                "folder": entry.name,
                **row.to_dict()
            })

    if not records:
        raise ValueError(f"No valid CSV data found in: {root_dir}")

    combined = pd.DataFrame(records)

    # Convert numeric columns safely
    for col in ["identity", "separability", "avg_time_sec"]:

        if col in combined.columns:
            combined[col] = pd.to_numeric(
                combined[col],
                errors="coerce"
            )

    return combined


# ──────────────────────────────────────────────────────────────
# Printing averages
# ──────────────────────────────────────────────────────────────

def print_metric_averages(df, metrics):

    print("\nOVERALL AVERAGES")
    print("-" * 60)

    for metric in metrics:

        if metric in df.columns:

            avg = df[metric].mean(skipna=True)

            print(f"{metric:20s}: {avg:.6f}")

    print("\nSHAP AVERAGES")
    print("-" * 60)

    shap_df = df[df["method"] == "shap"]

    for metric in metrics:

        if metric in shap_df.columns:

            avg = shap_df[metric].mean(skipna=True)

            print(f"{metric:20s}: {avg:.6f}")

    print("\nGRADCAM AVERAGES")
    print("-" * 60)

    gradcam_df = df[df["method"] == "gradcam"]

    for metric in metrics:

        if metric in gradcam_df.columns:

            avg = gradcam_df[metric].mean(skipna=True)

            print(f"{metric:20s}: {avg:.6f}")


# ──────────────────────────────────────────────────────────────
# Histogram plotting
# ──────────────────────────────────────────────────────────────

def plot_time_histogram(df, method, output_dir, bins=10):

    sub = df[df["method"] == method]

    values = sub["avg_time_sec"].dropna()

    if len(values) == 0:
        print(f"[WARN] No avg_time_sec values for method='{method}'")
        return

    color = METHOD_COLORS.get(method, "#999999")

    with plt.rc_context(STYLE):

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            values,
            bins=bins,
            color=color,
            edgecolor="black"
        )

        ax.set_title(
            f"{method.upper()} avg_time_sec Distribution",
            fontsize=18,
            fontweight="bold"
        )

        ax.set_xlabel("avg_time_sec", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)

        ax.grid(axis="y")

        out_path = os.path.join(
            output_dir,
            f"{method}_avg_time_sec_histogram.png"
        )

        fig.tight_layout()

        fig.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    print(f"Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="Summarize identity/separability/avg_time_sec metrics."
    )

    parser.add_argument(
        "--root_dir",
        default="fixed_fidelity_interpretability/",
        help="Directory containing result folders"
    )

    parser.add_argument(
        "--output_dir",
        default="plots/",
        help="Directory for output plots"
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of histogram bins"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning: {args.root_dir}")

    df = collect_data(args.root_dir)

    print(f"Loaded {len(df)} rows")

    metrics = [
        "identity",
        "separability",
        "avg_time_sec"
    ]

    # ──────────────────────────────────────────
    # Print averages
    # ──────────────────────────────────────────

    print_metric_averages(df, metrics)

    # ──────────────────────────────────────────
    # Histograms
    # ──────────────────────────────────────────

    if "avg_time_sec" in df.columns:

        plot_time_histogram(
            df,
            method="shap",
            output_dir=args.output_dir,
            bins=args.bins
        )

        plot_time_histogram(
            df,
            method="gradcam",
            output_dir=args.output_dir,
            bins=args.bins
        )

    else:
        print("\n[WARN] avg_time_sec column not found.")

    print("\nDone!")


if __name__ == "__main__":
    main()