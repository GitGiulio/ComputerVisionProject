"""
plot_model_metrics.py

Generates metric comparison plots across models from interpretability result folders.
Folder naming convention:
  interpretability_results_dogs_k=[5,9,{KERNEL_SIZE}]_{CONV_FILTER}_{CONV_LAYER}_{DENSE_NEURON}_{DENSE_LAYER}_wd{WEIGHT_DECAY}_do{DROPOUT}

Usage:
  python plotting.py --root_dir ./.. --output_dir ./plots
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

# Metrics to plot: (title, method, column, y_label)
METRICS = [
    ("SHAP - Insertion AUC",      "shap",    "insertion_auc",  "Insertion AUC"),
    ("Model - Test Accuracy",        None,      "model_test_acc", "Test Accuracy"),
    ("SHAP - Deletion AUC",        "shap",    "deletion_auc",   "Deletion AUC"),
    ("SHAP - Stability",           "shap",    "stability",      "Stability"),
    ("GradCAM - Deletion AUC",     "gradcam", "deletion_auc",   "Deletion AUC"),
    ("GradCAM - Stability",        "gradcam", "stability",      "Stability"),
    ("GradCAM - Insertion AUC",    "gradcam", "insertion_auc",  "Insertion AUC"),
]

FOLDER_RE = re.compile(
    r"interpretability_results_dogs_k=\[5,9,(?P<kernel_size>[^\]]+)\]"
    r"_(?P<conv_filter>[^_]+)"
    r"_(?P<conv_layer>[^_]+)"
    r"_(?P<dense_neuron>[^_]+)"
    r"_(?P<dense_layer>[^_]+)"
    r"_wd(?P<weight_decay>[^_]+)"
    r"_do(?P<dropout>.+)$"
)

STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#a0a0b0",
    "ytick.color":       "#a0a0b0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#2a2d3d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "sans-serif",
}

ACCENT_COLOR  = "#7c9ef7"   # blue bars
ACCENT2_COLOR = "#f7a07c"   # fallback second colour


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_folder_name(folder_name: str) -> dict | None:
    """Extract hyperparameters from folder name."""
    base = os.path.basename(folder_name.rstrip("/\\"))
    m = FOLDER_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    try:
        d["weight_decay"] = float(d["weight_decay"])
    except ValueError:
        pass
    return d


def load_folder(folder_path: str) -> pd.DataFrame | None:
    """Load the CSV from a model folder (first .csv found)."""
    df = pd.read_csv(folder_path + "/" + "metrics_summary.csv")
    print(df)
    df.columns = df.columns.str.strip()
    df["method"] = df["method"].str.strip().str.lower()
    return df


def collect_data(root_dir: str) -> pd.DataFrame:
    """Walk root_dir, load every valid model folder, return combined DataFrame."""
    records = []
    print(os.getcwd())
    for entry in os.scandir(os.getcwd()):
        if not entry.is_dir():
            continue
        params = parse_folder_name(entry.name)
        if params is None:
            continue
        df = load_folder(entry.path)
        if df is None:
            print(f"  [warn] no CSV in {entry.name}")
            continue
        for _, row in df.iterrows():
            rec = {"folder": entry.name, **params, **row.to_dict()}
            records.append(rec)

    if not records:
        raise ValueError(f"No valid model folders found under '{root_dir}'")

    combined = pd.DataFrame(records)
    # Ensure numeric types
    for col in ["model_tot_param", "weight_decay", "insertion_auc",
                "deletion_auc", "stability", "model_test_acc", "model_val_acc"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def make_x_labels(sorted_folders: list[str], params_map: dict) -> list[str]:
    """Create short, readable x-axis labels from folder params."""
    labels = []
    for f in sorted_folders:
        p = params_map[f]
        labels.append(
            f"ks{p['kernel_size']}\n"
            f"cf{p['conv_filter']} cl{p['conv_layer']}\n"
            f"dn{p['dense_neuron']} dl{p['dense_layer']}\n"
            f"wd{p['weight_decay']} do{p['dropout']}"
        )
    return labels


def sort_folders(df: pd.DataFrame) -> list[str]:
    """Return folder names sorted by (model_tot_param ASC, weight_decay ASC)."""
    ref = (
        df[["folder", "model_tot_param", "weight_decay"]]
        .drop_duplicates("folder")
        .sort_values(["model_tot_param", "weight_decay"])
    )
    return ref["folder"].tolist()


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_metric(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    title: str,
    method: str | None,
    column: str,
    y_label: str,
    output_path: str,
):
    if method is not None:
        sub = df[df["method"] == method]
    else:
        # model-level columns are duplicated per method – take first
        sub = df.drop_duplicates("folder")

    # Build ordered y values
    y_vals = []
    for f in sorted_folders:
        row = sub[sub["folder"] == f]
        if row.empty or pd.isna(row[column].values[0]):
            y_vals.append(np.nan)
        else:
            y_vals.append(row[column].values[0])

    x = np.arange(len(sorted_folders))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(10, len(sorted_folders) * 1.4), 5.5))

        bars = ax.bar(x, y_vals, color=ACCENT_COLOR, width=0.6,
                      edgecolor="#ffffff22", linewidth=0.5, zorder=3)

        # Value labels on bars
        for bar, val in zip(bars, y_vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(v for v in y_vals if not np.isnan(v)) * 0.01),
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=7, color="#c8cfe8",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, ha="center")
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", zorder=0)
        ax.set_xlim(-0.6, len(sorted_folders) - 0.4)

        # Param count secondary x annotation
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        params_labels = []
        ref = df.drop_duplicates("folder").set_index("folder")
        for f in sorted_folders:
            p = ref.loc[f, "model_tot_param"] if f in ref.index else np.nan
            params_labels.append(f"{int(p):,}" if not np.isnan(p) else "?")
        ax2.set_xticklabels(params_labels, fontsize=6.5, color="#7090c0")
        ax2.set_xlabel("Total parameters →", fontsize=8, color="#7090c0", labelpad=4)
        ax2.tick_params(axis="x", colors="#7090c0")
        for spine in ax2.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {output_path}")


def plot_all(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    for title, method, column, y_label in METRICS:
        safe = title.lower().replace(" ", "_").replace("-", "").replace("__", "_")
        out = os.path.join(output_dir, f"{safe}.png")
        plot_metric(df, sorted_folders, x_labels, title, method, column, y_label, out)


def plot_summary_grid(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    output_dir: str,
):
    """All 7 metrics in one figure."""
    n = len(METRICS)
    cols = 2
    rows = (n + 1) // cols

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * max(8, len(sorted_folders) * 0.9),
                                          rows * 4.5))
        axes = axes.flatten()
        x = np.arange(len(sorted_folders))

        for idx, (title, method, column, y_label) in enumerate(METRICS):
            ax = axes[idx]

            if method is not None:
                sub = df[df["method"] == method]
            else:
                sub = df.drop_duplicates("folder")

            y_vals = []
            for f in sorted_folders:
                row = sub[sub["folder"] == f]
                y_vals.append(row[column].values[0]
                              if not row.empty and not pd.isna(row[column].values[0])
                              else np.nan)

            color = ACCENT_COLOR if method != "gradcam" else ACCENT2_COLOR
            ax.bar(x, y_vals, color=color, width=0.6,
                   edgecolor="#ffffff18", linewidth=0.4, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=6, ha="center")
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.grid(axis="y", zorder=0)
            ax.set_xlim(-0.6, len(sorted_folders) - 0.4)

        # Hide spare axes
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Model Interpretability Metrics (sorted by parameter count)",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        out = os.path.join(output_dir, "summary_grid.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot AI model interpretability metrics.")
    parser.add_argument("--root_dir",   default=".",     help="Directory containing model folders")
    parser.add_argument("--output_dir", default="plots", help="Where to save output PNGs")
    args = parser.parse_args()

    print(f"Scanning '{args.root_dir}' for model folders …")
    df = collect_data(args.root_dir)
    print(f"  found {df['folder'].nunique()} model(s), {len(df)} rows total")

    # Build params map for labels
    params_map = {}
    for folder in df["folder"].unique():
        params_map[folder] = parse_folder_name(folder)

    sorted_folders = sort_folders(df)
    x_labels = make_x_labels(sorted_folders, params_map)

    print(f"\nGenerating individual plots …")
    plot_all(df, sorted_folders, x_labels, args.output_dir)

    print(f"\nGenerating summary grid …")
    plot_summary_grid(df, sorted_folders, x_labels, args.output_dir)

    print(f"\nDone! All plots saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()