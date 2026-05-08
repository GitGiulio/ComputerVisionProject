"""
plot_model_metrics.py

Generates metric comparison plots across models from interpretability result folders.
Folder naming convention:
  interpretability_results_dogs_k=[5,9,{KERNEL_SIZE}]_{CONV_FILTER}_{CONV_LAYER}_{DENSE_NEURON}_{DENSE_LAYER}_wd{WEIGHT_DECAY}_do{DROPOUT}

Usage:
  python plot_model_metrics.py --root_dir ./.. --output_dir ./plots
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
    ("SHAP - Insertion AUC",    "shap",    "insertion_auc",  "Insertion AUC"),
    ("Model - Test Accuracy",   None,      "model_test_acc", "Test Accuracy"),
    ("SHAP - Deletion AUC",     "shap",    "deletion_auc",   "Deletion AUC"),
    ("SHAP - Stability",        "shap",    "stability",      "Stability"),
    ("GradCAM - Deletion AUC",  "gradcam", "deletion_auc",   "Deletion AUC"),
    ("GradCAM - Stability",     "gradcam", "stability",      "Stability"),
    ("GradCAM - Insertion AUC", "gradcam", "insertion_auc",  "Insertion AUC"),
]

# Multi-metric line plot groups
LINE_PLOT_CONFIGS = [
    {
        "title":   "SHAP Metrics + Test Accuracy  [sorted by parameter count]",
        "method":  "shap",
        "sort_by": "params",
        "metrics": [
            ("Insertion AUC", "insertion_auc",  "#7c9ef7"),
            ("Deletion AUC",  "deletion_auc",   "#f7a07c"),
            ("Stability",     "stability",      "#7cf7a0"),
            ("Test Accuracy", "model_test_acc", "#f7e07c"),
        ],
    },
    {
        "title":   "GradCAM Metrics + Test Accuracy  [sorted by parameter count]",
        "method":  "gradcam",
        "sort_by": "params",
        "metrics": [
            ("Insertion AUC", "insertion_auc",  "#7c9ef7"),
            ("Deletion AUC",  "deletion_auc",   "#f7a07c"),
            ("Stability",     "stability",      "#7cf7a0"),
            ("Test Accuracy", "model_test_acc", "#f7e07c"),
        ],
    },
    {
        "title":   "SHAP Metrics + Test Accuracy  [sorted by test accuracy]",
        "method":  "shap",
        "sort_by": "accuracy",
        "metrics": [
            ("Insertion AUC", "insertion_auc",  "#7c9ef7"),
            ("Deletion AUC",  "deletion_auc",   "#f7a07c"),
            ("Stability",     "stability",      "#7cf7a0"),
            ("Test Accuracy", "model_test_acc", "#f7e07c"),
        ],
    },
    {
        "title":   "GradCAM Metrics + Test Accuracy  [sorted by test accuracy]",
        "method":  "gradcam",
        "sort_by": "accuracy",
        "metrics": [
            ("Insertion AUC", "insertion_auc",  "#7c9ef7"),
            ("Deletion AUC",  "deletion_auc",   "#f7a07c"),
            ("Stability",     "stability",      "#7cf7a0"),
            ("Test Accuracy", "model_test_acc", "#f7e07c"),
        ],
    },
]

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
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
}

ACCENT_COLOR  = "#7c9ef7"
ACCENT2_COLOR = "#f7a07c"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_folder_name(folder_name: str) -> dict | None:
    base = os.path.basename(folder_name.rstrip("/\\"))
    m = FOLDER_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    d["kernel_size"] = d.pop("kernel_size_a") or d.pop("kernel_size_b")
    try:
        d["weight_decay"] = float(d["weight_decay"])
    except ValueError:
        pass
    return d


def load_folder(folder_path: str) -> pd.DataFrame | None:
    df = pd.read_csv(folder_path + "/" + "metrics_summary.csv")
    df.columns = df.columns.str.strip()
    df["method"] = df["method"].str.strip().str.lower()
    return df


def collect_data(root_dir: str) -> pd.DataFrame:
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
    for col in ["model_tot_param", "weight_decay", "insertion_auc",
                "deletion_auc", "stability", "model_test_acc", "model_val_acc"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def make_x_labels(sorted_folders: list[str], params_map: dict) -> list[str]:
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


def sort_folders_by_params(df: pd.DataFrame) -> list[str]:
    """Sort by (model_tot_param ASC, weight_decay ASC)."""
    ref = (
        df[["folder", "model_tot_param", "weight_decay"]]
        .drop_duplicates("folder")
        .sort_values(["model_tot_param", "weight_decay"])
    )
    return ref["folder"].tolist()


def sort_folders_by_accuracy(df: pd.DataFrame) -> list[str]:
    """Sort by model_test_acc ASC (left = worst, right = best)."""
    ref = (
        df[["folder", "model_test_acc"]]
        .drop_duplicates("folder")
        .sort_values("model_test_acc")
    )
    return ref["folder"].tolist()


# ── Secondary axes helpers ────────────────────────────────────────────────────

def _add_param_axis(ax, df, sorted_folders, x):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ref = df.drop_duplicates("folder").set_index("folder")
    labels = []
    for f in sorted_folders:
        p = ref.loc[f, "model_tot_param"] if f in ref.index else np.nan
        labels.append(f"{int(p):,}" if not np.isnan(p) else "?")
    ax2.set_xticklabels(labels, fontsize=6.5, color="#7090c0")
    ax2.set_xlabel("Total parameters →", fontsize=8, color="#7090c0", labelpad=4)
    ax2.tick_params(axis="x", colors="#7090c0")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    return ax2


def _add_accuracy_axis(ax, df, sorted_folders, x):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ref = df.drop_duplicates("folder").set_index("folder")
    labels = []
    for f in sorted_folders:
        a = ref.loc[f, "model_test_acc"] if f in ref.index else np.nan
        labels.append(f"{a:.4f}" if not np.isnan(a) else "?")
    ax2.set_xticklabels(labels, fontsize=6.5, color="#90c070")
    ax2.set_xlabel("Test accuracy →", fontsize=8, color="#90c070", labelpad=4)
    ax2.tick_params(axis="x", colors="#90c070")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    return ax2


# ── Single-metric line plots ───────────────────────────────────────────────────

def plot_metric(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    title: str,
    method: str | None,
    column: str,
    y_label: str,
    output_path: str,
    secondary_axis: str = "params",   # "params" or "accuracy"
):
    sub = df[df["method"] == method] if method else df.drop_duplicates("folder")

    y_vals = []
    for f in sorted_folders:
        row = sub[sub["folder"] == f]
        y_vals.append(
            row[column].values[0]
            if not row.empty and not pd.isna(row[column].values[0])
            else np.nan
        )

    x = np.arange(len(sorted_folders))
    valid = [v for v in y_vals if not np.isnan(v)]
    y_max = max(valid) if valid else 1.0

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(10, len(sorted_folders) * 1.4), 5.5))

        ax.plot(x, y_vals, color=ACCENT_COLOR, linewidth=2,
                marker="o", markersize=6, zorder=3)

        for xi, val in zip(x, y_vals):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.4f}",
                    xy=(xi, val),
                    xytext=(0, 8),
                    textcoords="offset points",
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

        if secondary_axis == "params":
            _add_param_axis(ax, df, sorted_folders, x)
        else:
            _add_accuracy_axis(ax, df, sorted_folders, x)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {output_path}")


def plot_all(df, sorted_folders_params, sorted_folders_acc,
             x_labels_params, x_labels_acc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for title, method, column, y_label in METRICS:
        safe = title.lower().replace(" ", "_").replace("-", "").replace("__", "_")

        plot_metric(df, sorted_folders_params, x_labels_params,
                    title + "  [sorted by parameter count]",
                    method, column, y_label,
                    os.path.join(output_dir, f"{safe}_by_params.png"),
                    secondary_axis="params")

        plot_metric(df, sorted_folders_acc, x_labels_acc,
                    title + "  [sorted by test accuracy]",
                    method, column, y_label,
                    os.path.join(output_dir, f"{safe}_by_accuracy.png"),
                    secondary_axis="accuracy")


# ── Summary grids ─────────────────────────────────────────────────────────────

def plot_summary_grid(df, sorted_folders, x_labels, output_dir, suffix="by_params"):
    n = len(METRICS)
    cols = 2
    rows = (n + 1) // cols
    secondary  = "params" if suffix == "by_params" else "accuracy"
    sort_label = "parameter count" if secondary == "params" else "test accuracy"

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * max(8, len(sorted_folders) * 0.9), rows * 4.5)
        )
        axes = axes.flatten()
        x = np.arange(len(sorted_folders))

        for idx, (title, method, column, y_label) in enumerate(METRICS):
            ax = axes[idx]
            sub = df[df["method"] == method] if method else df.drop_duplicates("folder")

            y_vals = []
            for f in sorted_folders:
                row = sub[sub["folder"] == f]
                y_vals.append(
                    row[column].values[0]
                    if not row.empty and not pd.isna(row[column].values[0])
                    else np.nan
                )

            color = ACCENT_COLOR if method != "gradcam" else ACCENT2_COLOR
            ax.plot(x, y_vals, color=color, linewidth=2,
                    marker="o", markersize=4, zorder=3)
            for xi, val in zip(x, y_vals):
                if not np.isnan(val):
                    ax.annotate(f"{val:.3f}", xy=(xi, val), xytext=(0, 6),
                                textcoords="offset points", ha="center",
                                fontsize=5.5, color=color)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=6, ha="center")
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.grid(axis="both", zorder=0)
            ax.set_xlim(-0.6, len(sorted_folders) - 0.4)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"Model Interpretability Metrics (sorted by {sort_label})",
            fontsize=13, fontweight="bold", y=1.01
        )
        fig.tight_layout()
        out = os.path.join(output_dir, f"summary_grid_{suffix}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {out}")


# ── Multi-metric line plots ───────────────────────────────────────────────────

def plot_multi_metric_line(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    config: dict,
    output_path: str,
):
    """
    Line plot: multiple metrics on the same y-axis, one line per metric.
    model_test_acc is model-level (same value regardless of method row).
    """
    method     = config["method"]
    metrics    = config["metrics"]   # [(label, column, color), ...]
    sort_by    = config["sort_by"]

    method_sub = df[df["method"] == method].set_index("folder")
    model_sub  = df.drop_duplicates("folder").set_index("folder")

    x = np.arange(len(sorted_folders))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(12, len(sorted_folders) * 1.6), 6))

        for label, column, color in metrics:
            source = model_sub if column == "model_test_acc" else method_sub

            y_vals = []
            for f in sorted_folders:
                if f in source.index and not pd.isna(source.loc[f, column]):
                    y_vals.append(float(source.loc[f, column]))
                else:
                    y_vals.append(np.nan)

            ax.plot(x, y_vals,
                    color=color, linewidth=2, marker="o", markersize=5,
                    label=label, zorder=3)

            for xi, val in zip(x, y_vals):
                if not np.isnan(val):
                    ax.annotate(
                        f"{val:.3f}",
                        xy=(xi, val),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center", fontsize=6.5, color=color,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, ha="center")
        ax.set_ylabel("Metric value", fontsize=10)
        ax.set_title(config["title"], fontsize=12, pad=12, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="both", zorder=0)
        ax.set_xlim(-0.6, len(sorted_folders) - 0.4)
        ax.legend(loc="upper left", fontsize=9,
                  facecolor="#1a1d27", edgecolor="#3a3d4d",
                  labelcolor="#e0e0e0")

        if sort_by == "params":
            _add_param_axis(ax, df, sorted_folders, x)
        else:
            _add_accuracy_axis(ax, df, sorted_folders, x)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {output_path}")


def plot_all_line_plots(df, sorted_folders_params, sorted_folders_acc,
                        x_labels_params, x_labels_acc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    folder_map = {"params": sorted_folders_params, "accuracy": sorted_folders_acc}
    label_map  = {"params": x_labels_params,       "accuracy": x_labels_acc}

    for cfg in LINE_PLOT_CONFIGS:
        safe = (
            cfg["title"]
            .lower()
            .replace(" ", "_")
            .replace("[", "").replace("]", "")
            .replace("  ", "_")
            .replace("+", "plus")
            .replace("/", "_")
        )
        out = os.path.join(output_dir, f"line_{safe}.png")
        plot_multi_metric_line(
            df,
            folder_map[cfg["sort_by"]],
            label_map[cfg["sort_by"]],
            cfg,
            out,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot AI model interpretability metrics.")
    parser.add_argument("--root_dir",   default=".",     help="Directory containing model folders")
    parser.add_argument("--output_dir", default="plots", help="Where to save output PNGs")
    args = parser.parse_args()

    print(f"Scanning '{args.root_dir}' for model folders ...")
    df = collect_data(args.root_dir)
    print(f"  found {df['folder'].nunique()} model(s), {len(df)} rows total")

    params_map = {f: parse_folder_name(f) for f in df["folder"].unique()}

    sorted_by_params   = sort_folders_by_params(df)
    sorted_by_accuracy = sort_folders_by_accuracy(df)
    x_labels_params    = make_x_labels(sorted_by_params,   params_map)
    x_labels_acc       = make_x_labels(sorted_by_accuracy, params_map)

    print("\nGenerating single-metric line plots (both orderings) ...")
    plot_all(df, sorted_by_params, sorted_by_accuracy,
             x_labels_params, x_labels_acc, args.output_dir)

    print("\nGenerating summary grids ...")
    plot_summary_grid(df, sorted_by_params,   x_labels_params, args.output_dir, suffix="by_params")
    plot_summary_grid(df, sorted_by_accuracy, x_labels_acc,    args.output_dir, suffix="by_accuracy")

    print("\nGenerating multi-metric line plots ...")
    plot_all_line_plots(df, sorted_by_params, sorted_by_accuracy,
                        x_labels_params, x_labels_acc, args.output_dir)

    print(f"\nDone! All plots saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()