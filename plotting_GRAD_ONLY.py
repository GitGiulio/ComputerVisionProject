"""
plot_model_metrics.py

Generates GradCAM fidelity metric comparison plots across models.

Folder naming convention:
  interpretability_results_dogs_k=[5,9,{KERNEL_SIZE}]_{CONV_FILTER}_{CONV_LAYER}_{DENSE_NEURON}_{DENSE_LAYER}_wd{WEIGHT_DECAY}_do{DROPOUT}

Expected folder structure:
  interpretability_results_dogs_.../
      black/
          metrics_summary.csv
      mean/
          metrics_summary.csv
      blur/
          metrics_summary.csv

Usage:
  python plot_model_metrics.py --root_dir . --output_dir ./plots
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Config ──────────────────────────────────────────────────────────────────

MASK_TYPES = ["black", "mean", "blur"]

MASK_COLORS = {
    "black": "#2563fe",
    "mean":  "#ff7b43",
    "blur":  "#00A832",
}

# Only GradCAM fidelity metrics
METRICS = [
    ("GradCAM - Insertion AUC", "insertion_auc", "Insertion AUC"),
    ("GradCAM - Deletion AUC",  "deletion_auc",  "Deletion AUC"),
    # ("GradCAM - Stability",     "gradcam", "stability",      "Stability"),
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


# ── Folder Parsing ──────────────────────────────────────────────────────────

def parse_folder_name(folder_name: str) -> dict | None:

    base = os.path.basename(folder_name.rstrip("/\\"))

    m = FOLDER_RE.match(base)

    if not m:
        return None

    d = m.groupdict()

    d["kernel_size"] = (
        d.pop("kernel_size_a")
        or d.pop("kernel_size_b")
    )

    try:
        d["weight_decay"] = float(d["weight_decay"])
    except ValueError:
        pass

    return d


# ── Data Loading ────────────────────────────────────────────────────────────

def load_folder(folder_path: str) -> list[pd.DataFrame]:
    """
    Loads:
        black/metrics_summary.csv
        mean/metrics_summary.csv
        blur/metrics_summary.csv
    """

    dfs = []

    for mask_type in MASK_TYPES:

        csv_path = os.path.join(
            folder_path,
            mask_type,
            "metrics_summary.csv"
        )

        if not os.path.exists(csv_path):
            print(f"  [warn] missing: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [warn] failed reading {csv_path}: {e}")
            continue

        df.columns = df.columns.str.strip()

        if "method" in df.columns:
            df["method"] = (
                df["method"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        df["mask_type"] = mask_type

        dfs.append(df)

    return dfs


def collect_data(root_dir: str) -> pd.DataFrame:

    records = []

    for entry in os.scandir(root_dir):

        if not entry.is_dir():
            continue

        params = parse_folder_name(entry.name)

        if params is None:
            continue

        dfs = load_folder(entry.path)

        if not dfs:
            print(f"  [warn] no metric CSVs in {entry.name}")
            continue

        for df in dfs:

            for _, row in df.iterrows():

                rec = {
                    "folder": entry.name,
                    **params,
                    **row.to_dict(),
                }

                records.append(rec)

    if not records:
        raise ValueError(
            f"No valid model folders found under '{root_dir}'"
        )

    combined = pd.DataFrame(records)

    numeric_cols = [
        "model_tot_param",
        "weight_decay",
        "insertion_auc",
        "deletion_auc",
        "model_test_acc",
        "model_val_acc",
    ]

    for col in numeric_cols:

        if col in combined.columns:
            combined[col] = pd.to_numeric(
                combined[col],
                errors="coerce"
            )

    return combined


# ── Sorting Helpers ─────────────────────────────────────────────────────────

def sort_folders_by_params(df: pd.DataFrame) -> list[str]:

    ref = (
        df[["folder", "model_tot_param", "weight_decay"]]
        .drop_duplicates("folder")
        .sort_values(
            ["model_tot_param", "weight_decay"],
            ascending=True,
        )
    )

    return ref["folder"].tolist()


def sort_folders_by_accuracy(df: pd.DataFrame) -> list[str]:

    ref = (
        df[["folder", "model_test_acc"]]
        .drop_duplicates("folder")
        .sort_values(
            "model_test_acc",
            ascending=True,
        )
    )

    return ref["folder"].tolist()


# ── Label Helpers ───────────────────────────────────────────────────────────

def make_x_labels(
    sorted_folders: list[str],
    params_map: dict,
) -> list[str]:

    labels = []

    for f in sorted_folders:

        p = params_map[f]

        labels.append(
            f"k{p['kernel_size']}\n"
            f"cf{p['conv_filter']} cl{p['conv_layer']}\n"
            f"dn{p['dense_neuron']} dl{p['dense_layer']}\n"
            f"wd{p['weight_decay']}"
        )

    return labels


# ── Secondary Axes ──────────────────────────────────────────────────────────

def _add_param_axis(ax, df, sorted_folders, x):

    ax2 = ax.twiny()

    ax2.set_xlim(ax.get_xlim())

    ax2.set_xticks(x)

    ref = (
        df
        .drop_duplicates("folder")
        .set_index("folder")
    )

    labels = []

    for f in sorted_folders:

        p = (
            ref.loc[f, "model_tot_param"]
            if f in ref.index
            else np.nan
        )

        labels.append(
            f"{int(p)/1_000_000:.2f}M"
            if not np.isnan(p)
            else "?"
        )

    ax2.set_xticklabels(
        labels,
        fontsize=6.5,
        color="#4466aa",
    )

    ax2.set_xlabel(
        "Total parameters ->",
        fontsize=8,
        color="#4466aa",
        labelpad=4,
    )

    ax2.tick_params(axis="x", colors="#4466aa")

    for spine in ax2.spines.values():
        spine.set_visible(False)

    return ax2


def _add_accuracy_axis(ax, df, sorted_folders, x):

    ax2 = ax.twiny()

    ax2.set_xlim(ax.get_xlim())

    ax2.set_xticks(x)

    ref = (
        df
        .drop_duplicates("folder")
        .set_index("folder")
    )

    labels = []

    for f in sorted_folders:

        a = (
            ref.loc[f, "model_test_acc"]
            if f in ref.index
            else np.nan
        )

        labels.append(
            f"{a:.2f}"
            if not np.isnan(a)
            else "?"
        )

    ax2.set_xticklabels(
        labels,
        fontsize=20,
        color="#000000",
    )

    ax2.set_xlabel(
        "Test accuracy ->",
        fontsize=24,
        color="#000000",
        labelpad=4,
    )

    ax2.tick_params(axis="x", colors="#000000")

    for spine in ax2.spines.values():
        spine.set_visible(False)

    return ax2


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_metric_with_masks(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    metric_column: str,
    metric_title: str,
    y_label: str,
    output_path: str,
    secondary_axis: str = "params",
):

    x = np.arange(len(sorted_folders))

    with plt.rc_context(STYLE):

        fig, ax = plt.subplots(
            figsize=(max(12, len(sorted_folders) * 1.0), 8)
        )

        for mask_type in MASK_TYPES:

            sub = (
                df[df["mask_type"] == mask_type]
                .drop_duplicates("folder")
                .set_index("folder")
            )

            y_vals = []

            for f in sorted_folders:

                if (
                    f in sub.index
                    and not pd.isna(sub.loc[f, metric_column])
                ):
                    y_vals.append(
                        float(sub.loc[f, metric_column])
                    )
                else:
                    y_vals.append(np.nan)

            color = MASK_COLORS[mask_type]

            ax.plot(
                x,
                y_vals,
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                label=mask_type,
                zorder=3,
            )

            for xi, val in zip(x, y_vals):

                if not np.isnan(val):

                    ax.annotate(
                        f"{val:.2f}",
                        xy=(xi, val),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=25,
                        color=color,
                    )

        ax.set_xticks(x)

        ax.set_xticklabels(
            x_labels,
            fontsize=9,
            ha="center",
        )

        ax.set_ylabel(y_label, fontsize=35)

        ax.set_title(
            metric_title,
            fontsize=25,
            pad=12,
            fontweight="bold",
        )

        ax.yaxis.set_major_formatter(
            ticker.FormatStrFormatter("%.3f")
        )

        ax.tick_params(axis="y", labelsize=16)

        ax.grid(axis="both", zorder=0)

        ax.set_xlim(-0.6, len(sorted_folders) - 0.4)

        ax.legend(
            loc="upper left",
            fontsize=35,
            title_fontsize=35,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            labelcolor="#111111",
            title="Mask Type",
        )

        if secondary_axis == "params":
            _add_param_axis(
                ax,
                df,
                sorted_folders,
                x,
            )
        else:
            _add_accuracy_axis(
                ax,
                df,
                sorted_folders,
                x,
            )

        fig.tight_layout()

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

        plt.close(fig)

    print(f"  saved -> {output_path}")


def plot_all(
    df,
    sorted_folders_params,
    sorted_folders_acc,
    x_labels_params,
    x_labels_acc,
    output_dir,
):

    os.makedirs(output_dir, exist_ok=True)

    for title, column, y_label in METRICS:

        safe = (
            title.lower()
            .replace(" ", "_")
            .replace("-", "")
        )

        # Sorted by parameter count
        plot_metric_with_masks(
            df=df,
            sorted_folders=sorted_folders_params,
            x_labels=x_labels_params,
            metric_column=column,
            metric_title=f"{title} [sorted by parameter count]",
            y_label=y_label,
            output_path=os.path.join(
                output_dir,
                f"{safe}_by_params.png"
            ),
            secondary_axis="params",
        )

        # Sorted by test accuracy
        plot_metric_with_masks(
            df=df,
            sorted_folders=sorted_folders_acc,
            x_labels=x_labels_acc,
            metric_column=column,
            metric_title=f"{title} [sorted by test accuracy]",
            y_label=y_label,
            output_path=os.path.join(
                output_dir,
                f"{safe}_by_accuracy.png"
            ),
            secondary_axis="accuracy",
        )


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="Plot GradCAM fidelity metrics."
    )

    parser.add_argument(
        "--root_dir",
        default="gradcam_dogs_only_results",
        help="Directory containing model folders"
    )

    parser.add_argument(
        "--output_dir",
        default="gradcam_dogs_only_plots",
        help="Where to save output PNGs"
    )

    args = parser.parse_args()

    print(f"Scanning '{args.root_dir}' for model folders ...")

    df = collect_data(args.root_dir)

    print(
        f"  found {df['folder'].nunique()} model(s), "
        f"{len(df)} rows total"
    )

    params_map = {
        f: parse_folder_name(f)
        for f in df["folder"].unique()
    }

    sorted_by_params = sort_folders_by_params(df)

    sorted_by_accuracy = sort_folders_by_accuracy(df)

    x_labels_params = make_x_labels(
        sorted_by_params,
        params_map
    )

    x_labels_acc = make_x_labels(
        sorted_by_accuracy,
        params_map
    )

    print("\nGenerating fidelity plots ...")

    plot_all(
        df,
        sorted_by_params,
        sorted_by_accuracy,
        x_labels_params,
        x_labels_acc,
        args.output_dir,
    )

    print(
        f"\nDone! All plots saved to "
        f"'{args.output_dir}/'"
    )


if __name__ == "__main__":
    main()