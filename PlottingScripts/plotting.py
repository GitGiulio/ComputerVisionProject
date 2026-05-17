"""
@author: Giulio Lo Cigno

Generates metric comparison plots across models from interpretability result folders.
Folder naming convention:
  interpretability_results_dogs_k=[5,9,{KERNEL_SIZE}]_{CONV_FILTER}_{CONV_LAYER}_{DENSE_NEURON}_{DENSE_LAYER}_wd{WEIGHT_DECAY}_do{DROPOUT}

Optionally place a "baseline/" folder (same structure as the model folders) in the
root directory.  Its metrics_summary.csv will be read and drawn as a horizontal
reference line on every plot (same hue as the metric line, slightly desaturated and
semi-transparent).
"""

import os
import re
import argparse
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
            ("Insertion AUC", "insertion_auc",  "#3a6fd8"),
            ("Deletion AUC",  "deletion_auc",   "#d85a2a"),
           # ("Stability",     "stability",      "#c49a00"),
            ("Test Accuracy", "model_test_acc", "#2ab85a"),
        ],
    },
    {
        "title":   "GradCAM Metrics + Test Accuracy  [sorted by parameter count]",
        "method":  "gradcam",
        "sort_by": "params",
        "metrics": [
            ("Insertion AUC", "insertion_auc",  "#3a6fd8"),
            ("Deletion AUC",  "deletion_auc",   "#d85a2a"),
        #    ("Stability",     "stability",      "#c49a00"),
            ("Test Accuracy", "model_test_acc", "#2ab85a"),
        ],
    },
    {
        "title":   "SHAP Metrics + Test Accuracy  [sorted by test accuracy]",
        "method":  "shap",
        "sort_by": "accuracy",
        "metrics": [
            ("Insertion AUC", "insertion_auc",  "#3a6fd8"),
            ("Deletion AUC",  "deletion_auc",   "#d85a2a"),
       #     ("Stability",     "stability",      "#c49a00"),
            ("Test Accuracy", "model_test_acc", "#2ab85a"),
        ],
    },
    {
        "title":   "GradCAM Metrics + Test Accuracy  [sorted by test accuracy]",
        "method":  "gradcam",
        "sort_by": "accuracy",
        "metrics": [
            ("Insertion AUC", "insertion_auc",  "#3a6fd8"),
            ("Deletion AUC",  "deletion_auc",   "#d85a2a"),
       #     ("Stability",     "stability",      "#c49a00"),
            ("Test Accuracy", "model_test_acc", "#2ab85a"),
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
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f7f8fc",
    "axes.edgecolor":   "#c0c4d0",
    "axes.labelcolor":  "#222233",
    "xtick.color":      "#444455",
    "ytick.color":      "#444455",
    "text.color":       "#222233",
    "grid.color":       "#d8dae8",
    "grid.linestyle":   "--",
    "grid.alpha":       0.8,
    "font.family":      "sans-serif",
}

ACCENT_COLOR  = "#3a6fd8"   # blue  (SHAP / default)
ACCENT2_COLOR = "#d85a2a"   # orange (GradCAM)


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255), int(g * 255), int(b * 255)
    )


def desaturate(hex_color: str, factor: float = 0.45) -> str:
    """Return a desaturated version of *hex_color* (factor 0 = grey, 1 = original)."""
    r, g, b = _hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s * factor, v)
    return _rgb_to_hex(r2, g2, b2)


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


def load_csv(folder_path: str) -> pd.DataFrame | None:
    csv_path = os.path.join(folder_path, "metrics_summary.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["method"] = df["method"].str.strip().str.lower()
    return df


def load_baseline(root_dir: str) -> dict:
    """
    Read <root_dir>/baseline/metrics_summary.csv and return a nested dict:
        { method_or_None: { column: value } }
    where method_or_None is None for model-level columns (model_test_acc etc.).

    Returns an empty dict if no baseline folder / CSV is found.
    """
    baseline_dir = os.path.join(root_dir, "baseline")
    df = load_csv(baseline_dir)
    if df is None:
        print("  [info] No baseline/ folder found – skipping baseline lines.")
        return {}

    baseline: dict = {}
    numeric_cols = ["insertion_auc", "deletion_auc", "stability",
                    "model_test_acc", "model_val_acc", "model_tot_param"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Model-level columns (same across methods) → key None
    model_cols = ["model_test_acc", "model_val_acc", "model_tot_param"]
    baseline[None] = {}
    for col in model_cols:
        if col in df.columns:
            val = df[col].dropna().mean()
            if not np.isnan(val):
                baseline[None][col] = val

    # Method-level columns
    for method, grp in df.groupby("method"):
        baseline[method] = {}
        for col in ["insertion_auc", "deletion_auc", "stability"]:
            if col in grp.columns:
                val = grp[col].dropna().mean()
                if not np.isnan(val):
                    baseline[method][col] = val

    print(f"  [info] Baseline loaded from '{baseline_dir}'.")
    return baseline


def get_baseline_value(baseline: dict, method: str | None, column: str) -> float | None:
    """Return the baseline scalar for (method, column), or None if not available."""
    # model-level columns live under key None
    if column in ("model_test_acc", "model_val_acc", "model_tot_param"):
        return baseline.get(None, {}).get(column)
    if method is not None:
        return baseline.get(method, {}).get(column)
    return None


def collect_data(root_dir: str) -> pd.DataFrame:
    records = []
    scan_dir = os.getcwd() if root_dir == "." else root_dir
    for entry in os.scandir(scan_dir):
        if not entry.is_dir():
            continue
        if entry.name == "baseline":      # skip – handled separately
            continue
        params = parse_folder_name(entry.name)
        if params is None:
            continue
        df = load_csv(entry.path)
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
            f"k{p['kernel_size']}\n"
            f"cf{p['conv_filter']} cl{p['conv_layer']}\n"
            f"dn{p['dense_neuron']} dl{p['dense_layer']}\n"
            f"wd{p['weight_decay']}"
        )
    return labels


def sort_folders_by_kernels(df: pd.DataFrame) -> list[str]:
    #df = df[df["kernel_size"] != "0"]
    #df = df[df["model_test_acc"] > 0.66]
    ref = (
        df[["folder", "kernel_size", "model_test_acc"]]
        .drop_duplicates("folder")
        .sort_values(["kernel_size", "model_test_acc"])
    )
    return ref["folder"].tolist()


def sort_folders_by_accuracy(df: pd.DataFrame) -> list[str]:
    ref = (
        df[["folder", "model_test_acc"]]
        .drop_duplicates("folder")
        .sort_values("model_test_acc")
    )
    return ref["folder"].tolist()


def _add_param_axis(ax, df, sorted_folders, x):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ref = df.drop_duplicates("folder").set_index("folder")
    labels = []
    for f in sorted_folders:
        p = ref.loc[f, "model_tot_param"] if f in ref.index else np.nan
        labels.append(f"{int(p)/1e6:.2f}M" if not np.isnan(p) else "?")
    ax2.set_xticklabels(labels, fontsize=6.5, color="#2255aa")
    ax2.set_xlabel("Total parameters →", fontsize=8, color="#2255aa", labelpad=4)
    ax2.tick_params(axis="x", colors="#2255aa")
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
        labels.append(f".{int(a*1000)}" if not np.isnan(a) else "?")
    ax2.set_xticklabels(labels, fontsize=12, color="#000000")
    ax2.set_xlabel("Test accuracy ->", fontsize=19, color="#000000", labelpad=4)
    ax2.tick_params(axis="x", colors="#000000")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    return ax2


def _draw_baseline_hline(ax, value: float, color: str,
                         label: str = "baseline",
                         alpha: float = 1.0,
                         desaturate_factor: float = 0.45):
    """Draw a horizontal dashed line for a baseline value."""
    ds_color = desaturate(color, desaturate_factor)
    ax.axhline(
        y=value,
        color=ds_color,
        linewidth=1.4,
        linestyle="--",
        alpha=alpha,
        zorder=2,
        label=label,
    )
    # Small text annotation on the right edge
    xlim = ax.get_xlim()
    ax.text(
        xlim[1] - 1.5,
        value,
        f" {value:.3f}",
        color=ds_color,
        fontsize=16,
        va="center",
        alpha=alpha,
        clip_on=False,
    )


def plot_metric(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    title: str,
    method: str | None,
    column: str,
    y_label: str,
    output_path: str,
    baseline: dict,
    secondary_axis: str = "params",
):
    sub_grad = df[df["method"] == "gradcam"] if method else df.drop_duplicates("folder")
    sub_shap = df[df["method"] == "shap"] if method else df.drop_duplicates("folder")

    y_vals_grad = []
    y_vals_shap = []
    for f in sorted_folders:
        row_grad = sub_grad[sub_grad["folder"] == f]
        row_shap = sub_shap[sub_shap["folder"] == f]
        y_vals_grad.append(
            row_grad[column].values[0]
            if not row_grad.empty and not pd.isna(row_grad[column].values[0])
            else np.nan
        )
        y_vals_shap.append(
            row_shap[column].values[0]
            if not row_shap.empty and not pd.isna(row_shap[column].values[0])
            else np.nan
        )

    x = np.arange(len(sorted_folders))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(10, len(sorted_folders) * 1.0), 8))

        ax.plot(x, y_vals_grad, color=ACCENT2_COLOR, linewidth=2,
                marker="o", markersize=6, zorder=3, label="shap - stability")
        ax.plot(x, y_vals_shap, color=ACCENT_COLOR, linewidth=2,
                marker="o", markersize=6, zorder=3, label="gradcam - stability")
        
        for xi, val in zip(x, y_vals_grad):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.4f}",
                    xy=(xi, val),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=13, color="#983b16",
                )
        for xi, val in zip(x, y_vals_shap):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.4f}",
                    xy=(xi, val),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=13, color="#333355",
                )

        # Baseline line
        bval_shap = get_baseline_value(baseline, "shap", column)
        if bval_shap is not None:
            _draw_baseline_hline(ax, bval_shap, ACCENT_COLOR, label="shap - baseline")
            ax.legend(loc="upper left", fontsize=27,
                      facecolor="#ffffff", edgecolor="#c0c4d0")
        bval_grad = get_baseline_value(baseline, "gradcam", column)
        if bval_grad is not None:
            _draw_baseline_hline(ax, bval_grad, ACCENT2_COLOR, label="gradcam - baseline")
            ax.legend(loc="upper left", fontsize=27,
                      facecolor="#ffffff", edgecolor="#c0c4d0")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, ha="center")
        ax.set_ylabel("Stability", fontsize=16)
        ax.set_title("Stability comparison", fontsize=30, pad=12, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
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
             x_labels_kernels, x_labels_acc, output_dir, baseline):
    os.makedirs(output_dir, exist_ok=True)
    for title, method, column, y_label in METRICS:
        if column != "stability":
            continue
        safe = title.lower().replace(" ", "_").replace("-", "").replace("__", "_")

        plot_metric(df, sorted_folders_params, x_labels_kernels,
                    title + "  [sorted by parameter count]",
                    method, column, y_label,
                    os.path.join(output_dir, f"{safe}_by_params.png"),
                    baseline, secondary_axis="params")

        plot_metric(df, sorted_folders_acc, x_labels_acc,
                    title + "  [sorted by test accuracy]",
                    method, column, y_label,
                    os.path.join(output_dir, f"{safe}_by_accuracy.png"),
                    baseline, secondary_axis="accuracy")


def plot_summary_grid(df, sorted_folders, x_labels, output_dir,
                      baseline: dict, suffix="by_params"):
    n = len(METRICS)
    cols = 2
    rows = (n + 1) // cols
    secondary  = "params" if suffix == "by_params" else "accuracy"
    sort_label = "parameter count" if secondary == "params" else "test accuracy"

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * max(8, len(sorted_folders) * 1.0), rows * 4.5)
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

            # Baseline line
            bval = get_baseline_value(baseline, method, column)
            if bval is not None:
                _draw_baseline_hline(ax, bval, color)

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


def plot_multi_metric_line(
    df: pd.DataFrame,
    sorted_folders: list[str],
    x_labels: list[str],
    config: dict,
    output_path: str,
    baseline: dict,
):
    method     = config["method"]
    metrics    = config["metrics"]   # [(label, column, color), ...]
    sort_by    = config["sort_by"]

    method_sub = df[df["method"] == method].set_index("folder")
    model_sub  = df.drop_duplicates("folder").set_index("folder")

    x = np.arange(len(sorted_folders))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(12, len(sorted_folders) * 1.0), 8))

        baseline_handles = []   # collect for legend dedup

        for label, column, color in metrics:
            source = model_sub if column == "model_test_acc" else method_sub

            y_vals = []
            for f in sorted_folders:
                if f in source.index and not pd.isna(source.loc[f, column]):
                    y_vals.append(float(source.loc[f, column]))
                else:
                    y_vals.append(np.nan)

            display_label = label
            bval = get_baseline_value(baseline, method, column)

            if label == "Deletion AUC":
                y_vals = list(1 - np.array(y_vals))
                display_label = "1 - Deletion AUC"
                if bval is not None:
                    bval = 1 - bval

            ax.plot(x, y_vals,
                    color=color, linewidth=2, marker="o", markersize=5,
                    label=display_label, zorder=3)

            for xi, val in zip(x, y_vals):
                if not np.isnan(val):
                    ax.annotate(
                        f"{val:.3f}",
                        xy=(xi, val),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center", fontsize=13, color=color,
                    )

            # Baseline horizontal line (same colour, desaturated)
            if bval is not None:
                _draw_baseline_hline(
                    ax, bval, color,
                    label=f"{display_label} (baseline)"
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9, ha="center")
        ax.set_ylabel("Metric value", fontsize=20)
        ax.set_title("SHAP fidelity x model accuracy", fontsize=30, pad=12, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="both", zorder=0)
        ax.set_xlim(-0.6, len(sorted_folders) - 0.4)
        ax.legend(loc="lower right", fontsize=27,
                  facecolor="#ffffff", edgecolor="#c0c4d0",
                  labelcolor="#222233")

        if sort_by == "params":
            _add_param_axis(ax, df, sorted_folders, x)
        else:
            _add_accuracy_axis(ax, df, sorted_folders, x)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ref = df.drop_duplicates("folder").set_index("folder")
        labels = []
        for i,f in enumerate(sorted_folders):
            if i == len(sorted_folders) - 7:
                labels.append("11")
            elif i == len(sorted_folders) - 2:
                labels.append("15")
            elif i == len(sorted_folders) - 1:
                labels.append("19")
            else:
                labels.append("")

        ax2.set_xticklabels(labels, fontsize=12, color="#000000")
        #ax2.set_xlabel("Best for kernel size:", fontsize=19, color="#000000", labelpad=4)
        ax2.tick_params(axis="x", colors="#000000")
        for spine in ax2.spines.values():
            spine.set_visible(False)

        highlight_positions = [i for i, lbl in enumerate(labels) if lbl != ""]

        for xi in highlight_positions:
            ax.axvline(x=xi, color="#E54AC3", linestyle="-", linewidth=17.0, zorder=2, alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {output_path}")


def plot_all_line_plots(df, sorted_folders_params, sorted_folders_acc,
                        x_labels_kernels, x_labels_acc, output_dir, baseline):
    os.makedirs(output_dir, exist_ok=True)
    folder_map = {"params": sorted_folders_params, "accuracy": sorted_folders_acc}
    label_map  = {"params": x_labels_kernels,       "accuracy": x_labels_acc}

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
            baseline,
        )


def main():
    parser = argparse.ArgumentParser(description="Plot AI model interpretability metrics.")
    parser.add_argument("--root_dir",   default=".",     help="Directory containing model folders")
    parser.add_argument("--output_dir", default="plots", help="Where to save output PNGs")
    args = parser.parse_args()

    print(f"Scanning '{args.root_dir}' for model folders ...")

    # Load baseline BEFORE collecting model data (uses the same root_dir)
    baseline = load_baseline(args.root_dir)

    df = collect_data(args.root_dir)
    print(f"  found {df['folder'].nunique()} model(s), {len(df)} rows total")

    params_map = {f: parse_folder_name(f) for f in df["folder"].unique()}

    sorted_by_kernels   = sort_folders_by_kernels(df)
    sorted_by_accuracy = sort_folders_by_accuracy(df)
    x_labels_kernels    = make_x_labels(sorted_by_kernels,   params_map)
    x_labels_acc       = make_x_labels(sorted_by_accuracy, params_map)

    #print("\nGenerating single-metric line plots (both orderings) ...")
    #plot_all(df, sorted_by_kernels, sorted_by_accuracy,x_labels_kernels, x_labels_acc, args.output_dir, baseline)

    #print("\nGenerating summary grids ...")
    #plot_summary_grid(df, sorted_by_kernels,   x_labels_kernels, args.output_dir, baseline, suffix="by_params")
    #plot_summary_grid(df, sorted_by_accuracy, x_labels_acc,    args.output_dir, baseline, suffix="by_accuracy")

    print("\nGenerating multi-metric line plots ...")
    plot_all_line_plots(df, sorted_by_kernels, sorted_by_accuracy,x_labels_kernels, x_labels_acc, args.output_dir, baseline)

    print(f"\nDone! All plots saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()