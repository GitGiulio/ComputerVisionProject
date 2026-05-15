"""
@author: Giulio Lo Cigno
variation of finetune_plotting.py

Run this script from inside a model result folder that contains:
  - base.csv        (metrics before finetuning  -> epoch 0)
  - epoch1.csv      (metrics after epoch 1)
  - epoch2.csv      (metrics after epoch 2)
  - ...             (any number of epochs)

Output PNGs are saved to ./epoch_plots/ inside the same folder.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

STYLE = {
    "figure.facecolor": "#f7f8fc",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#c8ccd8",
    "axes.labelcolor":  "#2a2d3d",
    "xtick.color":      "#555870",
    "ytick.color":      "#555870",
    "text.color":       "#2a2d3d",
    "grid.color":       "#dde0ea",
    "grid.linestyle":   "--",
    "grid.alpha":       0.8,
    "font.family":      "sans-serif",
}

COLOR_BASE     = "#9098b0"   # muted blue-grey - base reference line
EPOCH_PALETTE  = [           # one colour per metric line in the overview
    "#3a6fd8",
    "#d85a20",
    "#1e9e5a",
    "#b8860b",
    "#7a3ab8",
    "#c0306a",
    "#1898a8",
]
COLOR_IMPROVE  = "#1e9e5a"
COLOR_REGRESS  = "#d83040"

# (display_title, method_filter, csv_column, y_label, higher_is_better)
METRICS = [
    ("Test Accuracy",           None,      "model_test_acc", "Accuracy",      True),
    ("SHAP - Insertion AUC",    "shap",    "insertion_auc",  "Insertion AUC", True),
    ("SHAP - Deletion AUC",     "shap",    "deletion_auc",   "Deletion AUC",  False),
    ("GradCAM - Insertion AUC", "gradcam", "insertion_auc",  "Insertion AUC", True),
    ("GradCAM - Deletion AUC",  "gradcam", "deletion_auc",   "Deletion AUC",  False),
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "method" in df.columns:
        df["method"] = df["method"].str.strip().str.lower()
    for col in ["model_tot_param", "model_test_acc", "model_val_acc",
                "insertion_auc", "deletion_auc", "stability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def extract_value(df: pd.DataFrame, method: str | None, column: str) -> float:
    if method is None:
        vals = df[column].dropna() if column in df.columns else pd.Series(dtype=float)
    else:
        sub  = df[df["method"] == method]
        vals = sub[column].dropna() if column in sub.columns else pd.Series(dtype=float)
    return float(vals.iloc[0]) if not vals.empty else np.nan


def discover_epochs(cwd: str) -> list[tuple[int, pd.DataFrame]]:
    """
    Returns [(epoch_number, dataframe), ...] sorted by epoch number.
    base.csv is epoch 0.
    """
    entries = []

    base_path = os.path.join(cwd, "base.csv")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"base.csv not found in {cwd}")
    entries.append((0, load_csv(base_path)))

    epoch_re = re.compile(r"^epoch(\d+)\.csv$", re.IGNORECASE)
    for fname in os.listdir(cwd):
        m = epoch_re.match(fname)
        if m:
            n  = int(m.group(1))
            df = load_csv(os.path.join(cwd, fname))
            entries.append((n, df))

    if len(entries) == 1:
        raise FileNotFoundError("No epoch*.csv files found beside base.csv")

    entries.sort(key=lambda t: t[0])
    return entries


def build_series(
    epochs: list[tuple[int, pd.DataFrame]],
    method: str | None,
    column: str,
) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for epoch_n, df in epochs:
        val = extract_value(df, method, column)
        xs.append(epoch_n)
        ys.append(val)
    return xs, ys


def _annotate_change(ax, xs, ys, color_up, color_dn, fontsize=7.5):
    """Annotate each point with its value and colour the marker by direction."""
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        if np.isnan(yi):
            continue
        ax.annotate(
            f"{yi:.4f}",
            xy=(xi, yi),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center", fontsize=fontsize, color="#2a2d3d",
        )
        if i == 0:
            ax.plot(xi, yi, "o", color=COLOR_BASE, markersize=7, zorder=5)
        else:
            prev = next((ys[j] for j in range(i - 1, -1, -1) if not np.isnan(ys[j])), None)
            if prev is not None:
                mc = color_up if yi >= prev else color_dn
            else:
                mc = "#e0e0e0"
            ax.plot(xi, yi, "o", color=mc, markersize=7, zorder=5)


def _x_ticks(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["Base" if x == 0 else f"Epoch {x}" for x in xs],
        fontsize=8,
    )
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)


def plot_metric_epoch(
    epochs: list[tuple[int, pd.DataFrame]],
    title: str,
    method: str | None,
    column: str,
    y_label: str,
    higher_is_better: bool,
    output_path: str,
    color: str = "#7c9ef7",
):
    xs, ys = build_series(epochs, method, column)

    if all(np.isnan(v) for v in ys):
        print(f"  [skip] {title} - no data")
        return

    color_up = COLOR_IMPROVE if higher_is_better else COLOR_REGRESS
    color_dn = COLOR_REGRESS if higher_is_better else COLOR_IMPROVE

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(xs) * 1.3), 5.5))

        # Shade area under the curve from the base value
        base_val = ys[0] if not np.isnan(ys[0]) else 0
        ax.axhline(base_val, color=COLOR_BASE, linewidth=1.2,
                   linestyle=":", alpha=0.6, label=f"Base ({base_val:.4f})")

        # Fill between curve and baseline
        ys_arr = np.array([v if not np.isnan(v) else base_val for v in ys])
        above  = np.where(ys_arr >= base_val, ys_arr, base_val)
        below  = np.where(ys_arr <  base_val, ys_arr, base_val)
        ax.fill_between(xs, base_val, above, alpha=0.12, color=color_up, zorder=1)
        ax.fill_between(xs, base_val, below, alpha=0.12, color=color_dn, zorder=1)

        ax.plot(xs, ys, color=color, linewidth=2.2, zorder=4, label=title)

        _annotate_change(ax, xs, ys, color_up, color_dn)
        _x_ticks(ax, xs)

        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(
            f"{title}  -  Epoch progression\n"
            f"{'↑ higher is better' if higher_is_better else '↓ lower is better'}",
            fontsize=11, fontweight="bold", pad=10,
        )
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="y", zorder=0)
        ax.legend(fontsize=8, facecolor="#ffffff", edgecolor="#c8ccd8",
                  labelcolor="#2a2d3d", loc="best")

        # Delta from base in corner
        valid_ys = [v for v in ys[1:] if not np.isnan(v)]
        if valid_ys and not np.isnan(ys[0]):
            final_delta = valid_ys[-1] - ys[0]
            sign  = "+" if final_delta >= 0 else ""
            good  = (final_delta >= 0) == higher_is_better
            dcol  = COLOR_IMPROVE if good else COLOR_REGRESS
            ax.text(
                0.99, 0.04,
                f"Total Δ: {sign}{final_delta:.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=dcol, fontweight="bold",
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {output_path}")


def plot_overview(
    epochs: list[tuple[int, pd.DataFrame]],
    output_path: str,
    folder_name: str,
):
    """
    One subplot per metric, all sharing the same x-axis epoch scale.
    Deletion AUC is shown as-is but annotated '↓ lower is better'.
    """
    n_metrics  = len(METRICS)
    cols       = 2
    rows       = (n_metrics + 1) // cols
    xs_global  = [e[0] for e in epochs]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * max(7, len(xs_global) * 1.2), rows * 4.5),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for idx, (title, method, column, y_label, hib) in enumerate(METRICS):
            ax    = axes_flat[idx]
            color = EPOCH_PALETTE[idx % len(EPOCH_PALETTE)]
            xs, ys = build_series(epochs, method, column)

            if all(np.isnan(v) for v in ys):
                ax.set_visible(False)
                continue

            base_val = ys[0] if not np.isnan(ys[0]) else 0
            ax.axhline(base_val, color=COLOR_BASE, linewidth=1,
                       linestyle=":", alpha=0.5)

            color_up = COLOR_IMPROVE if hib else COLOR_REGRESS
            color_dn = COLOR_REGRESS if hib else COLOR_IMPROVE
            ys_arr   = np.array([v if not np.isnan(v) else base_val for v in ys])
            ax.fill_between(xs, base_val,
                            np.where(ys_arr >= base_val, ys_arr, base_val),
                            alpha=0.10, color=color_up)
            ax.fill_between(xs, base_val,
                            np.where(ys_arr <  base_val, ys_arr, base_val),
                            alpha=0.10, color=color_dn)

            ax.plot(xs, ys, color=color, linewidth=2, zorder=4)
            _annotate_change(ax, xs, ys, color_up, color_dn, fontsize=6.5)
            _x_ticks(ax, xs)

            direction = "↑ higher better" if hib else "↓ lower better"
            ax.set_title(f"{title}  [{direction}]", fontsize=9,
                         fontweight="bold", pad=6)
            ax.set_ylabel(y_label, fontsize=8)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.grid(axis="y", zorder=0)

            valid_ys = [v for v in ys[1:] if not np.isnan(v)]
            if valid_ys and not np.isnan(ys[0]):
                fd   = valid_ys[-1] - ys[0]
                good = (fd >= 0) == hib
                dcol = COLOR_IMPROVE if good else COLOR_REGRESS
                ax.text(0.98, 0.04, f"Δ {fd:+.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color=dcol, fontweight="bold")

        # Hide unused subplots
        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            f"Finetuning epoch progression - {folder_name}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {output_path}")


def main():
    cwd         = os.getcwd()
    folder_name = os.path.basename(cwd)
    output_dir  = os.path.join(cwd, "epoch_plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning: {cwd}")
    epochs = discover_epochs(cwd)
    epoch_labels = ["base" if n == 0 else f"epoch{n}" for n, _ in epochs]
    print(f"  found {len(epochs)} checkpoints: {epoch_labels}")
    print(f"Saving plots to: {output_dir}\n")

    print("── Per-metric line plots ──")
    for idx, (title, method, column, y_label, hib) in enumerate(METRICS):
        color    = EPOCH_PALETTE[idx % len(EPOCH_PALETTE)]
        safe     = (title.lower()
                        .replace(" ", "_")
                        .replace("-", "")
                        .replace("__", "_")
                        .strip("_"))
        out_path = os.path.join(output_dir, f"epoch_{safe}.png")
        plot_metric_epoch(epochs, title, method, column, y_label, hib,
                          out_path, color=color)

    print("\n── Overview grid ──")
    plot_overview(epochs, os.path.join(output_dir, "overview_all_metrics.png"),
                  folder_name)

    print(f"\nDone! All plots saved to '{output_dir}/'")


if __name__ == "__main__":
    main()