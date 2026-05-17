"""
@author: Giulio Lo Cigno

Run this script from inside a model result folder that contains:
  - base.csv       (metrics before finetuning)
  - finetuned.csv  (metrics after finetuning)

Output PNGs are saved to ./finetune_plots/ inside the same folder.
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

COLOR_BASE      = "#3a6fd8"   # blue  - base model
COLOR_FINETUNED = "#d85a20"   # orange - finetuned model
COLOR_POSITIVE  = "#1e9e5a"   # green  - improvement annotation
COLOR_NEGATIVE  = "#d83040"   # red    - regression annotation

# Each entry: (display_title, method_filter, csv_column, y_label, higher_is_better)
METRICS = [
    ("Test Accuracy",        None,      "model_test_acc", "Accuracy",      True),
    ("SHAP: iAUC", "shap",    "insertion_auc",  "Insertion AUC", True),
    ("SHAP: -dAUC",  "shap",    "deletion_auc",   "Deletion AUC",  False),  # lower = better
    ("GradCAM: iAUC", "gradcam", "insertion_auc", "Insertion AUC", True),
    ("GradCAM: -dAUC",  "gradcam", "deletion_auc",  "Deletion AUC",  False),
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
    """Pull a single scalar from a metrics dataframe."""
    if method is None:
        # model-level metric - take first non-NaN value
        vals = df[column].dropna() if column in df.columns else pd.Series(dtype=float)
    else:
        sub = df[df["method"] == method]
        vals = sub[column].dropna() if column in sub.columns else pd.Series(dtype=float)

    return float(vals.iloc[0]) if not vals.empty else np.nan


def delta_color(delta: float, higher_is_better: bool) -> str:
    improved = delta > 0 if higher_is_better else delta < 0
    return COLOR_POSITIVE if improved else COLOR_NEGATIVE


def delta_arrow(delta: float, higher_is_better: bool) -> str:
    improved = delta > 0 if higher_is_better else delta < 0
    return "▲" if improved else "▼"


def plot_bar_comparison(
    base_df: pd.DataFrame,
    ft_df: pd.DataFrame,
    output_dir: str,
    folder_name: str,
) -> None:
    """One grouped-bar plot per metric."""
    os.makedirs(output_dir, exist_ok=True)

    for title, method, column, y_label, higher_is_better in METRICS:
        v_base = extract_value(base_df, method, column)
        v_ft   = extract_value(ft_df,   method, column)

        if np.isnan(v_base) and np.isnan(v_ft):
            print(f"  [skip] {title} - no data in either CSV")
            continue

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(5, 5.5))

            bars = ax.bar(
                ["Base", "Finetuned"],
                [v_base if not np.isnan(v_base) else 0,
                 v_ft   if not np.isnan(v_ft)   else 0],
                color=[COLOR_BASE, COLOR_FINETUNED],
                width=0.4,
                zorder=3,
            )

            # Value labels on bars
            for bar, val in zip(bars, [v_base, v_ft]):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (ax.get_ylim()[1] * 0.01),
                        f"{val:.4f}",
                        ha="center", va="bottom", fontsize=10,
                        color="#2a2d3d", fontweight="bold",
                    )

            # Delta annotation
            if not np.isnan(v_base) and not np.isnan(v_ft):
                delta = v_ft - v_base
                pct   = (delta / v_base * 100) if v_base != 0 else 0
                col   = delta_color(delta, higher_is_better)
                arrow = delta_arrow(delta, higher_is_better)
                ax.text(
                    0.5, 0.97,
                    f"{arrow}  Δ {delta:+.4f}  ({pct:+.2f}%)",
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=10, color=col, fontweight="bold",
                )

            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(f"{title}\n{folder_name}", fontsize=10, fontweight="bold", pad=10)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.grid(axis="y", zorder=0)

            # Tight y-range
            all_vals = [v for v in [v_base, v_ft] if not np.isnan(v)]
            if all_vals:
                span = max(all_vals) - min(all_vals)
                pad  = max(span * 0.3, 0.02)
                ax.set_ylim(max(0, min(all_vals) - pad), max(all_vals) + pad * 2.5)

            fig.tight_layout()
            safe = title.lower().replace(" ", "_").replace("-", "").replace("__", "_")
            out  = os.path.join(output_dir, f"bar_{safe}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
        print(f"  saved -> {out}")


def plot_summary(
    base_df: pd.DataFrame,
    ft_df: pd.DataFrame,
    output_dir: str,
    folder_name: str,
) -> None:
    """
    Single figure with all metrics side-by-side as grouped bars,
    plus a delta table at the bottom.
    """
    os.makedirs(output_dir, exist_ok=True)

    titles, base_vals, ft_vals, hib_flags = [], [], [], []
    for title, method, column, _, higher_is_better in METRICS:
        v_base = extract_value(base_df, method, column)
        v_ft   = extract_value(ft_df,   method, column)
        if np.isnan(v_base) and np.isnan(v_ft):
            continue
        titles.append(title)
        base_vals.append(v_base)
        ft_vals.append(v_ft)
        hib_flags.append(higher_is_better)

    n = len(titles)
    if n == 0:
        print("  [warn] no metrics to plot in summary")
        return

    x = np.arange(n)
    w = 0.32

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(10, n * 2.2), 6.5))

        bars_b = ax.bar(x - w / 2, base_vals, width=w, color=COLOR_BASE,
                        label="Base", zorder=3, alpha=0.92)
        bars_f = ax.bar(x + w / 2, ft_vals,   width=w, color=COLOR_FINETUNED,
                        label="Finetuned", zorder=3, alpha=0.92)

        # Value labels
        for bar, val in list(zip(bars_b, base_vals)) + list(zip(bars_f, ft_vals)):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#2a2d3d",
                )

        # Delta labels between bar pairs
        for i, (bv, fv, hib) in enumerate(zip(base_vals, ft_vals, hib_flags)):
            if not (np.isnan(bv) or np.isnan(fv)):
                delta = fv - bv
                col   = delta_color(delta, hib)
                arrow = delta_arrow(delta, hib)
                y_top = max(bv, fv) + 0.025
                ax.text(
                    x[i], y_top,
                    f"{arrow}{abs(delta):.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color=col, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(titles, fontsize=9, ha="center", wrap=True)
        ax.set_ylabel("Metric value", fontsize=11)
        ax.set_title(
            f"Base vs Finetuned - All Metrics\n{folder_name}",
            fontsize=12, fontweight="bold", pad=12,
        )
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", zorder=0)
        ax.set_xlim(-0.7, n - 0.3)
        ax.legend(loc="upper right", fontsize=10,
                  facecolor="#ffffff", edgecolor="#c8ccd8", labelcolor="#2a2d3d")

        fig.tight_layout()
        out = os.path.join(output_dir, "summary_all_metrics.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {out}")


def plot_radar(
    base_df: pd.DataFrame,
    ft_df: pd.DataFrame,
    output_dir: str,
    folder_name: str,
) -> None:
    """
    Radar chart. Deletion AUC is inverted so that 'outward = better' always holds.
    """
    os.makedirs(output_dir, exist_ok=True)

    radar_metrics = []
    for title, method, column, _, higher_is_better in METRICS:
        v_base = extract_value(base_df, method, column)
        v_ft   = extract_value(ft_df,   method, column)
        if np.isnan(v_base) or np.isnan(v_ft):
            continue
        # Invert deletion AUC so outward = better
        if not higher_is_better:
            v_base = 1 - v_base
            v_ft   = 1 - v_ft
            label  = f"1-{title.split('-')[-1].strip()}\n({title.split('-')[0].strip()})"
        else:
            label = title.replace(" - ", "\n")
        radar_metrics.append((label, v_base, v_ft))

    n = len(radar_metrics)
    if n < 3:
        print("  [skip] radar - need ≥3 metrics")
        return

    labels  = [m[0] for m in radar_metrics]
    v_base  = [m[1] for m in radar_metrics]
    v_ft    = [m[2] for m in radar_metrics]

    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    v_base  += v_base[:1]
    v_ft    += v_ft[:1]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        fig.patch.set_facecolor("#f7f8fc")
        ax.set_facecolor("#ffffff")

        ax.plot(angles, v_base, color=COLOR_BASE,      linewidth=2, label="Base")
        ax.fill(angles, v_base, color=COLOR_BASE,      alpha=0.12)
        ax.plot(angles, v_ft,   color=COLOR_FINETUNED, linewidth=2, label="Finetuned")
        ax.fill(angles, v_ft,   color=COLOR_FINETUNED, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8, color="#2a2d3d")
        ax.tick_params(colors="#555870")
        ax.grid(color="#dde0ea", linestyle="--", alpha=0.8)
        ax.spines["polar"].set_color("#c8ccd8")

        ax.set_title(
            f"Radar - Base vs Finetuned\n{folder_name}",
            fontsize=11, fontweight="bold", pad=20, color="#2a2d3d",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15),
                  fontsize=9, facecolor="#ffffff",
                  edgecolor="#c8ccd8", labelcolor="#2a2d3d")

        fig.tight_layout()
        out = os.path.join(output_dir, "radar_comparison.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {out}")


def plot_delta_waterfall(
    base_df: pd.DataFrame,
    ft_df: pd.DataFrame,
    output_dir: str,
    folder_name: str,
) -> None:
    """
    Horizontal bar chart of absolute deltas (finetuned - base).
    Deletion AUC delta is negated so positive always means improvement.
    """
    os.makedirs(output_dir, exist_ok=True)

    items = []
    for title, method, column, _, higher_is_better in METRICS:
        v_base = extract_value(base_df, method, column)
        v_ft   = extract_value(ft_df,   method, column)
        if np.isnan(v_base) or np.isnan(v_ft):
            continue
        delta = v_ft - v_base
        if not higher_is_better:
            delta = -delta
            label = f"{title}"
        else:
            label = title
        items.append((label, delta))

    if not items:
        return

    labels = [i[0] for i in items]
    deltas = [i[1] for i in items]
    colors = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE for d in deltas]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, max(4, len(items) * 0.85)))

        bars = ax.barh(labels, deltas, color=colors, zorder=3, height=0.55)

        for bar, val in zip(bars, deltas):
            x_pos = val + (0.0005 if val >= 0 else -0.0005)
            ha    = "left" if val >= 0 else "right"
            ax.text(
                x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center", ha=ha, fontsize=9,
                color=COLOR_POSITIVE if val >= 0 else COLOR_NEGATIVE,
                fontweight="bold",
            )

        ax.axvline(0, color="#9098b0", linewidth=1.2, zorder=4)
        ax.set_xlabel("Δ (finetuned - base)", fontsize=10)
        ax.set_title(
            f"Finetuning result model B",
            fontsize=12, fontweight="bold", pad=10,
        )
        ax.grid(axis="x", zorder=0)
        ax.set_xlim([-0.021,0.055])
        ax.invert_yaxis()

        fig.tight_layout()
        out = os.path.join(output_dir, "delta_waterfall.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved -> {out}")


def main():
    cwd = os.getcwd()
    folder_name = os.path.basename(cwd)

    base_path = os.path.join(cwd, "base.csv")
    ft_path   = os.path.join(cwd, "finetuned.csv")

    for p in [base_path, ft_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")

    print(f"Loading CSVs from: {cwd}")
    base_df = load_csv(base_path)
    ft_df   = load_csv(ft_path)
    print(f"  base.csv     - {len(base_df)} rows")
    print(f"  finetuned.csv - {len(ft_df)} rows")

    output_dir = os.path.join(cwd, "finetune_plots")
    print(f"\nSaving plots to: {output_dir}")

    #print("\n── Bar comparisons (one per metric) ──")
    #plot_bar_comparison(base_df, ft_df, output_dir, folder_name)

    #print("\n── Summary overview ──")
    #plot_summary(base_df, ft_df, output_dir, folder_name)

    #print("\n── Radar chart ──")
    #plot_radar(base_df, ft_df, output_dir, folder_name)

    print("\n── Delta waterfall ──")
    plot_delta_waterfall(base_df, ft_df, output_dir, folder_name)

    print(f"\nDone! All plots saved to '{output_dir}/'")


if __name__ == "__main__":
    main()