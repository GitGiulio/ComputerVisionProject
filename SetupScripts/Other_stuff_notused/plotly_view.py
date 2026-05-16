# csv_to_plotly_report.py

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# EDIT ONLY THESE VALUES
# ============================================================

CSV_PATH = "./baseline_bias_check/baseline_bias_summary.csv"
OUTPUT_HTML = "./baseline_bias_check/baseline_bias_plotly.html"

# Good options now that the CSV includes accuracy:
#
# Classification performance:
#   "accuracy"
#   "balanced_accuracy"
#   "f1_class_1"
#   "class_0_accuracy"
#   "class_1_accuracy"
#
# Baseline bias:
#   "mean_p_class_1"
#   "mean_logit"
#   "neutral_distance"
#   "cat_bias_if_class_0_cat"
#   "dog_bias_if_class_1_dog"
#   "baseline_pred_class_0_frac"
#   "baseline_pred_class_1_frac"
#
MODEL_SORT_BY = "accuracy"

# Which rows should define the model order?
# For accuracy sorting, these filters do not matter much because accuracy is repeated
# across all baseline rows for the same model.
#
# For baseline-bias sorting, use:
#   SORT_CLASS = "ALL"
#   SORT_BASELINE = "blur"
#
SORT_CLASS = "ALL"
SORT_BASELINE = "blur"

# True = low to high
# False = high to low
SORT_ASCENDING = False

# If multiple rows remain per model after filtering, how should sorting value be aggregated?
# Options: "mean", "median", "max", "min", "first"
SORT_AGG = "mean"

# Show only top N models after sorting.
# Use None to show all models.
TOP_N_MODELS = None

# ============================================================


def shorten_model_name(name: str) -> str:
    return (
        str(name)
        .replace("model_kernel=", "k=")
        .replace(".pth", "")
    )


def normalize_class_names(df: pd.DataFrame) -> pd.DataFrame:
    if "original_class_name" in df.columns:
        df["original_class_name"] = (
            df["original_class_name"]
            .astype(str)
            .str.replace("^all$", "ALL", regex=True)
        )
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds useful columns if they are missing.

    The new baseline script already creates:
      neutral_distance
      cat_bias_if_class_0_cat
      dog_bias_if_class_1_dog

    This function keeps compatibility if some columns are missing.
    """
    if "mean_p_class_1" in df.columns:
        if "neutral_distance" not in df.columns:
            df["neutral_distance"] = (df["mean_p_class_1"] - 0.5).abs()

        if "cat_bias_if_class_0_cat" not in df.columns:
            df["cat_bias_if_class_0_cat"] = 0.5 - df["mean_p_class_1"]

        if "dog_bias_if_class_1_dog" not in df.columns:
            df["dog_bias_if_class_1_dog"] = df["mean_p_class_1"] - 0.5

        # Short aliases, useful for plotting/table compatibility.
        df["cat_bias"] = df["cat_bias_if_class_0_cat"]
        df["dog_bias"] = df["dog_bias_if_class_1_dog"]

    if {"fp", "fn", "tp", "tn"}.issubset(df.columns):
        total = df["tn"] + df["fp"] + df["fn"] + df["tp"]
        total = total.replace(0, np.nan)

        df["error_rate"] = (df["fp"] + df["fn"]) / total
        df["false_positive_rate"] = df["fp"] / (df["fp"] + df["tn"]).replace(0, np.nan)
        df["false_negative_rate"] = df["fn"] / (df["fn"] + df["tp"]).replace(0, np.nan)

    return df


def aggregate_sort_values(sort_df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if SORT_AGG == "mean":
        out = sort_df.groupby("model_short", as_index=False)[sort_by].mean()
    elif SORT_AGG == "median":
        out = sort_df.groupby("model_short", as_index=False)[sort_by].median()
    elif SORT_AGG == "max":
        out = sort_df.groupby("model_short", as_index=False)[sort_by].max()
    elif SORT_AGG == "min":
        out = sort_df.groupby("model_short", as_index=False)[sort_by].min()
    elif SORT_AGG == "first":
        out = sort_df.groupby("model_short", as_index=False)[sort_by].first()
    else:
        raise ValueError(
            f"Unknown SORT_AGG={SORT_AGG}. "
            "Use: mean, median, max, min, or first."
        )

    return out


def get_model_order(df: pd.DataFrame) -> list[str]:
    if MODEL_SORT_BY not in df.columns:
        available = "\n".join(df.columns)
        raise ValueError(
            f"MODEL_SORT_BY='{MODEL_SORT_BY}' not found in CSV.\n\n"
            f"Available columns:\n{available}"
        )

    sort_df = df.copy()

    if "original_class_name" in sort_df.columns and SORT_CLASS is not None:
        sort_df = sort_df[sort_df["original_class_name"].eq(SORT_CLASS)]

    if "baseline" in sort_df.columns and SORT_BASELINE is not None:
        sort_df = sort_df[sort_df["baseline"].eq(SORT_BASELINE)]

    if sort_df.empty:
        raise ValueError(
            "No rows remain after applying sorting filters:\n"
            f"  SORT_CLASS={SORT_CLASS}\n"
            f"  SORT_BASELINE={SORT_BASELINE}\n\n"
            "Try SORT_CLASS='ALL' and SORT_BASELINE='blur'."
        )

    if pd.api.types.is_numeric_dtype(sort_df[MODEL_SORT_BY]):
        order_df = aggregate_sort_values(sort_df, MODEL_SORT_BY)
        order_df = order_df.sort_values(
            MODEL_SORT_BY,
            ascending=SORT_ASCENDING,
            na_position="last",
        )
    else:
        order_df = (
            sort_df[["model_short", MODEL_SORT_BY]]
            .drop_duplicates("model_short")
            .sort_values(
                MODEL_SORT_BY,
                ascending=SORT_ASCENDING,
                na_position="last",
            )
        )

    model_order = order_df["model_short"].tolist()

    if TOP_N_MODELS is not None:
        model_order = model_order[:TOP_N_MODELS]

    return model_order


def filter_to_ordered_models(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    return df[df["model_short"].isin(model_order)].copy()


def ordered_category(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["model_short"] = pd.Categorical(
        df["model_short"],
        categories=model_order,
        ordered=True,
    )
    return df


def add_neutral_line(fig):
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        annotation_text="neutral = 0.5",
        annotation_position="top left",
    )


def all_hover_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "model_short"]


def format_numeric_table(table_df: pd.DataFrame) -> pd.DataFrame:
    out = table_df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: f"{x:.5g}" if pd.notna(x) else "")
    return out


def add_table_figure(figures: list, heading: str, table_df: pd.DataFrame, height: int = 800):
    table_df = format_numeric_table(table_df)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_df.columns),
                    align="left",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[table_df[col].tolist() for col in table_df.columns],
                    align="left",
                    font=dict(size=10),
                    height=24,
                ),
            )
        ]
    )

    fig_table.update_layout(
        title=heading,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    figures.append((heading, fig_table))


def make_plotly_report():
    csv_path = Path(CSV_PATH)
    output_path = Path(OUTPUT_HTML)

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV file is empty.")

    df = normalize_class_names(df)

    if "model" in df.columns:
        df["model_short"] = df["model"].apply(shorten_model_name)
    else:
        df["model_short"] = np.arange(len(df)).astype(str)

    df = add_derived_columns(df)

    model_order = get_model_order(df)
    df_plot = filter_to_ordered_models(df, model_order)
    df_plot = ordered_category(df_plot, model_order)

    figures = []

    required_bar_cols = {
        "model_short",
        "baseline",
        "mean_p_class_1",
        "original_class_name",
    }

    # ------------------------------------------------------------
    # Plot 1: accuracy by model
    # ------------------------------------------------------------
    if {"model_short", "accuracy", "balanced_accuracy", "f1_class_1"}.issubset(df_plot.columns):
        metric_df = (
            df_plot[[
                "model_short",
                "architecture",
                "accuracy",
                "balanced_accuracy",
                "precision_class_1",
                "recall_class_1",
                "f1_class_1",
                "class_0_accuracy",
                "class_1_accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ]]
            .drop_duplicates("model_short")
            .sort_values("model_short")
        )

        long_metric_df = metric_df.melt(
            id_vars=["model_short", "architecture", "tn", "fp", "fn", "tp"],
            value_vars=[
                "accuracy",
                "balanced_accuracy",
                "f1_class_1",
                "class_0_accuracy",
                "class_1_accuracy",
            ],
            var_name="metric",
            value_name="score",
        )

        fig_acc = px.bar(
            long_metric_df,
            x="model_short",
            y="score",
            color="metric",
            barmode="group",
            category_orders={
                "model_short": model_order,
                "metric": [
                    "accuracy",
                    "balanced_accuracy",
                    "f1_class_1",
                    "class_0_accuracy",
                    "class_1_accuracy",
                ],
            },
            hover_data=["architecture", "tn", "fp", "fn", "tp"],
            title=(
                "Original test-image performance by model "
                f"<br><sup>Sorted by {MODEL_SORT_BY}, ascending={SORT_ASCENDING}</sup>"
            ),
            labels={
                "model_short": "Model",
                "score": "Score",
                "metric": "Metric",
            },
        )

        fig_acc.update_layout(
            height=750,
            xaxis_tickangle=-70,
            yaxis=dict(range=[0, 1]),
            margin=dict(l=60, r=30, t=100, b=260),
        )

        figures.append(("Original image accuracy / F1 / class-wise accuracy", fig_acc))

    # ------------------------------------------------------------
    # Plot 2: confusion matrix values by model
    # ------------------------------------------------------------
    if {"model_short", "tn", "fp", "fn", "tp"}.issubset(df_plot.columns):
        cm_df = (
            df_plot[["model_short", "architecture", "tn", "fp", "fn", "tp"]]
            .drop_duplicates("model_short")
            .sort_values("model_short")
        )

        cm_long = cm_df.melt(
            id_vars=["model_short", "architecture"],
            value_vars=["tn", "fp", "fn", "tp"],
            var_name="confusion_cell",
            value_name="count",
        )

        fig_cm = px.bar(
            cm_long,
            x="model_short",
            y="count",
            color="confusion_cell",
            barmode="group",
            category_orders={
                "model_short": model_order,
                "confusion_cell": ["tn", "fp", "fn", "tp"],
            },
            hover_data=["architecture"],
            title=(
                "Confusion matrix counts on original test images "
                "<br><sup>tn=Cat correct, fp=Cat predicted Dog, "
                "fn=Dog predicted Cat, tp=Dog correct</sup>"
            ),
            labels={
                "model_short": "Model",
                "count": "Count",
                "confusion_cell": "Cell",
            },
        )

        fig_cm.update_layout(
            height=750,
            xaxis_tickangle=-70,
            margin=dict(l=60, r=30, t=100, b=260),
        )

        figures.append(("Original image confusion matrix counts", fig_cm))

    # ------------------------------------------------------------
    # Plot 3: baseline plot for ALL rows
    # ------------------------------------------------------------
    if required_bar_cols.issubset(df_plot.columns):
        all_df = df_plot[df_plot["original_class_name"].eq("ALL")].copy()

        if not all_df.empty:
            fig_all = px.bar(
                all_df,
                x="model_short",
                y="mean_p_class_1",
                color="baseline",
                barmode="group",
                category_orders={
                    "model_short": model_order,
                    "baseline": sorted(df_plot["baseline"].dropna().unique()),
                },
                hover_data=all_hover_columns(df_plot),
                title=(
                    "Baseline bias across models: mean predicted probability of class 1 / Dog "
                    f"<br><sup>Sorted by {MODEL_SORT_BY}, class={SORT_CLASS}, "
                    f"baseline={SORT_BASELINE}, ascending={SORT_ASCENDING}</sup>"
                ),
                labels={
                    "model_short": "Model",
                    "mean_p_class_1": "Mean p(class 1 / Dog)",
                    "baseline": "Baseline",
                },
            )

            add_neutral_line(fig_all)

            fig_all.update_layout(
                height=750,
                xaxis_tickangle=-70,
                yaxis=dict(range=[0, 1]),
                margin=dict(l=60, r=30, t=100, b=260),
            )

            figures.append(("ALL images baseline-bias grouped bar chart", fig_all))

    # ------------------------------------------------------------
    # Plot 4: Cat/Dog split baseline plot
    # ------------------------------------------------------------
    if required_bar_cols.issubset(df_plot.columns):
        class_df = df_plot[df_plot["original_class_name"].isin(["Cat", "Dog"])].copy()

        if not class_df.empty:
            fig_class = px.bar(
                class_df,
                x="model_short",
                y="mean_p_class_1",
                color="baseline",
                facet_row="original_class_name",
                barmode="group",
                category_orders={
                    "model_short": model_order,
                    "baseline": sorted(df_plot["baseline"].dropna().unique()),
                    "original_class_name": ["Cat", "Dog"],
                },
                hover_data=all_hover_columns(df_plot),
                title=(
                    "Baseline bias split by original image class "
                    f"<br><sup>Sorted by {MODEL_SORT_BY}</sup>"
                ),
                labels={
                    "model_short": "Model",
                    "mean_p_class_1": "Mean p(class 1 / Dog)",
                    "baseline": "Baseline",
                    "original_class_name": "Original class",
                },
            )

            add_neutral_line(fig_class)

            fig_class.update_layout(
                height=900,
                xaxis_tickangle=-70,
                yaxis=dict(range=[0, 1]),
                margin=dict(l=60, r=30, t=100, b=260),
            )

            figures.append(("Cat/Dog split baseline-bias grouped bar chart", fig_class))

    # ------------------------------------------------------------
    # Plot 5: bias strength plot
    # ------------------------------------------------------------
    bias_cols = {"model_short", "neutral_distance", "baseline", "original_class_name"}

    if bias_cols.issubset(df_plot.columns):
        bias_df = df_plot[df_plot["original_class_name"].eq("ALL")].copy()

        if not bias_df.empty:
            fig_bias = px.bar(
                bias_df,
                x="model_short",
                y="neutral_distance",
                color="baseline",
                barmode="group",
                category_orders={
                    "model_short": model_order,
                    "baseline": sorted(df_plot["baseline"].dropna().unique()),
                },
                hover_data=all_hover_columns(df_plot),
                title=(
                    "Baseline bias strength: distance from neutral probability 0.5 "
                    "<br><sup>Higher means stronger baseline bias, regardless of Cat/Dog direction</sup>"
                ),
                labels={
                    "model_short": "Model",
                    "neutral_distance": "|mean p(class 1) - 0.5|",
                    "baseline": "Baseline",
                },
            )

            fig_bias.update_layout(
                height=750,
                xaxis_tickangle=-70,
                yaxis=dict(range=[0, 0.5]),
                margin=dict(l=60, r=30, t=100, b=260),
            )

            figures.append(("Baseline bias strength from neutral", fig_bias))

    # ------------------------------------------------------------
    # Plot 6: baseline prediction fractions
    # ------------------------------------------------------------
    pred_frac_cols = {
        "model_short",
        "baseline",
        "original_class_name",
        "baseline_pred_class_0_frac",
        "baseline_pred_class_1_frac",
    }

    if pred_frac_cols.issubset(df_plot.columns):
        pred_df = df_plot[df_plot["original_class_name"].eq("ALL")].copy()

        if not pred_df.empty:
            pred_long = pred_df.melt(
                id_vars=["model_short", "baseline", "architecture", "accuracy"],
                value_vars=[
                    "baseline_pred_class_0_frac",
                    "baseline_pred_class_1_frac",
                ],
                var_name="predicted_side",
                value_name="fraction",
            )

            pred_long["predicted_side"] = pred_long["predicted_side"].replace({
                "baseline_pred_class_0_frac": "Predicted Cat / class 0",
                "baseline_pred_class_1_frac": "Predicted Dog / class 1",
            })

            fig_pred_frac = px.bar(
                pred_long,
                x="model_short",
                y="fraction",
                color="predicted_side",
                facet_row="baseline",
                barmode="stack",
                category_orders={
                    "model_short": model_order,
                    "predicted_side": [
                        "Predicted Cat / class 0",
                        "Predicted Dog / class 1",
                    ],
                },
                hover_data=["architecture", "accuracy"],
                title=(
                    "Baseline predicted class fractions "
                    "<br><sup>For ALL test images; stacked bars show whether baseline images become Cat or Dog</sup>"
                ),
                labels={
                    "model_short": "Model",
                    "fraction": "Fraction",
                    "predicted_side": "Predicted class",
                },
            )

            fig_pred_frac.update_layout(
                height=900,
                xaxis_tickangle=-70,
                yaxis=dict(range=[0, 1]),
                margin=dict(l=60, r=30, t=100, b=260),
            )

            figures.append(("Baseline predicted class fractions", fig_pred_frac))

    # ------------------------------------------------------------
    # Plot 7: accuracy vs baseline-bias strength
    # ------------------------------------------------------------
    if {"accuracy", "neutral_distance", "baseline", "original_class_name"}.issubset(df_plot.columns):
        acc_bias_df = df_plot[df_plot["original_class_name"].eq("ALL")].copy()

        if not acc_bias_df.empty:
            fig_acc_bias = px.scatter(
                acc_bias_df,
                x="accuracy",
                y="neutral_distance",
                color="baseline",
                symbol="architecture" if "architecture" in acc_bias_df.columns else None,
                hover_name="model_short",
                hover_data=all_hover_columns(acc_bias_df),
                title=(
                    "Accuracy vs baseline-bias strength "
                    "<br><sup>Useful for checking whether better classifiers are also more/less baseline-biased</sup>"
                ),
                labels={
                    "accuracy": "Original test accuracy",
                    "neutral_distance": "Baseline bias strength |p(class 1) - 0.5|",
                    "baseline": "Baseline",
                },
            )

            fig_acc_bias.update_layout(
                height=650,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 0.5]),
                margin=dict(l=60, r=30, t=100, b=80),
            )

            figures.append(("Accuracy vs baseline-bias strength", fig_acc_bias))

    # ------------------------------------------------------------
    # Plot 8: accuracy vs mean p(class 1)
    # ------------------------------------------------------------
    if {"accuracy", "mean_p_class_1", "baseline", "original_class_name"}.issubset(df_plot.columns):
        acc_p_df = df_plot[df_plot["original_class_name"].eq("ALL")].copy()

        if not acc_p_df.empty:
            fig_acc_p = px.scatter(
                acc_p_df,
                x="accuracy",
                y="mean_p_class_1",
                color="baseline",
                symbol="architecture" if "architecture" in acc_p_df.columns else None,
                hover_name="model_short",
                hover_data=all_hover_columns(acc_p_df),
                title=(
                    "Accuracy vs baseline tendency toward Dog / class 1 "
                    "<br><sup>y > 0.5 means baseline tends toward Dog; y < 0.5 means Cat</sup>"
                ),
                labels={
                    "accuracy": "Original test accuracy",
                    "mean_p_class_1": "Mean p(class 1 / Dog)",
                    "baseline": "Baseline",
                },
            )

            add_neutral_line(fig_acc_p)

            fig_acc_p.update_layout(
                height=650,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                margin=dict(l=60, r=30, t=100, b=80),
            )

            figures.append(("Accuracy vs baseline class tendency", fig_acc_p))

    # ------------------------------------------------------------
    # Plot 9: mean logit vs mean p(class 1)
    # ------------------------------------------------------------
    scatter_cols = {"mean_logit", "mean_p_class_1"}

    if scatter_cols.issubset(df_plot.columns):
        fig_scatter = px.scatter(
            df_plot,
            x="mean_logit",
            y="mean_p_class_1",
            color="baseline" if "baseline" in df_plot.columns else None,
            symbol="original_class_name" if "original_class_name" in df_plot.columns else None,
            hover_name="model_short",
            hover_data=all_hover_columns(df_plot),
            title="All CSV rows: mean logit vs mean p(class 1 / Dog)",
            labels={
                "mean_logit": "Mean logit",
                "mean_p_class_1": "Mean p(class 1 / Dog)",
            },
        )

        add_neutral_line(fig_scatter)

        fig_scatter.add_vline(
            x=0,
            line_dash="dot",
            annotation_text="logit = 0",
        )

        fig_scatter.update_layout(
            height=650,
            yaxis=dict(range=[0, 1]),
            margin=dict(l=60, r=30, t=80, b=80),
        )

        figures.append(("All rows scatter overview", fig_scatter))

    # ------------------------------------------------------------
    # Table 1: model-level metrics
    # ------------------------------------------------------------
    model_table_cols = [
        col for col in [
            "model_short",
            "architecture",
            "kernel_size",
            "conv_filter",
            "conv_layer",
            "dense_neuron",
            "dense_layer",
            "weight_decay",
            "dropout",
            "original_n",
            "accuracy",
            "balanced_accuracy",
            "precision_class_1",
            "recall_class_1",
            "f1_class_1",
            "tn",
            "fp",
            "fn",
            "tp",
            "class_0_accuracy",
            "class_1_accuracy",
            "mean_original_logit",
            "mean_original_p_class_1",
        ]
        if col in df_plot.columns
    ]

    if model_table_cols:
        model_table_df = (
            df_plot[model_table_cols]
            .drop_duplicates("model_short")
            .sort_values("model_short")
        )
        add_table_figure(
            figures,
            "Model-level original-image performance table",
            model_table_df,
            height=750,
        )

    # ------------------------------------------------------------
    # Table 2: rows used to determine model sorting
    # ------------------------------------------------------------
    sort_info_df = df_plot.copy()

    if "original_class_name" in sort_info_df.columns and SORT_CLASS is not None:
        sort_info_df = sort_info_df[sort_info_df["original_class_name"].eq(SORT_CLASS)]

    if "baseline" in sort_info_df.columns and SORT_BASELINE is not None:
        sort_info_df = sort_info_df[sort_info_df["baseline"].eq(SORT_BASELINE)]

    sort_table_cols = [
        col for col in [
            "model_short",
            "architecture",
            "baseline",
            "original_class_name",
            "accuracy",
            "balanced_accuracy",
            "f1_class_1",
            "class_0_accuracy",
            "class_1_accuracy",
            "tn",
            "fp",
            "fn",
            "tp",
            "mean_logit",
            "mean_p_class_1",
            "neutral_distance",
            "cat_bias_if_class_0_cat",
            "dog_bias_if_class_1_dog",
            "baseline_pred_class_0_count",
            "baseline_pred_class_1_count",
            "baseline_pred_class_0_frac",
            "baseline_pred_class_1_frac",
            "bias_strength",
        ]
        if col in sort_info_df.columns
    ]

    if sort_table_cols:
        sort_table_df = sort_info_df[sort_table_cols].copy()
        sort_table_df = sort_table_df.sort_values("model_short")

        add_table_figure(
            figures,
            "Rows used to determine model sorting",
            sort_table_df,
            height=750,
        )

    # ------------------------------------------------------------
    # Table 3: full CSV data table
    # ------------------------------------------------------------
    table_df = df_plot.copy()
    table_df = table_df.sort_values(["model_short", "original_class_name", "baseline"])

    add_table_figure(
        figures,
        "Full CSV data table",
        table_df,
        height=950,
    )

    # ------------------------------------------------------------
    # Save HTML
    # ------------------------------------------------------------
    html_parts = [
        """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Baseline Bias Plotly Report</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 28px;
}
h1 {
    margin-bottom: 0.2rem;
}
p {
    max-width: 1100px;
    line-height: 1.4;
}
.section {
    margin-top: 42px;
}
code {
    background: #f2f2f2;
    padding: 2px 4px;
    border-radius: 4px;
}
</style>
</head>
<body>
"""
    ]

    html_parts.append("<h1>Baseline Bias + Accuracy Plotly Report</h1>")
    html_parts.append(f"<p>Source CSV: <code>{csv_path.name}</code></p>")
    html_parts.append(
        f"<p>Rows plotted: <b>{len(df_plot)}</b> | Columns: <b>{len(df_plot.columns)}</b> | "
        f"Models: <b>{df_plot['model_short'].nunique()}</b></p>"
    )
    html_parts.append(
        "<p>"
        f"Model sorting: <code>{MODEL_SORT_BY}</code>, "
        f"class filter: <code>{SORT_CLASS}</code>, "
        f"baseline filter: <code>{SORT_BASELINE}</code>, "
        f"ascending: <code>{SORT_ASCENDING}</code>, "
        f"aggregation: <code>{SORT_AGG}</code>"
        "</p>"
    )
    html_parts.append(
        "<p>"
        "Interpretation: <code>mean_p_class_1</code> is the model's mean probability for "
        "class 1, which is usually Dog when ImageFolder classes are alphabetical. "
        "Values above 0.5 lean Dog/class 1; values below 0.5 lean Cat/class 0. "
        "<code>neutral_distance</code> measures how far the baseline prediction is from 0.5."
        "</p>"
    )

    for i, (heading, fig) in enumerate(figures):
        include_js = "cdn" if i == 0 else False

        html_parts.append('<div class="section">')
        html_parts.append(f"<h2>{heading}</h2>")
        html_parts.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=include_js,
            )
        )
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"Read CSV: {csv_path}")
    print(f"Saved Plotly report: {output_path}")
    print(f"Rows plotted: {len(df_plot)}")
    print(f"Models plotted: {df_plot['model_short'].nunique()}")
    print(f"Columns: {len(df_plot.columns)}")
    print(f"Sorted by: {MODEL_SORT_BY}")
    print(f"Sort class: {SORT_CLASS}")
    print(f"Sort baseline: {SORT_BASELINE}")
    print(f"Ascending: {SORT_ASCENDING}")


if __name__ == "__main__":
    make_plotly_report()