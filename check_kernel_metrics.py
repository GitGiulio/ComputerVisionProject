"""
check_kernel_metrics.py

For each kernel size (11, 15, 19), prints the BEST and WORST
insertion_auc, deletion_auc, and stability for each method (shap, gradcam).

Usage:
  python check_kernel_metrics.py --root_dir fixed_fidelity_interpretability/
"""

import os
import re
import argparse
import pandas as pd

FOLDER_RE = re.compile(
    r"interpretability_results_dogs_k=(?:\[5,9,(?P<kernel_size_a>[^\]]+)\]|(?P<kernel_size_b>[^_]+))"
    r"_(?P<conv_filter>[^_]+)"
    r"_(?P<conv_layer>[^_]+)"
    r"_(?P<dense_neuron>[^_]+)"
    r"_(?P<dense_layer>[^_]+)"
    r"_wd(?P<weight_decay>[^_]+)"
    r"_do(?P<dropout>.+)$"
)

KERNELS  = ["11", "15", "19"]
METHODS  = ["shap", "gradcam"]
METRICS  = ["insertion_auc", "deletion_auc", "stability"]


def parse_folder_name(name: str) -> dict | None:
    m = FOLDER_RE.match(os.path.basename(name.rstrip("/\\")))
    if not m:
        return None
    d = m.groupdict()
    d["kernel_size"] = d.pop("kernel_size_a") or d.pop("kernel_size_b")
    return d


def collect(root_dir: str) -> pd.DataFrame:
    records = []
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        params = parse_folder_name(entry.name)
        if params is None:
            continue
        csv_path = os.path.join(entry.path, "metrics_summary.csv")
        if not os.path.exists(csv_path):
            print(f"  [warn] missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df["method"] = df["method"].astype(str).str.strip().str.lower()
        df["folder"] = entry.name
        df["kernel_size"] = params["kernel_size"]
        records.append(df)

    if not records:
        raise ValueError(f"No valid folders found under '{root_dir}'")

    combined = pd.concat(records, ignore_index=True)
    for col in METRICS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def print_section(title: str, row: pd.Series, metrics: list[str]) -> None:
    print(f"    {title}")
    for m in metrics:
        val = row.get(m, float("nan"))
        print(f"      {m:<16} = {val:.4f}   [{row['folder']}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="fixed_fidelity_interpretability/")
    args = parser.parse_args()

    df = collect(args.root_dir)

    for kernel in KERNELS:
        kdf = df[df["kernel_size"] == kernel]
        if kdf.empty:
            print(f"\n── Kernel {kernel} ── (no data)\n")
            continue

        print(f"\n{'─'*60}")
        print(f"  Kernel size: {kernel}")
        print(f"{'─'*60}")

        for method in METHODS:
            mdf = kdf[kdf["method"] == method]
            if mdf.empty:
                print(f"\n  [{method.upper()}]  (no data)")
                continue

            print(f"\n  [{method.upper()}]")

            for metric in METRICS:
                col_data = mdf.dropna(subset=[metric])
                if col_data.empty:
                    print(f"    {metric}: no data")
                    continue

                best_row  = col_data.loc[col_data[metric].idxmax()]
                worst_row = col_data.loc[col_data[metric].idxmin()]

                print(f"\n    {metric}")
                print(f"      BEST   = {best_row[metric]:.4f}   [{best_row['folder']}]")
                print(f"      WORST  = {worst_row[metric]:.4f}   [{worst_row['folder']}]")

    print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    main()