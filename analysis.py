#!/usr/bin/env python3
"""
Analyze experimental results CSV with columns:
dataset, regime, model, method, ratio, seed, accuracy, f1, auc

Usage:
  python analyze_results.py results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["accuracy", "f1", "auc"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "dataset", "regime", "model", "method", "ratio", "seed",
        "accuracy", "f1", "auc"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Normalize types
    df["ratio"] = df["ratio"].astype(float)
    df["seed"] = df["seed"].astype(str)
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def summarize_by_group(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "regime", "model", "method", "ratio"]
    summary = (
        df.groupby(group_cols)[METRICS]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    # Flatten columns
    summary.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in summary.columns
    ]
    return summary


def best_rows(summary: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    metric_mean_col = f"{metric}_mean"
    group_cols = ["dataset", "regime", "model", "method"]

    idx = summary.groupby(group_cols)[metric_mean_col].idxmax()
    best = summary.loc[idx].sort_values(group_cols).reset_index(drop=True)
    return best


def pairwise_method_compare(summary: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    """
    For each dataset/regime/model/ratio, rank methods by metric.
    """
    metric_mean_col = f"{metric}_mean"
    cols = ["dataset", "regime", "model", "ratio", "method", metric_mean_col]

    ranked = summary[cols].copy()
    ranked["rank"] = ranked.groupby(["dataset", "regime", "model", "ratio"])[metric_mean_col] \
                           .rank(ascending=False, method="dense")
    return ranked.sort_values(["dataset", "regime", "model", "ratio", "rank", "method"])


def make_plots(summary: pd.DataFrame, outdir: str = "analysis_plots") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    for (dataset, regime, model), g in summary.groupby(["dataset", "regime", "model"]):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

        for ax, metric in zip(axes, METRICS):
            y = f"{metric}_mean"
            err = f"{metric}_std"

            for method, m in g.groupby("method"):
                m = m.sort_values("ratio")
                ax.errorbar(
                    m["ratio"], m[y], yerr=m[err],
                    marker="o", linewidth=1.5, capsize=3,
                    label=method
                )

            ax.set_title(metric.upper())
            ax.set_xlabel("ratio")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("score")
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"{dataset} | {regime} | {model}")
        fig.tight_layout()
        fig.savefig(out / f"{dataset}_{regime}_{model}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to results CSV")
    parser.add_argument("--outdir", default="analysis_out", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Save plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv_path)

    # 1) Full grouped summary across seeds
    summary = summarize_by_group(df)
    summary_path = outdir / "summary_by_group.csv"
    summary.to_csv(summary_path, index=False)

    # 2) Best ratio per dataset/regime/model/method by AUC
    best_auc = best_rows(summary, metric="auc")
    best_auc_path = outdir / "best_by_auc.csv"
    best_auc.to_csv(best_auc_path, index=False)

    # 3) Best ratio per dataset/regime/model/method by F1
    best_f1 = best_rows(summary, metric="f1")
    best_f1_path = outdir / "best_by_f1.csv"
    best_f1.to_csv(best_f1_path, index=False)

    # 4) Method ranking at each ratio
    ranking = pairwise_method_compare(summary, metric="auc")
    ranking_path = outdir / "method_ranking_by_auc.csv"
    ranking.to_csv(ranking_path, index=False)

    # 5) Print compact report
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("\n=== Summary saved ===")
    print(summary_path)
    print(best_auc_path)
    print(best_f1_path)
    print(ranking_path)

    print("\n=== Top rows by AUC (per dataset/regime/model/method) ===")
    cols = [
        "dataset", "regime", "model", "method", "ratio",
        "auc_mean", "auc_std", "f1_mean", "accuracy_mean"
    ]
    print(best_auc[cols].sort_values(["dataset", "regime", "model", "method"]).to_string(index=False))

    print("\n=== Top rows by F1 (per dataset/regime/model/method) ===")
    print(best_f1[cols].sort_values(["dataset", "regime", "model", "method"]).to_string(index=False))

    # Optional plots
    if args.plot:
        make_plots(summary, outdir=str(outdir / "plots"))
        print(f"\nPlots saved to: {outdir / 'plots'}")


if __name__ == "__main__":
    main()