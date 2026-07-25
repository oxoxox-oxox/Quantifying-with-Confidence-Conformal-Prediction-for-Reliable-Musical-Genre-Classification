"""Generate GTZAN vs FMA comparison figures.

Compares coverage and set size across datasets, highlighting
CP's distribution-free guarantee holding for both small (GTZAN 999)
and large (FMA 8000) datasets.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "figures"
GTZAN_DATA = ROOT / "outputs" / "cp_results" / "baseline_comparison.csv"
FMA_DATA = ROOT / "outputs" / "cp_results_fma" / "baseline_comparison_fma.csv"

ALPHAS = [0.01, 0.05, 0.10]
METHODS = ["CP", "Bootstrap", "Gaussian"]
COLORS = {"CP": "#2196F3", "Bootstrap": "#FF9800", "Gaussian": "#4CAF50"}
DATASET_COLORS = {"GTZAN": "#2196F3", "FMA": "#E91E63"}
MARKERS = {"CP": "o", "Bootstrap": "s", "Gaussian": "D"}
DATASET_MARKERS = {"GTZAN": "o", "FMA": "s"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def load_results():
    gtzan = pd.read_csv(GTZAN_DATA)
    gtzan["dataset"] = "GTZAN"
    if FMA_DATA.exists():
        fma = pd.read_csv(FMA_DATA)
        fma["dataset"] = "FMA"
        return pd.concat([gtzan, fma], ignore_index=True)
    print("Warning: FMA results not found. Generating GTZAN-only plots.")
    return gtzan


def plot_dataset_coverage_comparison(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = df["dataset"].unique()

    x_alpha = np.array(ALPHAS)
    ax.plot(x_alpha, 1.0 - x_alpha, "k--", linewidth=1, label="Nominal (1-α)")

    for ds in datasets:
        sub = df[(df["dataset"] == ds) & (df["method"] == "CP")]
        means = [sub[sub["alpha"] == a]["coverage"].mean() for a in ALPHAS]
        stds = [sub[sub["alpha"] == a]["coverage"].std() for a in ALPHAS]
        ax.errorbar(
            x_alpha, means, yerr=stds,
            marker=DATASET_MARKERS[ds], color=DATASET_COLORS[ds],
            capsize=5, linewidth=2, markersize=8,
            label=f"{ds} (CP)"
        )

    ax.set_xlabel("Significance level α")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("CP Coverage: GTZAN vs FMA")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.005, 0.115)
    ax.set_ylim(0.84, 1.02)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ALPHAS))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fma_coverage_comparison.png")
    plt.close(fig)
    print(f"Saved: fma_coverage_comparison.png")


def plot_dataset_set_size_comparison(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = df["dataset"].unique()
    methods_to_show = ["CP"]

    x = np.arange(len(ALPHAS))
    width = 0.3

    for j, ds in enumerate(datasets):
        sub = df[(df["dataset"] == ds) & (df["method"].isin(methods_to_show))]
        means = []
        stds = []
        for a in ALPHAS:
            a_sub = sub[sub["alpha"] == a]
            means.append(a_sub["avg_set_size"].mean())
            stds.append(a_sub["avg_set_size"].std())
        bars = ax.bar(
            x + j * width, means, width,
            yerr=stds, color=DATASET_COLORS.get(ds, "#999"),
            alpha=0.75, label=f"{ds} (CP)", capsize=4,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"α = {a:.2f}" for a in ALPHAS])
    ax.set_ylabel("Average prediction set size")
    ax.set_title("CP Set Size: GTZAN vs FMA")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fma_set_size_comparison.png")
    plt.close(fig)
    print(f"Saved: fma_set_size_comparison.png")


def plot_all_methods_fma(df_fma_only):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_alpha = np.array(ALPHAS)
    ax = axes[0]
    ax.plot(x_alpha, 1.0 - x_alpha, "k--", linewidth=1, label="Nominal (1-α)")
    for method in METHODS:
        sub = df_fma_only[df_fma_only["method"] == method]
        means = [sub[sub["alpha"] == a]["coverage"].mean() for a in ALPHAS]
        stds = [sub[sub["alpha"] == a]["coverage"].std() for a in ALPHAS]
        ax.errorbar(
            x_alpha, means, yerr=stds,
            marker=MARKERS[method], color=COLORS[method],
            capsize=4, linewidth=1.5, markersize=6,
            label=method
        )
    ax.set_xlabel("Significance level α")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("FMA: Coverage vs α")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.005, 0.115)
    ax.set_ylim(0.84, 1.02)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ALPHAS))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(ALPHAS))
    width = 0.22
    for j, method in enumerate(METHODS):
        sub = df_fma_only[df_fma_only["method"] == method]
        devs = []
        for a in ALPHAS:
            cov = sub[sub["alpha"] == a]["coverage"].mean()
            devs.append(cov - (1.0 - a))
        ax.bar(x + j * width, devs, width, color=COLORS[method],
               alpha=0.75, label=method)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"α = {a:.2f}" for a in ALPHAS])
    ax.set_ylabel("Coverage deviation")
    ax.set_title("FMA: Coverage Deviation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fma_full_results.png")
    plt.close(fig)
    print(f"Saved: fma_full_results.png")


def print_summary(df):
    print("\n=== DATASET COMPARISON SUMMARY ===")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        n = sub.iloc[0]["n_test"] if "n_test" in sub.columns else "?"
        print(f"\n  {ds}:")
        for method in METHODS:
            ms = sub[sub["method"] == method]
            for a in ALPHAS:
                a_ms = ms[ms["alpha"] == a]
                cov = a_ms["coverage"].mean()
                sz = a_ms["avg_set_size"].mean()
                target = 1.0 - a
                ok = "OK" if abs(cov - target) < 0.03 else "XX"
                print(f"    {method:12s} α={a:.2f}: cov={cov:.3f} (target {target:.2f}) {ok}  "
                      f"size={sz:.2f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    if "FMA" in df["dataset"].unique():
        print_summary(df)
        print("\nPlotting GTZAN vs FMA comparison ...")
        plot_dataset_coverage_comparison(df)
        plot_dataset_set_size_comparison(df)

        fma_only = df[df["dataset"] == "FMA"]
        print("\nPlotting FMA full results ...")
        plot_all_methods_fma(fma_only)
    else:
        print("Only GTZAN results available. Run cp_fma_experiment.py first.")

    print("\n=== COMPARISON PLOTS DONE ===")


if __name__ == "__main__":
    main()
