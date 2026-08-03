"""Step 13: Generate publication-quality figures for CP experiments.

Figures:
  1. Coverage vs alpha — all methods vs nominal diagonal
  2. Prediction set size boxplot — per alpha level
  3. Method comparison — coverage deviation from nominal
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
DATA_DIR = ROOT / "outputs" / "cp_results"

ALPHAS = [0.01, 0.05, 0.10]
METHODS = ["CP", "Bootstrap", "Gaussian"]
COLORS = {"CP": "#2196F3", "Bootstrap": "#FF9800", "Gaussian": "#4CAF50"}
MARKERS = {"CP": "o", "Bootstrap": "s", "Gaussian": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def load_baseline_results():
    df = pd.read_csv(DATA_DIR / "baseline_comparison.csv")
    return df


def plot_coverage_vs_alpha(df):
    fig, ax = plt.subplots(figsize=(7, 5))

    x_alpha = np.array(ALPHAS)
    ax.plot(x_alpha, 1.0 - x_alpha, "k--", linewidth=1, label="Nominal (1-α)")

    for method in METHODS:
        sub = df[df["method"] == method]
        means = [sub[sub["alpha"] == a]["coverage"].mean() for a in ALPHAS]
        stds = [sub[sub["alpha"] == a]["coverage"].std() for a in ALPHAS]
        ax.errorbar(
            x_alpha, means, yerr=stds,
            marker=MARKERS[method], color=COLORS[method],
            capsize=4, linewidth=1.5, markersize=6,
            label=f"{method} coverage"
        )

    ax.set_xlabel("Significance level α")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage vs. Significance Level")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.005, 0.115)
    ax.set_ylim(0.85, 1.02)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ALPHAS))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_vs_alpha.png")
    plt.close(fig)
    print(f"Saved: coverage_vs_alpha.png")


def plot_set_size_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = []
    data = []
    colors_list = []
    labels = []

    width = 0.15
    n_methods = len(METHODS)
    for i, alpha in enumerate(ALPHAS):
        for j, method in enumerate(METHODS):
            sub = df[(df["alpha"] == alpha) & (df["method"] == method)]
            pos = i * (n_methods + 1) * width + j * width
            sz_means = sub["avg_set_size"].values
            positions.append(pos)
            data.append(sz_means)
            colors_list.append(COLORS[method])
            if i == 0:
                labels.append(method)

    bp = ax.boxplot(
        data, positions=positions, widths=width * 0.8,
        patch_artist=True, showfliers=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markersize=4),
    )

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_xticks([
        (n_methods - 1) * width / 2 + i * (n_methods + 1) * width
        for i in range(len(ALPHAS))
    ])
    ax.set_xticklabels([f"α = {a:.2f}" for a in ALPHAS])

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=COLORS[m], alpha=0.5, label=m) for m in METHODS]
    ax.legend(handles=legend_patches, loc="upper right")

    ax.set_ylabel("Prediction set size")
    ax.set_title("Prediction Set Size Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "set_size_boxplot.png")
    plt.close(fig)
    print(f"Saved: set_size_boxplot.png")


def plot_coverage_deviation(df):
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(ALPHAS))
    width = 0.22

    for j, method in enumerate(METHODS):
        sub = df[df["method"] == method]
        devs = []
        for a in ALPHAS:
            cov = sub[sub["alpha"] == a]["coverage"].mean()
            devs.append(cov - (1.0 - a))
        bars = ax.bar(
            x + j * width, devs, width, color=COLORS[method],
            alpha=0.75, label=method
        )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels([f"α = {a:.2f}" for a in ALPHAS])
    ax.set_ylabel("Coverage deviation from nominal")
    ax.set_title("Coverage Deviation (Empirical - Nominal)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_deviation.png")
    plt.close(fig)
    print(f"Saved: coverage_deviation.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading baseline results ...")
    df = load_baseline_results()

    print("Plotting coverage vs alpha ...")
    plot_coverage_vs_alpha(df)

    print("Plotting set size boxplot ...")
    plot_set_size_boxplot(df)

    print("Plotting coverage deviation ...")
    plot_coverage_deviation(df)

    print("\n=== ALL PLOTS GENERATED ===")


if __name__ == "__main__":
    main()
