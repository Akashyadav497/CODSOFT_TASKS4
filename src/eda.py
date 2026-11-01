"""EDA script to generate all required visual exploratory charts."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR, RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import load_raw_data

# Ensure clean styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_eda_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data()
    print("Generating EDA figures into", FIGURES_DIR)

    # 1. Distribution of Sales (Histogram & Box Plot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, ax=axes[0], color="#2b5c8f", bins=15)
    axes[0].set_title("Sales Distribution (Histogram & KDE)", fontweight="bold")
    axes[0].set_xlabel("Sales (Thousands of Units / $k)")
    axes[0].set_ylabel("Frequency")

    sns.boxplot(y=df[TARGET_COLUMN], ax=axes[1], color="#4ca1af", width=0.4)
    axes[1].set_title("Sales Distribution (Box Plot)", fontweight="bold")
    axes[1].set_ylabel("Sales (Thousands of Units / $k)")
    plt.suptitle("Figure 1: Distribution Analysis of Target Variable (Sales)", fontsize=14, y=1.03)
    plt.savefig(FIGURES_DIR / "eda_sales_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("- Saved eda_sales_distribution.png")

    # 2. Distribution of Each Advertising Factor
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    channels = [("TV", "#e74c3c"), ("Radio", "#3498db"), ("Newspaper", "#2ecc71")]
    for idx, (col, color) in enumerate(channels):
        sns.histplot(df[col], kde=True, ax=axes[idx], color=color, bins=15)
        axes[idx].set_title(f"{col} Budget Distribution", fontweight="bold")
        axes[idx].set_xlabel(f"{col} Spend ($k)")
        axes[idx].set_ylabel("Count")
    plt.suptitle("Figure 2: Distribution of Media Expenditures across Channels", fontsize=14, y=1.04)
    plt.savefig(FIGURES_DIR / "eda_features_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("- Saved eda_features_distribution.png")

    # 3. Scatter plot of each advertising factor against Sales with trend line
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (col, color) in enumerate(channels):
        sns.regplot(
            data=df,
            x=col,
            y=TARGET_COLUMN,
            ax=axes[idx],
            color=color,
            scatter_kws={"alpha": 0.6, "s": 35},
            line_kws={"color": "black", "linewidth": 2},
        )
        corr_val = df[col].corr(df[TARGET_COLUMN])
        axes[idx].set_title(f"{col} vs Sales (Pearson r = {corr_val:.3f})", fontweight="bold")
        axes[idx].set_xlabel(f"{col} Spend ($k)")
        axes[idx].set_ylabel("Sales ($k)")
    plt.suptitle("Figure 3: Scatter Plots with Regression Trend Line against Sales", fontsize=14, y=1.04)
    plt.savefig(FIGURES_DIR / "eda_scatter_regression.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("- Saved eda_scatter_regression.png")

    # 4. Correlation Heatmap
    plt.figure(figsize=(7, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Figure 4: Pearson Correlation Matrix Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("- Saved eda_correlation_heatmap.png")

    # 5. Pairplot of All Features and Target
    g = sns.pairplot(
        df,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 30, "color": "#1f77b4"},
        diag_kws={"color": "#ff7f0e", "fill": True},
    )
    g.fig.suptitle("Figure 5: Pairwise Scatter Matrix and Kernel Densities", y=1.02, fontsize=14)
    g.savefig(FIGURES_DIR / "eda_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("- Saved eda_pairplot.png")

    # 6. Missing Value Overview Chart
    plt.figure(figsize=(7, 4))
    null_counts = df.isnull().sum()
    bars = plt.bar(null_counts.index, null_counts.values, color="#2ca02c", width=0.5)
    plt.title("Figure 6: Missing Values per Feature (Verification)", fontsize=13, fontweight="bold")
    plt.xlabel("Variables")
    plt.ylabel("Missing Count")
    plt.ylim(0, 10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{int(yval)}", ha="center", va="bottom")
    plt.savefig(FIGURES_DIR / "eda_missing_values.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("- Saved eda_missing_values.png")

    # 7. Channel Spend Share Composition
    channel_totals = df[["TV", "Radio", "Newspaper"]].sum()
    plt.figure(figsize=(7, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    plt.pie(
        channel_totals,
        labels=[f"{col} ({channel_totals[col]:,.1f}$k)" for col in channel_totals.index],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0.05, 0.05, 0.05),
        shadow=True,
    )
    plt.title("Figure 7: Total Budget Allocation Share Across Advertising Channels", fontsize=13, fontweight="bold")
    plt.savefig(FIGURES_DIR / "eda_budget_shares.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("- Saved eda_budget_shares.png")

    print("All EDA charts generated successfully.")


if __name__ == "__main__":
    generate_eda_figures()
