"""
eda.py
------
Module 3: Exploratory Data Analysis (EDA) for Cookie Cats A/B Testing Dataset.

Purpose:
    Generate 12 professional, publication-quality visualizations to understand
    player behavior, retention patterns, and the impact of the A/B test
    (gate_30 vs gate_40) on player engagement.

    Every plot is saved automatically to: outputs/plots/

Author : (Your Name)
Project: Cookie Cats Retention Prediction - Game District Portfolio Project
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
# pandas      -> load and manipulate the dataset (DataFrames)
# numpy       -> numerical operations (used internally by seaborn/pandas)
# matplotlib  -> low-level plotting engine; gives us full control over figures
# seaborn     -> built on top of matplotlib; makes statistical plots easier
#                and more visually appealing with less code
# os          -> used to build file paths that work on any operating system
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------------------------
# 2. GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
# We define these ONCE at the top instead of repeating them in every function.
# This is a professional practice -> "configure once, reuse everywhere".

# seaborn theme: 'whitegrid' gives a clean white background with light grid
# lines, which is the standard look used in professional data science reports.
sns.set_theme(style="whitegrid")

# A consistent, professional color palette used across all plots.
# Using the SAME palette everywhere makes the whole report look cohesive,
# like a single designed document instead of random colors per chart.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# Default figure size (width, height in inches). 10x6 is a good balance:
# readable on screen, and not too large when saved as an image file.
FIGSIZE = (10, 6)

# Resolution for saved images. 300 DPI (dots per inch) is "print quality" -
# sharp enough for a portfolio/report, unlike the default 100 DPI which looks
# blurry when zoomed in.
DPI = 300

# Path to the folder where every plot will be saved.
# We build this path relative to THIS file's location (src/eda.py), so the
# script works correctly no matter where you run it from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> Cookie_Cats_Project/
DATA_PATH = os.path.join(BASE_DIR, "data", "cookie_cats.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# Create the plots folder if it doesn't already exist.
# exist_ok=True means: "don't throw an error if the folder is already there".
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. HELPER FUNCTION: SAVE PLOT
# ---------------------------------------------------------------------------
def save_plot(filename: str):
    """
    Save the current matplotlib figure to outputs/plots/ and then close it.

    Why a helper function?
        Every single plot needs the same 3 steps: tight_layout -> savefig ->
        close. Instead of repeating these 3 lines 12 times, we write them
        once here and call save_plot() everywhere. This is the DRY principle
        ("Don't Repeat Yourself") - a core professional coding habit.

    Parameters
    ----------
    filename : str
        Name of the output image file, e.g. "01_version_count.png"
    """
    # tight_layout() automatically adjusts spacing so titles/labels don't
    # get cut off at the edges of the figure.
    plt.tight_layout()

    # Build the full save path: outputs/plots/<filename>
    full_path = os.path.join(PLOTS_DIR, filename)

    # Save the figure to disk at high resolution.
    plt.savefig(full_path, dpi=DPI, bbox_inches="tight")

    # Close the figure to free up memory. Without this, generating 12 plots
    # in one script run can silently consume a lot of RAM.
    plt.close()

    print(f"Saved: {full_path}")


# ---------------------------------------------------------------------------
# 4. LOAD DATA
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Load the Cookie Cats dataset from data/cookie_cats.csv

    Returns
    -------
    pd.DataFrame
        The raw dataset with columns:
        userid, version, sum_gamerounds, retention_1, retention_7
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 5. VISUALIZATION FUNCTIONS (1 function = 1 graph)
# ---------------------------------------------------------------------------
# Each function follows the SAME pattern:
#   1. Create a figure with our standard size
#   2. Draw the plot using seaborn
#   3. Add title, axis labels, and value labels
#   4. Save the figure using our save_plot() helper
#
# Keeping each graph in its own function makes the code modular - you can
# call just ONE graph for debugging, instead of re-running the whole script.


def plot_version_count(df: pd.DataFrame):
    """Graph 1: Count of players in each A/B test group (gate_30 vs gate_40)."""
    plt.figure(figsize=FIGSIZE)

    # countplot automatically counts how many rows belong to each category
    ax = sns.countplot(data=df, x="version", hue="version",
                        palette=PALETTE[:2], legend=False)

    ax.set_title("Player Distribution by A/B Test Group", fontsize=14, fontweight="bold")
    ax.set_xlabel("Game Version (Gate Position)", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)

    # Add the exact count on top of each bar - very useful for reports,
    # since the reader doesn't have to guess the value from the bar height.
    for container in ax.containers:
        ax.bar_label(container, fontsize=11)

    save_plot("01_version_count.png")


def plot_retention1_count(df: pd.DataFrame):
    """Graph 2: How many players returned 1 day after install (True/False)."""
    plt.figure(figsize=FIGSIZE)

    ax = sns.countplot(data=df, x="retention_1", hue="retention_1",
                        palette=PALETTE[2:4], legend=False)

    ax.set_title("Day 1 Retention Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Returned After 1 Day?", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fontsize=11)

    save_plot("02_retention1_count.png")


def plot_retention7_count(df: pd.DataFrame):
    """Graph 3: How many players returned 7 days after install (target variable)."""
    plt.figure(figsize=FIGSIZE)

    ax = sns.countplot(data=df, x="retention_7", hue="retention_7",
                        palette=PALETTE[4:6], legend=False)

    ax.set_title("Day 7 Retention Count (Target Variable)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Returned After 7 Days?", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fontsize=11)

    save_plot("03_retention7_count.png")


def plot_gamerounds_distribution(df: pd.DataFrame):
    """Graph 4: Distribution (histogram) of total game rounds played."""
    plt.figure(figsize=FIGSIZE)

    # We clip at the 99th percentile only for VISUALIZATION, because a few
    # extreme outliers (players with 1000s of rounds) would otherwise
    # squash the whole chart into one unreadable bar.
    upper_limit = df["sum_gamerounds"].quantile(0.99)
    filtered = df[df["sum_gamerounds"] <= upper_limit]

    ax = sns.histplot(filtered["sum_gamerounds"], bins=40, color=PALETTE[0], kde=False)

    ax.set_title("Distribution of Total Game Rounds Played (up to 99th percentile)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Sum of Game Rounds", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)

    save_plot("04_gamerounds_distribution.png")


def plot_gamerounds_boxplot(df: pd.DataFrame):
    """Graph 5: Boxplot of game rounds - shows median, quartiles, and outliers."""
    plt.figure(figsize=FIGSIZE)

    upper_limit = df["sum_gamerounds"].quantile(0.99)
    filtered = df[df["sum_gamerounds"] <= upper_limit]

    ax = sns.boxplot(x=filtered["sum_gamerounds"], color=PALETTE[1])

    ax.set_title("Boxplot of Game Rounds Played (Outliers Beyond 99th Percentile Removed)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Sum of Game Rounds", fontsize=12)

    save_plot("05_gamerounds_boxplot.png")


def plot_gamerounds_by_version(df: pd.DataFrame):
    """Graph 6: Compare game rounds played between gate_30 and gate_40."""
    plt.figure(figsize=FIGSIZE)

    upper_limit = df["sum_gamerounds"].quantile(0.99)
    filtered = df[df["sum_gamerounds"] <= upper_limit]

    ax = sns.boxplot(data=filtered, x="version", y="sum_gamerounds",
                      hue="version", palette=PALETTE[:2], legend=False)

    ax.set_title("Game Rounds Played by A/B Test Version", fontsize=14, fontweight="bold")
    ax.set_xlabel("Game Version", fontsize=12)
    ax.set_ylabel("Sum of Game Rounds", fontsize=12)

    save_plot("06_gamerounds_by_version.png")


def plot_retention1_by_version(df: pd.DataFrame):
    """Graph 7: Day 1 retention RATE compared across the two A/B groups."""
    plt.figure(figsize=FIGSIZE)

    # groupby + mean on a boolean column gives the RATE (proportion of True)
    rates = df.groupby("version")["retention_1"].mean().reset_index()

    ax = sns.barplot(data=rates, x="version", y="retention_1",
                      hue="version", palette=PALETTE[:2], legend=False)

    ax.set_title("Day 1 Retention Rate by Version", fontsize=14, fontweight="bold")
    ax.set_xlabel("Game Version", fontsize=12)
    ax.set_ylabel("Retention Rate", fontsize=12)
    ax.set_ylim(0, max(rates["retention_1"]) * 1.3)  # extra headroom for labels

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=11)

    save_plot("07_retention1_by_version.png")


def plot_retention7_by_version(df: pd.DataFrame):
    """Graph 8: Day 7 retention RATE compared across the two A/B groups."""
    plt.figure(figsize=FIGSIZE)

    rates = df.groupby("version")["retention_7"].mean().reset_index()

    ax = sns.barplot(data=rates, x="version", y="retention_7",
                      hue="version", palette=PALETTE[2:4], legend=False)

    ax.set_title("Day 7 Retention Rate by Version", fontsize=14, fontweight="bold")
    ax.set_xlabel("Game Version", fontsize=12)
    ax.set_ylabel("Retention Rate", fontsize=12)
    ax.set_ylim(0, max(rates["retention_7"]) * 1.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=11)

    save_plot("08_retention7_by_version.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Graph 9: Correlation heatmap between numeric/boolean features."""
    plt.figure(figsize=FIGSIZE)

    # Select only the columns relevant to correlation analysis.
    # Booleans are converted to int (True->1, False->0) so corr() can use them.
    corr_df = df[["sum_gamerounds", "retention_1", "retention_7"]].astype(
        {"retention_1": int, "retention_7": int}
    )

    corr_matrix = corr_df.corr()

    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                      vmin=-1, vmax=1, linewidths=0.5, square=True)

    ax.set_title("Correlation Heatmap: Game Rounds & Retention", fontsize=14, fontweight="bold")

    save_plot("09_correlation_heatmap.png")


def plot_pairplot(df: pd.DataFrame):
    """Graph 10: Pairplot - pairwise relationships between key variables."""
    upper_limit = df["sum_gamerounds"].quantile(0.99)
    filtered = df[df["sum_gamerounds"] <= upper_limit].copy()

    # pairplot manages its own figure internally (it's a "Grid" object,
    # not a single figure), so we do NOT call plt.figure() before it.
    grid = sns.pairplot(
        filtered[["sum_gamerounds", "retention_1", "retention_7", "version"]],
        hue="version",
        palette=PALETTE[:2],
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 15},
    )
    grid.fig.suptitle("Pairplot: Game Rounds & Retention by Version", y=1.02, fontsize=14, fontweight="bold")

    # pairplot has its own figure (grid.fig), so we save that directly
    # instead of using our save_plot() helper, which targets plt's current figure.
    full_path = os.path.join(PLOTS_DIR, "10_pairplot.png")
    grid.fig.savefig(full_path, dpi=DPI, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"Saved: {full_path}")


def plot_violin_gamerounds_by_retention7(df: pd.DataFrame):
    """Graph 11: Violin plot - distribution shape of game rounds, split by retention_7."""
    plt.figure(figsize=FIGSIZE)

    upper_limit = df["sum_gamerounds"].quantile(0.99)
    filtered = df[df["sum_gamerounds"] <= upper_limit]

    ax = sns.violinplot(data=filtered, x="retention_7", y="sum_gamerounds",
                         hue="retention_7", palette=PALETTE[4:6], legend=False)

    ax.set_title("Game Rounds Distribution by Day 7 Retention (Violin Plot)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Returned After 7 Days?", fontsize=12)
    ax.set_ylabel("Sum of Game Rounds", fontsize=12)

    save_plot("11_violin_gamerounds_retention7.png")


def plot_histogram_kde(df: pd.DataFrame):
    """Graph 12: Histogram + KDE (smooth curve) overlay for game rounds."""
    plt.figure(figsize=FIGSIZE)

    upper_limit = df["sum_gamerounds"].quantile(0.99)
    filtered = df[df["sum_gamerounds"] <= upper_limit]

    ax = sns.histplot(filtered["sum_gamerounds"], bins=40, color=PALETTE[3],
                       kde=True, line_kws={"linewidth": 2})

    ax.set_title("Game Rounds: Histogram with KDE Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sum of Game Rounds", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    save_plot("12_histogram_kde.png")


# ---- BONUS: an extra gaming-specific insight beyond the requested 12 ----
def plot_retention_funnel(df: pd.DataFrame):
    """
    Bonus Graph: Retention Funnel.
    Shows how many players drop off from Total -> Day 1 -> Day 7.
    This is a classic gaming/product-analytics chart that recruiters
    at game companies immediately recognize.
    """
    plt.figure(figsize=FIGSIZE)

    total = len(df)
    day1 = df["retention_1"].sum()
    day7 = df["retention_7"].sum()

    funnel_df = pd.DataFrame({
        "stage": ["Total Players", "Returned Day 1", "Returned Day 7"],
        "count": [total, day1, day7]
    })

    ax = sns.barplot(data=funnel_df, x="stage", y="count",
                      hue="stage", palette=PALETTE[:3], legend=False)

    ax.set_title("Player Retention Funnel", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Players", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fontsize=11)

    save_plot("13_retention_funnel.png")


# ---------------------------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------------------------
def run_eda():
    """
    Run the full EDA pipeline: load data, then generate and save every plot.
    This is the single entry point - running `python eda.py` calls this.
    """
    df = load_data()

    print("\nGenerating visualizations...\n")

    plot_version_count(df)
    plot_retention1_count(df)
    plot_retention7_count(df)
    plot_gamerounds_distribution(df)
    plot_gamerounds_boxplot(df)
    plot_gamerounds_by_version(df)
    plot_retention1_by_version(df)
    plot_retention7_by_version(df)
    plot_correlation_heatmap(df)
    plot_pairplot(df)
    plot_violin_gamerounds_by_retention7(df)
    plot_histogram_kde(df)
    plot_retention_funnel(df)  # bonus

    print("\nEDA completed. All plots saved in outputs/plots/")


# This block only runs when the script is executed directly
# (python eda.py), NOT when it's imported into another file like
# train_model.py. This is standard professional Python practice.
if __name__ == "__main__":
    run_eda()

    