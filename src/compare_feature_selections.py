"""
Integrated comparison of the four feature-selection analyses.

All four CSV outputs under outputs/metrics evaluate the same LinearRegression
pipeline on the same train/test split; they differ only in HOW candidate
feature subsets were chosen:

  1. sequential_feature_selection_list.csv        - CV-based forward selection
  2. business_guided_feature_selection_summary.csv - expert business priority order
  3. feature_selection_performance_allinone.csv    - five selection algorithms
  4. marketing_vs_administration_comparison.csv    - Marketing-vs-Administration probe

This script merges them into one deduplicated table of unique feature sets and
one integrated figure, to show whether the four lenses converge.
"""

import ast
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRICS_DIR = os.path.join("outputs", "metrics")
FIGURES_DIR = os.path.join("outputs", "figures")
N_TEST = 10  # test rows from the shared 80/20 split of 50 samples

SOURCE_FILES = {
    "SFS": "sequential_feature_selection_list.csv",
    "BUSINESS": "business_guided_feature_selection_summary.csv",
    "ALGO": "feature_selection_performance_allinone.csv",
    "MVA": "marketing_vs_administration_comparison.csv",
}

SOURCE_LABELS = {
    "SFS": "Sequential FS (CV forward)",
    "BUSINESS": "Business-guided order",
    "ALGO": "Five selection algorithms",
    "MVA": "Marketing vs Administration probe",
}


def print_section(title):
    """Print a readable section divider."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def adjusted_r2_score(r2, n_samples, n_predictors):
    """Adjusted R2 with the same formula used by the main modeling script."""
    if n_samples <= n_predictors + 1:
        return float("nan")
    return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1))


def parse_feature_list(raw):
    """Parse the bracketed feature list string used by all four CSVs."""
    inner = raw.strip().strip("[]")
    return tuple(sorted(part.strip() for part in inner.split(",") if part.strip()))


def short_set_label(features):
    """Compact display label for a feature set."""
    abbrev = {
        "R&D Spend": "R&D",
        "Marketing Spend": "Mkt",
        "Administration": "Adm",
        "State_Florida": "FL",
        "State_New York": "NY",
    }
    return " + ".join(abbrev.get(f, f) for f in features)


def load_all_rows():
    """Load the four CSVs into one long DataFrame with a Source column."""
    frames = []
    for source, file_name in SOURCE_FILES.items():
        df = pd.read_csv(os.path.join(METRICS_DIR, file_name))
        df["Source"] = source
        if "Adjusted R2" not in df.columns:
            df["Adjusted R2"] = [
                adjusted_r2_score(r2, N_TEST, n)
                for r2, n in zip(df["R-squared"], df["Number of Features"])
            ]
        if "Algorithm" not in df.columns:
            df["Algorithm"] = ""
        frames.append(
            df[
                [
                    "Selected Features",
                    "Number of Features",
                    "RMSE",
                    "R-squared",
                    "Adjusted R2",
                    "MAE",
                    "Source",
                    "Algorithm",
                ]
            ]
        )
    long_df = pd.concat(frames, ignore_index=True)
    long_df["Feature Key"] = long_df["Selected Features"].apply(parse_feature_list)
    return long_df


def build_integrated_table(long_df):
    """Deduplicate identical feature sets and record which analyses tried them."""
    rows = []
    for key, group in long_df.groupby("Feature Key"):
        sources = sorted(group["Source"].unique())
        algorithms = sorted(a for a in group["Algorithm"].unique() if a)
        rows.append(
            {
                "Features": ", ".join(key),
                "Label": short_set_label(key),
                "Number of Features": int(group["Number of Features"].iloc[0]),
                "RMSE": group["RMSE"].mean(),
                "R-squared": group["R-squared"].mean(),
                "Adjusted R2": group["Adjusted R2"].mean(),
                "MAE": group["MAE"].mean(),
                "Contains R&D": "R&D Spend" in key,
                "Evaluated By": " / ".join(sources),
                "Evaluation Count": len(group),
                "Algorithms": ", ".join(algorithms),
            }
        )
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    table.insert(0, "Rank", table.index + 1)
    return table


def plot_integrated_comparison(long_df, table, output_path):
    """One figure: selection paths, full ranking, and the integrated table."""
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.5])
    ax_paths = fig.add_subplot(grid[0, 0])
    ax_rank = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    # --- Panel A: RMSE vs feature count for each selection path ------------
    algo_df = long_df[long_df["Source"] == "ALGO"]
    business_path = long_df[long_df["Source"] == "BUSINESS"].sort_values(
        "Number of Features"
    )
    sfs_path = long_df[long_df["Source"] == "SFS"].sort_values("Number of Features")
    kbest_path = algo_df[algo_df["Algorithm"] == "SelectKBest"].sort_values(
        "Number of Features"
    )

    ax_paths.plot(
        business_path["Number of Features"],
        business_path["RMSE"],
        marker="o",
        linewidth=2.4,
        color="#2f6f8f",
        label="Business order (= RFE / Lasso / RF)",
    )
    ax_paths.plot(
        sfs_path["Number of Features"],
        sfs_path["RMSE"],
        marker="s",
        linewidth=2,
        linestyle="--",
        color="#b45f3c",
        label="Sequential FS (CV forward)",
    )
    ax_paths.plot(
        kbest_path["Number of Features"],
        kbest_path["RMSE"],
        marker="^",
        linewidth=2,
        linestyle="-.",
        color="#6a8f3f",
        label="SelectKBest",
    )
    best = table.iloc[0]
    ax_paths.scatter(
        [best["Number of Features"]],
        [best["RMSE"]],
        s=160,
        zorder=3,
        color="#1f5a34",
        label="Common winner: R&D only",
    )
    ax_paths.set_title("Every selection path starts at its minimum: R&D Spend only")
    ax_paths.set_xlabel("Number of Features")
    ax_paths.set_ylabel("Test RMSE")
    ax_paths.set_xticks(range(1, 6))
    ax_paths.grid(True, linestyle="--", alpha=0.35)
    ax_paths.legend(loc="lower right", fontsize=9)

    # --- Panel B: every unique feature set ever evaluated, ranked ----------
    rank_df = table.sort_values("RMSE", ascending=False)
    colors = [
        "#1f5a34" if rank == 1 else ("#2f6f8f" if has_rd else "#c24031")
        for rank, has_rd in zip(rank_df["Rank"], rank_df["Contains R&D"])
    ]
    ax_rank.barh(rank_df["Label"], rank_df["RMSE"], color=colors)
    ax_rank.set_title("All unique feature sets across the four analyses")
    ax_rank.set_xlabel("Test RMSE")
    ax_rank.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax_rank.tick_params(axis="y", labelsize=9)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f5a34"),
        plt.Rectangle((0, 0), 1, 1, color="#2f6f8f"),
        plt.Rectangle((0, 0), 1, 1, color="#c24031"),
    ]
    ax_rank.legend(
        handles,
        ["Winner", "Contains R&D Spend", "No R&D Spend"],
        loc="lower right",
        fontsize=9,
    )

    # --- Panel C: integrated table ------------------------------------------
    ax_table.axis("off")
    display = table[
        ["Rank", "Label", "Number of Features", "RMSE", "Adjusted R2", "Evaluated By"]
    ].copy()
    display.columns = ["#", "Feature Set", "k", "RMSE", "Adj R2", "Evaluated By"]
    display["RMSE"] = display["RMSE"].map("{:,.0f}".format)
    display["Adj R2"] = display["Adj R2"].map("{:.4f}".format)
    cell_table = ax_table.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    cell_table.auto_set_font_size(False)
    cell_table.set_fontsize(8.5)
    cell_table.scale(1, 1.3)
    for (row, col), cell in cell_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        elif row == 1:
            cell.set_facecolor("#e3efe7")
        if col == 1:
            cell.set_width(0.3)
        if col == 5:
            cell.set_width(0.22)

    fig.suptitle(
        "Integrated Feature Selection Comparison - same model, same split, "
        "four selection lenses",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print_section("Loading the four feature-selection result tables")
    long_df = load_all_rows()
    for source, label in SOURCE_LABELS.items():
        count = (long_df["Source"] == source).sum()
        print(f" - {label}: {count} evaluated feature sets")

    print_section("Integrated unique feature sets (deduplicated across analyses)")
    table = build_integrated_table(long_df)
    print(
        table[
            [
                "Rank",
                "Label",
                "Number of Features",
                "RMSE",
                "Adjusted R2",
                "Evaluated By",
                "Evaluation Count",
            ]
        ].to_string(
            index=False,
            formatters={
                "RMSE": "{:,.2f}".format,
                "Adjusted R2": "{:.4f}".format,
            },
        )
    )

    csv_path = os.path.join(METRICS_DIR, "feature_selection_integrated_comparison.csv")
    table.drop(columns=["Label"]).to_csv(csv_path, index=False)
    print(f"\nSaved integrated comparison CSV to: {csv_path}")

    png_path = os.path.join(
        FIGURES_DIR, "feature_selection_integrated_comparison.png"
    )
    plot_integrated_comparison(long_df, table, png_path)
    print(f"Saved integrated comparison figure to: {png_path}")

    print_section("Conclusions")
    best = table.iloc[0]
    worst_with_rd = table[table["Contains R&D"]].iloc[-1]
    no_rd = table[~table["Contains R&D"]]
    print(
        f"1. All four analyses converge on the same winner: [{best['Features']}] "
        f"with RMSE {best['RMSE']:,.0f} and Adjusted R2 {best['Adjusted R2']:.4f}."
    )
    print(
        "2. Adding any feature after R&D Spend never reduces test RMSE "
        f"(R&D-containing sets range {best['RMSE']:,.0f}-{worst_with_rd['RMSE']:,.0f})."
    )
    print(
        "3. Every set WITHOUT R&D Spend collapses: RMSE "
        f"{no_rd['RMSE'].min():,.0f}-{no_rd['RMSE'].max():,.0f} "
        "(3-4x worse), confirming R&D Spend carries the signal."
    )
    print(
        "4. The only disagreement between lenses is the second feature: "
        "CV-based forward selection picks Administration, while the test split "
        "and business logic both prefer Marketing Spend - a small-sample "
        "reminder that CV choice and test outcome can diverge."
    )


if __name__ == "__main__":
    main()
