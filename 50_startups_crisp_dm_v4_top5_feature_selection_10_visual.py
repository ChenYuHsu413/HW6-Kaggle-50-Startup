"""
CRISP-DM v4 feature-selection experiment for the Kaggle 50 Startups dataset.

This script keeps the existing project direction: Linear Regression, sklearn
Pipeline, ColumnTransformer, OneHotEncoder, and expert business interpretation.
The v4 upgrade expands the encoded feature-selection comparison to 10 methods,
evaluates top-k subsets from k=1 to k=6, and creates a combined visual summary.
"""

# -----------------------------------------------------------------------------
# Step 0: Import libraries
# -----------------------------------------------------------------------------
import os
from collections import Counter

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    SequentialFeatureSelector,
    f_regression,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = (
    "https://github.com/harimittapalli/Mulitple-Linear-Reggression/raw/master/"
    "50_Startups.csv"
)
LOCAL_DATA_PATH = os.path.join("sources", "50_Startups.csv")
TARGET_COLUMN = "Profit"

ORIGINAL_FEATURES = ["R&D Spend", "Marketing Spend", "Administration", "State"]
NUMERICAL_FEATURES = ["R&D Spend", "Marketing Spend", "Administration"]
CATEGORICAL_FEATURES = ["State"]
STATE_CATEGORIES = ["California", "Florida", "New York"]
CANDIDATE_FEATURES = [
    "R&D Spend",
    "Marketing Spend",
    "Administration",
    "State_California",
    "State_Florida",
    "State_New York",
]
BUSINESS_PRIORITY = {
    "R&D Spend": 0,
    "Marketing Spend": 1,
    "Administration": 2,
    "State_California": 3,
    "State_Florida": 4,
    "State_New York": 5,
}

OUTPUT_DIR = os.path.join("outputs", "crisp_dm_v4")
PLOT_DIR = os.path.join("plots", "crisp_dm_v4")
RANDOM_STATE = 42
TEST_SIZE = 0.2

ALGORITHM_TYPES = {
    "SelectKBest_f_regression": "Filter",
    "SelectKBest_mutual_info_regression": "Filter",
    "RidgeCV_Coefficients": "Embedded",
    "GradientBoosting_Feature_Importance": "Embedded",
    "Sequential_Forward_Selection": "Wrapper",
    "Sequential_Backward_Selection": "Wrapper",
    "LassoCV": "Embedded",
    "ElasticNetCV": "Embedded",
    "RandomForest_Feature_Importance": "Embedded",
    "Permutation_Importance": "Model Inspection",
}


def print_section(title):
    """Print a readable CRISP-DM section divider."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def make_one_hot_encoder():
    """Create OneHotEncoder(drop=None) with stable State category order."""
    kwargs = {
        "drop": None,
        "handle_unknown": "ignore",
        "categories": [STATE_CATEGORIES],
    }
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def build_preprocessing_pipeline():
    """Build the required Pipeline + ColumnTransformer preprocessing step."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("state", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])


def clean_feature_names(raw_feature_names):
    """Convert ColumnTransformer names to report-ready model-ready names."""
    names = []
    for name in raw_feature_names:
        clean = name.replace("num__", "").replace("state__", "")
        names.append(clean)
    return names


def load_dataset():
    """Load from the required URL, with local fallback for offline execution."""
    try:
        print(f"Loading dataset from URL:\n{DATA_URL}")
        return pd.read_csv(DATA_URL)
    except Exception as error:
        print(f"URL load failed: {error}")
        print(f"Loading local fallback dataset from: {LOCAL_DATA_PATH}")
        return pd.read_csv(LOCAL_DATA_PATH)


def encode_train_test(X_train, X_test):
    """Fit preprocessing only on training data and transform train/test splits."""
    preprocessing_pipeline = build_preprocessing_pipeline()
    X_train_encoded = preprocessing_pipeline.fit_transform(X_train)
    X_test_encoded = preprocessing_pipeline.transform(X_test)
    feature_names = clean_feature_names(
        preprocessing_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    X_train_df = pd.DataFrame(
        X_train_encoded, columns=feature_names, index=X_train.index
    )[CANDIDATE_FEATURES]
    X_test_df = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)[
        CANDIDATE_FEATURES
    ]
    return preprocessing_pipeline, X_train_df, X_test_df


def adjusted_r2_score(r2, n_samples, n_predictors):
    """Calculate Adjusted R2 for the test set."""
    if n_samples <= n_predictors + 1:
        return np.nan
    return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1))


def evaluate_feature_set(algorithm_name, selected_features, X_train, X_test, y_train, y_test):
    """Train LinearRegression on one selected encoded feature set and score it."""
    model = LinearRegression()
    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "Algorithm": algorithm_name,
        "Type": ALGORITHM_TYPES[algorithm_name],
        "k": len(selected_features),
        "Selected Features": selected_features,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adjusted_r2_score(r2, len(y_test), len(selected_features)),
        "MAE": mae,
        "MSE": mse,
    }


def sort_features_by_scores(scores, descending=True):
    """Sort score dictionary with business-priority tie breaking."""
    if descending:
        return sorted(
            CANDIDATE_FEATURES,
            key=lambda feature: (-scores.get(feature, -np.inf), BUSINESS_PRIORITY[feature]),
        )
    return sorted(
        CANDIDATE_FEATURES,
        key=lambda feature: (scores.get(feature, np.inf), BUSINESS_PRIORITY[feature]),
    )


def rank_select_kbest_f(X_train, y_train):
    selector = SelectKBest(score_func=f_regression, k="all")
    selector.fit(X_train, y_train)
    scores = dict(zip(CANDIDATE_FEATURES, np.nan_to_num(selector.scores_, nan=-np.inf)))
    return sort_features_by_scores(scores, descending=True)


def rank_select_kbest_mutual_info(X_train, y_train):
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_regression(
            X, y, random_state=RANDOM_STATE, n_neighbors=3
        ),
        k="all",
    )
    selector.fit(X_train, y_train)
    scores = dict(zip(CANDIDATE_FEATURES, np.nan_to_num(selector.scores_, nan=-np.inf)))
    return sort_features_by_scores(scores, descending=True)


def rank_ridge_cv_coefficients(X_train, y_train):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=np.logspace(-3, 6, 30), cv=5)),
        ]
    )
    model.fit(X_train, y_train)
    coefs = np.abs(model.named_steps["regressor"].coef_)
    scores = dict(zip(CANDIDATE_FEATURES, coefs))
    return sort_features_by_scores(scores, descending=True)


def rank_gradient_boosting_importance(X_train, y_train):
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    scores = dict(zip(CANDIDATE_FEATURES, model.feature_importances_))
    return sort_features_by_scores(scores, descending=True)


def rank_sequential_forward(X_train, y_train):
    selected = []
    previous = set()
    for k in range(1, len(CANDIDATE_FEATURES)):
        selector = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=k,
            direction="forward",
            scoring="neg_root_mean_squared_error",
            cv=5,
        )
        selector.fit(X_train, y_train)
        support = set(X_train.columns[selector.get_support()])
        added = sorted(support - previous, key=lambda f: BUSINESS_PRIORITY[f])
        selected.extend(added)
        previous = support
    selected.extend([f for f in CANDIDATE_FEATURES if f not in selected])
    return selected


def rank_sequential_backward(X_train, y_train):
    removal_order = []
    previous = set(CANDIDATE_FEATURES)
    for k in range(len(CANDIDATE_FEATURES) - 1, 0, -1):
        selector = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=k,
            direction="backward",
            scoring="neg_root_mean_squared_error",
            cv=5,
        )
        selector.fit(X_train, y_train)
        support = set(X_train.columns[selector.get_support()])
        removed = sorted(previous - support, key=lambda f: BUSINESS_PRIORITY[f])
        removal_order.extend(removed)
        previous = support
    removal_order.extend(sorted(previous, key=lambda f: BUSINESS_PRIORITY[f]))
    return list(reversed(removal_order))


def rank_lasso_cv(X_train, y_train):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=100000)),
        ]
    )
    model.fit(X_train, y_train)
    coefs = np.abs(model.named_steps["regressor"].coef_)
    scores = dict(zip(CANDIDATE_FEATURES, coefs))
    return sort_features_by_scores(scores, descending=True)


def rank_elastic_net_cv(X_train, y_train):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                ElasticNetCV(
                    cv=5,
                    random_state=RANDOM_STATE,
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    max_iter=100000,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    coefs = np.abs(model.named_steps["regressor"].coef_)
    scores = dict(zip(CANDIDATE_FEATURES, coefs))
    return sort_features_by_scores(scores, descending=True)


def rank_random_forest_importance(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    scores = dict(zip(CANDIDATE_FEATURES, model.feature_importances_))
    return sort_features_by_scores(scores, descending=True)


def rank_permutation_importance(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    result = permutation_importance(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        n_repeats=50,
        random_state=RANDOM_STATE,
    )
    scores = dict(zip(CANDIDATE_FEATURES, result.importances_mean))
    return sort_features_by_scores(scores, descending=True)


RANKERS = {
    "SelectKBest_f_regression": rank_select_kbest_f,
    "SelectKBest_mutual_info_regression": rank_select_kbest_mutual_info,
    "RidgeCV_Coefficients": rank_ridge_cv_coefficients,
    "GradientBoosting_Feature_Importance": rank_gradient_boosting_importance,
    "Sequential_Forward_Selection": rank_sequential_forward,
    "Sequential_Backward_Selection": rank_sequential_backward,
    "LassoCV": rank_lasso_cv,
    "ElasticNetCV": rank_elastic_net_cv,
    "RandomForest_Feature_Importance": rank_random_forest_importance,
    "Permutation_Importance": rank_permutation_importance,
}


def run_feature_selection_experiment(X_train, X_test, y_train, y_test):
    """Rank features with 10 algorithms and evaluate top-k subsets from 1 to 6."""
    ranking_rows = []
    result_rows = []

    for algorithm_name, ranker in RANKERS.items():
        ranked_features = ranker(X_train, y_train)
        ranking_rows.append(
            {
                "Algorithm": algorithm_name,
                "Type": ALGORITHM_TYPES[algorithm_name],
                "Ranking": ranked_features,
                "Top 5 Selected Features": ranked_features[:5],
            }
        )

        for k in range(1, len(CANDIDATE_FEATURES) + 1):
            selected_features = ranked_features[:k]
            result_rows.append(
                evaluate_feature_set(
                    algorithm_name,
                    selected_features,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
            )

    return pd.DataFrame(ranking_rows), pd.DataFrame(result_rows)


def feature_list_to_string(features):
    return "[" + ", ".join(features) + "]"


def compact_feature_list(features):
    """Short labels for dense plot annotations."""
    abbreviations = {
        "R&D Spend": "R&D",
        "Marketing Spend": "Mkt",
        "Administration": "Adm",
        "State_California": "CA",
        "State_Florida": "FL",
        "State_New York": "NY",
    }
    return " + ".join(abbreviations.get(feature, feature) for feature in features)


def build_summary_tables(ranking_df, top_k_df):
    """Create algorithm, top-k, frequency, top-5, and best-by-k tables."""
    summary_rows = []
    for _, ranking_row in ranking_df.iterrows():
        algorithm = ranking_row["Algorithm"]
        algorithm_results = top_k_df[top_k_df["Algorithm"] == algorithm].sort_values(
            ["RMSE", "R2"], ascending=[True, False]
        )
        best = algorithm_results.iloc[0]
        top5 = top_k_df[(top_k_df["Algorithm"] == algorithm) & (top_k_df["k"] == 5)].iloc[
            0
        ]
        summary_rows.append(
            {
                "Algorithm": algorithm,
                "Type": ranking_row["Type"],
                "Top 5 Selected Features": top5["Selected Features"],
                "Best k": int(best["k"]),
                "Best Feature Set": best["Selected Features"],
                "Best RMSE": best["RMSE"],
                "Best R2": best["R2"],
                "Best Adjusted R2": best["Adjusted R2"],
                "MAE": best["MAE"],
            }
        )
    algorithm_summary = pd.DataFrame(summary_rows)

    top_k_table = top_k_df[
        [
            "Algorithm",
            "Type",
            "k",
            "Selected Features",
            "RMSE",
            "R2",
            "Adjusted R2",
            "MAE",
            "MSE",
        ]
    ].copy()

    top1_counter = Counter()
    top3_counter = Counter()
    top5_counter = Counter()
    rank_sum = Counter()
    for ranking in ranking_df["Ranking"]:
        top1_counter[ranking[0]] += 1
        for feature in ranking[:3]:
            top3_counter[feature] += 1
        for feature in ranking[:5]:
            top5_counter[feature] += 1
        for index, feature in enumerate(ranking, start=1):
            rank_sum[feature] += index

    frequency_rows = []
    for feature in CANDIDATE_FEATURES:
        frequency_rows.append(
            {
                "Feature": feature,
                "Times Ranked Top 1": top1_counter[feature],
                "Times Selected in Top 3": top3_counter[feature],
                "Times Selected in Top 5": top5_counter[feature],
                "Average Rank": rank_sum[feature] / len(ranking_df),
            }
        )
    frequency_table = pd.DataFrame(frequency_rows).sort_values(
        ["Times Selected in Top 5", "Average Rank"],
        ascending=[False, True],
    )

    best_by_k = (
        top_k_df.sort_values(["k", "RMSE", "R2"], ascending=[True, True, False])
        .groupby("k", as_index=False)
        .first()
    )
    best_by_k = best_by_k[
        ["k", "Algorithm", "Selected Features", "RMSE", "R2", "Adjusted R2", "MAE"]
    ].rename(columns={"k": "Number of Features", "R2": "R-squared"})

    top5_by_algorithm = top_k_df[top_k_df["k"] == 5][
        [
            "Algorithm",
            "Type",
            "k",
            "Selected Features",
            "RMSE",
            "R2",
            "Adjusted R2",
            "MAE",
            "MSE",
        ]
    ].copy()
    top5_by_algorithm = top5_by_algorithm.rename(
        columns={"k": "Number of Features", "R2": "R-squared"}
    )

    return (
        algorithm_summary,
        top_k_table,
        frequency_table,
        best_by_k,
        top5_by_algorithm,
    )


def csv_ready(df):
    """Convert feature-list columns to strings for readable CSV output."""
    out = df.copy()
    for column in ["Selected Features", "Top 5 Selected Features", "Best Feature Set", "Ranking"]:
        if column in out.columns:
            out[column] = out[column].apply(
                lambda value: feature_list_to_string(value)
                if isinstance(value, list)
                else value
            )
    return out


def save_tables(
    algorithm_summary,
    top_k_table,
    frequency_table,
    best_by_k,
    top5_by_algorithm,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = {
        "algorithm_summary": os.path.join(
            OUTPUT_DIR, "feature_selection_algorithm_summary.csv"
        ),
        "top_k": os.path.join(OUTPUT_DIR, "top_k_performance_table.csv"),
        "frequency": os.path.join(
            OUTPUT_DIR, "feature_ranking_frequency_table.csv"
        ),
        "best_by_k": os.path.join(
            OUTPUT_DIR, "best_subset_per_feature_count_table.csv"
        ),
        "top5_by_algorithm": os.path.join(
            OUTPUT_DIR, "top5_selected_features_by_algorithm.csv"
        ),
    }
    csv_ready(algorithm_summary).to_csv(paths["algorithm_summary"], index=False)
    csv_ready(top_k_table).to_csv(paths["top_k"], index=False)
    frequency_table.to_csv(paths["frequency"], index=False)
    csv_ready(best_by_k).to_csv(paths["best_by_k"], index=False)
    csv_ready(top5_by_algorithm).to_csv(paths["top5_by_algorithm"], index=False)
    return paths


def style_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_combined_summary(top_k_df, top5_by_algorithm):
    """Create the all-in-one figure with per-algorithm selected-feature table."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 13))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.25, 1.45], hspace=0.28, wspace=0.2)
    ax_rmse = fig.add_subplot(grid[0, 0])
    ax_r2 = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(RANKERS)))
    for color, (algorithm, group) in zip(color_cycle, top_k_df.groupby("Algorithm")):
        group = group.sort_values("k")
        label = algorithm.replace("_", " ")
        ax_rmse.plot(
            group["k"],
            group["RMSE"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=label,
        )
        ax_r2.plot(
            group["k"],
            group["R2"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=label,
        )

    ax_rmse.set_title("RMSE by Number of Features", fontsize=13, fontweight="bold")
    ax_rmse.set_xlabel("Number of Features")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_xticks(range(1, 7))
    style_axes(ax_rmse)

    ax_r2.set_title("R-squared by Number of Features", fontsize=13, fontweight="bold")
    ax_r2.set_xlabel("Number of Features")
    ax_r2.set_ylabel("R-squared")
    ax_r2.set_xticks(range(1, 7))
    style_axes(ax_r2)

    handles, labels = ax_r2.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        fontsize=8,
        frameon=False,
    )

    ax_table.axis("off")
    display = top5_by_algorithm[
        ["Algorithm", "Number of Features", "Selected Features", "RMSE", "R-squared"]
    ].copy()
    display["Algorithm"] = display["Algorithm"].str.replace("_", " ", regex=False)
    display["Selected Features"] = display["Selected Features"].apply(feature_list_to_string)
    display["RMSE"] = display["RMSE"].map("{:,.2f}".format)
    display["R-squared"] = display["R-squared"].map("{:.4f}".format)

    table = ax_table.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.32)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_facecolor("#e9edf2")
        elif row % 2 == 1:
            cell.set_facecolor("#f7f8fa")
        if col == 0:
            cell.set_width(0.23)
        elif col == 1:
            cell.set_width(0.11)
        elif col == 2:
            cell.set_width(0.48)
        elif col in [3, 4]:
            cell.set_width(0.09)

    fig.suptitle(
        "Each Algorithm's Top-5 Selected Features and Top-k Performance",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    output_path = os.path.join(
        PLOT_DIR, "feature_selection_performance_allinone_summary.png"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_best_rmse_by_algorithm(algorithm_summary):
    plot_df = algorithm_summary.sort_values("Best RMSE", ascending=True)
    plt.figure(figsize=(11, 6))
    plt.barh(plot_df["Algorithm"], plot_df["Best RMSE"], color="#2f6f8f")
    plt.title("Best RMSE Across 10 Feature Selection Algorithms")
    plt.xlabel("Best RMSE")
    style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "best_rmse_by_algorithm.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_best_adjusted_r2_by_algorithm(algorithm_summary):
    plot_df = algorithm_summary.sort_values("Best Adjusted R2", ascending=True)
    plt.figure(figsize=(11, 6))
    plt.barh(plot_df["Algorithm"], plot_df["Best Adjusted R2"], color="#607d3b")
    plt.title("Best Adjusted R2 Across 10 Feature Selection Algorithms")
    plt.xlabel("Best Adjusted R2")
    style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "best_adjusted_r2_by_algorithm.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_feature_ranking_frequency(frequency_table):
    plot_df = frequency_table.sort_values("Times Selected in Top 5", ascending=True)
    plt.figure(figsize=(10, 5.5))
    plt.barh(plot_df["Feature"], plot_df["Times Selected in Top 5"], color="#b45f3c")
    plt.title("Feature Ranking Frequency: Times Selected in Top 5")
    plt.xlabel("Count across 10 algorithms")
    plt.xlim(0, 10)
    style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_ranking_frequency.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_selection_heatmap(ranking_df):
    matrix = []
    labels = []
    for _, row in ranking_df.iterrows():
        selected = set(row["Top 5 Selected Features"])
        matrix.append([1 if feature in selected else 0 for feature in CANDIDATE_FEATURES])
        labels.append(row["Algorithm"])
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CANDIDATE_FEATURES)))
    ax.set_xticklabels(CANDIDATE_FEATURES, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Top-5 Feature Selection Heatmap by Algorithm")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                "Y" if matrix[row, col] else "",
                ha="center",
                va="center",
                color="#111111",
                fontsize=8,
            )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "algorithm_feature_selection_heatmap.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_top5_feature_selection_comparison(top_k_df):
    plot_df = top_k_df[top_k_df["k"] == 5].sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["Algorithm"], plot_df["RMSE"], color="#2f6f8f")
    ax.set_title("Top-5 Feature Subset Comparison")
    ax.set_xlabel("Feature Selection Algorithm")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=35)
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "top5_feature_selection_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_best_selected_features_by_algorithm(top5_by_algorithm):
    plot_df = top5_by_algorithm.sort_values("RMSE", ascending=True).copy()
    labels = plot_df["Algorithm"].str.replace("_", " ", regex=False)
    feature_labels = plot_df["Selected Features"].apply(
        lambda features: compact_feature_list(features)
    )

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    bars = ax.barh(labels, plot_df["RMSE"], color="#2f6f8f", alpha=0.92)
    ax.set_title("Top-5 Selected Feature Set by Algorithm")
    ax.set_xlabel("RMSE of each algorithm's own top-5 selected features")
    ax.set_ylabel("Feature Selection Algorithm")
    style_axes(ax)

    x_offset = max(plot_df["RMSE"]) * 0.012
    for bar, features, r2 in zip(bars, feature_labels, plot_df["R-squared"]):
        ax.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{features} | R2={r2:.4f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(plot_df["RMSE"]) * 1.28)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "best_selected_features_by_algorithm.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_best_model_actual_vs_predicted(best_row, X_train, X_test, y_train, y_test):
    selected_features = best_row["Selected Features"]
    model = LinearRegression()
    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, color="#607d3b", alpha=0.85)
    plt.plot([min_value, max_value], [min_value, max_value], color="#333333", linestyle="--")
    plt.title("Actual vs Predicted Profit - Best v4 Model")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "best_model_actual_vs_predicted.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_plots(
    ranking_df,
    top_k_df,
    algorithm_summary,
    frequency_table,
    best_by_k,
    top5_by_algorithm,
    X_train,
    X_test,
    y_train,
    y_test,
):
    os.makedirs(PLOT_DIR, exist_ok=True)
    best_overall = top_k_df.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
    return {
        "combined": plot_combined_summary(top_k_df, top5_by_algorithm),
        "best_rmse": plot_best_rmse_by_algorithm(algorithm_summary),
        "best_adjusted_r2": plot_best_adjusted_r2_by_algorithm(algorithm_summary),
        "frequency": plot_feature_ranking_frequency(frequency_table),
        "heatmap": plot_selection_heatmap(ranking_df),
        "top5": plot_top5_feature_selection_comparison(top_k_df),
        "best_selected_features": plot_best_selected_features_by_algorithm(
            top5_by_algorithm
        ),
        "actual_vs_predicted": plot_best_model_actual_vs_predicted(
            best_overall, X_train, X_test, y_train, y_test
        ),
    }


def print_top5_by_algorithm(top5_by_algorithm):
    print("\nEach algorithm's own top-5 selected feature set:")
    rows = top5_by_algorithm.copy()
    rows["Selected Features"] = rows["Selected Features"].apply(feature_list_to_string)
    print(
        rows[
            [
                "Algorithm",
                "Number of Features",
                "Selected Features",
                "RMSE",
                "R-squared",
                "Adjusted R2",
                "MAE",
            ]
        ].to_string(
            index=False,
            formatters={
                "RMSE": "{:,.2f}".format,
                "R-squared": "{:.4f}".format,
                "Adjusted R2": "{:.4f}".format,
                "MAE": "{:,.2f}".format,
            },
        )
    )


def print_final_conclusion(algorithm_summary, top_k_df, frequency_table, best_by_k):
    best_overall = top_k_df.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
    top5_frequency = frequency_table.sort_values(
        ["Times Selected in Top 5", "Average Rank"], ascending=[False, True]
    )
    most_frequent_5 = top5_frequency.head(5)["Feature"].tolist()
    best5_rows = top_k_df[top_k_df["k"] == 5]
    state_top5_counts = {
        feature: int(best5_rows["Selected Features"].apply(lambda x: feature in x).sum())
        for feature in CANDIDATE_FEATURES
        if feature.startswith("State_")
    }

    rd_top1_count = int(
        frequency_table.loc[
            frequency_table["Feature"] == "R&D Spend", "Times Ranked Top 1"
        ].iloc[0]
    )
    marketing_top3_count = int(
        frequency_table.loc[
            frequency_table["Feature"] == "Marketing Spend", "Times Selected in Top 3"
        ].iloc[0]
    )
    admin_top5_count = int(
        frequency_table.loc[
            frequency_table["Feature"] == "Administration", "Times Selected in Top 5"
        ].iloc[0]
    )

    print_section("Step 6: Expert-Based Business Conclusion")
    print(
        "RMSE and R-squared are shown as two separate line charts inside one figure "
        "because they answer different questions and use different scales: RMSE is "
        "prediction error in Profit units, while R-squared is explained variance."
    )
    print(
        f"\nMost frequently selected 5 features across the 10 algorithms: "
        f"{feature_list_to_string(most_frequent_5)}"
    )
    print(
        f"R&D Spend remains the strongest feature: it was ranked top 1 by "
        f"{rd_top1_count}/10 algorithms."
    )
    print(
        f"Marketing Spend remains an important supporting amplifier: it appeared in "
        f"the top 3 for {marketing_top3_count}/10 algorithms."
    )
    print(
        f"Administration is mixed: it appeared in top-5 subsets for "
        f"{admin_top5_count}/10 algorithms, but should be judged by Adjusted R2, "
        "RMSE, and MAE because it may act like operating-cost noise."
    )
    print("State dummy variables in the 10 top-5 subsets:")
    for feature, count in state_top5_counts.items():
        print(f" - {feature}: {count}/10")
    print(
        "\nBest overall model by RMSE:"
        f"\n - Algorithm: {best_overall['Algorithm']}"
        f"\n - k: {int(best_overall['k'])}"
        f"\n - Features: {feature_list_to_string(best_overall['Selected Features'])}"
        f"\n - RMSE: {best_overall['RMSE']:,.2f}"
        f"\n - MAE: {best_overall['MAE']:,.2f}"
        f"\n - R-squared: {best_overall['R2']:.4f}"
        f"\n - Adjusted R2: {best_overall['Adjusted R2']:.4f}"
    )

    best_k = int(best_overall["k"])
    if best_k == 5:
        print("\nThe best test-set model prefers exactly 5 features.")
    elif best_k < 5:
        print(
            "\nThe best test-set model prefers fewer than 5 features, which supports "
            "the small-sample warning that simpler models can outperform more complex "
            "models on this 50-row dataset."
        )
    else:
        print(
            "\nThe best test-set model prefers all 6 encoded features, but State "
            "should still not be over-interpreted because the dataset has only 50 rows."
        )

    rmse_by_k = best_by_k.sort_values("Number of Features")
    first_rmse = rmse_by_k.iloc[0]["RMSE"]
    last_rmse = rmse_by_k.iloc[-1]["RMSE"]
    trend = "decreases" if last_rmse < first_rmse else "increases"
    print(
        f"\nAcross the best subset at each feature count, error {trend} from k=1 "
        f"to k=6 ({first_rmse:,.2f} -> {last_rmse:,.2f}). The practical choice "
        "should balance RMSE, MAE, Adjusted R2, and business interpretability."
    )
    print(
        "\nFinal business recommendation: keep R&D Spend as the core predictor, "
        "test Marketing Spend as the most defensible supporting business feature, "
        "treat Administration as conditional, and use State only as a cautious "
        "regional control. For a final pure Linear Regression deployment, "
        "OneHotEncoder(drop='first') may be used later to avoid the dummy variable "
        "trap; v4 intentionally keeps all 3 State dummies visible for comparison."
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Business Understanding
    # -------------------------------------------------------------------------
    print_section("Step 1: Business Understanding")
    print(
        "Goal: predict Profit from the Kaggle 50 Startups dataset using Linear "
        "Regression and interpretable feature selection.\n"
        "Expert consensus: R&D Spend is the core profit driver; Marketing Spend "
        "is a supporting amplifier; Administration is uncertain because it may "
        "represent operating cost; State is a supporting regional feature and "
        "must not be over-interpreted with only 50 rows."
    )

    # -------------------------------------------------------------------------
    # Step 2: Data Understanding
    # -------------------------------------------------------------------------
    print_section("Step 2: Data Understanding")
    df = load_dataset()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nState value counts:")
    print(df["State"].value_counts())
    print("\nNumerical correlation with Profit:")
    print(df[NUMERICAL_FEATURES + [TARGET_COLUMN]].corr()[TARGET_COLUMN])

    # -------------------------------------------------------------------------
    # Step 3: Data Preparation
    # -------------------------------------------------------------------------
    print_section("Step 3: Data Preparation")
    X = df[ORIGINAL_FEATURES]
    y = df[TARGET_COLUMN]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    preprocessing_pipeline, X_train, X_test = encode_train_test(X_train_raw, X_test_raw)

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print("Original input features:", ORIGINAL_FEATURES)
    print("Preprocessing pipeline:", preprocessing_pipeline)
    print("OneHotEncoder setting: drop=None, all 3 State dummy variables kept.")
    print("Candidate model-ready features:", CANDIDATE_FEATURES)
    print(
        "Note: keeping all 3 State dummies is intentional for feature-selection "
        "comparison. A final deployment model may later use drop='first' if needed."
    )

    # -------------------------------------------------------------------------
    # Step 4: Modeling
    # -------------------------------------------------------------------------
    print_section("Step 4: Modeling - 10 Feature Selection Algorithms")
    ranking_df, top_k_df = run_feature_selection_experiment(
        X_train, X_test, y_train, y_test
    )
    print("Completed rankings and top-k LinearRegression evaluations.")

    # -------------------------------------------------------------------------
    # Step 5: Evaluation
    # -------------------------------------------------------------------------
    print_section("Step 5: Evaluation")
    (
        algorithm_summary,
        top_k_table,
        frequency_table,
        best_by_k,
        top5_by_algorithm,
    ) = build_summary_tables(ranking_df, top_k_df)
    table_paths = save_tables(
        algorithm_summary,
        top_k_table,
        frequency_table,
        best_by_k,
        top5_by_algorithm,
    )
    plot_paths = save_plots(
        ranking_df,
        top_k_df,
        algorithm_summary,
        frequency_table,
        best_by_k,
        top5_by_algorithm,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print_top5_by_algorithm(top5_by_algorithm)

    best_overall = top_k_df.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
    print("\nBest overall algorithm and feature subset:")
    print(
        f" - Algorithm: {best_overall['Algorithm']}\n"
        f" - k: {int(best_overall['k'])}\n"
        f" - Features: {feature_list_to_string(best_overall['Selected Features'])}\n"
        f" - RMSE: {best_overall['RMSE']:,.2f}\n"
        f" - R-squared: {best_overall['R2']:.4f}\n"
        f" - Adjusted R2: {best_overall['Adjusted R2']:.4f}\n"
        f" - MAE: {best_overall['MAE']:,.2f}"
    )

    print("\nSummary table of best subsets by feature count:")
    display_best_by_k = best_by_k.copy()
    display_best_by_k["Selected Features"] = display_best_by_k[
        "Selected Features"
    ].apply(feature_list_to_string)
    print(
        display_best_by_k.to_string(
            index=False,
            formatters={
                "RMSE": "{:,.2f}".format,
                "R-squared": "{:.4f}".format,
                "Adjusted R2": "{:.4f}".format,
                "MAE": "{:,.2f}".format,
            },
        )
    )

    print("\nSaved CSV tables:")
    for path in table_paths.values():
        print(f" - {path}")
    print("\nSaved PNG plots:")
    for path in plot_paths.values():
        print(f" - {path}")

    print_final_conclusion(algorithm_summary, top_k_df, frequency_table, best_by_k)


if __name__ == "__main__":
    main()
