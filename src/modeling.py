"""
CRISP-DM sklearn solution for the Kaggle 50 Startups regression problem.

This script predicts startup Profit from spending features and State, then
compares five feature experiments guided by expert business discussion.
"""

# -----------------------------------------------------------------------------
# Step 0: Import libraries
# -----------------------------------------------------------------------------
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


DATA_URL = (
    "https://github.com/harimittapalli/Mulitple-Linear-Reggression/raw/master/"
    "50_Startups.csv"
)
LOCAL_DATA_PATH = os.path.join("sources", "50_Startups.csv")
FIGURES_DIR = os.path.join("outputs", "figures")
METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
TARGET_COLUMN = "Profit"
NUMERICAL_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]


def print_section(title):
    """Print a readable section divider."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def make_one_hot_encoder():
    """Create a OneHotEncoder compatible with newer and older sklearn versions."""
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def adjusted_r2_score(r2, n_samples, n_predictors):
    """
    Calculate Adjusted R2.

    n_predictors is the number of model input columns after preprocessing.
    """
    if n_samples <= n_predictors + 1:
        return np.nan
    return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1))


def build_preprocessor(selected_features, categorical_features):
    """Build a ColumnTransformer for the selected experiment features."""
    numeric_features = [
        feature for feature in selected_features if feature not in categorical_features
    ]

    transformers = []
    if numeric_features:
        transformers.append(("num", "passthrough", numeric_features))
    if categorical_features:
        transformers.append(("cat", make_one_hot_encoder(), categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def count_model_predictors(pipeline, X_test):
    """Count transformed predictor columns used by the regression model."""
    transformed_X = pipeline.named_steps["preprocessor"].transform(X_test)
    return transformed_X.shape[1]


def run_experiment(model_name, purpose, features, categorical_features, X_train, X_test, y_train, y_test):
    """Train one sklearn Pipeline experiment and return the model plus metrics."""
    preprocessor = build_preprocessor(features, categorical_features)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])

    r2 = r2_score(y_test, y_pred)
    n_predictors = count_model_predictors(model, X_test[features])
    adj_r2 = adjusted_r2_score(r2, len(y_test), n_predictors)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    result = {
        "Model": model_name,
        "Purpose": purpose,
        "Features": ", ".join(features),
        "Predictor Count": n_predictors,
        "R2": r2,
        "Adjusted R2": adj_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }
    return model, result


def get_model_coefficients(model, features):
    """Return a DataFrame of fitted feature names and linear coefficients."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out(features)
    clean_names = [
        name.replace("num__", "").replace("cat__", "") for name in feature_names
    ]

    coefficients = pd.DataFrame(
        {
            "Feature": clean_names,
            "Coefficient": regressor.coef_,
            "Absolute Coefficient": np.abs(regressor.coef_),
        }
    )
    return coefficients.sort_values("Absolute Coefficient", ascending=False)


def load_dataset():
    """Load from the required URL, with a local fallback for offline execution."""
    try:
        print(f"Loading dataset from URL:\n{DATA_URL}")
        return pd.read_csv(DATA_URL)
    except Exception as error:
        print(f"URL load failed: {error}")
        print(f"Loading local fallback dataset from: {LOCAL_DATA_PATH}")
        return pd.read_csv(LOCAL_DATA_PATH)


def save_data_understanding_plots(df, output_dir):
    """Save simple exploratory plots for the main business features."""
    os.makedirs(output_dir, exist_ok=True)

    for feature in NUMERICAL_FEATURES:
        file_name = feature.lower().replace(" ", "_").replace("&", "and")
        plt.figure(figsize=(8, 5))
        plt.scatter(df[feature], df[TARGET_COLUMN], alpha=0.75, color="#2f6f8f")
        plt.title(f"{feature} vs Profit")
        plt.xlabel(feature)
        plt.ylabel("Profit")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_vs_profit.png"), dpi=150)
        plt.close()

    state_profit = df.groupby("State")[TARGET_COLUMN].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    state_profit.plot(kind="bar", color="#6a8f3f")
    plt.title("Average Profit by State")
    plt.xlabel("State")
    plt.ylabel("Average Profit")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_profit_by_state.png"), dpi=150)
    plt.close()

    corr = df[NUMERICAL_FEATURES + [TARGET_COLUMN]].corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="RdYlBu", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for row in range(len(corr.columns)):
        for col in range(len(corr.columns)):
            plt.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def save_model_evaluation_plots(results_df, best_model, best_features, best_coefficients, X_test, y_test, output_dir):
    """Save model comparison, prediction, and coefficient charts."""
    os.makedirs(output_dir, exist_ok=True)
    plot_df = results_df.sort_values("Adjusted R2", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Model"], plot_df["Adjusted R2"], color="#2f6f8f")
    plt.title("Model Comparison by Adjusted R2")
    plt.xlabel("Adjusted R2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_adjusted_r2.png"), dpi=150)
    plt.close()

    plot_df = results_df.sort_values("RMSE", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Model"], plot_df["RMSE"], color="#b45f3c")
    plt.title("Model Comparison by RMSE")
    plt.xlabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_rmse.png"), dpi=150)
    plt.close()

    y_pred = best_model.predict(X_test[best_features])
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.8, color="#6a8f3f")
    plt.plot([min_value, max_value], [min_value, max_value], color="#333333", linestyle="--")
    plt.title("Actual vs Predicted Profit - Best Model")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_model_actual_vs_predicted.png"), dpi=150)
    plt.close()

    coef_df = best_coefficients.sort_values("Absolute Coefficient", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="#2f6f8f")
    plt.title("Best Model Coefficients")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_model_coefficients.png"), dpi=150)
    plt.close()


def save_feature_selection_performance(results_df, output_dir):
    """Save feature selection performance as a CSV table and charts."""
    os.makedirs(output_dir, exist_ok=True)

    performance_df = results_df.sort_values("Experiment Order").copy()
    performance_df["Feature Set"] = performance_df["Model"].str.replace(
        r"^Model [A-E]: ", "", regex=True
    )
    performance_columns = [
        "Experiment Order",
        "Model",
        "Features",
        "Predictor Count",
        "R2",
        "Adjusted R2",
        "MAE",
        "RMSE",
    ]
    csv_path = os.path.join(METRICS_DIR, "feature_selection_performance.csv")
    performance_df[performance_columns].to_csv(csv_path, index=False)

    best_adj_row = performance_df.loc[performance_df["Adjusted R2"].idxmax()]
    best_rmse_row = performance_df.loc[performance_df["RMSE"].idxmin()]
    state_only_row = performance_df[
        performance_df["Model"] == "Model E: State Only"
    ].iloc[0]
    result_comment = (
        "Result meaning:\n"
        f"- Best feature set: {best_adj_row['Feature Set']}\n"
        f"- Adjusted R2 = {best_adj_row['Adjusted R2']:.4f}, "
        f"RMSE = {best_rmse_row['RMSE']:,.2f}\n"
        "- Adding Marketing, Administration, and State did not improve this test split.\n"
        f"- State only performs poorly: Adjusted R2 = {state_only_row['Adjusted R2']:.4f}."
    )

    plt.figure(figsize=(12, 7))
    plt.plot(
        performance_df["Feature Set"],
        performance_df["Adjusted R2"],
        marker="o",
        linewidth=2,
        color="#2f6f8f",
    )
    best_position = performance_df.index.get_loc(best_adj_row.name)
    plt.scatter(
        best_position,
        best_adj_row["Adjusted R2"],
        s=120,
        color="#1f5a34",
        zorder=3,
        label="Best feature set",
    )
    plt.title("Feature Selection Performance - Adjusted R2")
    plt.xlabel("Feature Set")
    plt.ylabel("Adjusted R2")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.annotate(
        "Best: R&D Spend Only",
        xy=(best_position, best_adj_row["Adjusted R2"]),
        xytext=(best_position + 0.3, best_adj_row["Adjusted R2"] - 0.25),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
    )
    plt.gcf().text(
        0.53,
        0.23,
        result_comment,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f6f6", "edgecolor": "#999999"},
    )
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "feature_selection_adjusted_r2.png"), dpi=150
    )
    plt.close()

    rmse_comment = (
        "Result meaning:\n"
        f"- Lowest error: {best_rmse_row['Feature Set']}\n"
        f"- RMSE = {best_rmse_row['RMSE']:,.2f}\n"
        "- Lower RMSE means predictions are closer to actual Profit.\n"
        "- More features increased error, so simpler is better here."
    )

    plt.figure(figsize=(12, 7))
    plt.plot(
        performance_df["Feature Set"],
        performance_df["RMSE"],
        marker="o",
        linewidth=2,
        color="#b45f3c",
    )
    best_rmse_position = performance_df.index.get_loc(best_rmse_row.name)
    plt.scatter(
        best_rmse_position,
        best_rmse_row["RMSE"],
        s=120,
        color="#1f5a34",
        zorder=3,
        label="Lowest RMSE",
    )
    plt.title("Feature Selection Performance - RMSE")
    plt.xlabel("Feature Set")
    plt.ylabel("RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.annotate(
        "Lowest error: R&D Spend Only",
        xy=(best_rmse_position, best_rmse_row["RMSE"]),
        xytext=(best_rmse_position + 0.3, best_rmse_row["RMSE"] + 9000),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
    )
    plt.gcf().text(
        0.53,
        0.58,
        rmse_comment,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f6f6", "edgecolor": "#999999"},
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_selection_rmse.png"), dpi=150)
    plt.close()

    return performance_df[performance_columns], csv_path


def make_encoded_feature_frames(X_train, X_test):
    """One-hot encode State and return train/test DataFrames for feature selection."""
    preprocessor = build_preprocessor(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES, CATEGORICAL_FEATURES
    )
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    return (
        pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index),
        pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index),
    )


def run_sequential_feature_selection(X_train, X_test, y_train, y_test):
    """Evaluate forward-selected feature subsets with 1 to all encoded predictors."""
    X_train_encoded, X_test_encoded = make_encoded_feature_frames(X_train, X_test)
    max_features = X_train_encoded.shape[1]
    selection_results = []

    for feature_count in range(1, max_features + 1):
        if feature_count == max_features:
            selected_features = X_train_encoded.columns.tolist()
        else:
            selector = SequentialFeatureSelector(
                LinearRegression(),
                n_features_to_select=feature_count,
                direction="forward",
                scoring="neg_root_mean_squared_error",
                cv=5,
            )
            selector.fit(X_train_encoded, y_train)
            selected_features = X_train_encoded.columns[selector.get_support()].tolist()

        model = LinearRegression()
        model.fit(X_train_encoded[selected_features], y_train)
        y_pred = model.predict(X_test_encoded[selected_features])

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        selection_results.append(
            {
                "Number of Features": feature_count,
                "Selected Features": selected_features,
                "RMSE": rmse,
                "R-squared": r2,
                "MAE": mae,
            }
        )

    return pd.DataFrame(selection_results)


def save_sequential_feature_selection_outputs(selection_df, output_dir):
    """Save side-by-side performance plots and a list of selected features."""
    os.makedirs(output_dir, exist_ok=True)

    table_df = selection_df.copy()
    table_df["Selected Features"] = table_df["Selected Features"].apply(
        lambda features: "[" + ", ".join(features) + "]"
    )
    csv_path = os.path.join(METRICS_DIR, "sequential_feature_selection_list.csv")
    table_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        selection_df["Number of Features"],
        selection_df["RMSE"],
        marker="o",
        linewidth=2,
        color="#2f6f8f",
    )
    axes[0].set_title("RMSE by Number of Features")
    axes[0].set_xlabel("Number of Features")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(
        selection_df["Number of Features"],
        selection_df["R-squared"],
        marker="o",
        linewidth=2,
        color="#2f6f8f",
    )
    axes[1].set_title("R-squared by Number of Features")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("R-squared")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "sequential_feature_selection_plots.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.45])
    ax_rmse = fig.add_subplot(grid[0, 0])
    ax_r2 = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    ax_rmse.plot(
        selection_df["Number of Features"],
        selection_df["RMSE"],
        marker="o",
        linewidth=2,
        color="#2f6f8f",
    )
    ax_rmse.set_title("RMSE by Number of Features")
    ax_rmse.set_xlabel("Number of Features")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.grid(True, linestyle="--", alpha=0.35)

    ax_r2.plot(
        selection_df["Number of Features"],
        selection_df["R-squared"],
        marker="o",
        linewidth=2,
        color="#2f6f8f",
    )
    ax_r2.set_title("R-squared by Number of Features")
    ax_r2.set_xlabel("Number of Features")
    ax_r2.set_ylabel("R-squared")
    ax_r2.grid(True, linestyle="--", alpha=0.35)

    ax_table.axis("off")
    table_display = table_df[
        ["Number of Features", "Selected Features", "RMSE", "R-squared"]
    ].copy()
    table_display["RMSE"] = table_display["RMSE"].map("{:,.2f}".format)
    table_display["R-squared"] = table_display["R-squared"].map("{:.4f}".format)
    table = ax_table.table(
        cellText=table_display.values,
        colLabels=table_display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if col == 1:
            cell.set_width(0.62)

    combined_path = os.path.join(
        output_dir, "sequential_feature_selection_summary.png"
    )
    fig.tight_layout()
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    return table_df, csv_path, plot_path, combined_path


def evaluate_encoded_feature_set(
    feature_set_name,
    selected_features,
    X_train_encoded,
    X_test_encoded,
    y_train,
    y_test,
):
    """Train and evaluate one encoded feature set."""
    model = LinearRegression()
    model.fit(X_train_encoded[selected_features], y_train)
    y_pred = model.predict(X_test_encoded[selected_features])

    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2_score(r2, len(y_test), len(selected_features))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "Feature Set Name": feature_set_name,
        "Selected Features": selected_features,
        "Number of Features": len(selected_features),
        "R-squared": r2,
        "Adjusted R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
    }


def run_marketing_vs_administration_comparison(X_train, X_test, y_train, y_test):
    """Compare Marketing Spend and Administration directly after R&D Spend."""
    X_train_encoded, X_test_encoded = make_encoded_feature_frames(X_train, X_test)
    comparison_sets = [
        ("R&D only", ["R&D Spend"]),
        ("Marketing only", ["Marketing Spend"]),
        ("Administration only", ["Administration"]),
        ("R&D + Marketing", ["R&D Spend", "Marketing Spend"]),
        ("R&D + Administration", ["R&D Spend", "Administration"]),
        ("Marketing + Administration", ["Marketing Spend", "Administration"]),
        ("R&D + Marketing + Administration", ["R&D Spend", "Marketing Spend", "Administration"]),
        (
            "All encoded features",
            [
                "R&D Spend",
                "Administration",
                "Marketing Spend",
                "State_Florida",
                "State_New York",
            ],
        ),
    ]

    rows = [
        evaluate_encoded_feature_set(
            feature_set_name,
            selected_features,
            X_train_encoded,
            X_test_encoded,
            y_train,
            y_test,
        )
        for feature_set_name, selected_features in comparison_sets
    ]
    return pd.DataFrame(rows)


def save_marketing_vs_administration_outputs(comparison_df, output_dir):
    """Save Marketing vs Administration comparison as CSV and summary image."""
    os.makedirs(output_dir, exist_ok=True)

    table_df = comparison_df.sort_values(
        ["Adjusted R2", "RMSE"], ascending=[False, True]
    ).copy()
    table_df["Selected Features"] = table_df["Selected Features"].apply(
        lambda features: "[" + ", ".join(features) + "]"
    )
    csv_path = os.path.join(METRICS_DIR, "marketing_vs_administration_comparison.csv")
    table_df.to_csv(csv_path, index=False)

    plot_df = comparison_df.copy()
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.7])
    ax_rmse = fig.add_subplot(grid[0, 0])
    ax_r2 = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    x_positions = np.arange(len(plot_df))

    ax_rmse.bar(x_positions, plot_df["RMSE"], color="#b45f3c")
    ax_rmse.set_title("RMSE by Feature Set")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_xticks(x_positions)
    ax_rmse.set_xticklabels(plot_df["Feature Set Name"], rotation=30, ha="right")
    ax_rmse.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax_r2.bar(x_positions, plot_df["Adjusted R2"], color="#2f6f8f")
    ax_r2.set_title("Adjusted R2 by Feature Set")
    ax_r2.set_ylabel("Adjusted R2")
    ax_r2.set_xticks(x_positions)
    ax_r2.set_xticklabels(plot_df["Feature Set Name"], rotation=30, ha="right")
    ax_r2.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax_table.axis("off")
    display_df = table_df[
        [
            "Feature Set Name",
            "Selected Features",
            "Number of Features",
            "RMSE",
            "R-squared",
            "Adjusted R2",
        ]
    ].copy()
    display_df["RMSE"] = display_df["RMSE"].map("{:,.2f}".format)
    display_df["R-squared"] = display_df["R-squared"].map("{:.4f}".format)
    display_df["Adjusted R2"] = display_df["Adjusted R2"].map("{:.4f}".format)

    table = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if col == 1:
            cell.set_width(0.35)

    summary_path = os.path.join(
        output_dir, "marketing_vs_administration_comparison.png"
    )
    fig.tight_layout()
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)

    return table_df, csv_path, summary_path


def run_business_guided_feature_selection(X_train, X_test, y_train, y_test):
    """Evaluate feature selection in expert-guided business priority order."""
    X_train_encoded, X_test_encoded = make_encoded_feature_frames(X_train, X_test)
    feature_sets = [
        ("1 feature: R&D", ["R&D Spend"]),
        ("2 features: R&D + Marketing", ["R&D Spend", "Marketing Spend"]),
        (
            "3 features: R&D + Marketing + Administration",
            ["R&D Spend", "Marketing Spend", "Administration"],
        ),
        (
            "4 features: Add State_Florida",
            ["R&D Spend", "Marketing Spend", "Administration", "State_Florida"],
        ),
        (
            "5 features: Add all State dummies",
            [
                "R&D Spend",
                "Marketing Spend",
                "Administration",
                "State_Florida",
                "State_New York",
            ],
        ),
    ]

    rows = [
        evaluate_encoded_feature_set(
            feature_set_name,
            selected_features,
            X_train_encoded,
            X_test_encoded,
            y_train,
            y_test,
        )
        for feature_set_name, selected_features in feature_sets
    ]
    return pd.DataFrame(rows)


def save_business_guided_feature_selection_outputs(selection_df, output_dir):
    """Save a feature-selection summary with Marketing before Administration."""
    os.makedirs(output_dir, exist_ok=True)

    table_df = selection_df.copy()
    table_df["Selected Features"] = table_df["Selected Features"].apply(
        lambda features: "[" + ", ".join(features) + "]"
    )
    csv_path = os.path.join(
        METRICS_DIR, "business_guided_feature_selection_summary.csv"
    )
    table_df.to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.55])
    ax_rmse = fig.add_subplot(grid[0, 0])
    ax_r2 = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    x_values = selection_df["Number of Features"]
    ax_rmse.plot(
        x_values,
        selection_df["RMSE"],
        marker="o",
        linewidth=2,
        color="#2f6f8f",
    )
    ax_rmse.set_title("RMSE by Number of Features")
    ax_rmse.set_xlabel("Number of Features")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.grid(True, linestyle="--", alpha=0.35)

    ax_r2.plot(
        x_values,
        selection_df["R-squared"],
        marker="o",
        linewidth=2,
        color="#2f6f8f",
    )
    ax_r2.set_title("R-squared by Number of Features")
    ax_r2.set_xlabel("Number of Features")
    ax_r2.set_ylabel("R-squared")
    ax_r2.grid(True, linestyle="--", alpha=0.35)

    ax_table.axis("off")
    display_df = table_df[
        ["Number of Features", "Selected Features", "RMSE", "R-squared"]
    ].copy()
    display_df["RMSE"] = display_df["RMSE"].map("{:,.2f}".format)
    display_df["R-squared"] = display_df["R-squared"].map("{:.4f}".format)

    table = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if col == 1:
            cell.set_width(0.62)

    summary_path = os.path.join(
        output_dir, "business_guided_feature_selection_summary.png"
    )
    fig.tight_layout()
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)

    return table_df, csv_path, summary_path


def rank_features_by_algorithm(algorithm_name, X_train_encoded, y_train):
    """Return encoded feature names ranked by one feature selection algorithm."""
    feature_names = X_train_encoded.columns.tolist()
    max_features = len(feature_names)

    if algorithm_name == "Sequential Feature Selection":
        ranked_features = []
        for feature_count in range(1, max_features):
            selector = SequentialFeatureSelector(
                LinearRegression(),
                n_features_to_select=feature_count,
                direction="forward",
                scoring="neg_root_mean_squared_error",
                cv=5,
            )
            selector.fit(X_train_encoded, y_train)
            selected = X_train_encoded.columns[selector.get_support()].tolist()
            for feature in selected:
                if feature not in ranked_features:
                    ranked_features.append(feature)
        for feature in feature_names:
            if feature not in ranked_features:
                ranked_features.append(feature)
        return ranked_features

    if algorithm_name == "RFE":
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        selector = RFE(
            estimator,
            n_features_to_select=1,
            importance_getter="named_steps.regressor.coef_",
        )
        selector.fit(X_train_encoded, y_train)
        ranking_df = pd.DataFrame(
            {"Feature": feature_names, "Rank": selector.ranking_}
        )
        return ranking_df.sort_values("Rank")["Feature"].tolist()

    if algorithm_name == "Lasso":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_encoded)
        model = Lasso(alpha=0.01, max_iter=20000, random_state=42)
        model.fit(X_scaled, y_train)
        ranking_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": np.abs(model.coef_),
            }
        )
        return ranking_df.sort_values("Importance", ascending=False)["Feature"].tolist()

    if algorithm_name == "SelectKBest":
        selector = SelectKBest(score_func=f_regression, k="all")
        selector.fit(X_train_encoded, y_train)
        ranking_df = pd.DataFrame(
            {"Feature": feature_names, "Score": selector.scores_}
        )
        return ranking_df.sort_values("Score", ascending=False)["Feature"].tolist()

    if algorithm_name == "Random Forest Importance":
        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(X_train_encoded, y_train)
        ranking_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        )
        return ranking_df.sort_values("Importance", ascending=False)["Feature"].tolist()

    raise ValueError(f"Unknown feature selection algorithm: {algorithm_name}")


def run_all_feature_selection_algorithms(X_train, X_test, y_train, y_test):
    """Run five feature selection algorithms and evaluate top-k feature subsets."""
    X_train_encoded, X_test_encoded = make_encoded_feature_frames(X_train, X_test)
    algorithms = [
        "Sequential Feature Selection",
        "RFE",
        "Lasso",
        "SelectKBest",
        "Random Forest Importance",
    ]
    rows = []

    for algorithm_name in algorithms:
        ranked_features = rank_features_by_algorithm(
            algorithm_name, X_train_encoded, y_train
        )
        for feature_count in range(1, len(ranked_features) + 1):
            selected_features = ranked_features[:feature_count]
            result = evaluate_encoded_feature_set(
                algorithm_name,
                selected_features,
                X_train_encoded,
                X_test_encoded,
                y_train,
                y_test,
            )
            result["Algorithm"] = algorithm_name
            result["Number of Features"] = feature_count
            rows.append(result)

    return pd.DataFrame(rows)


def save_all_feature_selection_algorithms_outputs(results_df, output_dir):
    """Save all five feature selection algorithms in one CSV and one PNG."""
    os.makedirs(output_dir, exist_ok=True)

    table_df = results_df.copy()
    table_df["Selected Features"] = table_df["Selected Features"].apply(
        lambda features: "[" + ", ".join(features) + "]"
    )
    csv_path = os.path.join(
        METRICS_DIR, "feature_selection_performance_allinone.csv"
    )
    table_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    for algorithm_name, group_df in results_df.groupby("Algorithm"):
        group_df = group_df.sort_values("Number of Features")
        axes[0].plot(
            group_df["Number of Features"],
            group_df["RMSE"],
            marker="o",
            linewidth=2,
            label=algorithm_name,
        )
        axes[1].plot(
            group_df["Number of Features"],
            group_df["R-squared"],
            marker="o",
            linewidth=2,
            label=algorithm_name,
        )

    axes[0].set_title("Feature Selection Performance: RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].set_title("Feature Selection Performance: R-squared")
    axes[1].set_xlabel("Number of Selected Features")
    axes[1].set_ylabel("R-squared")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    png_path = os.path.join(
        output_dir, "feature_selection_performance_allinone.png"
    )
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    best_df = results_df.sort_values(["RMSE", "R-squared"], ascending=[True, False])
    best_csv_path = os.path.join(
        METRICS_DIR, "feature_selection_performance_allinone_best.csv"
    )
    best_table = best_df.groupby("Algorithm", as_index=False).first()
    best_table["Selected Features"] = best_table["Selected Features"].apply(
        lambda features: "[" + ", ".join(features) + "]"
    )
    best_table.to_csv(best_csv_path, index=False)

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.55])
    ax_rmse = fig.add_subplot(grid[0, 0])
    ax_r2 = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    for algorithm_name, group_df in results_df.groupby("Algorithm"):
        group_df = group_df.sort_values("Number of Features")
        ax_rmse.plot(
            group_df["Number of Features"],
            group_df["RMSE"],
            marker="o",
            linewidth=2,
            label=algorithm_name,
        )
        ax_r2.plot(
            group_df["Number of Features"],
            group_df["R-squared"],
            marker="o",
            linewidth=2,
            label=algorithm_name,
        )

    ax_rmse.set_title("RMSE by Number of Features")
    ax_rmse.set_xlabel("Number of Features")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.grid(True, linestyle="--", alpha=0.35)
    ax_rmse.legend(loc="best", fontsize=8)

    ax_r2.set_title("R-squared by Number of Features")
    ax_r2.set_xlabel("Number of Features")
    ax_r2.set_ylabel("R-squared")
    ax_r2.grid(True, linestyle="--", alpha=0.35)
    ax_r2.legend(loc="best", fontsize=8)

    ax_table.axis("off")
    best_by_count = (
        table_df.sort_values(["Number of Features", "RMSE", "R-squared"], ascending=[True, True, False])
        .groupby("Number of Features", as_index=False)
        .first()
    )
    display_all = best_by_count[
        ["Number of Features", "Selected Features", "RMSE", "R-squared"]
    ].copy()
    display_all["RMSE"] = display_all["RMSE"].map("{:,.2f}".format)
    display_all["R-squared"] = display_all["R-squared"].map("{:.4f}".format)
    table = ax_table.table(
        cellText=display_all.values,
        colLabels=display_all.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if col == 1:
            cell.set_width(0.62)

    summary_png_path = os.path.join(
        output_dir, "feature_selection_performance_allinone_summary.png"
    )
    fig.tight_layout()
    fig.savefig(summary_png_path, dpi=150)
    plt.close(fig)

    return table_df, csv_path, png_path, best_table, best_csv_path, summary_png_path


# -----------------------------------------------------------------------------
# Step 1: Business Understanding
# -----------------------------------------------------------------------------
print_section("Step 1: Business Understanding")
print(
    "Goal: Predict startup Profit using business spending features and State.\n"
    "Business questions:\n"
    " - Which feature has the strongest influence on startup profit?\n"
    " - Is R&D Spend the most important predictor of Profit?\n"
    " - Does Marketing Spend improve prediction performance?\n"
    " - Does Administration help the model, or does it add noise?\n"
    " - Does State provide useful regional information?\n\n"
    "Expert logic: R&D Spend is the core profit driver, Marketing Spend is a "
    "possible value amplifier, Administration is uncertain because it represents "
    "operating cost, and State is a supporting regional feature that should not "
    "be over-interpreted because the dataset is small."
)


# -----------------------------------------------------------------------------
# Step 2: Data Understanding
# -----------------------------------------------------------------------------
print_section("Step 2: Data Understanding")
df = load_dataset()

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate row count:")
print(df.duplicated().sum())

print("\nState value counts:")
print(df["State"].value_counts())

print("\nCorrelation matrix for numerical features and Profit:")
print(df[NUMERICAL_FEATURES + [TARGET_COLUMN]].corr())

print(
    "\nFeature understanding:\n"
    " - R&D Spend: expected strongest predictor because it reflects innovation "
    "and product development.\n"
    " - Marketing Spend: may improve customer reach and market exposure.\n"
    " - Administration: may be weak or noisy because it is an operating cost.\n"
    " - State: may capture regional differences, but only as supporting evidence.\n"
    " - Profit: continuous target variable, so this is a regression problem."
)

save_data_understanding_plots(df, FIGURES_DIR)
print(f"\nSaved data understanding visualizations to: {FIGURES_DIR}")


# -----------------------------------------------------------------------------
# Step 3: Data Preparation
# -----------------------------------------------------------------------------
print_section("Step 3: Data Preparation")
X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training rows: {X_train.shape[0]}")
print(f"Test rows: {X_test.shape[0]}")
print("Numerical features:", NUMERICAL_FEATURES)
print("Categorical features:", CATEGORICAL_FEATURES)
print("Preprocessing: OneHotEncoder(drop='first') for State inside ColumnTransformer.")


# -----------------------------------------------------------------------------
# Step 4: Modeling
# -----------------------------------------------------------------------------
print_section("Step 4: Modeling - Feature Experiments")
experiments = [
    {
        "model_name": "Model A: R&D Spend Only",
        "features": ["R&D Spend"],
        "categorical_features": [],
        "purpose": "Test whether R&D alone explains most of Profit.",
    },
    {
        "model_name": "Model B: R&D Spend + Marketing Spend",
        "features": ["R&D Spend", "Marketing Spend"],
        "categorical_features": [],
        "purpose": "Test whether Marketing Spend adds value beyond R&D Spend.",
    },
    {
        "model_name": "Model C: R&D Spend + Marketing Spend + Administration",
        "features": ["R&D Spend", "Marketing Spend", "Administration"],
        "categorical_features": [],
        "purpose": "Test whether Administration improves prediction or adds noise.",
    },
    {
        "model_name": "Model D: All Features Including State",
        "features": ["R&D Spend", "Marketing Spend", "Administration", "State"],
        "categorical_features": ["State"],
        "purpose": "Test whether State improves performance after one-hot encoding.",
    },
    {
        "model_name": "Model E: State Only",
        "features": ["State"],
        "categorical_features": ["State"],
        "purpose": "Test whether location alone can predict Profit.",
    },
]

trained_models = {}
experiment_results = []

for experiment_order, experiment in enumerate(experiments, start=1):
    model, result = run_experiment(
        model_name=experiment["model_name"],
        purpose=experiment["purpose"],
        features=experiment["features"],
        categorical_features=experiment["categorical_features"],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    result["Experiment Order"] = experiment_order
    trained_models[experiment["model_name"]] = {
        "model": model,
        "features": experiment["features"],
    }
    experiment_results.append(result)
    print(f"Trained {experiment['model_name']}")


# -----------------------------------------------------------------------------
# Step 5: Evaluation
# -----------------------------------------------------------------------------
print_section("Step 5: Evaluation")
results_df = pd.DataFrame(experiment_results)
results_df = results_df.sort_values(
    by=["Adjusted R2", "RMSE"], ascending=[False, True]
).reset_index(drop=True)

metric_columns = [
    "Model",
    "Predictor Count",
    "R2",
    "Adjusted R2",
    "MAE",
    "MSE",
    "RMSE",
    "Features",
]
print("\nModel comparison table:")
print(
    results_df[metric_columns].to_string(
        index=False,
        formatters={
            "R2": "{:.4f}".format,
            "Adjusted R2": "{:.4f}".format,
            "MAE": "{:,.2f}".format,
            "MSE": "{:,.2f}".format,
            "RMSE": "{:,.2f}".format,
        },
    )
)

best_model_name = results_df.iloc[0]["Model"]
best_bundle = trained_models[best_model_name]
best_model = best_bundle["model"]
best_features = best_bundle["features"]
best_coefficients = get_model_coefficients(best_model, best_features)

print(f"\nBest model selected by Adjusted R2 and RMSE: {best_model_name}")
print("\nBest model coefficients:")
print(
    best_coefficients.to_string(
        index=False,
        formatters={
            "Coefficient": "{:,.4f}".format,
            "Absolute Coefficient": "{:,.4f}".format,
        },
    )
)

print("\nCoefficient interpretation note:")
print(
    "Positive coefficients mean the model predicts higher Profit as that feature "
    "increases, holding other included features constant. State coefficients are "
    "relative to the dropped reference State from OneHotEncoder(drop='first')."
)

feature_selection_df, feature_selection_csv_path = save_feature_selection_performance(
    results_df, FIGURES_DIR
)
print("\nFeature selection performance table:")
print(
    feature_selection_df.to_string(
        index=False,
        formatters={
            "R2": "{:.4f}".format,
            "Adjusted R2": "{:.4f}".format,
            "MAE": "{:,.2f}".format,
            "RMSE": "{:,.2f}".format,
        },
    )
)
print(f"\nSaved feature selection performance CSV to: {feature_selection_csv_path}")

sequential_selection_df = run_sequential_feature_selection(
    X_train, X_test, y_train, y_test
)
(
    sequential_selection_table,
    sequential_selection_csv_path,
    sequential_selection_plot_path,
    sequential_selection_summary_path,
) = save_sequential_feature_selection_outputs(
    sequential_selection_df, FIGURES_DIR
)
print("\nSequential feature selection list:")
print(
    sequential_selection_table.to_string(
        index=False,
        formatters={
            "RMSE": "{:,.2f}".format,
            "R-squared": "{:.4f}".format,
            "MAE": "{:,.2f}".format,
        },
    )
)
print(f"\nSaved sequential feature selection CSV to: {sequential_selection_csv_path}")
print(f"Saved sequential feature selection plots to: {sequential_selection_plot_path}")
print(f"Saved sequential feature selection summary to: {sequential_selection_summary_path}")

marketing_admin_df = run_marketing_vs_administration_comparison(
    X_train, X_test, y_train, y_test
)
(
    marketing_admin_table,
    marketing_admin_csv_path,
    marketing_admin_summary_path,
) = save_marketing_vs_administration_outputs(marketing_admin_df, FIGURES_DIR)
print("\nMarketing Spend vs Administration direct comparison:")
print(
    marketing_admin_table.to_string(
        index=False,
        formatters={
            "RMSE": "{:,.2f}".format,
            "R-squared": "{:.4f}".format,
            "Adjusted R2": "{:.4f}".format,
            "MAE": "{:,.2f}".format,
        },
    )
)
print(f"\nSaved Marketing vs Administration CSV to: {marketing_admin_csv_path}")
print(f"Saved Marketing vs Administration summary to: {marketing_admin_summary_path}")

business_guided_df = run_business_guided_feature_selection(
    X_train, X_test, y_train, y_test
)
(
    business_guided_table,
    business_guided_csv_path,
    business_guided_summary_path,
) = save_business_guided_feature_selection_outputs(
    business_guided_df, FIGURES_DIR
)
print("\nBusiness-guided feature selection summary:")
print(
    business_guided_table.to_string(
        index=False,
        formatters={
            "RMSE": "{:,.2f}".format,
            "R-squared": "{:.4f}".format,
            "Adjusted R2": "{:.4f}".format,
            "MAE": "{:,.2f}".format,
        },
    )
)
print(f"\nSaved business-guided feature selection CSV to: {business_guided_csv_path}")
print(f"Saved business-guided feature selection summary to: {business_guided_summary_path}")

all_selection_results = run_all_feature_selection_algorithms(
    X_train, X_test, y_train, y_test
)
(
    all_selection_table,
    all_selection_csv_path,
    all_selection_png_path,
    all_selection_best_table,
    all_selection_best_csv_path,
    all_selection_summary_png_path,
) = save_all_feature_selection_algorithms_outputs(
    all_selection_results, FIGURES_DIR
)
print("\nBest result from each feature selection algorithm:")
print(
    all_selection_best_table[
        [
            "Algorithm",
            "Number of Features",
            "Selected Features",
            "RMSE",
            "R-squared",
            "Adjusted R2",
        ]
    ].to_string(
        index=False,
        formatters={
            "RMSE": "{:,.2f}".format,
            "R-squared": "{:.4f}".format,
            "Adjusted R2": "{:.4f}".format,
        },
    )
)
print(f"\nSaved all-in-one feature selection CSV to: {all_selection_csv_path}")
print(f"Saved all-in-one feature selection PNG to: {all_selection_png_path}")
print(f"Saved all-in-one best-results CSV to: {all_selection_best_csv_path}")
print(f"Saved all-in-one summary PNG to: {all_selection_summary_png_path}")

save_model_evaluation_plots(
    results_df=results_df,
    best_model=best_model,
    best_features=best_features,
    best_coefficients=best_coefficients,
    X_test=X_test,
    y_test=y_test,
    output_dir=FIGURES_DIR,
)
print(f"\nSaved model evaluation visualizations to: {FIGURES_DIR}")


# -----------------------------------------------------------------------------
# Step 6: Expert-Based Business Conclusion
# -----------------------------------------------------------------------------
print_section("Step 6: Expert-Based Business Conclusion")

state_only_row = results_df[results_df["Model"] == "Model E: State Only"].iloc[0]
all_features_row = results_df[
    results_df["Model"] == "Model D: All Features Including State"
].iloc[0]
no_state_row = results_df[
    results_df["Model"] == "Model C: R&D Spend + Marketing Spend + Administration"
].iloc[0]

print("Feature ranking based on coefficients and business logic:")
print("1. R&D Spend: Very high importance. Strongly keep as the core predictor.")
print("2. Marketing Spend: Medium to high importance. Keep and test as an amplifier.")
print("3. Administration: Uncertain. Keep only if metrics and coefficients support it.")
print("4. State: Supporting only. Encode and test, but do not over-interpret.")

print("\nState evaluation:")
print(
    f" - All-features Adjusted R2: {all_features_row['Adjusted R2']:.4f}, "
    f"RMSE: {all_features_row['RMSE']:,.2f}"
)
print(
    f" - No-State Adjusted R2: {no_state_row['Adjusted R2']:.4f}, "
    f"RMSE: {no_state_row['RMSE']:,.2f}"
)
print(
    f" - State-only Adjusted R2: {state_only_row['Adjusted R2']:.4f}, "
    f"RMSE: {state_only_row['RMSE']:,.2f}"
)

print("\nFinal conclusion:")
print(
    "The expert discussion is consistent with the regression experiment design: "
    "Profit is mainly driven by resource allocation, especially R&D Spend. "
    "Marketing Spend may add predictive value as a market exposure and sales "
    "support feature. Administration should be treated carefully because it may "
    "represent operating cost rather than direct profit generation. State can be "
    "useful as a supporting regional signal, but with only 50 rows it should not "
    "be over-interpreted. The final model should therefore be chosen using "
    "Adjusted R2, RMSE, MAE, coefficient reasonableness, and business logic."
)
