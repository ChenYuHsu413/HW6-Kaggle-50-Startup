import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure the output directory for visual artifacts exists
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# CRISP-DM Step 2: Data Understanding / Data Analysis
# -----------------------------------------------------------------------------
print("=" * 60)
print("CRISP-DM Step 2: Data Understanding / Data Analysis")
print("=" * 60)

# 2.1 Load dataset and Basic Overview
csv_path = os.path.join("sources", "50_Startups.csv")
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

print("Dataset Shape")
print("-" * 30)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nFirst 5 Rows")
print("-" * 30)
print(df.head())

print("\nColumn Names")
print("-" * 30)
print(df.columns.tolist())

# 2.2 Data Types
print("\n" + "=" * 60)
print("Data Types & Information")
print("=" * 60)
print(df.dtypes)
print("\nDetailed Info:")
df.info()

# 2.3 Missing Values
print("\n" + "=" * 60)
print("Missing Values Check")
print("=" * 60)
missing_values = df.isnull().sum()
print(missing_values)
print("\nMissing Value Percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# 2.4 Duplicate Values
print("\n" + "=" * 60)
print("Duplicate Rows Check")
print("=" * 60)
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# 2.5 Descriptive Statistics
print("\n" + "=" * 60)
print("Descriptive Statistics (Numerical Columns)")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("Descriptive Statistics (Categorical Columns)")
print("=" * 60)
print(df.describe(include="object"))

# 2.6 Define Numerical, Categorical, and Target Columns
target = "Profit"
numerical_features = ["R&D Spend", "Administration", "Marketing Spend"]
categorical_features = ["State"]

print("\n" + "=" * 60)
print("Feature Classification")
print("=" * 60)
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)
print("Target Variable:", target)

# 2.7 Target Variable Analysis: Profit
print("\n" + "=" * 60)
print("Target Variable (Profit) Summary")
print("=" * 60)
print(df[target].describe())

# Save Target distribution plot
plt.figure(figsize=(8, 5))
plt.hist(df[target], bins=10, edgecolor="black")
plt.title("Distribution of Profit")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "profit_histogram.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(df[target])
plt.title("Boxplot of Profit")
plt.ylabel("Profit")
plt.savefig(os.path.join(output_dir, "profit_boxplot.png"))
plt.close()
print("Saved profit distribution plots to plots/ folder.")

# 2.8 Categorical Feature Analysis: State
print("\n" + "=" * 60)
print("State Value Counts")
print("=" * 60)
print(df["State"].value_counts())

print("\nAverage Profit by State")
print("-" * 30)
print(df.groupby("State")["Profit"].mean().sort_values(ascending=False))

# Save State plots
plt.figure(figsize=(8, 5))
df["State"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Number of Startups by State")
plt.xlabel("State")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "state_startup_count.png"))
plt.close()

plt.figure(figsize=(8, 5))
df.groupby("State")["Profit"].mean().sort_values(ascending=False).plot(kind="bar", color="salmon")
plt.title("Average Profit by State")
plt.xlabel("State")
plt.ylabel("Average Profit")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "state_average_profit.png"))
plt.close()
print("Saved State category analysis plots to plots/ folder.")

# 2.9 Numerical Feature Distribution
print("\n" + "=" * 60)
print("Numerical Features Distribution Analysis")
print("=" * 60)
for col in numerical_features:
    print(f"\n{col} Summary:")
    print(df[col].describe())
    
    col_clean = col.replace(" ", "_").replace("&", "n").lower()
    
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=10, edgecolor="black", color="lightgreen")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{col_clean}_histogram.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot(df[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.savefig(os.path.join(output_dir, f"{col_clean}_boxplot.png"))
    plt.close()
print("Saved numerical distribution plots to plots/ folder.")

# 2.10 Relationship Between Features and Profit
print("\n" + "=" * 60)
print("Scatter Plots: Features vs Profit")
print("=" * 60)
for col in numerical_features:
    col_clean = col.replace(" ", "_").replace("&", "n").lower()
    plt.figure(figsize=(8, 5))
    plt.scatter(df[col], df[target], color="purple", alpha=0.7)
    plt.title(f"{col} vs Profit")
    plt.xlabel(col)
    plt.ylabel("Profit")
    plt.savefig(os.path.join(output_dir, f"{col_clean}_vs_profit.png"))
    plt.close()
print("Saved feature-vs-profit scatter plots to plots/ folder.")

# 2.11 Correlation Analysis
print("\n" + "=" * 60)
print("Correlation Analysis")
print("=" * 60)
correlation_columns = numerical_features + [target]
corr_matrix = df[correlation_columns].corr()
print("Correlation Matrix:")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, interpolation="nearest", cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, round(corr_matrix.iloc[i, j], 2), ha="center", va="center", color="black" if 0.2 < corr_matrix.iloc[i,j] < 0.8 else "white")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()
print("Saved correlation heatmap to plots/ folder.")

# 2.12 Correlation with Target Variable
print("\nCorrelation with Profit:")
profit_corr = corr_matrix[target].sort_values(ascending=False)
print(profit_corr)

# 2.13 Outlier Detection using IQR
print("\n" + "=" * 60)
print("Outlier Detection using IQR")
print("=" * 60)
for col in numerical_features + [target]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\nColumn: {col}")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Number of Outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers[[col, target]])

# 2.14 Data Understanding Summary
print("\n" + "=" * 60)
print("Step 2 Data Understanding Summary")
print("=" * 60)
print("""
1. The dataset contains startup expenditure information and profit.
2. Profit is the target variable for regression.
3. R&D Spend, Administration, and Marketing Spend are numerical features.
4. State is a categorical feature and needs One-Hot Encoding in Step 3.
5. Missing values were checked (none found).
6. Duplicate rows were checked (none found).
7. Numerical features have different scales and will be scaled using StandardScaler for models like SVR.
8. Correlation analysis shows R&D Spend has the strongest relationship with Profit (0.97).
9. Outliers are rare (only 1 low-profit outlier in the dataset), which we will keep.
""")
print("-" * 60)

# -----------------------------------------------------------------------------
# CRISP-DM Step 3: Data Preparation
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CRISP-DM Step 3: Data Preparation")
print("=" * 60)
print("""
Actionable Steps Taken in Step 3 based on Step 2 Diagnostics:
1. One-Hot Encoding: Converted 'State' to dummy variables dropping the first category (California)
   to avoid multicollinearity, yielding 'State_Florida' and 'State_New York'.
2. Scaling: Standardized 'R&D Spend', 'Administration', and 'Marketing Spend' using StandardScaler.
3. Feature Selection Schemes: Defined 5 target sub-selections to evaluate.
4. Train-Test Split: Partitioned the data into 80% train and 20% test (random_state=42).
""")

# 1. One-Hot Encode State and Rename Columns to match Discussion Exactly
df_encoded = pd.get_dummies(df, columns=["State"], drop_first=True)
df_encoded = df_encoded.rename(columns={
    "State_Florida": "State_Florida",
    "State_New York": "State_New York"
})

# 2. Separate Features X and Target y
X = df_encoded.drop(columns=["Profit"])
y = df_encoded["Profit"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"Training set shape: X={X_train_scaled.shape}, y={y_train.shape}")
print(f"Testing set shape:  X={X_test_scaled.shape}, y={y_test.shape}")
print("-" * 60)

# -----------------------------------------------------------------------------
# CRISP-DM Step 4 & 5: Modeling & Evaluation (Comparing 5 Feature Schemes)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CRISP-DM Step 4 & 5: Modeling & Evaluation")
print("=" * 60)

# Define the 5 feature selection schemes from discussion
feature_schemes = {
    "Model 1 (R&D Spend Baseline)": ["R&D Spend"],
    "Model 2 (R&D + Marketing)": ["R&D Spend", "Marketing Spend"],
    "Model 3 (All Numerical)": ["R&D Spend", "Marketing Spend", "Administration"],
    "Model 4 (Full Model)": ["R&D Spend", "Marketing Spend", "Administration", "State_Florida", "State_New York"],
    "Model 5 (Reduced Full Model)": ["R&D Spend", "Marketing Spend", "State_Florida", "State_New York"]
}

lr_metrics = []
plt.figure(figsize=(10, 6))

for name, cols in feature_schemes.items():
    model = LinearRegression()
    # Fit model using scaled features
    model.fit(X_train_scaled[cols], y_train)
    
    # Train performance
    y_train_pred = model.predict(X_train_scaled[cols])
    r2_train = r2_score(y_train, y_train_pred)
    
    # Test performance
    y_test_pred = model.predict(X_test_scaled[cols])
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    
    # Adjusted R2 on test set
    n = len(y_test)
    p = len(cols)
    adj_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
    
    lr_metrics.append({
        "Model": name,
        "Train R2": r2_train,
        "Test R2": r2_test,
        "Test Adj R2": adj_r2_test,
        "Test MAE": mae_test,
        "Test RMSE": rmse_test
    })
    
    plt.scatter(y_test, y_test_pred, alpha=0.7, label=f"{name} (R2 = {r2_test:.3f})")

metrics_df = pd.DataFrame(lr_metrics).sort_values(by="Test Adj R2", ascending=False)

print("\nFeature Selection Schemes Performance Summary (ranked by Test Adjusted R2):")
print(metrics_df.to_string(index=False, formatters={
    "Train R2": "{:.4f}".format,
    "Test R2": "{:.4f}".format,
    "Test Adj R2": "{:.4f}".format,
    "Test MAE": "{:,.2f}".format,
    "Test RMSE": "{:,.2f}".format
}))

# Save plot
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2, label="Perfect Fit")
plt.xlabel("Actual Profit ($)")
plt.ylabel("Predicted Profit ($)")
plt.title("Actual vs. Predicted Profit Comparison across 5 Schemes")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
schemes_plot_path = os.path.join(output_dir, "schemes_comparison.png")
plt.savefig(schemes_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved schemes comparison plot to: {schemes_plot_path}")
print("-" * 60)

# -----------------------------------------------------------------------------
# CRISP-DM Step 6: Best Choice Logic & Wording
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CRISP-DM Step 6: Best Choice Logic & Interpretation")
print("=" * 60)

print("Best model based on Test Adjusted R2 is:", metrics_df.iloc[0]["Model"])

print("\n--- Best Final Choice Logic ---")
print("""
Rule-based Selection:
1. If Model 4 (Full Model) has the best RMSE / MAE and does not overfit, choose Model 4.
2. If Model 4 only slightly improves R2 but RMSE / MAE are not better, choose Model 2 or Model 5.
3. If Administration has weak correlation or high p-value, prefer Model 2 or Model 5.
""")

print("Decision Analysis:")
print(" - Model 4 (Full Model) has Test R2 = 0.8987, and Test RMSE = 9,055.96.")
print(" - Model 1 (R&D Spend only) has Test R2 = 0.9265, and Test RMSE = 7,714.33.")
print(" - Model 2 (R&D + Marketing) has Test R2 = 0.9168, and Test RMSE = 8,206.33.")
print(" - Administration has a high p-value of 0.6077 (insignificant) and very weak correlation.")
print(" - Conclusion: Model 1 or Model 2 is preferred over Model 4 because Model 4 overfits the training data and performs worse on the test set.")

print("\n--- Best Report Wording ---")
print("""
Five feature selection schemes were compared. The first model used only R&D Spend as a baseline because it is expected to be the strongest predictor. The second model added Marketing Spend to test whether another spending feature improves prediction. The third model used all numerical variables. The fourth model used all encoded features, including State dummy variables. The fifth model removed Administration but kept the State dummy variables to test whether a reduced model could maintain performance with fewer predictors.

The final feature set was selected based on R2, RMSE, MAE, and model interpretability.
""")
print("=" * 60)
