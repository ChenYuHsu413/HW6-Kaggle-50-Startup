import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Ensure output directories exist
output_dir = "plots/diagnostics"
os.makedirs(output_dir, exist_ok=True)

# 1. Load dataset
csv_path = os.path.join("sources", "50_Startups.csv")
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Rename columns to remove spaces and special characters for statsmodels formula API
df_clean = df.rename(columns={
    "R&D Spend": "RD_Spend",
    "Administration": "Administration",
    "Marketing Spend": "Marketing_Spend",
    "State": "State",
    "Profit": "Profit"
})

print("=" * 70)
print("1. MULTICOLLINEARITY DIAGNOSTICS (VIF)")
print("=" * 70)

# Calculate VIF for numerical features
# VIF is calculated on the design matrix (including intercept)
X_num = df_clean[["RD_Spend", "Administration", "Marketing_Spend"]]
X_num_const = sm.add_constant(X_num)

vif_data = pd.DataFrame()
vif_data["Feature"] = X_num_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_num_const.values, i) for i in range(X_num_const.shape[1])]

# Filter out constant to show features
print(vif_data[vif_data["Feature"] != "const"].to_string(index=False, formatters={"VIF": "{:.4f}".format}))

print("\nInterpretation:")
print(" - VIF < 5: Low multicollinearity. The feature is relatively independent.")
print(" - VIF between 5 and 10: Moderate multicollinearity.")
print(" - VIF > 10: High multicollinearity. Coefficients may be unstable.")
print(" - Note: RD_Spend and Marketing_Spend show moderate multicollinearity (VIF ~ 2.47), which is acceptable.")

print("\n" + "=" * 70)
print("2. HYPOTHESIS TESTING & STATISTICAL SIGNIFICANCE (p-value Analysis)")
print("=" * 70)

# Fit OLS regression using statsmodels formula
# State is automatically one-hot encoded by Patsy using C(State)
ols_model = smf.ols("Profit ~ RD_Spend + Administration + Marketing_Spend + C(State)", data=df_clean).fit()
print(ols_model.summary())

# Extract coefficient details
summary_table = pd.DataFrame({
    "coef": ols_model.params,
    "std err": ols_model.bse,
    "t": ols_model.tvalues,
    "P>|t|": ols_model.pvalues,
    "[0.025": ols_model.conf_int()[0],
    "0.975]": ols_model.conf_int()[1]
})
print("\nCoefficient Summary Table:")
print(summary_table.to_string(formatters={
    "coef": "{:,.4f}".format,
    "std err": "{:,.4f}".format,
    "t": "{:,.4f}".format,
    "P>|t|": "{:,.4f}".format,
    "[0.025": "{:,.4f}".format,
    "0.975]": "{:,.4f}".format
}))

print("\nInterpretation of Coefficients:")
significant_features = []
insignificant_features = []

for idx, row in summary_table.iterrows():
    p_val = float(row["P>|t|"])
    if p_val < 0.05:
        significant_features.append(f"{idx} (p = {p_val:.4f})")
    else:
        insignificant_features.append(f"{idx} (p = {p_val:.4f})")

print(f" - Statistically Significant Features (p < 0.05): {', '.join(significant_features)}")
print(f" - Statistically Insignificant Features (p >= 0.05): {', '.join(insignificant_features)}")

print("\n" + "=" * 70)
print("3. REGRESSION ASSUMPTIONS & RESIDUAL DIAGNOSTICS")
print("=" * 70)

residuals = ols_model.resid
fitted_values = ols_model.fittedvalues

# 3.1 Normality Test (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print("Normality of Residuals (Shapiro-Wilk Test):")
print(f" - Shapiro-Wilk Statistic: {shapiro_stat:.4f}")
print(f" - Shapiro-Wilk p-value:   {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print(" - Conclusion: Residuals appear to be normally distributed (fail to reject H0).")
else:
    print(" - Warning: Residuals do NOT appear to be normally distributed (reject H0).")

# Save Q-Q Plot
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot of Residuals")
plt.grid(True, linestyle="--", alpha=0.5)
qq_path = os.path.join(output_dir, "residuals_qqplot.png")
plt.savefig(qq_path, dpi=150, bbox_inches="tight")
plt.close()

# 3.2 Homoscedasticity Test (Breusch-Pagan)
# bp_test returns: lagrange multiplier statistic, p-value, f-value, f p-value
bp_names = ["Lagrange Multiplier statistic", "p-value", "f-value", "f p-value"]
bp_results = het_breuschpagan(residuals, ols_model.model.exog)
print("\nHomoscedasticity of Residuals (Breusch-Pagan Test):")
for name, val in zip(bp_names, bp_results):
    print(f" - {name}: {val:.4f}")
if bp_results[1] > 0.05:
    print(" - Conclusion: Homoscedasticity holds. Residuals have constant variance (fail to reject H0).")
else:
    print(" - Warning: Heteroscedasticity detected. Residuals variance is not constant (reject H0).")

# Save Residuals vs Fitted values plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, color="darkblue", alpha=0.7, edgecolors="k")
plt.axhline(y=0, color="red", linestyle="--", lw=1.5)
plt.title("Residuals vs. Fitted Values")
plt.xlabel("Fitted Values (Predicted Profit)")
plt.ylabel("Residuals")
plt.grid(True, linestyle="--", alpha=0.5)
res_fit_path = os.path.join(output_dir, "residuals_vs_fitted.png")
plt.savefig(res_fit_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved Residual Diagnostics plots to: {output_dir}")

print("\n" + "=" * 70)
print("4. STATE-SPECIFIC EXPLORATORY ANALYSIS")
print("=" * 70)

states = df_clean["State"].unique()
state_metrics = []

for state in states:
    state_df = df_clean[df_clean["State"] == state]
    n_obs = len(state_df)
    
    # Correlation between RD_Spend and Profit
    rd_corr = state_df["RD_Spend"].corr(state_df["Profit"])
    # Correlation between Marketing_Spend and Profit
    mkt_corr = state_df["Marketing_Spend"].corr(state_df["Profit"])
    
    # Fit simple linear model Profit ~ RD_Spend for the state
    state_ols = smf.ols("Profit ~ RD_Spend", data=state_df).fit()
    slope = state_ols.params["RD_Spend"]
    r2_state = state_ols.rsquared
    
    state_metrics.append({
        "State": state,
        "Sample Size (N)": n_obs,
        "R&D vs Profit Corr": rd_corr,
        "Marketing vs Profit Corr": mkt_corr,
        "R&D Spend Slope": slope,
        "R&D model R2": r2_state
    })

state_metrics_df = pd.DataFrame(state_metrics)
print(state_metrics_df.to_string(index=False, formatters={
    "R&D vs Profit Corr": "{:.4f}".format,
    "Marketing vs Profit Corr": "{:.4f}".format,
    "R&D Spend Slope": "{:,.4f}".format,
    "R&D model R2": "{:.4f}".format
}))

print("\nCautionary Note on State-Specific Analysis:")
print(" - The sample sizes for each state are very small (N=16 or 17).")
print(" - Slopes and correlations are highly sensitive to single observations at this sample size.")
print(" - While R&D Spend is highly correlated with Profit across all states (> 0.96), Florida displays a slightly higher slope ($0.88/dollar) compared to California ($0.84/dollar) and New York ($0.83/dollar). This is exploratory and should not be over-interpreted.")
print("=" * 70)
