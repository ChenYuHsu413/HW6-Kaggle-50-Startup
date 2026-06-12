# Project Log: 50 Startups Machine Learning & Advanced Data Analysis

This log records the steps, results, and design decisions taken during today's paired-programming session on the Kaggle 50 Startups regression problem. The process adheres to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

---

## Step 1: Initial Discovery & Data Understanding (CRISP-DM Phase 2)

- **Objective:** Study files under the `sources` directory to understand the structure of the data.
- **Actions:**
  - Listed the workspace and found [50_Startups.csv](file:///d:/AI%20Class%20ChenYu/AIClass/L6/sources/50_Startups.csv) (51 lines total, containing 50 data rows and 1 header row).
  - Inspected column headers: `R&D Spend`, `Administration`, `Marketing Spend`, `State`, and `Profit`.
  - Wrote and executed [describe_data.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/describe_data.py) to check basic stats:
    - **Shape:** 50 rows, 5 columns.
    - **Data Types:** 4 numeric float columns, 1 object categorical column (`State`).
    - **Missing values:** 0 missing values across all columns.
    - **Category counts in State:** New York (17), California (17), Florida (16).
- **Deliverable:** Created [data_understanding_report.md](file:///C:/Users/admin/.gemini/antigravity-ide/brain/6ac4088c-f10a-41ab-b3cf-32fdedfcfcbf/data_understanding_report.md) outlining these baseline details.

---

## Step 2: Environment Configuration & Package Installation

- **Objective:** Install the required scientific computing and modeling libraries.
- **Actions:**
  - Ran `pip list` and verified that `scikit-learn`, `matplotlib`, and `statsmodels` were missing.
  - Executed `pip install scikit-learn matplotlib` to enable modeling and visualization.
  - Executed `pip install statsmodels` to enable advanced linear regression diagnostics (VIF, p-values, t-stats, and residual tests).

---

## Step 3: Initial End-to-End Pipeline Implementation

- **Objective:** Create a script comparing different ML models on the full dataset.
- **Actions:**
  - Created [solve_startups.py](file:///d:/AI%20Class%20ChenYu/AIClass/L6/solve_startups.py) containing:
    - A `ColumnTransformer` applying `OneHotEncoder(drop='first')` to the `State` column, and `StandardScaler` to numeric spends.
    - Train-test split (80/20, `random_state=42`).
    - Model comparison pipelines for: *Linear Regression, Ridge, Lasso, SVR,* and *Random Forest*.
  - **Debugging Windows Encoding Error:** Encountered a `UnicodeEncodeError` when trying to print the superscript 2 symbol (`R²` / `\xb2`) to the console under Windows CP950 encoding. Fixed the script by replacing all instances of `R²` with standard `R2` or `R-squared`.
- **Results:** 
  - Random Forest achieved the highest test performance (R2 = 0.9147, MAE = \$6,131.91) and the best cross-validation stability (CV RMSE = \$9,050.49).

---

## Step 4: Advanced Statistical Regression Diagnostics

- **Objective:** Validate the regression assumptions (collinearity, normality, homoscedasticity) and check statistical significance.
- **Actions:**
  - Wrote and executed [advanced_analysis.py](file:///d:/AI%20Class%20ChenYu/AIClass/L6/advanced_analysis.py).
  - Refactored coefficient table generation to use direct `statsmodels` object properties instead of `pd.read_html` (which threw an error due to a missing optional `lxml` dependency).
- **Statistical Findings:**
  1. **Multicollinearity (VIF):** All variables show low multicollinearity (R&D Spend VIF = 2.47, Marketing Spend VIF = 2.33, Administration VIF = 1.18). This is well below the collinearity threshold of 5, indicating stable coefficient estimations.
  2. **Statistical Significance ($p$-values):**
     - **R&D Spend** is highly significant ($p = 0.0000$).
     - **Marketing Spend** is not significant at 95% confidence ($p = 0.1227$) due to overlapping variance with R&D Spend.
     - **Administration** ($p = 0.6077$) and **State** ($p > 0.95$) are highly insignificant.
  3. **Residual Normality (Shapiro-Wilk):** Rejected normality ($p = 0.0105 < 0.05$). The model residuals were non-normal, caused by a single severe low-profit outlier (Index 49, Profit = \$14,681.40).
  4. **Homoscedasticity (Breusch-Pagan):** Homoscedasticity holds ($p = 0.6767 > 0.05$). The residuals have constant variance.
  5. **State-Specific Analysis:** R&D Spend is strongly correlated with Profit in all states ($r > 0.96$). However, due to small sample sizes ($N=16$ or $17$), these sub-group slopes are highly sensitive to single observations.
- **Deliverable:** Created [advanced_diagnostics_report.md](file:///C:/Users/admin/.gemini/antigravity-ide/brain/6ac4088c-f10a-41ab-b3cf-32fdedfcfcbf/advanced_diagnostics_report.md) with detailed tables and plots.

---

## Step 5: Preprocessing Strategy Simulation (CRISP-DM Phase 3)

- **Objective:** Evaluate different data preparation options in Step 3 to fix residual non-normality and improve performance.
- **Actions:**
  - Wrote and executed [test_prep_strategies.py](file:///C:/Users/admin/.gemini/antigravity-ide/scratch/test_prep_strategies.py) to simulate 5 strategies:
    - *Strategy 1 (Baseline):* All features, outlier kept (R2 = 0.9508, Residuals: Non-Normal).
    - *Strategy 2 (Drop Insignificant):* Drop Admin and State (R2 = 0.9505, Adj R2 improved to 0.9483, Residuals: Non-Normal).
    - *Strategy 3 (Remove Outlier):* Drop row 49 (R2 = 0.9618, Residuals: **Normal**, Shapiro $p=0.2833$).
    - *Strategy 4 (Drop + Remove Outlier):* Best strategy. Yielded **R2 = 0.9611**, **Adj R2 = 0.9594**, and **Normal Residuals** (Shapiro $p=0.3146$). It also made `Marketing Spend` statistically significant ($p = 0.0408 < 0.05$).
    - *Strategy 5 (Log Transform):* Degraded R2 to 0.7652 (Non-Normal). Not recommended.
- **Decision:** Recommended adopting **Strategy 4** (drop insignificant features, remove outlier at index 49) in Step 3.

---

## Step 6: Feature Selection Schemes Comparison

- **Objective:** Integrate a comparative evaluation of the top 5 feature selection schemes into the main script.
- **Actions:**
  - Refined [solve_startups.py](file:///d:/AIClass/L6/solve_startups.py) to compare the 5 schemes using Linear Regression:
    - **Model 1 (R&D Spend only):** Test R2 = 0.9265, Test Adj R2 = 0.9173, Test RMSE = 7,714.33
    - **Model 2 (R&D + Marketing):** Test R2 = 0.9168, Test Adj R2 = 0.8931, Test RMSE = 8,206.33
    - **Model 3 (All Numerical):** Test R2 = 0.9001, Test Adj R2 = 0.8501, Test RMSE = 8,995.91
    - **Model 4 (Full Model):** Test R2 = 0.8987, Test Adj R2 = 0.7721, Test RMSE = 9,055.96
    - **Model 5 (Reduced Full):** Test R2 = 0.9159, Test Adj R2 = 0.8485, Test RMSE = 8,254.69
  - Added Best Final Choice logic showing that Model 1 (R&D Spend only) or Model 2 (R&D + Marketing) is preferred over Model 4 because they prevent overfitting and generalize better.
  - Plotted and saved the comparative results to `plots/schemes_comparison.png`.
- **Deliverable:** Wrote the final walkthrough summary report: [walkthrough.md](file:///C:/Users/admin/.gemini/antigravity-ide/brain/6ac4088c-f10a-41ab-b3cf-32fdedfcfcbf/walkthrough.md).
