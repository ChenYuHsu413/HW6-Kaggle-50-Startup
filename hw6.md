# HW6: 50 Startups Regression with CRISP-DM

Date: June 11, 2026

## Overview

This homework implements a complete machine learning regression workflow for the Kaggle 50 Startups dataset. The goal is to predict startup `Profit` using spending features and State information.

The project follows the CRISP-DM process and uses scikit-learn pipelines to compare multiple feature sets and feature selection algorithms.

## Workflow [FLOW]

![HW6 CRISP-DM Workflow](plots/crisp_dm_v2/workflow.png)

Icon legend:

| Icon | Meaning |
|---|---|
| `[GOAL]` | Business goal |
| `[DATA]` | Data understanding |
| `[PREP]` | Data preparation |
| `[MODEL]` | Modeling |
| `[SELECT]` | Feature selection |
| `[CHART]` | Evaluation and visualization |
| `[DONE]` | Final conclusion |

## Work Completed Today

### 1. Built the Main CRISP-DM Script [MODEL]

Main file:

`50_startups_crisp_dm_v2.py`

The script includes:

- Step 1: Business Understanding
- Step 2: Data Understanding
- Step 3: Data Preparation
- Step 4: Modeling
- Step 5: Evaluation
- Step 6: Expert-Based Business Conclusion

The model uses:

- `LinearRegression`
- `Pipeline`
- `ColumnTransformer`
- `OneHotEncoder(drop="first")`
- `train_test_split(test_size=0.2, random_state=42)`

## Feature Experiments [SELECT]

Five business-guided feature experiments were implemented:

| Model | Feature Set | Purpose |
|---|---|---|
| Model A | `R&D Spend` | Test the strongest single business driver |
| Model B | `R&D Spend + Marketing Spend` | Test whether Marketing adds value after R&D |
| Model C | `R&D Spend + Marketing Spend + Administration` | Test whether Administration helps or adds noise |
| Model D | All features including State | Test whether regional information improves performance |
| Model E | State only | Test whether location alone explains Profit |

Best result:

| Model | RMSE | R-squared | Adjusted R2 |
|---|---:|---:|---:|
| Model A: R&D Spend Only | `7,714.33` | `0.9265` | `0.9173` |

## Feature Selection Algorithms [SELECT]

Five feature selection approaches were implemented and compared:

| Algorithm | Type |
|---|---|
| Sequential Feature Selection | Wrapper method |
| RFE | Wrapper method |
| Lasso | Embedded method |
| SelectKBest | Filter method |
| Random Forest Feature Importance | Embedded method |

All five methods found `R&D Spend` as the strongest one-feature result.

## Important Result Images [CHART]

| Output | File |
|---|---|
| Workflow diagram | `plots/crisp_dm_v2/workflow.png` |
| Business-guided feature selection summary | `plots/crisp_dm_v2/business_guided_feature_selection_summary.png` |
| Sequential feature selection summary | `plots/crisp_dm_v2/sequential_feature_selection_summary.png` |
| All-in-one feature selection comparison | `plots/crisp_dm_v2/feature_selection_performance_allinone_summary.png` |
| Marketing vs Administration comparison | `plots/crisp_dm_v2/marketing_vs_administration_comparison.png` |
| Actual vs predicted plot | `plots/crisp_dm_v2/best_model_actual_vs_predicted.png` |
| Correlation heatmap | `plots/crisp_dm_v2/correlation_heatmap.png` |

## Main Business Interpretation [GOAL]

The strongest predictor of startup profit is `R&D Spend`.

`Marketing Spend` is more useful than `Administration` when compared directly after `R&D Spend`, but the best test-set result still comes from using `R&D Spend` alone.

`Administration` appears weaker and less stable because it represents operating cost rather than direct profit generation.

`State` can be tested as a supporting regional feature, but it should not be over-interpreted because the dataset has only 50 rows.

## Conclusion [DONE]

The final conclusion matches the expert discussion in the design file:

- `R&D Spend` is the core predictor.
- `Marketing Spend` is an important supporting amplifier.
- `Administration` is uncertain and should be tested carefully.
- `State` is only a supporting feature.
- The best model should be chosen using Adjusted R2, RMSE, MAE, coefficient interpretation, and business logic.
