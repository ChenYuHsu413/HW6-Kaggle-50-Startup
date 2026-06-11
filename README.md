# 50 Startups CRISP-DM Regression Project

This project predicts startup `Profit` from business spending features and State data using a CRISP-DM machine learning workflow.

## Files [FILES]

| File or Folder | Description |
|---|---|
| `50_startups_crisp_dm_v2.py` | Main executable Python script |
| `design.md` | Project design requirements |
| `sources/50_Startups.csv` | Local dataset copy |
| `plots/crisp_dm_v2/` | Generated charts, workflow image, and CSV outputs |
| `hw6.md` | Homework report summary |

## How to Run [RUN]

```bash
python 50_startups_crisp_dm_v2.py
```

The script loads the dataset, performs CRISP-DM analysis, trains regression models, compares feature sets, runs feature selection algorithms, and saves visualization results.

## Main Libraries [TOOLS]

- pandas
- numpy
- matplotlib
- scikit-learn

## Workflow [FLOW]

1. `[GOAL]` Business Understanding
2. `[DATA]` Data Understanding
3. `[PREP]` Data Preparation
4. `[MODEL]` Modeling
5. `[SELECT]` Feature Selection
6. `[METRIC]` Evaluation
7. `[CHART]` Visualization
8. `[DONE]` Business Conclusion

Workflow image:

`plots/crisp_dm_v2/workflow.png`

## Key Outputs [CHART]

| Output | File |
|---|---|
| Feature selection all-in-one summary | `plots/crisp_dm_v2/feature_selection_performance_allinone_summary.png` |
| Business-guided feature selection | `plots/crisp_dm_v2/business_guided_feature_selection_summary.png` |
| Marketing vs Administration comparison | `plots/crisp_dm_v2/marketing_vs_administration_comparison.png` |
| Model comparison by Adjusted R2 | `plots/crisp_dm_v2/model_comparison_adjusted_r2.png` |
| Model comparison by RMSE | `plots/crisp_dm_v2/model_comparison_rmse.png` |
| Best model actual vs predicted | `plots/crisp_dm_v2/best_model_actual_vs_predicted.png` |

## Best Model Result [BEST]

The best model on the current test split is:

`Model A: R&D Spend Only`

| Metric | Value |
|---|---:|
| R-squared | `0.9265` |
| Adjusted R2 | `0.9173` |
| MAE | `6,077.36` |
| RMSE | `7,714.33` |

## Feature Selection Summary [SELECT]

Five feature selection methods were implemented:

- Sequential Feature Selection
- RFE
- Lasso
- SelectKBest
- Random Forest Feature Importance

All five methods identify `R&D Spend` as the best one-feature result.

## Business Conclusion [DONE]

`R&D Spend` is the strongest profit driver. `Marketing Spend` is a useful supporting feature and performs better than `Administration` when compared after R&D. `Administration` is less stable, and `State` should be treated as a supporting feature only because the dataset is small.
