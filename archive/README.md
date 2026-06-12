# Archive

This folder contains earlier draft scripts and outputs from the development session.
These files are kept for reference but are not part of the main project submission.

| File | Description |
|---|---|
| `solve_startups.py` | First exploration: CRISP-DM Step 2 EDA + multi-model comparison (LinearRegression, Ridge, Lasso, SVR, RandomForest) |
| `advanced_analysis.py` | Advanced statistical diagnostics using `statsmodels` (VIF, p-values, Shapiro-Wilk, Breusch-Pagan) |
| `log.md` | Session development log recording steps, decisions, and findings |
| `model_metrics.csv` | Output from `solve_startups.py` — model comparison across all five ML algorithms |

The main deliverable is `src/modeling.py` in the project root (formerly `50_startups_crisp_dm_v2.py`).

Note: these archived scripts still use the old `plots/` output paths and expect to be
run from the project root. Their original outputs were removed during cleanup; rerun
the scripts if those charts are needed again.
