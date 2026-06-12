# Kaggle 50 Startups Linear Regression Analysis

## Demo

| Updated Video Demo | URL |
|---|---|
| Presentation video V1 - PPTX element-cut version | <https://chenyuhsu413.github.io/HW6-Kaggle-50-Startup/startup-presentation-video-pptx/renders/startup-profit-presentation.mp4> |
| Presentation video V2 - dynamic cut version | <https://chenyuhsu413.github.io/HW6-Kaggle-50-Startup/startup-presentation-video-v2/renders/startup-profit-presentation-v2.mp4> |
| 0612 Gemini segmentation video - experimental | <https://chenyuhsu413.github.io/HW6-Kaggle-50-Startup/startup_profit_video/renders/startup_profit_presentation.mp4> |
| 0612 ChatGPT segmentation video - code deck, under 2 minutes | <https://chenyuhsu413.github.io/HW6-Kaggle-50-Startup/startup-profit-code-video/renders/startup-profit-code-video.mp4> |

| Demo | URL |
|---|---|
| 📖 Tutorial webpage (slide deck + notes) | <https://chenyuhsu413.github.io/HW6-Kaggle-50-Startup/tutorial/> |
| 🚀 Interactive Streamlit app | <https://chenyu-hw6-kaggle-50-startup.streamlit.app/> |
| 📖 技術白皮書 | <https://chenyuhsu413.github.io/HW6-Kaggle-50-Startup/sources/HW6_50_Startups_Technical_Whitepaper.pdf> |

> The GitHub Pages links require Pages to be enabled once:
> repo **Settings → Pages → Deploy from a branch → `main` / `(root)`**.

## Video Versions

| Version | Source / Prompt Note | Element Strategy | Result Note | Output |
|---|---|---|---|---|
| `startup-presentation-video-pptx/` | Earlier ChatGPT prompt explicitly requested getting elements from each slide. | Rectangular element cutting from the slide deck. | Current preferred older reference video. | `startup-presentation-video-pptx/renders/startup-profit-presentation.mp4` |
| `startup-presentation-video-v2/` | Claude Fable5 prompt did not explicitly request element separation. | More dynamic whole-slide / staged presentation approach. | Faster 76.5s dynamic cut. | `startup-presentation-video-v2/renders/startup-profit-presentation-v2.mp4` |
| `startup_profit_video/` | 0612 Gemini prompt requested segmentation for the 10-algorithm topic. | Segmentation-based extraction attempt. | Experimental; segmentation quality was weak and less suitable as the final reference. | `startup_profit_video/renders/startup_profit_presentation.mp4` |
| `startup-profit-code-video/` | 0612 ChatGPT prompt requested segmentation-style element separation from `Startup_Profit_Code.pptx` / PDF, upbeat female narration, and total duration under 2 minutes. | PDF-to-PNG, OpenCV element detection, transparent layer extraction, inpainted backgrounds, GSAP layer entrances, Edge TTS narration. | Rendered successfully; 97.99s final MP4 with narration. | `startup-profit-code-video/renders/startup-profit-code-video.mp4` |

## Project Overview

This project analyzes the Kaggle 50 Startups dataset using Multiple Linear Regression.
The goal is to predict company profit based on business spending features and state information,
following the CRISP-DM machine learning workflow.

![HW6 50 Startups CRISP-DM Visual Summary](sources/ChatGPT_02.png)

## Dataset

The dataset is stored in:

`sources/50_Startups.csv`

Columns:

- R&D Spend
- Administration
- Marketing Spend
- State
- Profit

The `sources/` folder contains the original dataset and should not be deleted.

## Project Structure

| File or Folder | Description |
|---|---|
| `src/modeling.py` | Main executable CRISP-DM analysis script |
| `src/compare_feature_selections.py` | Integrated comparison of the four feature-selection analyses |
| `50_startups_crisp_dm_v4_top5_feature_selection_10_visual.py` | v4 CRISP-DM feature-selection experiment with 10 algorithms and top-k visual summary |
| `streamlit_app.py` | Interactive Streamlit app (tutorial slides, data exploration, model comparison, profit predictor) |
| `tutorial/index.html` | Standalone tutorial webpage for the presentation deck |
| `sources/50_Startups.csv` | Original dataset |
| `sources/Startup_Profit_Code.pptx` | 0612 code-focused presentation source deck |
| `sources/Startup_Profit_Code.pdf` | PDF export used by the segmentation video pipeline |
| `scripts/build_startup_profit_code_video.py` | Automated PDF-to-HyperFrames video pipeline with layer detection and narration |
| `outputs/figures/` | Generated charts and the workflow image |
| `outputs/metrics/` | Generated CSV metric tables |
| `design.md` | Project design requirements |
| `hw6.md` | Homework report summary |
| `logs/` | Daily work reports and handoff notes |
| `archive/` | Earlier draft scripts and development log (reference only) |
| `startup-presentation-video-pptx/` | Presentation video project and final MP4 render |
| `startup-presentation-video-v2/` | V2 dynamic presentation video project and final MP4 render |
| `startup_profit_video/` | 0612 Gemini segmentation experiment video project |
| `startup-profit-code-video/` | 0612 ChatGPT segmentation-style code-deck video project under 2 minutes |

## Methodology

This project follows the CRISP-DM process:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment / Reporting

Workflow image: `outputs/figures/workflow.png`

## Experiments

The project includes experiments on:

- Using 1 to 4 numerical/business features (Models A–E)
- Evaluating the effect of the `State` categorical variable with `OneHotEncoder(drop="first")`
- Comparing model performance using RMSE, MAE, R², and Adjusted R²
- Five feature selection algorithms: Sequential Feature Selection, RFE, Lasso,
  SelectKBest, and Random Forest Feature Importance
- v4 feature-selection upgrade: 10 algorithms, 6 model-ready encoded features,
  top-1 to top-6 comparison, and one integrated RMSE/R-squared/table figure

## How to Run

Run from the project root:

```bash
pip install -r requirements.txt
python src/modeling.py
```

The script loads the dataset (URL first, local `sources/50_Startups.csv` as fallback),
performs CRISP-DM analysis, trains the regression models, compares feature sets,
runs the feature selection algorithms, and saves all results.

Run the v4 feature-selection experiment:

```bash
python 50_startups_crisp_dm_v4_top5_feature_selection_10_visual.py
```

The v4 script keeps CRISP-DM, Linear Regression, Pipeline, ColumnTransformer,
and OneHotEncoder. It keeps all 3 State dummy variables visible for comparison,
evaluates 10 feature-selection algorithms, compares top-k feature subsets from
k=1 to k=6, and saves report-ready CSV/PNG outputs.

Interactive app:

```bash
streamlit run streamlit_app.py
```

Tutorial webpage: open `tutorial/index.html` in a browser (slide images are loaded
from `startup-presentation-video-pptx/assets/slides/`, so keep the repo layout intact).

Build the 0612 ChatGPT segmentation-style code-deck video:

```bash
python scripts/build_startup_profit_code_video.py
cd startup-profit-code-video
npx hyperframes lint
npx hyperframes inspect --samples 12
npx hyperframes render --output renders/startup-profit-code-video.mp4 --quality standard
```

On this Windows workspace, rendering used the existing `ffmpeg-static` binary from
`startup-presentation-video-v2/node_modules/ffmpeg-static/` because system FFmpeg
was not installed globally.

## Outputs

Generated outputs are saved under:

- `outputs/figures/` — PNG charts
- `outputs/metrics/` — CSV metric tables
- `outputs/crisp_dm_v4/` - v4 CSV metric tables
- `plots/crisp_dm_v4/` - v4 PNG charts

Key result images:

| Output | File |
|---|---|
| **v4 all-in-one 10-algorithm top-k summary** | `plots/crisp_dm_v4/feature_selection_performance_allinone_summary.png` |
| v4 top-5 selected features by algorithm | `plots/crisp_dm_v4/best_selected_features_by_algorithm.png` |
| **Integrated feature-selection comparison (all four analyses)** | `outputs/figures/feature_selection_integrated_comparison.png` |
| Feature selection all-in-one summary | `outputs/figures/feature_selection_performance_allinone_summary.png` |
| Business-guided feature selection | `outputs/figures/business_guided_feature_selection_summary.png` |
| Marketing vs Administration comparison | `outputs/figures/marketing_vs_administration_comparison.png` |
| Model comparison by Adjusted R² | `outputs/figures/model_comparison_adjusted_r2.png` |
| Model comparison by RMSE | `outputs/figures/model_comparison_rmse.png` |
| Best model actual vs predicted | `outputs/figures/best_model_actual_vs_predicted.png` |

## Best Model Result

The best model on the current test split is:

`Model A: R&D Spend Only`

| Metric | Value |
|---|---:|
| R² | `0.9265` |
| Adjusted R² | `0.9173` |
| MAE | `6,077.36` |
| RMSE | `7,714.33` |

## Business Conclusion

`R&D Spend` is the strongest profit driver. `Marketing Spend` is a useful supporting
feature and performs better than `Administration` when compared after R&D.
`Administration` is less stable, and `State` should be treated as a supporting feature
only, because the dataset has just 50 rows.
