# 2026-06-12 Work Report - v4 Analysis and Video Pipeline

## Git / Project State

- Synced local `main` with `origin/main`.
- Pushed commit `a559694 Add v4 feature selection experiment`.
- Added daily handoff summaries:
  - `hw6-0612.yaml`
  - `hw6-0612.md`

## v4 Machine Learning Upgrade

- Added `50_startups_crisp_dm_v4_top5_feature_selection_10_visual.py`.
- Kept the original CRISP-DM workflow, `LinearRegression`, `Pipeline`, `ColumnTransformer`, and `OneHotEncoder`.
- Compared 10 feature-selection algorithms on 6 model-ready candidate features.
- Removed `RFE_LinearRegression` and `RFECV_LinearRegression` because their results distorted the chart scale.
- Updated the v4 report focus so each algorithm reports its own top-5 selected feature set.
- Generated v4 CSV outputs under `outputs/crisp_dm_v4/`.
- Generated v4 plots under `plots/crisp_dm_v4/`.

## Video Version Summary

| Video Project | Prompt / Agent Note | Element Strategy | Result |
|---|---|---|---|
| `startup-presentation-video-pptx/` | Earlier ChatGPT prompt explicitly requested getting elements from each slide. | Rectangular element cutting. | Current preferred older reference video. |
| `startup-presentation-video-v2/` | Claude Fable5 prompt did not explicitly request element separation. | Dynamic whole-slide / staged animation. | Faster 76.5s dynamic cut. |
| `startup_profit_video/` | 0612 Gemini prompt requested segmentation for the 10-algorithm topic. | Segmentation-based extraction attempt. | Experimental; segmentation quality was weak. |
| `startup-profit-code-video/` | 0612 ChatGPT prompt requested segmentation-style element separation from `Startup_Profit_Code.pptx` / PDF, upbeat female narration, and total duration under 2 minutes. | PDF-to-PNG, OpenCV element detection, transparent layer extraction, inpainted backgrounds, GSAP entrances, Edge TTS narration. | Rendered successfully; 97.99s final MP4. |

## New ChatGPT Segmentation-Style Video Pipeline

- Added `scripts/build_startup_profit_code_video.py`.
- Source deck:
  - `sources/Startup_Profit_Code.pptx`
  - `sources/Startup_Profit_Code.pdf`
- Pipeline behavior:
  - Uses PyMuPDF `fitz` to render PDF pages at `dpi=150`.
  - Uses OpenCV median brightness detection and Otsu thresholding.
  - Uses a `(120, 50)` dilation kernel to group text, tables, and icons into major regions.
  - Keeps up to 8 major elements per slide.
  - Produces transparent element PNGs and inpainted `background.png` files.
  - Generates per-slide `metadata.json` with element positions and animation suggestions.
  - Uses `edge-tts` voice `zh-TW-HsiaoChenNeural` at `+10%`.
  - Uses `tinytag` to measure narration durations.
  - Generates 12 HyperFrames sub-compositions and one root `index.html`.
  - Alternates slide tracks 1 and 2 to avoid overlap lint conflicts.
  - Adds 0.5s slide overlap transitions and final black fade.
- Validation:
  - `npx hyperframes lint`: 0 errors, 0 warnings.
  - `npx hyperframes inspect --samples 12`: 0 layout issues.
  - Rendered MP4: `startup-profit-code-video/renders/startup-profit-code-video.mp4`.
  - Final duration: about 97.99 seconds.

## README Updates

- Added updated video demo links.
- Added a `Video Versions` section explaining the four video projects and their prompt / segmentation differences.
- Added new project structure rows for the new source deck, pipeline script, and video projects.
- Added commands for rebuilding and rendering the 0612 ChatGPT segmentation-style code-deck video.
