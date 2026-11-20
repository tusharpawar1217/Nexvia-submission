===============================================================================
FIGURES DOCUMENTATION
===============================================================================

Generated: November 20, 2026
Location: reports/figures/
Purpose: Portfolio, documentation, presentations

===============================================================================
FIGURE 1: model_comparison.png (147.5 KB)
===============================================================================
Description:
  Bar chart comparing performance of all 4 models

Shows:
  - Naive: WAPE = 0.9993
  - Seasonal Naive: WAPE = 0.9700
  - Moving Average: WAPE = 0.7571
  - LightGBM: WAPE = 0.6595 ⭐

Highlights:
  - LightGBM bar highlighted in green
  - 13% improvement annotation with arrow
  - Professional color scheme
  - Value labels on each bar

Use Cases:
  - README header
  - Portfolio presentations
  - Model selection justification
  - Results summary slides

===============================================================================
FIGURE 2: forecast_example.png (349.5 KB)
===============================================================================
Description:
  28-day forecast visualization with prediction intervals

Shows:
  - Black line: Actual sales
  - Blue dashed line: Forecast (median Q50)
  - Light blue band: 90% prediction interval (Q05-Q95)
  - Medium blue band: 50% prediction interval (Q25-Q75)

Annotations:
  - Coverage: 94.52% (exceeds 90% target)
  - Series ID displayed in title
  - Legend for all elements

Use Cases:
  - Demonstrating forecast quality
  - Showing uncertainty quantification
  - Portfolio showcase
  - Technical presentations

===============================================================================
FIGURE 3: metrics_table.png (119.6 KB)
===============================================================================
Description:
  Comprehensive metrics comparison table

Shows:
  Metrics for all 4 models:
  - WAPE (lower is better)
  - RMSE (lower is better)
  - MAE (lower is better)
  - Coverage 90% (LightGBM only)

Styling:
  - Blue header row
  - Green highlight for best model (LightGBM)
  - Alternating row colors for readability
  - Professional table format

Use Cases:
  - Detailed results documentation
  - Academic papers
  - Technical reports
  - Model comparison discussions

===============================================================================
FIGURE 4: pipeline_diagram.png (223.3 KB)
===============================================================================
Description:
  Visual representation of entire data processing pipeline

Shows 6 Stages:
  1. M5 Raw Data (red) - 30,490 series × 1,941 days
  2. Data Preparation (orange) - 41M rows, filter/melt/join
  3. Feature Engineering (yellow) - 47 features
  4. Model Training (green) - Baselines + LightGBM
  5. Prediction Intervals (blue) - 5 quantiles, 94.52% coverage
  6. API & Dashboard (purple) - FastAPI + Plotly Dash

Features:
  - Color-coded stages
  - Execution time for each stage
  - Arrows showing flow
  - Key metrics in each box

Use Cases:
  - System architecture overview
  - Technical documentation
  - Project workflow explanation
  - Portfolio overview slide

===============================================================================
FIGURE 5: dataset_overview.png (301.3 KB)
===============================================================================
Description:
  4-panel overview of dataset and project statistics

Panel 1 - Series by Category (top left):
  - FOODS: 997 series
  - HOUSEHOLD: 753 series
  - HOBBIES: 384 series

Panel 2 - Pipeline Timing (top right):
  - Validation: 0.5s
  - Data Prep: 477s
  - Features: 631s
  - Baselines: 22s
  - LightGBM: 474s

Panel 3 - Model Performance Gap (bottom left):
  - Shows % worse than LightGBM
  - Naive: +32.0%
  - Seasonal Naive: +27.1%
  - Moving Average: +14.8%
  - LightGBM: BEST

Panel 4 - Data Volume (bottom right):
  - Raw CSV: 120 MB
  - Processed: 288 MB
  - Features: 1040 MB
  - Models: 3 MB

Use Cases:
  - Project statistics summary
  - Portfolio overview
  - Quick reference
  - Executive summary

===============================================================================
HOW TO USE THESE FIGURES
===============================================================================

1. README.md:
   - Already embedded in README
   - Will display on GitHub automatically
   - Provides visual appeal to documentation

2. Portfolio Website:
   - Use all 5 figures in project page
   - Show pipeline_diagram first for overview
   - Then model_comparison to show results
   - forecast_example for technical depth
   - dataset_overview for statistics

3. Presentations:
   - Slide 1: pipeline_diagram (overview)
   - Slide 2: dataset_overview (scope)
   - Slide 3: model_comparison (results)
   - Slide 4: forecast_example (quality)
   - Slide 5: metrics_table (details)

4. Job Applications:
   - Attach model_comparison in cover letter
   - Include pipeline_diagram in portfolio
   - Reference forecast_example in interviews

5. LinkedIn:
   - Post model_comparison with caption
   - Post forecast_example showing prediction intervals
   - Tag with #machinelearning #datascience #timeseries

===============================================================================
REGENERATING FIGURES
===============================================================================

To regenerate all figures:

  cd C:\Users\pawar\OneDrive\Desktop\Nexvia\DS-02\demand-forecast-project
  python generate_figures.py

This will:
  - Load latest model results
  - Create all 5 figures
  - Save to reports/figures/
  - Overwrite existing files

===============================================================================
FIGURE SPECIFICATIONS
===============================================================================

Format: PNG
Resolution: 300 DPI (high quality)
Color Scheme: Professional (blues, greens, reds, oranges, purples)
Font: Arial (system default)
Style: Seaborn whitegrid

All figures:
  - High resolution for printing
  - Suitable for presentations
  - Web-optimized file sizes
  - Professional appearance

===============================================================================
CREDITS
===============================================================================

Generated by: generate_figures.py
Libraries: matplotlib, seaborn, pandas, numpy
Data Source: M5 Walmart Dataset (Kaggle)
Project: Retail Demand Forecasting & Inventory Optimization

===============================================================================
