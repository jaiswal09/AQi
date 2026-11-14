# How to Run: Beijing Air Pollution Prediction (Group-3)
**Complete Step-by-Step Reproduction & Execution Guide**

This document provides clear, reproducible instructions to run the entire pipeline from exploratory data analysis to feature engineering, model training, benchmarking, and generating final test submissions.

---

## 1. Prerequisites & Environment Setup

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python Version**: Python 3.9, 3.10, 3.11, 3.12, or 3.15
- **RAM**: Minimum 8 GB recommended (16 GB optimal)
- **Disk Space**: ~200 MB free space for intermediate models, figures, and submissions

### Clone / Navigate to Workspace
Ensure your terminal / command prompt is in the project root directory:
```bash
cd c:\Users\ABCOM\Desktop\group-3-beijing-air
```

### Install Required Python Packages
A `requirements.txt` file is included in the project directory. Install dependencies using:

```bash
pip install -r requirements.txt
```

Alternatively, install the core libraries directly:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn openpyxl
```

---

## 2. Project Directory Structure

Ensure the dataset files are located in the project directory:

```text
group-3-beijing-air/
│
├── beijing_train.csv                 # Official historical training dataset (2013-2016)
├── beijing_test.csv                  # Official test dataset (Jan-Feb 2017)
├── sample_submission.csv.xlsx        # Benchmark sample submission format
│
├── requirements.txt                  # Python dependencies
├── run_eda.py                        # Script to generate 8 publication-quality EDA figures
├── benchmark_models.py               # Script to train & benchmark Ridge, RF, LightGBM, XGBoost
├── generate_clean_submission.py      # Master production training & verified submission generator
│
├── PROJECT_PRESENTATION.md           # PPT-style slide deck presentation in Markdown
├── HOW_TO_RUN.md                     # This step-by-step execution guide
│
├── eda_plots/                        # Directory where all 8 EDA figures are saved
│   ├── 01_pm25_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_pm25_vs_pm10_co.png
│   ├── 04_temporal_trends.png
│   ├── 05_station_distribution.png
│   ├── 06_wind_effects.png
│   ├── 07_yearly_trends.png
│   └── 08_pm_ratio_seasonal.png
│
├── model_plots/                      # Directory where model comparison & diagnostic plots are saved
│   ├── 01_model_benchmark_comparison.png
│   ├── 02_feature_importance.png
│   ├── 03_actual_vs_predicted_residuals.png
│   ├── 04_station_rmse_breakdown.png
│   ├── 05_test_predictions_timeseries.png
│   ├── benchmark_comparison.csv
│   └── station_performance.csv
│
└── submission.csv                    # Final generated test predictions (16,728 rows)
```

---

## 3. Step-by-Step Execution Guide

### Step 1: Run Exploratory Data Analysis (EDA)
Generates the 8 statistical visualization plots analyzing $\text{PM}_{2.5}$ distributions, multi-pollutant correlations, diurnal and seasonal patterns, spatial variations across stations, and wind aerodynamics.

Run:
```bash
python run_eda.py
```
*Expected Output*:
- Creates/populates the `eda_plots/` directory with figures `01_pm25_distribution.png` through `08_pm_ratio_seasonal.png`.
- Runtime: ~20–30 seconds.

---

### Step 2: Run Multi-Model Benchmarking & Validation
Performs domain feature engineering (101 features) and evaluates 5 regression models on a strict out-of-time holdout validation split (training on 2013–2015, validating on the full year 2016).

Run:
```bash
python benchmark_models.py
```
*Expected Output*:
- Evaluates Ridge, Random Forest, LightGBM, XGBoost, and the Blended Ensemble.
- Saves:
  - `model_plots/benchmark_comparison.csv`
  - `model_plots/01_model_rmse_comparison.png`
  - `model_plots/02_feature_importance.png`
  - `model_plots/03_actual_vs_predicted_residuals.png`
  - `model_plots/04_station_rmse_breakdown.png`
- Runtime: ~2–3 minutes.

---

### Step 3: Train Production Models & Generate Submissions
Trains the fine-tuned LightGBM and XGBoost models on **100% of the historical training data** (2013–2016, 395,301 valid records) and generates final predictions for `beijing_test.csv` (Jan–Feb 2017).

Run:
```bash
python generate_clean_submission.py
```
*Expected Output*:
- Generates 3 deliverable submission files:
  1. `submission.csv`: Standard headerless CSV (`REF_NO, PM2.5`) matching competition format.
  2. `submission_with_header.csv`: Labeled CSV with column names (`REF_NO, PM2.5`).
  3. `submission.xlsx`: Excel workbook format matching `sample_submission.csv.xlsx`.
- Prints verification summary confirming exact shape (`(16728, 2)`), 0 nulls, and row-for-row alignment with `sample_submission.csv.xlsx`.
- Runtime: ~2 minutes.

---

### Step 4: Verify Submissions (Sanity Check)
To independently verify the submission file integrity, run this quick Python verification command:

```bash
python -c "
import pandas as pd
sub = pd.read_csv('submission.csv', header=None)
sample = pd.read_excel('sample_submission.csv.xlsx', header=None)
print('Shape match:', sub.shape == (16728, 2))
print('REF_NO alignment match:', (sub[0] == sample[0]).all())
print('Null count:', sub[1].isnull().sum())
print('PM2.5 Stats:\n', sub[1].describe())
"
```

*Expected Verification Output*:
```text
Shape match: True
REF_NO alignment match: True
Null count: 0
PM2.5 Stats:
 count    16728.000000
mean        92.043581
std        109.326244
min          2.587841
25%         19.531204
50%         47.200192
75%        127.391823
max        674.302148
```

---

## 4. Key Performance Reference

When executing the validation benchmark (`benchmark_models.py`), the expected results are:

| Model | Validation RMSE ($\mu\text{g/m}^3$) | Validation MAE ($\mu\text{g/m}^3$) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Blended Ensemble (0.55 LGB + 0.45 XGB)** | **17.37** | **10.20** | **0.9473** |
| LightGBM Regressor | 17.45 | 10.31 | 0.9469 |
| XGBoost Regressor (Hist) | 17.55 | 10.25 | 0.9463 |
| Random Forest Regressor | 19.56 | 11.17 | 0.9333 |
| Ridge Regression (Baseline) | 22.64 | 14.09 | 0.9106 |

---

## 5. Troubleshooting & Notes

1. **Memory Usage**: The feature engineering step creates ~101 columns across ~420,000 concatenated rows. This uses approximately 1.5 GB of RAM. Ensure sufficient free RAM before running.
2. **Windows File Path Compatibility**: All scripts use `os.path.join(os.path.dirname(os.path.abspath(__file__)), ...)` to ensure seamless execution on Windows, macOS, and Linux regardless of working directory.
3. **No GPU Required**: Both LightGBM and XGBoost (`tree_method='hist'`) utilize optimized multi-threaded CPU histograms (`n_jobs=4`), executing in under 2 minutes without requiring CUDA or GPU drivers.
