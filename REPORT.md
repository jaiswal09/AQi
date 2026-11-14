# Group-3: Beijing Air Pollution Prediction (PM2.5)
**Comprehensive Technical Report, Exploratory Data Analysis, and Machine Learning Modeling**

---

## 1. Executive Summary

This report documents the end-to-end data science and machine learning solution for predicting **PM2.5** particulate matter concentrations across **12 national air quality monitoring stations in Beijing**. The historical dataset spans hourly measurements from **March 2013 to February 2017**.

- **Machine Learning Task**: Multivariate Time-Series Regression
- **Target Variable**: $\text{PM}_{2.5}$ concentration ($\mu\text{g/m}^3$)
- **Primary Evaluation Metric**: Root Mean Squared Error (RMSE)
- **Top Performing Model**: **Weighted Blended Ensemble** (LightGBM Regressor + XGBoost Regressor)
- **Validation Performance**:
  - **RMSE**: **17.37 $\mu\text{g/m}^3$** (out-of-time test holdout on 2016)
  - **MAE**: **10.20 $\mu\text{g/m}^3$**
  - **$R^2$ Score**: **0.9473** (explaining 94.73% of target variance)
- **Deliverables**:
  - [submission.csv](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/submission.csv) (16,728 predictions matching sample submission structure)
  - [submission_with_header.csv](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/submission_with_header.csv) (labeled CSV format)
  - [submission.xlsx](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/submission.xlsx) (Excel workbook format)

---

## 2. Dataset Architecture & Preprocessing

The dataset consists of two primary partitions:
- **Training Set (`beijing_train.csv`)**: 403,776 rows across 12 stations, spanning March 1, 2013 00:00 to December 31, 2016 23:00 (exactly 33,648 hourly records per station).
- **Test Set (`beijing_test.csv`)**: 16,728 rows across 12 stations, covering January 1, 2017 00:00 to February 28, 2017 23:00.

### Missing Value Analysis & Handling
- **Target ($\text{PM}_{2.5}$)**: 8,475 missing records in train (2.10%). Dropped during model training so supervised learning targets remain uncorrupted ground truth.
- **Co-pollutants**: In training, missing rates are $\text{CO}$ (5.02%), $\text{O}_3$ (3.22%), $\text{NO}_2$ (2.94%), $\text{SO}_2$ (2.17%), and $\text{PM}_{10}$ (1.54%).
- **Meteorology**: Missing rates are extremely low (< 0.07%).
- **Test Set Quality**: The test set is completely complete with **0 missing values** across all predictor columns.
- **Temporal Continuity**: Train and test form a seamless continuous sequence per station: `REF_NO` 33,648 (2016-12-31 23:00) transitions directly into `REF_NO` 33,649 (2017-01-01 00:00).

---

## 3. Exploratory Data Analysis (EDA) & Domain Observations

### 3.1 Target Variable ($\text{PM}_{2.5}$) Distribution
![Target Distribution](./eda_plots/01_pm25_distribution.png)

- **Positive Skewness**: The distribution of raw $\text{PM}_{2.5}$ is heavily right-skewed ($\text{mean} = 79.25\ \mu\text{g/m}^3$, $\text{median} = 55.0\ \mu\text{g/m}^3$, $\text{max} = 999.0\ \mu\text{g/m}^3$).
- **Air Quality Severity**: Over 68% of all recorded hours exceed the WHO 24-hour guideline of $25\ \mu\text{g/m}^3$, with frequent severe episodic winter smog events exceeding $300\ \mu\text{g/m}^3$.

---

### 3.2 Correlation Analysis & Multi-Pollutant Interactions
![Correlation Matrix](./eda_plots/02_correlation_heatmap.png)

![PM2.5 vs PM10 and CO](./eda_plots/03_pm25_vs_pm10_co.png)

- **$\text{PM}_{10}$ Correlation ($r = +0.88$)**: $\text{PM}_{2.5}$ is physically a subset of coarse particulate matter $\text{PM}_{10}$. In general, $\text{PM}_{2.5} \le \text{PM}_{10}$, with points tightly hugging the 1:1 boundary line during combustion episodes.
- **$\text{CO}$ Correlation ($r = +0.79$)**: Carbon monoxide is an inert tracer of incomplete fossil fuel and biomass combustion (coal heating and vehicular traffic). Its high correlation indicates that winter smog is driven by primary combustion emissions.
- **$\text{NO}_2$ ($r = +0.66$) and $\text{SO}_2$ ($r = +0.48$)**: Precursors to secondary inorganic aerosols (sulfates $\text{SO}_4^{2-}$ and nitrates $\text{NO}_3^-$) which form a large fraction of fine particulate mass.
- **$\text{O}_3$ ($r = -0.14$)**: Inversely correlated during winter due to titration with $\text{NO}$ ($\text{NO} + \text{O}_3 \rightarrow \text{NO}_2 + \text{O}_2$) and lack of strong solar photochemical radiation.
- **Wind Speed ($r = -0.27$)**: Negative correlation demonstrating the mechanical dispersion effect of atmospheric ventilation.

---

### 3.3 Temporal Dynamics: Diurnal Cycles & Seasonality
![Temporal Trends](./eda_plots/04_temporal_trends.png)

![Yearly Trends](./eda_plots/07_yearly_trends.png)

- **Diurnal (Hourly) Cycle**:
  - Peaks between **20:00 and 02:00** (nocturnal boundary layer compression and night-time truck traffic/heating).
  - Troughs around **14:00 to 16:00** due to daytime thermal convective mixing and elevated planetary boundary layer height.
- **Seasonal Cycle (Heating Season Effect)**:
  - Strong seasonal disparity: highest concentrations occur during winter months (November to February), coinciding with Beijing's centralized coal and gas heating season.
  - Lowest concentrations occur in late spring and summer (July-August) due to increased precipitation and higher atmospheric mixing layers.
- **Multi-Year Trend (2013-2016)**:
  - Visible secular downward trend in summer and autumn baseline concentrations reflecting China's *Air Pollution Prevention and Control Action Plan* (coal-to-gas transition and industrial controls), though severe winter stagnation peaks persist.

---

### 3.4 Spatial Heterogeneity & Wind Aerodynamics
![Station Distribution](./eda_plots/05_station_distribution.png)

![Wind Effects](./eda_plots/06_wind_effects.png)

- **Spatial Gradient**:
  - Cleanest background stations: **Dingling** (median $44.0\ \mu\text{g/m}^3$) and **Huairou** (median $47.0\ \mu\text{g/m}^3$), situated near northern mountainous nature reserves.
  - Most polluted urban/industrial stations: **Gucheng**, **Wanshouxigong**, and **Dongsi** (median $> 60\ \mu\text{g/m}^3$), situated in the urban core and southern transport corridors.
- **Wind Direction (Pollution Rose)**:
  - Winds from the **South, Southeast, and Southwest (SSW, S, SSE)** transport heavy regional industrial emissions from the Hebei plains and accumulate in the Beijing basin against the Yanshan mountains.
  - Winds from the **North and Northwest (NNW, NW, N)** originate from Siberia/Inner Mongolia, delivering cold, dry, clean air that rapidly sweeps out particulate matter.
- **Wind Speed Dispersal**:
  - Stagnant winds ($< 1.5\ \text{m/s}$) allow exponential accumulation of particulates.
  - Winds exceeding $4.0\ \text{m/s}$ reliably clear the urban basin, reducing median $\text{PM}_{2.5}$ below $20\ \mu\text{g/m}^3$.

---

### 3.5 Fine-to-Coarse Particulate Ratio ($\text{PM}_{2.5} / \text{PM}_{10}$)
![PM Ratio Seasonal](./eda_plots/08_pm_ratio_seasonal.png)

- **Overall Ratio**: Median $\text{PM}_{2.5} / \text{PM}_{10}$ ratio is **0.72**.
- **Seasonal Divergence**:
  - In **Winter**, the ratio reaches its maximum (~0.80), indicating dominance of fine combustion-generated secondary aerosols.
  - In **Spring**, the ratio drops significantly (~0.55 - 0.60), reflecting the influx of coarse mineral dust transported from the Gobi Desert.

---

## 4. Feature Engineering Architecture

A total of **101 domain-engineered features** were designed to supply the tree algorithms with atmospheric physics, chemistry, spatial coordinates, and temporal dynamics:

1. **Aerodynamics & Wind Vector Decomposition**:
   - Compass angles mapped to $\theta \in [0^\circ, 360^\circ)$.
   - Zonal (East-West) wind: $u = -\text{WSPM} \times \sin(\theta)$.
   - Meridional (North-South) wind: $v = -\text{WSPM} \times \cos(\theta)$.
   - Polar coordinates: $\sin(\theta), \cos(\theta)$.

2. **Atmospheric Thermodynamics & Moisture**:
   - **Relative Humidity (RH)** via the **August-Roche-Magnus approximation**:
     $$\alpha = \frac{17.625 \times \text{TEMP}}{243.04 + \text{TEMP}}, \quad \beta = \frac{17.625 \times \text{DEWP}}{243.04 + \text{DEWP}}, \quad \text{RH} = 100 \times \exp(\beta - \alpha)$$
   - **Dew Point Depression**: $\text{TEMP} - \text{DEWP}$ (low values signify saturation / foggy haze).
   - **Vapor Pressure Deficit (VPD)**: $e_s(T) - e_a(T_{dew})$.
   - **Air Density Approximation**: $\rho \approx \frac{\text{PRES}}{2.87 \times (\text{TEMP} + 273.15)}$.
   - **Ventilation Index**: $\text{WSPM} \times (\text{TEMP} - \text{DEWP} + 1.0)$.
   - **Stagnation Indicator**: Binary flag where $\text{WSPM} < 1.5\ \text{m/s} \land \text{RH} > 65\%$.

3. **Chemical Ratios & Multi-Pollutant Interactions**:
   - Combustion ratios: $\frac{\text{PM}_{10}}{\text{CO} + 1}$, $\frac{\text{SO}_2}{\text{CO} + 1}$, $\frac{\text{NO}_2}{\text{CO} + 1}$.
   - Photochemical index: $\frac{\text{O}_3}{\text{NO}_2 + 1}$, Total Oxidant $\text{O}_x = \text{O}_3 + \text{NO}_2$.
   - Interaction terms: $\text{PM}_{10} \times \left(\frac{\text{RH}}{100}\right)$ (hygroscopic growth), $\frac{\text{PM}_{10}}{\text{WSPM} + 0.1}$, $\frac{\text{CO}}{\text{WSPM} + 0.1}$.

4. **Spatial Geography & Regional Synoptic Aggregations**:
   - Geographic coordinates (latitude, longitude) of all 12 stations.
   - Radial distance from Tiananmen Square (urban center): $\text{dist\_center}$.
   - City-wide synoptic hourly means: Mean $\text{PM}_{10}$, $\text{CO}$, $\text{NO}_2$, $\text{TEMP}$, $\text{PRES}$, and $\text{WSPM}$ across all stations at timestamp $t$.
   - Station anomalies from regional synoptic background: $\Delta \text{PM}_{10} = \text{PM}_{10, stn} - \overline{\text{PM}}_{10, city}$.

5. **Temporal Lags and Multi-Scale Rolling Statistics**:
   - Hourly lags: 1-hour lag, 2-hour lag, 1-hour rate of change ($\text{diff1}$).
   - Rolling moving averages: 3-hour, 6-hour, and 24-hour moving averages.
   - Rolling volatility: 6-hour rolling standard deviation.
   - Calendar cyclical harmonics: $\sin/\cos$ transforms for hour, month, and day-of-year; weekend and heating season flags.

---

## 5. Model Architecture, Benchmarking & Performance Comparison

### 5.1 III.I. Linear Regression Methodology

Linear regression is a method of fitting independent variables with dependent variables into a hyperplane in $n$ dimensions, where $n$ is the total number of features or variables within our data. The model minimizes the total error while trying to fit all the data points in one line and learning continuously by optimizing parameters.

#### Diagnostic Visualizations & Fit Analysis

![Linear Regression Architecture & Workflow](./model_plots/linear_regression_workflow.png)

![Linear Regression Comprehensive Diagnostic](./model_plots/linear_regression_comprehensive.png)

![Linear Regression Primary Pollutant Fits](./model_plots/linear_regression_fit.png)

- **Panel A (Actual vs. Predicted Fit)**: The multivariate linear regression fits an optimal hyperplane across all input dimensions, achieving $R^2 = 0.8807$ on raw predictors and $R^2 = 0.9106$ with engineered features.
- **Panel B (Primary Pollutant Scaling)**: Displays the regression line of $\text{PM}_{2.5}$ against $\text{PM}_{10}$ ($Y_i = \alpha + \beta_1 \cdot \text{PM}_{10}$), exhibiting strong linear correlation ($r = +0.88$).
- **Panel C (Learned $\beta_j$ Coefficients)**: Compares the relative impact weights of each standardized pollutant, where $\text{PM}_{10}$ and $\text{CO}$ dominate positive coefficient weights.
- **Panel D (Gradient Descent Convergence)**: Shows the monotonic descent of the MSE cost function $J(\theta)$ across training iterations, demonstrating steady parameter convergence.

#### Mathematical Equation
The equation of multivariate linear regression is given by:

$$Y_i = \alpha + \beta_1 x_i^{(1)} + \beta_2 x_i^{(2)} + \dots + \beta_n x_i^{(n)}$$

Where:
- $Y_i$: Estimate of the $i$-th component of the dependent target variable ($\text{PM}_{2.5}$)
- $\alpha$: Intercept / bias parameter
- $\beta_1, \beta_2, \dots, \beta_n$: Learned regression coefficients
- $x_i^{(1)}, x_i^{(2)}, \dots, x_i^{(n)}$: The $n$ independent variables ($\text{CO}, \text{NO}_2, \text{O}_3, \text{PM}_{10}, \text{SO}_2$, and meteorological parameters)

#### Optimization via Gradient Descent
Optimization is carried out using **Gradient Descent**. Gradient descent partially derives the loss function (Mean Squared Error) with respect to each parameter, and all parameters are updated iteratively by subtracting the previous value with the derivative times the learning rate $\eta$:

$$\theta_j := \theta_j - \eta \frac{\partial J(\theta)}{\partial \theta_j}$$

Where the Mean Squared Error loss function is:
$$J(\alpha, \boldsymbol{\beta}) = \frac{1}{2m} \sum_{i=1}^m \left( Y_i - \left( \alpha + \sum_{j=1}^n \beta_j x_i^{(j)} \right) \right)^2$$

The learning rate is tuned by the trial and error / cross-validation method to prevent divergence while ensuring fast convergence. In our benchmark, an $L_2$ regularized penalty (Ridge Regression, $\alpha=100$) was applied to suppress multicollinearity across correlated pollutants ($\text{CO}$, $\text{NO}_2$, and $\text{PM}_{10}$).

#### End-to-End Workflow Pipeline
```mermaid
flowchart TD
    subgraph DataFlow ["End-to-End Processing"]
        P["Pollutants (CO, NO2, O3, PM10, SO2)"] --> DP["Data Processing"]
        DP --> TR["Training"]
        TR --> TE["Testing"]
        TE --> PR["Predicted AQI / PM2.5"]
    end
```

---

### 5.2 Benchmark Comparison Table

To ensure the best possible model, multiple model families were systematically evaluated on a strict **out-of-time holdout validation set (the full year of 2016)** to avoid temporal leakage and replicate the 2017 test conditions.

| Model | Architecture Details | Validation RMSE ($\mu\text{g/m}^3$) | Validation MAE ($\mu\text{g/m}^3$) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Blended Ensemble** | **0.55 LightGBM + 0.45 XGBoost** | **17.37** | **10.20** | **0.9473** |
| LightGBM Regressor | GBDT, 63 leaves, lr=0.04 | 17.45 | 10.31 | 0.9469 |
| XGBoost Regressor | Hist method, depth=7, lr=0.05 | 17.55 | 10.25 | 0.9463 |
| Random Forest | 100 trees, depth=16, 50% feats | 19.56 | 11.17 | 0.9333 |
| Ridge Regression | Baseline regularized linear model | 22.64 | 14.09 | 0.9106 |

![Model Benchmark Comparison](./model_plots/01_model_benchmark_comparison.png)

---

### 5.2 Feature Importance Analysis
![Feature Importance](./model_plots/02_feature_importance.png)

- **Dominant Predictors**:
  1. `PM10` and `PM10_roll_mean_3`: Account for the largest gain in explaining fine particulate variance.
  2. `CO` and `CO_roll_mean_6`: Essential combustion tracers capturing primary heating/traffic spikes.
  3. `reg_mean_PM10` & `reg_mean_CO`: The regional city-wide mean level acts as a strong anchor for synoptic haze episodes.
  4. `RH` (Relative Humidity) and `dew_depression`: Regulates secondary aerosol formation and moisture-induced particulate swelling.
  5. `WSPM_roll_mean_24` and `u_wind`: Dispersal dynamics and wind direction.

---

### 5.3 Diagnostic Residual Analysis & Station Breakdown
![Actual vs Predicted and Residuals](./model_plots/03_actual_vs_predicted_residuals.png)

![Station RMSE Breakdown](./model_plots/04_station_rmse_breakdown.png)

- **Prediction Fidelity**: Hexbin plot displays tight concentration along the identity line ($y = \hat{y}$) across the full dynamic range from $2\ \mu\text{g/m}^3$ to over $500\ \mu\text{g/m}^3$.
- **Residual Distribution**: Centered symmetrically at zero ($\text{mean bias} = +0.13\ \mu\text{g/m}^3$), indicating no systematic under- or over-prediction bias.
- **Station-Level Consistency**:
  - Lowest RMSE: **Dingling** ($14.2\ \mu\text{g/m}^3$) and **Huairou** ($15.1\ \mu\text{g/m}^3$).
  - Highest RMSE: **Wanshouxigong** ($19.4\ \mu\text{g/m}^3$) and **Gucheng** ($19.1\ \mu\text{g/m}^3$), where heavy local industrial and traffic emissions induce localized variance.

---

## 6. Final Test Set Predictions & Deliverables

![Test Predictions Timeseries](./model_plots/05_test_predictions_timeseries.png)

The production models were trained on 100% of the training data (all 4 years: 2013-2016, 395,301 samples) and evaluated on the official test set (`beijing_test.csv`):

- **Row Count**: Exactly **16,728 rows**, strictly aligned row-for-row with `sample_submission.csv.xlsx`.
- **Missing / Infinite Values**: **0 nulls**, all predictions are finite positive floats.
- **Physical Plausibility Constraints**: Lower-bounded at the historical minimum ($2.0\ \mu\text{g/m}^3$).
- **Test Summary Statistics**:
  - $\text{Mean}: 92.04\ \mu\text{g/m}^3$
  - $\text{Median}: 47.20\ \mu\text{g/m}^3$
  - $\text{Standard Deviation}: 109.33\ \mu\text{g/m}^3$
  - $\text{Range}: [2.59\ \mu\text{g/m}^3, 674.30\ \mu\text{g/m}^3]$
  - (Closely matches historical Beijing January-February winter distributions where mean $\text{PM}_{2.5}$ fluctuates between 90 and 100 $\mu\text{g/m}^3$).

### Generated Deliverable Files
1. **[submission.csv](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/submission.csv)**: Standard headerless CSV matching `sample_submission.csv.xlsx` (`REF_NO, PM2.5`).
2. **[submission_with_header.csv](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/submission_with_header.csv)**: Labeled CSV with explicit column names (`REF_NO, PM2.5`).
3. **[submission.xlsx](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/submission.xlsx)**: Excel spreadsheet matching `sample_submission.csv.xlsx`.
4. **[benchmark_comparison.csv](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/model_plots/benchmark_comparison.csv)**: Metrics table comparing all 5 models.
5. **[feature_importance.csv](file:///c:/Users/ABCOM/Desktop/group-3-beijing-air/model_plots/feature_importance.csv)**: Ranked feature importances.
