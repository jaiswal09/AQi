# Group-3: Beijing Air Quality Prediction (PM2.5)
## Machine Learning Regression & Environmental Data Analytics

---

<!-- SLIDE 1 -->
# Slide 1: Title Slide
### Group-3: Beijing Air Pollution Prediction
**Target**: Hourly Particulate Matter ($\text{PM}_{2.5}$) Concentration  
**Task**: Multivariate Environmental Time-Series Regression  
**Evaluation Metric**: Root Mean Squared Error (RMSE)  
**Dataset Time Span**: 2013 – 2017 (12 Monitoring Stations in Beijing)  
**Final Best Model**: Weighted Blended Ensemble (LightGBM + XGBoost)  
**Best Validation RMSE**: **17.37 $\mu\text{g/m}^3$** ($R^2 = 0.9473$)

> **Speaker Note**: Welcome everyone. Today we present our comprehensive machine learning pipeline for Beijing air pollution forecasting. Air pollution, specifically fine particulate matter $\text{PM}_{2.5}$, has severe cardiovascular and respiratory health consequences. Our objective is to accurately predict $\text{PM}_{2.5}$ using co-pollutants and meteorological dynamics across 12 stations in Beijing.

---

<!-- SLIDE 2 -->
# Slide 2: Project Objectives & Win Conditions

### Core Challenge
Predict hourly ground-level **$\text{PM}_{2.5}$** across 12 monitoring stations in Beijing for January – February 2017 using historical observations (2013 – 2016).

### Key Deliverables
- **Exploratory Data Analysis (EDA)**: Identify physical, chemical, and atmospheric mechanisms driving smog events, backed by 8 dedicated visualizations.
- **Physics-Informed Feature Engineering**: Model atmospheric stability, wind aerodynamics, moisture thermodynamics, and chemical precursor ratios.
- **Multi-Model Benchmark**: Systematically evaluate Linear Baselines, Random Forests, XGBoost, LightGBM, and Ensembles.
- **Test Submissions**: Output 16,728 clean predictions strictly matching the test format.
- **Evaluation Metric**: Root Mean Squared Error (RMSE).

> **Speaker Note**: Our evaluation metric is RMSE. Because particulate matter spikes can reach lethal levels during winter haze episodes, minimizing large errors is both mathematically rewarded by RMSE and environmentally critical.

---

<!-- SLIDE 3 -->
# Slide 3: Dataset Architecture & Data Quality

| Characteristic | Training Set (`beijing_train.csv`) | Test Set (`beijing_test.csv`) |
| :--- | :--- | :--- |
| **Observation Period** | March 1, 2013 – December 31, 2016 | January 1, 2017 – February 28, 2017 |
| **Duration** | 46 months (~3.83 years) | 2 winter months (59 days) |
| **Total Rows** | 403,776 observations | 16,728 observations |
| **Stations Covered** | Exactly 12 stations (33,648 rows each) | Exactly 12 stations (~1,394 rows each) |
| **Predictor Columns** | 16 (year, month, day, hour, pollutants, weather) | 16 (all except target $\text{PM}_{2.5}$) |
| **Missing Data in Predictors** | $\text{CO}$ (5.0%), $\text{O}_3$ (3.2%), $\text{NO}_2$ (2.9%), $\text{SO}_2$ (2.2%), $\text{PM}_{10}$ (1.5%) | **0 missing values (100% complete)** |
| **Target Missing Values** | 8,475 rows (2.10%) &rarr; dropped for training | Target to be predicted |

### Stations Monitored (12)
- **Urban Core**: Dongsi, Tiantan, Guanyuan, Wanshouxigong, Aotizhongxin, Nongzhanguan
- **Suburban / Developing**: Wanliu, Gucheng, Shunyi, Changping
- **Regional Background / Nature**: Dingling, Huairou

> **Speaker Note**: Notice the clean structure: exactly 12 stations. The training set ends December 31, 2016 at 23:00, and the test set begins January 1, 2017 at 00:00. This continuity allows us to calculate seamless rolling lags across the train-test boundary.

---

<!-- SLIDE 4 -->
# Slide 4: Target Variable ($\text{PM}_{2.5}$) Insights

### Distribution Characteristics
- **Mean Concentration**: $79.25\ \mu\text{g/m}^3$ (WHO Annual guideline is $5\ \mu\text{g/m}^3$; WHO Interim Target-1 is $35\ \mu\text{g/m}^3$).
- **Median Concentration**: $55.00\ \mu\text{g/m}^3$ (reflecting heavy right-skewness).
- **Extreme Range**: Minimum $2.0\ \mu\text{g/m}^3$ to Maximum $999.0\ \mu\text{g/m}^3$.
- **Smog Episodic Nature**: More than 68% of all recorded hours exceed the WHO 24-hour advisory threshold ($25\ \mu\text{g/m}^3$).

### Key Visualizations Generated
- `eda_plots/01_pm25_distribution.png`: Compares raw distribution against $\log(1 + \text{PM}_{2.5})$.
- `eda_plots/08_pm_ratio_seasonal.png`: Fine-to-coarse ratio ($\text{PM}_{2.5} / \text{PM}_{10}$).

> **Speaker Note**: While median $\text{PM}_{2.5}$ is 55, during severe winter stagnation events it frequently exceeds 400. In spring, coarse dust brings the $\text{PM}_{2.5}/\text{PM}_{10}$ ratio down to 0.55, but in winter, fine combustion aerosols push this ratio up to 0.80+.

---

<!-- SLIDE 5 -->
# Slide 5: Multi-Pollutant & Chemical Synergy

### Correlation Highlights with $\text{PM}_{2.5}$
- **$\text{PM}_{10}$ ($r = +0.88$)**: $\text{PM}_{2.5}$ is physically embedded within $\text{PM}_{10}$. Strongest single linear predictor.
- **$\text{CO}$ ($r = +0.79$)**: Carbon monoxide acts as an inert chemical tracer for incomplete combustion (coal boilers and vehicle exhaust).
- **$\text{NO}_2$ ($r = +0.66$) & $\text{SO}_2$ ($r = +0.48$)**: Precursors that oxidize in the atmosphere to form secondary sulfate and nitrate fine particulates.
- **$\text{O}_3$ ($r = -0.14$)**: Inversely correlated in winter due to titration with nitric oxide ($\text{NO} + \text{O}_3 \rightarrow \text{NO}_2 + \text{O}_2$) and weak photochemistry.

### Key Visualizations Generated
- `eda_plots/02_correlation_heatmap.png`: Full Pearson correlation triangular heatmap.
- `eda_plots/03_pm25_vs_pm10_co.png`: Hexbin density maps showing the 1:1 boundary with $\text{PM}_{10}$ and linear relationship with $\text{CO}$.

> **Speaker Note**: Notice the synergy between $\text{PM}_{10}$ and $\text{CO}$. If $\text{PM}_{10}$ is high but $\text{CO}$ is low, it indicates coarse desert dust. If both are high, it indicates extreme coal/combustion winter smog. This insight informed our non-linear ratio features.

---

<!-- SLIDE 6 -->
# Slide 6: Meteorological & Aerodynamic Mechanisms

### 1. Wind Direction ("Pollution Rose")
- **Southerly Winds (S, SSW, SSE)**: Carry warm, humid, heavily polluted industrial plumes from the southern Hebei industrial corridor, damming against the northern mountains &rarr; **Highest $\text{PM}_{2.5}$ ($> 100\ \mu\text{g/m}^3$)**.
- **Northerly Winds (NW, NNW, N)**: Transport cold, dry, clean air from the Siberian/Mongolian plateau &rarr; **Rapid clearing ($< 40\ \mu\text{g/m}^3$)**.

### 2. Atmospheric Ventilation & Dispersal
- **Stagnation Threshold**: Wind speeds $< 1.5\ \text{m/s}$ lead to exponential particulate accumulation.
- **Clearing Velocity**: Wind speeds $> 4.0\ \text{m/s}$ reliably dilute and purge the Beijing basin.

### Key Visualizations Generated
- `eda_plots/06_wind_effects.png`: Compass rose bar chart and wind speed vs. $\text{PM}_{2.5}$ dispersal curve.

> **Speaker Note**: The geography of Beijing forms a semi-basin surrounded by mountains on the west and north. Southerly winds trap pollutants in the basin; northerly winds blow them out.

---

<!-- SLIDE 7 -->
# Slide 7: Temporal Cycles & Spatial Gradients

### Temporal Cycles
- **Diurnal (Hourly) Rhythm**:
  - Troughs at **14:00 – 16:00**: High daytime solar heating expands the planetary boundary layer height, facilitating vertical dispersion.
  - Peaks at **21:00 – 02:00**: Nocturnal boundary layer collapse, temperature inversions, night-time heavy diesel transit, and heating.
- **Seasonal Rhythm (The Heating Season)**:
  - Official heating season (Nov 15 – Mar 15) experiences the highest $\text{PM}_{2.5}$ due to coal/gas boilers and meteorological inversions.

### Spatial Gradients
- **Cleanest Stations**: Dingling (median 44.0) and Huairou (median 47.0) near northern reservoirs.
- **Most Polluted Stations**: Gucheng (industrial west) and Wanshouxigong / Dongsi (urban core, median > 60).

### Key Visualizations Generated
- `eda_plots/04_temporal_trends.png`: Hourly diurnal curve and monthly seasonal curve.
- `eda_plots/05_station_distribution.png`: Station boxplots ordered by median concentration.
- `eda_plots/07_yearly_trends.png`: Yearly trends demonstrating policy impacts.

> **Speaker Note**: Because the test dataset is in January and February 2017, it falls entirely within Beijing's heating season. This made seasonal modeling essential.

---

<!-- SLIDE 8 -->
# Slide 8: Domain-Specific Feature Engineering (101 Features)

We engineered **101 domain-specific features** categorized into 5 physical domains:

### 1. Aerodynamics & Wind Vector Decomposition
- Zonal (East-West) component: $u = -\text{WSPM} \times \sin(\theta)$
- Meridional (North-South) component: $v = -\text{WSPM} \times \cos(\theta)$
- Directional harmonics: $\sin(\theta), \cos(\theta)$

### 2. Moisture Thermodynamics (Magnus-Tetens Equation)
- **Relative Humidity (RH)**: $\text{RH} = 100 \times \exp(\beta - \alpha)$
- **Dew Point Depression**: $\text{TEMP} - \text{DEWP}$ (fog/haze saturation indicator)
- **Vapor Pressure Deficit (VPD)**: $e_s(T) - e_a(T_{dew})$
- **Ventilation Index**: $\text{WSPM} \times (\text{TEMP} - \text{DEWP} + 1.0)$
- **Stagnation Indicator**: Binary flag $(\text{WSPM} < 1.5\ \text{m/s}) \land (\text{RH} > 65\%)$

### 3. Chemical Interactions & Precursor Ratios
- Combustion ratios: $\frac{\text{PM}_{10}}{\text{CO} + 1}$, $\frac{\text{SO}_2}{\text{CO} + 1}$, $\frac{\text{NO}_2}{\text{CO} + 1}$
- Photochemical titration index: $\frac{\text{O}_3}{\text{NO}_2 + 1}$, Total Oxidant $\text{O}_x = \text{O}_3 + \text{NO}_2$
- Secondary aerosol growth interaction: $\text{PM}_{10} \times \left(\frac{\text{RH}}{100}\right)$

### 4. Regional Synoptic Aggregations & Spatial Geography
- City-wide hourly means: $\overline{\text{PM}}_{10, city}, \overline{\text{CO}}_{city}, \overline{\text{TEMP}}_{city}$ (captures regional smog blankets)
- Station anomalies: $\Delta \text{PM}_{10} = \text{PM}_{10, stn} - \overline{\text{PM}}_{10, city}$ (isolates local source emissions)
- Station coordinates (latitude, longitude) & distance from Tiananmen Square ($\text{dist\_center}$)

### 5. Temporal Multi-Scale Lags & Rolling Dynamics
- Lags 1h, 2h, and 1h rate of change ($\text{diff1}$)
- Multi-scale rolling moving averages: 3h, 6h, and 24h windows
- Rolling volatility: 6-hour rolling standard deviation ($\text{roll\_std\_6}$)
- Harmonic calendar features: $\sin/\cos$ of hour, month, day of year

> **Speaker Note**: These features provide the decision trees with physics that raw sensor readings cannot provide alone, such as whether high humidity is accelerating sulfate aerosol growth.

---

<!-- SLIDE 9 -->
# Slide 9: Model Benchmarking & Performance Comparison

### Validation Strategy: Out-of-Time Holdout (Full Year 2016)
To prevent temporal data leakage and replicate the 2017 test conditions, all models were trained on 2013–2015 data and evaluated on the out-of-time 2016 dataset (103,151 samples).

| Model Family | Model Name & Hyperparameters | RMSE ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Ensemble (Top)** | **Weighted Blend (0.55 LGBM + 0.45 XGB)** | **17.37** | **10.20** | **0.9473** |
| Gradient Boosting | LightGBM Regressor (num_leaves=63, lr=0.04) | 17.45 | 10.31 | 0.9469 |
| Gradient Boosting | XGBoost Regressor (Hist, max_depth=7, lr=0.05) | 17.55 | 10.25 | 0.9463 |
| Bagging Ensemble | Random Forest Regressor (100 trees, depth=16) | 19.56 | 11.17 | 0.9333 |
| Linear Baseline | Ridge Regression ($\alpha=100$) | 22.64 | 14.09 | 0.9106 |

### Performance Insights
- **Tree-Based Superiority**: Non-linear tree ensembles reduce RMSE by **5.27 $\mu\text{g/m}^3$ (23.3%)** over regularized linear regression.
- **Ensemble Gain**: Combining LightGBM and XGBoost diversifies structural tree splits, achieving the lowest overall error (**17.37 $\mu\text{g/m}^3$**) and explaining **94.73% of total variance**.

### Key Visualizations Generated
- `model_plots/01_model_benchmark_comparison.png`: Horizontal bar chart benchmarking all 5 models.

> **Speaker Note**: Our blended ensemble achieved an $R^2$ of 0.9473. On a highly volatile environmental metric like particulate matter, explaining 94.7% of variance is considered state-of-the-art.

---

<!-- SLIDE 10 -->
# Slide 10: Model Interpretability & Feature Importance

### Top 10 Most Influential Features (LightGBM Gain)
1. **`PM10` & `PM10_roll_mean_3`**: Primary mass anchor; fine particles scale closely with total particulate mass.
2. **`CO` & `CO_roll_mean_6`**: Direct combustion fingerprint; differentiates clean dust from toxic combustion smog.
3. **`reg_mean_PM10` & `reg_mean_CO`**: Captures regional synoptic weather fronts moving across Northern China.
4. **`RH` & `dew_depression`**: Regulates secondary hygroscopic particle growth and fog/haze chemistry.
5. **`WSPM_roll_mean_24` & `u_wind`**: Macro ventilation index over the preceding day.
6. **`total_oxidant` ($\text{O}_x = \text{O}_3 + \text{NO}_2$)**: Atmospheric oxidation potential.

### Key Visualizations Generated
- `model_plots/02_feature_importance.png`: Top 25 feature importance gain ranking.

> **Speaker Note**: The model did not rely solely on instantaneous values. Rolling 3-hour and 6-hour moving averages and regional city-wide levels were ranked higher than raw temperature or pressure.

---

<!-- SLIDE 11 -->
# Slide 11: Error Diagnostics & Station-Level Reliability

### Residual Diagnostics
- **Symmetric Centering**: Residuals ($\hat{y} - y$) are centered at $+0.13\ \mu\text{g/m}^3$, confirming zero systematic bias.
- **Dynamic Range Tracking**: Hexbin actual vs. predicted plot follows the $y = x$ identity line accurately across low, medium, and extreme smog ranges ($> 500\ \mu\text{g/m}^3$).

### Station-by-Station RMSE Breakdown
- **Lowest Error**: Dingling ($14.2\ \mu\text{g/m}^3$) and Huairou ($15.1\ \mu\text{g/m}^3$) &rarr; Stable background air with fewer micro-local emissions.
- **Moderate Error**: Tiantan ($16.8\ \mu\text{g/m}^3$), Aotizhongxin ($17.1\ \mu\text{g/m}^3$), Shunyi ($17.3\ \mu\text{g/m}^3$).
- **Higher Error**: Wanshouxigong ($19.4\ \mu\text{g/m}^3$) and Gucheng ($19.1\ \mu\text{g/m}^3$) &rarr; Intense urban road networks and heavy legacy industrial sources.

### Key Visualizations Generated
- `model_plots/03_actual_vs_predicted_residuals.png`: Dual-panel hexbin scatter and residual histogram.
- `model_plots/04_station_rmse_breakdown.png`: Bar chart of RMSE across all 12 stations.

> **Speaker Note**: The model is resilient across all 12 stations. Even in the most volatile urban stations like Gucheng, RMSE remains under 20 $\mu\text{g/m}^3$.

---

<!-- SLIDE 12 -->
# Slide 12: Final Production Model & Test Deliverables

### Production Training Strategy
- Models retrained on **100% of historical data** (March 2013 – December 2016, 395,301 valid rows) using optimal weights (0.55 LightGBM + 0.45 XGBoost).
- Generates predictions on the unseen test dataset (`beijing_test.csv`: Jan 1 – Feb 28, 2017).

### Test Prediction Verification & Quality
- **Exact Shape**: Exactly **16,728 rows**, 0 missing values.
- **Row-by-Row Alignment**: Strictly aligned with `REF_NO` and station sequence of `sample_submission.csv.xlsx`.
- **Physical Bounding**: Predictions clipped to $\ge 2.0\ \mu\text{g/m}^3$.
- **Test Distribution Stats**:
  - Mean: $92.04\ \mu\text{g/m}^3$ (consistent with severe winter 2017 smog).
  - Median: $47.20\ \mu\text{g/m}^3$.
  - Range: $2.59\ \mu\text{g/m}^3$ to $674.30\ \mu\text{g/m}^3$.

### Deliverable Files Generated
- `submission.csv`: Standard headerless CSV matching the competition format.
- `submission_with_header.csv`: Labeled CSV format (`REF_NO, PM2.5`).
- `submission.xlsx`: Excel workbook format.
- `model_plots/05_test_predictions_timeseries.png`: Prediction timeline for test period.

> **Speaker Note**: The test predictions exhibit the exact episodic wave characteristics of Beijing winter smog: sharp spikes during multi-day stagnant inversions, followed by rapid clearing as northerly cold fronts arrive.

---

<!-- SLIDE 13 -->
# Slide 13: Environmental Impact & Practical Application

### Operational Real-World Deployment
1. **Public Health Early Warnings**: Enables municipal health authorities to issue school closure and outdoor activity advisories 24 hours in advance.
2. **Dynamic Traffic & Industrial Routing**: Can guide emergency industrial curtailment and vehicle odd-even restrictions when stagnation thresholds are detected.
3. **Sensor Quality Control & Imputation**: The high correlation and ensemble structure allows detection of faulty physical sensors when recorded $\text{PM}_{2.5}$ deviates by $> 3\sigma$ from predictions.

### Key Takeaways
- **Physics + ML Synergy**: Combining atmospheric domain physics (relative humidity, wind vectors, combustion ratios) with gradient boosting yielded a 23.3% performance boost over baselines.
- **Robust Generalization**: Rigorous out-of-time validation ensured zero temporal leakage, proving model reliability for operational forecasting.

> **Speaker Note**: By integrating atmospheric physics into machine learning, we built an accurate, explainable system ready for municipal environmental monitoring.

---

<!-- SLIDE 14 -->
# Slide 14: Q&A & Thank You
### Group-3: Beijing Air Pollution Prediction
**Final Summary**:
- **Dataset**: 2013–2017 across 12 Beijing Monitoring Stations
- **Winning Model**: 0.55 LightGBM + 0.45 XGBoost Ensemble
- **Validation RMSE**: **17.37 $\mu\text{g/m}^3$** ($R^2 = 0.9473$)
- **Status**: Production Submissions & Artifacts Generated and Verified

**Open for Questions and Discussion!**
