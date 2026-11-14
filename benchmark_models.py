import pandas as pd
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

os.makedirs('model_plots', exist_ok=True)
print("=== STARTING COMPREHENSIVE FEATURE ENGINEERING & MODEL BENCHMARK ===")
start_time = time.time()

# 1. Load Data
train = pd.read_csv('beijing_train.csv')
test = pd.read_csv('beijing_test.csv')

train['datetime'] = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
test['datetime'] = pd.to_datetime(test[['year', 'month', 'day', 'hour']])

test['PM2.5'] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(['station', 'datetime']).reset_index(drop=True)

# 2. Wind Direction & Aerodynamics
wd_angles = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}
combined['wd_angle'] = combined['wd'].map(wd_angles)
combined['sin_wd'] = np.sin(np.radians(combined['wd_angle']))
combined['cos_wd'] = np.cos(np.radians(combined['wd_angle']))
combined['u_wind'] = -combined['WSPM'] * combined['sin_wd'] # East-West
combined['v_wind'] = -combined['WSPM'] * combined['cos_wd'] # North-South

# 3. Atmospheric Thermodynamics & Moisture
def calc_rh(temp, dewp):
    a, b = 17.625, 243.04
    alpha = (a * temp) / (b + temp)
    beta = (a * dewp) / (b + dewp)
    rh = 100.0 * np.exp(beta - alpha)
    return np.clip(rh, 0, 100)

combined['RH'] = calc_rh(combined['TEMP'], combined['DEWP'])
combined['dew_depression'] = combined['TEMP'] - combined['DEWP']

# Saturation and actual vapor pressure (kPa)
es = 0.61078 * np.exp((17.27 * combined['TEMP']) / (combined['TEMP'] + 237.3))
ea = 0.61078 * np.exp((17.27 * combined['DEWP']) / (combined['DEWP'] + 237.3))
combined['VPD'] = np.maximum(0, es - ea)
combined['air_density'] = combined['PRES'] / (2.87 * (combined['TEMP'] + 273.15))

# Atmospheric dispersion / ventilation
combined['dispersion_factor'] = combined['WSPM'] / (combined['dew_depression'] + 1.0)
combined['stagnation_flag'] = ((combined['WSPM'] < 1.5) & (combined['RH'] > 70)).astype(int)

# 4. Temporal & Calendar Features
combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24.0)
combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24.0)
combined['month_sin'] = np.sin(2 * np.pi * combined['month'] / 12.0)
combined['month_cos'] = np.cos(2 * np.pi * combined['month'] / 12.0)
doy = combined['datetime'].dt.day_of_year
combined['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
combined['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
combined['dayofweek'] = combined['datetime'].dt.day_of_week
combined['is_weekend'] = (combined['dayofweek'] >= 5).astype(int)
combined['is_heating'] = ((combined['month'] >= 11) | (combined['month'] <= 3)).astype(int)

# 5. Chemical Indices & Ratios
combined['pm10_co_ratio'] = combined['PM10'] / (combined['CO'] + 1.0)
combined['pm10_so2_ratio'] = combined['PM10'] / (combined['SO2'] + 1.0)
combined['pm10_no2_ratio'] = combined['PM10'] / (combined['NO2'] + 1.0)
combined['so2_no2_ratio'] = combined['SO2'] / (combined['NO2'] + 1.0)
combined['co_no2_ratio'] = combined['CO'] / (combined['NO2'] + 1.0)
combined['o3_no2_ratio'] = combined['O3'] / (combined['NO2'] + 1.0)
combined['total_oxidant'] = combined['O3'] + combined['NO2']
combined['pm10_rh_inter'] = combined['PM10'] * (combined['RH'] / 100.0)
combined['pm10_temp_inter'] = combined['PM10'] * combined['TEMP']
combined['pm10_wspm_inter'] = combined['PM10'] / (combined['WSPM'] + 0.1)
combined['co_wspm_inter'] = combined['CO'] / (combined['WSPM'] + 0.1)

# 6. Station Geographic Metadata
station_coords = {
    'Dongsi': (39.929, 116.417),
    'Tiantan': (39.886, 116.407),
    'Guanyuan': (39.929, 116.339),
    'Wanshouxigong': (39.878, 116.352),
    'Aotizhongxin': (39.982, 116.397),
    'Nongzhanguan': (39.937, 116.461),
    'Wanliu': (39.987, 116.287),
    'Gucheng': (39.914, 116.184),
    'Shunyi': (40.127, 116.655),
    'Changping': (40.217, 116.230),
    'Dingling': (40.292, 116.220),
    'Huairou': (40.328, 116.628)
}
combined['lat'] = combined['station'].map(lambda s: station_coords[s][0])
combined['lon'] = combined['station'].map(lambda s: station_coords[s][1])
combined['dist_center'] = np.sqrt((combined['lat'] - 39.9042)**2 + (combined['lon'] - 116.4074)**2) * 111.0

# 7. Regional City-Wide Synoptic Means & Station Anomalies
regional_cols = ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'PRES', 'WSPM']
hourly_regional = combined.groupby('datetime')[regional_cols].transform('mean')
hourly_regional.columns = [f'regional_mean_{c}' for c in regional_cols]
combined = pd.concat([combined, hourly_regional], axis=1)

combined['pm10_anomaly'] = combined['PM10'] - combined['regional_mean_PM10']
combined['co_anomaly'] = combined['CO'] - combined['regional_mean_CO']
combined['no2_anomaly'] = combined['NO2'] - combined['regional_mean_NO2']

# 8. Temporal Lags and Multi-scale Rolling Aggregations per Station
tracking_cols = ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'PRES', 'WSPM']
for col in tracking_cols:
    combined[f'{col}_lag1'] = combined.groupby('station')[col].shift(1)
    combined[f'{col}_lag2'] = combined.groupby('station')[col].shift(2)
    combined[f'{col}_diff1'] = combined[col] - combined[f'{col}_lag1']
    combined[f'{col}_diff3'] = combined[col] - combined.groupby('station')[col].shift(3)
    combined[f'{col}_roll_mean_3'] = combined.groupby('station')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    combined[f'{col}_roll_mean_6'] = combined.groupby('station')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
    combined[f'{col}_roll_std_6'] = combined.groupby('station')[col].transform(lambda x: x.rolling(6, min_periods=1).std())
    combined[f'{col}_roll_mean_12'] = combined.groupby('station')[col].transform(lambda x: x.rolling(12, min_periods=1).mean())
    combined[f'{col}_roll_mean_24'] = combined.groupby('station')[col].transform(lambda x: x.rolling(24, min_periods=1).mean())
    combined[f'{col}_roll_max_24'] = combined.groupby('station')[col].transform(lambda x: x.rolling(24, min_periods=1).max())
    combined[f'{col}_roll_min_24'] = combined.groupby('station')[col].transform(lambda x: x.rolling(24, min_periods=1).min())

# Frequency encoding for station
station_freq = combined['station'].value_counts(normalize=True).to_dict()
combined['station_freq'] = combined['station'].map(station_freq)

# Categorical mapping for tree models
combined['station_code'] = combined['station'].astype('category').cat.codes
combined['wd_code'] = combined['wd'].astype('category').cat.codes

# Split back
train_df = combined[combined['REF_NO'].isin(train['REF_NO'])].sort_values('REF_NO').reset_index(drop=True)
test_df = combined[combined['REF_NO'].isin(test['REF_NO'])].sort_values('REF_NO').reset_index(drop=True)

# Drop missing target in train
train_df_clean = train_df.dropna(subset=['PM2.5']).copy()

# Feature column selection (exclude IDs and leakable strings)
excluded_cols = ['REF_NO', 'PM2.5', 'datetime', 'station', 'wd', 'year']
feature_cols = [c for c in train_df_clean.columns if c not in excluded_cols]
print(f"Total features created: {len(feature_cols)}")

# Split into Out-of-Time Train (2013-2015) and Validation (2016)
tr_mask = train_df_clean['year'] < 2016
val_mask = train_df_clean['year'] == 2016

X_tr = train_df_clean.loc[tr_mask, feature_cols].copy()
y_tr = train_df_clean.loc[tr_mask, 'PM2.5'].copy()

X_val = train_df_clean.loc[val_mask, feature_cols].copy()
y_val = train_df_clean.loc[val_mask, 'PM2.5'].copy()

print(f"Training split: {X_tr.shape}, Validation split: {X_val.shape}")

# Impute for linear model (median imputation)
medians = X_tr.median()
X_tr_imp = X_tr.fillna(medians)
X_val_imp = X_val.fillna(medians)

models_eval = {}

# Model 1: Baseline Ridge Regression
print("\n--- 1. Evaluating Ridge Regression Baseline ---")
ridge = Ridge(alpha=100.0)
ridge.fit(X_tr_imp, y_tr)
y_pred_ridge = np.clip(ridge.predict(X_val_imp), 0, None)
models_eval['Ridge'] = {
    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_ridge)),
    'mae': mean_absolute_error(y_val, y_pred_ridge),
    'r2': r2_score(y_val, y_pred_ridge),
    'preds': y_pred_ridge
}
print(f"Ridge -> RMSE: {models_eval['Ridge']['rmse']:.4f}, MAE: {models_eval['Ridge']['mae']:.4f}, R²: {models_eval['Ridge']['r2']:.4f}")

# Model 2: Random Forest Regressor (Sample for efficiency)
print("\n--- 2. Evaluating Random Forest Regressor ---")
rf = RandomForestRegressor(n_estimators=100, max_depth=16, max_features=0.5, random_state=42, n_jobs=4)
# Train on 80,000 random samples to train in reasonable time
rf_sample_idx = np.random.RandomState(42).choice(len(X_tr_imp), size=min(80000, len(X_tr_imp)), replace=False)
rf.fit(X_tr_imp.iloc[rf_sample_idx], y_tr.iloc[rf_sample_idx])
y_pred_rf = np.clip(rf.predict(X_val_imp), 0, None)
models_eval['Random Forest'] = {
    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_rf)),
    'mae': mean_absolute_error(y_val, y_pred_rf),
    'r2': r2_score(y_val, y_pred_rf),
    'preds': y_pred_rf
}
print(f"Random Forest -> RMSE: {models_eval['Random Forest']['rmse']:.4f}, MAE: {models_eval['Random Forest']['mae']:.4f}, R²: {models_eval['Random Forest']['r2']:.4f}")

# Model 3: LightGBM Regressor
print("\n--- 3. Evaluating LightGBM Regressor ---")
lgb_tr_data = lgb.Dataset(X_tr, label=y_tr)
lgb_val_data = lgb.Dataset(X_val, label=y_val, reference=lgb_tr_data)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.04,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 30,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'seed': 42,
    'verbose': -1,
    'n_jobs': 4
}

lgb_model = lgb.train(
    lgb_params,
    lgb_tr_data,
    num_boost_round=1500,
    valid_sets=[lgb_tr_data, lgb_val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)]
)

y_pred_lgb = np.clip(lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration), 0, None)
models_eval['LightGBM'] = {
    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_lgb)),
    'mae': mean_absolute_error(y_val, y_pred_lgb),
    'r2': r2_score(y_val, y_pred_lgb),
    'preds': y_pred_lgb
}
print(f"LightGBM -> RMSE: {models_eval['LightGBM']['rmse']:.4f}, MAE: {models_eval['LightGBM']['mae']:.4f}, R²: {models_eval['LightGBM']['r2']:.4f}")

# Model 4: XGBoost Regressor (Hist method)
print("\n--- 4. Evaluating XGBoost Regressor ---")
xgb_tr_data = xgb.DMatrix(X_tr, label=y_tr)
xgb_val_data = xgb.DMatrix(X_val, label=y_val)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'max_depth': 8,
    'colsample_bytree': 0.75,
    'subsample': 0.8,
    'seed': 42,
    'nthread': 4
}

xgb_model = xgb.train(
    xgb_params,
    xgb_tr_data,
    num_boost_round=1200,
    evals=[(xgb_tr_data, 'train'), (xgb_val_data, 'val')],
    early_stopping_rounds=50,
    verbose_eval=200
)

y_pred_xgb = np.clip(xgb_model.predict(xgb_val_data), 0, None)
models_eval['XGBoost'] = {
    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_xgb)),
    'mae': mean_absolute_error(y_val, y_pred_xgb),
    'r2': r2_score(y_val, y_pred_xgb),
    'preds': y_pred_xgb
}
print(f"XGBoost -> RMSE: {models_eval['XGBoost']['rmse']:.4f}, MAE: {models_eval['XGBoost']['mae']:.4f}, R²: {models_eval['XGBoost']['r2']:.4f}")

# Model 5: Weighted Blended Ensemble (LightGBM + XGBoost)
print("\n--- 5. Evaluating Blended Ensemble (LightGBM + XGBoost) ---")
best_w = 0.5
best_ensemble_rmse = 999.0
for w in np.linspace(0.1, 0.9, 17):
    blend = w * y_pred_lgb + (1 - w) * y_pred_xgb
    rmse_blend = np.sqrt(mean_squared_error(y_val, blend))
    if rmse_blend < best_ensemble_rmse:
        best_ensemble_rmse = rmse_blend
        best_w = w

y_pred_ensemble = best_w * y_pred_lgb + (1 - best_w) * y_pred_xgb
models_eval['Ensemble (LGB+XGB)'] = {
    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_ensemble)),
    'mae': mean_absolute_error(y_val, y_pred_ensemble),
    'r2': r2_score(y_val, y_pred_ensemble),
    'preds': y_pred_ensemble
}
print(f"Optimal LightGBM weight: {best_w:.2f}, XGBoost weight: {1-best_w:.2f}")
print(f"Ensemble -> RMSE: {models_eval['Ensemble (LGB+XGB)']['rmse']:.4f}, MAE: {models_eval['Ensemble (LGB+XGB)']['mae']:.4f}, R²: {models_eval['Ensemble (LGB+XGB)']['r2']:.4f}")

# Summary Table
summary_df = pd.DataFrame({
    'Model': list(models_eval.keys()),
    'RMSE': [models_eval[m]['rmse'] for m in models_eval],
    'MAE': [models_eval[m]['mae'] for m in models_eval],
    'R2': [models_eval[m]['r2'] for m in models_eval]
}).sort_values('RMSE').reset_index(drop=True)

print("\n=== MODEL BENCHMARK COMPARISON TABLE ===")
print(summary_df.to_string(index=False))
summary_df.to_csv('model_plots/model_comparison.csv', index=False)

# Visualizations:
# 1. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(summary_df['Model'], summary_df['RMSE'], color=['#1a9641', '#2ca25f', '#41b6c4', '#fdae61', '#d7191c'][::-1])
ax.set_xlabel('Validation RMSE (µg/m³)', fontsize=12)
ax.set_title('Beijing PM2.5 Prediction: Model Benchmark Comparison (Out-of-Time 2016)', fontsize=13, fontweight='bold')
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.2f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlim(0, max(summary_df['RMSE']) * 1.15)
plt.tight_layout()
plt.savefig('model_plots/01_model_rmse_comparison.png', bbox_inches='tight')
plt.close()

# 2. Feature Importance (LightGBM top 25)
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
importance.to_csv('model_plots/feature_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 8))
top25 = importance.head(25)
sns.barplot(data=top25, x='importance', y='feature', palette='viridis', ax=ax)
ax.set_title('Top 25 Feature Importances (LightGBM Gain Metric)', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Gain (Predictive Contribution)', fontsize=12)
ax.set_ylabel('Engineered Feature', fontsize=12)
plt.tight_layout()
plt.savefig('model_plots/02_feature_importance.png', bbox_inches='tight')
plt.close()

# 3. Actual vs Predicted & Residual Distribution for Ensemble
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Hexbin Actual vs Pred
sample_eval = np.random.RandomState(42).choice(len(y_val), size=30000, replace=False)
y_val_sub = y_val.iloc[sample_eval]
y_pred_sub = y_pred_ensemble[sample_eval]

hb = axes[0].hexbin(y_val_sub, y_pred_sub, gridsize=60, cmap='plasma', mincnt=1, bins='log')
axes[0].plot([0, 600], [0, 600], 'r--', linewidth=2, label='Identity Line (y = ŷ)')
axes[0].set_xlim(0, 600)
axes[0].set_ylim(0, 600)
axes[0].set_title(f'Actual vs Predicted PM2.5 (Ensemble)\nRMSE: {models_eval["Ensemble (LGB+XGB)"]["rmse"]:.2f} µg/m³, R²: {models_eval["Ensemble (LGB+XGB)"]["r2"]:.4f}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual PM2.5 (µg/m³)', fontsize=11)
axes[0].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=11)
axes[0].legend()
fig.colorbar(hb, ax=axes[0], label='log10(count)')

# Residuals
residuals = y_pred_ensemble - y_val.values
sns.histplot(residuals, bins=100, kde=True, color='#2b5c8f', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--', label='Zero Error')
axes[1].axvline(np.mean(residuals), color='green', linestyle=':', label=f'Mean Error: {np.mean(residuals):.2f}')
axes[1].set_xlim(-100, 100)
axes[1].set_title('Residual Distribution (Predicted - Actual)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual (µg/m³)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].legend()
plt.tight_layout()
plt.savefig('model_plots/03_actual_vs_predicted_residuals.png', bbox_inches='tight')
plt.close()

# 4. Station-by-Station RMSE Breakdown
stn_eval = []
for stn in train_df_clean['station'].unique():
    stn_mask = (X_val['station_code'] == train_df_clean[train_df_clean['station'] == stn]['station_code'].iloc[0])
    if stn_mask.sum() > 0:
        rmse_stn = np.sqrt(mean_squared_error(y_val[stn_mask], y_pred_ensemble[stn_mask]))
        mae_stn = mean_absolute_error(y_val[stn_mask], y_pred_ensemble[stn_mask])
        stn_eval.append({'station': stn, 'RMSE': rmse_stn, 'MAE': mae_stn, 'count': stn_mask.sum()})

stn_eval_df = pd.DataFrame(stn_eval).sort_values('RMSE')
stn_eval_df.to_csv('model_plots/station_performance.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(stn_eval_df['station'], stn_eval_df['RMSE'], color='#2b5c8f')
ax.set_title('Validation RMSE Across 12 Stations (Ensemble Model)', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE (µg/m³)', fontsize=11)
ax.set_xticklabels(stn_eval_df['station'], rotation=30, ha='right')
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 0.3, f'{y:.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('model_plots/04_station_rmse_breakdown.png', bbox_inches='tight')
plt.close()

print(f"\nAll models trained and evaluated in {time.time() - start_time:.2f} seconds!")
