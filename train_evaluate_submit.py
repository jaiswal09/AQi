import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(base_dir, 'model_plots'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'eda_plots'), exist_ok=True)

print("="*70)
print(" BEIJING AIR POLLUTION PREDICTION (PM2.5) - MASTER PRODUCTION PIPELINE ")
print("="*70)
total_start = time.time()

# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------
print("\n[Step 1/6] Loading Datasets...")
train_path = os.path.join(base_dir, 'beijing_train.csv')
test_path = os.path.join(base_dir, 'beijing_test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train Raw Shape: {train.shape}")
print(f"Test Raw Shape:  {test.shape}")

train['datetime'] = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
test['datetime'] = pd.to_datetime(test[['year', 'month', 'day', 'hour']])

test['PM2.5'] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(['station', 'datetime']).reset_index(drop=True)

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
print("\n[Step 2/6] Engineering Domain-Specific Atmospheric & Chemical Features...")

# A. Aerodynamics & Wind Vector Decomposition
wd_angles = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
    'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
    'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
    'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
}
combined['wd_angle'] = combined['wd'].map(wd_angles)
combined['sin_wd'] = np.sin(np.radians(combined['wd_angle']))
combined['cos_wd'] = np.cos(np.radians(combined['wd_angle']))
# Zonal (u) and Meridional (v) wind speed components
combined['u_wind'] = -combined['WSPM'] * combined['sin_wd'] # East-West wind
combined['v_wind'] = -combined['WSPM'] * combined['cos_wd'] # North-South wind

# B. Atmospheric Thermodynamics & Humidity (Magnus-Tetens Formula)
a, b = 17.625, 243.04
alpha = (a * combined['TEMP']) / (b + combined['TEMP'])
beta = (a * combined['DEWP']) / (b + combined['DEWP'])
combined['RH'] = np.clip(100.0 * np.exp(beta - alpha), 0.0, 100.0)
combined['dew_depression'] = combined['TEMP'] - combined['DEWP']

# Vapor pressures (kPa) & Vapor Pressure Deficit (VPD)
es = 0.61078 * np.exp((17.27 * combined['TEMP']) / (combined['TEMP'] + 237.3))
ea = 0.61078 * np.exp((17.27 * combined['DEWP']) / (combined['DEWP'] + 237.3))
combined['VPD'] = np.maximum(0.0, es - ea)
combined['air_density'] = combined['PRES'] / (2.87 * (combined['TEMP'] + 273.15))

# Atmospheric stagnation & ventilation index
combined['ventilation_index'] = combined['WSPM'] * (combined['dew_depression'] + 1.0)
combined['stagnation_flag'] = ((combined['WSPM'] < 1.5) & (combined['RH'] > 65.0)).astype(int)

# C. Temporal & Solar Cyclical Features
combined['hour_sin'] = np.sin(2.0 * np.pi * combined['hour'] / 24.0)
combined['hour_cos'] = np.cos(2.0 * np.pi * combined['hour'] / 24.0)
combined['month_sin'] = np.sin(2.0 * np.pi * combined['month'] / 12.0)
combined['month_cos'] = np.cos(2.0 * np.pi * combined['month'] / 12.0)
doy = combined['datetime'].dt.day_of_year
combined['doy_sin'] = np.sin(2.0 * np.pi * doy / 365.25)
combined['doy_cos'] = np.cos(2.0 * np.pi * doy / 365.25)
combined['dayofweek'] = combined['datetime'].dt.day_of_week
combined['is_weekend'] = (combined['dayofweek'] >= 5).astype(int)
# Official Beijing Heating Season (Nov 15 - Mar 15)
combined['is_heating'] = ((combined['month'] >= 11) | (combined['month'] <= 3)).astype(int)

# D. Chemical Ratios & Multi-Pollutant Interactions
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

# E. Spatial Station Geography
station_coords = {
    'Dongsi': (39.929, 116.417), 'Tiantan': (39.886, 116.407),
    'Guanyuan': (39.929, 116.339), 'Wanshouxigong': (39.878, 116.352),
    'Aotizhongxin': (39.982, 116.397), 'Nongzhanguan': (39.937, 116.461),
    'Wanliu': (39.987, 116.287), 'Gucheng': (39.914, 116.184),
    'Shunyi': (40.127, 116.655), 'Changping': (40.217, 116.230),
    'Dingling': (40.292, 116.220), 'Huairou': (40.328, 116.628)
}
combined['lat'] = combined['station'].map(lambda s: station_coords[s][0])
combined['lon'] = combined['station'].map(lambda s: station_coords[s][1])
combined['dist_center'] = np.sqrt((combined['lat'] - 39.9042)**2 + (combined['lon'] - 116.4074)**2) * 111.0

# F. Regional Synoptic City-Wide Means & Station Anomalies
reg_cols = ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'PRES', 'WSPM']
reg_df = combined.groupby('datetime')[reg_cols].transform('mean')
reg_df.columns = [f'reg_mean_{c}' for c in reg_cols]
combined = pd.concat([combined, reg_df], axis=1)

combined['pm10_anomaly'] = combined['PM10'] - combined['reg_mean_PM10']
combined['co_anomaly'] = combined['CO'] - combined['reg_mean_CO']
combined['no2_anomaly'] = combined['NO2'] - combined['reg_mean_NO2']

# G. Temporal Lags and Multi-Scale Rolling Statistics (Station-Wise)
lag_features = {}
for col in ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'WSPM']:
    grp = combined.groupby('station')[col]
    lag_features[f'{col}_lag1'] = grp.shift(1)
    lag_features[f'{col}_lag2'] = grp.shift(2)
    lag_features[f'{col}_diff1'] = combined[col] - grp.shift(1)
    lag_features[f'{col}_roll_mean_3'] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    lag_features[f'{col}_roll_mean_6'] = grp.transform(lambda x: x.rolling(6, min_periods=1).mean())
    lag_features[f'{col}_roll_std_6'] = grp.transform(lambda x: x.rolling(6, min_periods=1).std())
    lag_features[f'{col}_roll_mean_24'] = grp.transform(lambda x: x.rolling(24, min_periods=1).mean())

lag_df = pd.DataFrame(lag_features, index=combined.index)
combined = pd.concat([combined, lag_df], axis=1)

# Categorical encodings
combined['station_code'] = combined['station'].astype('category').cat.codes
combined['wd_code'] = combined['wd'].astype('category').cat.codes

# Split back to train and test
train_df = combined[combined['REF_NO'].isin(train['REF_NO'])].sort_values('REF_NO').reset_index(drop=True)
test_df = combined[combined['REF_NO'].isin(test['REF_NO'])].sort_values('REF_NO').reset_index(drop=True)

# Clean training set (drop rows with missing target PM2.5)
train_clean = train_df.dropna(subset=['PM2.5']).copy()

# Feature list
drop_cols = ['REF_NO', 'PM2.5', 'datetime', 'station', 'wd', 'year']
feature_cols = [c for c in train_clean.columns if c not in drop_cols]
print(f"Total Model Features: {len(feature_cols)}")
print(f"Clean Training Samples: {len(train_clean)}")
print(f"Test Samples to Predict: {len(test_df)}")

# ---------------------------------------------------------
# 3. OUT-OF-TIME VALIDATION & BENCHMARK COMPARISON
# ---------------------------------------------------------
print("\n[Step 3/6] Running Out-of-Time Model Benchmark (Train 2013-2015, Val 2016)...")
tr_mask = train_clean['year'] < 2016
val_mask = train_clean['year'] == 2016

X_tr = train_clean.loc[tr_mask, feature_cols].copy()
y_tr = train_clean.loc[tr_mask, 'PM2.5'].copy()
X_val = train_clean.loc[val_mask, feature_cols].copy()
y_val = train_clean.loc[val_mask, 'PM2.5'].copy()

# Benchmark 1: Ridge
medians = X_tr.median()
X_tr_imp = X_tr.fillna(medians)
X_val_imp = X_val.fillna(medians)

ridge = Ridge(alpha=100.0)
ridge.fit(X_tr_imp, y_tr)
pred_ridge = np.clip(ridge.predict(X_val_imp), 2.0, None)
rmse_ridge = np.sqrt(mean_squared_error(y_val, pred_ridge))
mae_ridge = mean_absolute_error(y_val, pred_ridge)
r2_ridge = r2_score(y_val, pred_ridge)

# Benchmark 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=16, max_features=0.5, random_state=42, n_jobs=4)
rf_idx = np.random.RandomState(42).choice(len(X_tr_imp), size=min(80000, len(X_tr_imp)), replace=False)
rf.fit(X_tr_imp.iloc[rf_idx], y_tr.iloc[rf_idx])
pred_rf = np.clip(rf.predict(X_val_imp), 2.0, None)
rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))
mae_rf = mean_absolute_error(y_val, pred_rf)
r2_rf = r2_score(y_val, pred_rf)

# Benchmark 3: LightGBM
lgb_tr = lgb.Dataset(X_tr, label=y_tr)
lgb_va = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.04,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'seed': 42,
    'verbose': -1,
    'n_jobs': 4
}
m_lgb = lgb.train(lgb_params, lgb_tr, num_boost_round=1500, valid_sets=[lgb_tr, lgb_va],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
pred_lgb = np.clip(m_lgb.predict(X_val), 2.0, None)
rmse_lgb = np.sqrt(mean_squared_error(y_val, pred_lgb))
mae_lgb = mean_absolute_error(y_val, pred_lgb)
r2_lgb = r2_score(y_val, pred_lgb)

# Benchmark 4: XGBoost
xgb_tr = xgb.DMatrix(X_tr, label=y_tr)
xgb_va = xgb.DMatrix(X_val, label=y_val)
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'max_depth': 7,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'seed': 42,
    'nthread': 4
}
m_xgb = xgb.train(xgb_params, xgb_tr, num_boost_round=1200, evals=[(xgb_tr, 'train'), (xgb_va, 'val')],
                  early_stopping_rounds=50, verbose_eval=False)
pred_xgb = np.clip(m_xgb.predict(xgb_va), 2.0, None)
rmse_xgb = np.sqrt(mean_squared_error(y_val, pred_xgb))
mae_xgb = mean_absolute_error(y_val, pred_xgb)
r2_xgb = r2_score(y_val, pred_xgb)

# Benchmark 5: Blended Ensemble
pred_ens = 0.55 * pred_lgb + 0.45 * pred_xgb
rmse_ens = np.sqrt(mean_squared_error(y_val, pred_ens))
mae_ens = mean_absolute_error(y_val, pred_ens)
r2_ens = r2_score(y_val, pred_ens)

benchmark_df = pd.DataFrame([
    {'Model': 'Ridge Regression (Baseline)', 'RMSE': rmse_ridge, 'MAE': mae_ridge, 'R2': r2_ridge},
    {'Model': 'Random Forest Regressor', 'RMSE': rmse_rf, 'MAE': mae_rf, 'R2': r2_rf},
    {'Model': 'XGBoost Regressor (Hist)', 'RMSE': rmse_xgb, 'MAE': mae_xgb, 'R2': r2_xgb},
    {'Model': 'LightGBM Regressor', 'RMSE': rmse_lgb, 'MAE': mae_lgb, 'R2': r2_lgb},
    {'Model': 'Blended Ensemble (LGB+XGB)', 'RMSE': rmse_ens, 'MAE': mae_ens, 'R2': r2_ens}
]).sort_values('RMSE').reset_index(drop=True)

print("\n--- Out-of-Time 2016 Validation Results ---")
print(benchmark_df.to_string(index=False))
benchmark_df.to_csv(os.path.join(base_dir, 'model_plots', 'benchmark_comparison.csv'), index=False)

# ---------------------------------------------------------
# 4. FULL 5-FOLD ENSEMBLE TRAINING & TEST PREDICTION
# ---------------------------------------------------------
print("\n[Step 4/6] Training Full 5-Fold Cross-Validation Ensemble on All 4 Years (2013-2016)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(train_clean))
oof_preds_xgb = np.zeros(len(train_clean))
test_preds_lgb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))

X_full = train_clean[feature_cols].reset_index(drop=True)
y_full = train_clean['PM2.5'].reset_index(drop=True)
X_test = test_df[feature_cols].reset_index(drop=True)

xgb_test = xgb.DMatrix(X_test)

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
    fold_start = time.time()
    X_f_tr, y_f_tr = X_full.iloc[train_idx], y_full.iloc[train_idx]
    X_f_va, y_f_va = X_full.iloc[val_idx], y_full.iloc[val_idx]
    
    # LightGBM Fold
    ds_tr = lgb.Dataset(X_f_tr, label=y_f_tr)
    ds_va = lgb.Dataset(X_f_va, label=y_f_va, reference=ds_tr)
    fold_lgb = lgb.train(
        lgb_params, ds_tr, num_boost_round=1600, valid_sets=[ds_tr, ds_va],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_preds_lgb[val_idx] = fold_lgb.predict(X_f_va)
    test_preds_lgb += fold_lgb.predict(X_test) / 5.0
    
    # XGBoost Fold
    d_tr = xgb.DMatrix(X_f_tr, label=y_f_tr)
    d_va = xgb.DMatrix(X_f_va, label=y_f_va)
    fold_xgb = xgb.train(
        xgb_params, d_tr, num_boost_round=1300, evals=[(d_tr, 'train'), (d_va, 'val')],
        early_stopping_rounds=50, verbose_eval=False
    )
    oof_preds_xgb[val_idx] = fold_xgb.predict(d_va)
    test_preds_xgb += fold_xgb.predict(xgb_test) / 5.0
    
    fold_rmse = np.sqrt(mean_squared_error(y_f_va, 0.55 * oof_preds_lgb[val_idx] + 0.45 * oof_preds_xgb[val_idx]))
    fold_metrics.append(fold_rmse)
    print(f"  Fold {fold+1}/5 finished in {time.time() - fold_start:.1f}s - Ensemble RMSE: {fold_rmse:.4f}")

oof_blend = 0.55 * oof_preds_lgb + 0.45 * oof_preds_xgb
oof_blend = np.clip(oof_blend, 2.0, None)

cv_rmse = np.sqrt(mean_squared_error(y_full, oof_blend))
cv_mae = mean_absolute_error(y_full, oof_blend)
cv_r2 = r2_score(y_full, oof_blend)

print(f"\n>>> 5-Fold Cross-Validation Overall Results:")
print(f"    RMSE: {cv_rmse:.4f} µg/m³")
print(f"    MAE:  {cv_mae:.4f} µg/m³")
print(f"    R²:   {cv_r2:.4f}")
print(f"    Per-fold RMSE: {[round(m, 4) for m in fold_metrics]}")

# Final Test Predictions
final_test_preds = 0.55 * test_preds_lgb + 0.45 * test_preds_xgb
final_test_preds = np.clip(final_test_preds, 2.0, None)

# ---------------------------------------------------------
# 5. DIAGNOSTIC PLOTS & VISUALIZATIONS
# ---------------------------------------------------------
print("\n[Step 5/6] Generating Publication-Quality Evaluation Charts...")

# Plot 1: Model Benchmark Comparison
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#1a9641', '#41b6c4', '#2b83ba', '#fdae61', '#d7191c']
bars = ax.barh(benchmark_df['Model'], benchmark_df['RMSE'], color=colors[::-1])
ax.set_xlabel('Validation RMSE (µg/m³)', fontsize=12, fontweight='bold')
ax.set_title('Beijing Air Pollution Prediction: Model Comparison (Out-of-Time 2016)', fontsize=13, fontweight='bold')
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.2f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlim(0, max(benchmark_df['RMSE']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_plots', '01_model_benchmark_comparison.png'), bbox_inches='tight')
plt.close()

# Plot 2: Top 30 Feature Importances
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': m_lgb.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
importance_df.to_csv(os.path.join(base_dir, 'model_plots', 'feature_importance.csv'), index=False)

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(data=importance_df.head(30), x='importance', y='feature', palette='viridis', ax=ax)
ax.set_title('Top 30 Most Important Features in LightGBM (Gain Metric)', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Gain (Predictive Power)', fontsize=12, fontweight='bold')
ax.set_ylabel('Engineered Feature', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_plots', '02_feature_importance_top30.png'), bbox_inches='tight')
plt.close()

# Plot 3: Actual vs Predicted & Residuals on Full OOF Dataset
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sample_idx = np.random.RandomState(42).choice(len(y_full), size=40000, replace=False)
y_sub = y_full.iloc[sample_idx]
pred_sub = oof_blend[sample_idx]

hb = axes[0].hexbin(y_sub, pred_sub, gridsize=65, cmap='plasma', mincnt=1, bins='log')
axes[0].plot([0, 700], [0, 700], 'r--', linewidth=2, label='Perfect Identity (y = ŷ)')
axes[0].set_xlim(0, 700)
axes[0].set_ylim(0, 700)
axes[0].set_title(f'Actual vs Predicted PM2.5 (5-Fold OOF CV)\nRMSE: {cv_rmse:.2f} µg/m³, R²: {cv_r2:.4f}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
axes[0].legend()
fig.colorbar(hb, ax=axes[0], label='log10(sample count)')

residuals = oof_blend - y_full.values
sns.histplot(residuals, bins=120, kde=True, color='#2b5c8f', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--', label='Zero Error')
axes[1].axvline(np.mean(residuals), color='green', linestyle=':', label=f'Mean Bias: {np.mean(residuals):.2f}')
axes[1].set_xlim(-80, 80)
axes[1].set_title(f'Residual Distribution (Mean Error: {np.mean(residuals):.2f} µg/m³)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual (ŷ - y in µg/m³)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency Density', fontsize=11, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_plots', '03_actual_vs_predicted_residuals.png'), bbox_inches='tight')
plt.close()

# Plot 4: Station-by-Station RMSE Breakdown
stn_eval = []
for stn in train_clean['station'].unique():
    mask = (train_clean['station'] == stn).values
    stn_rmse = np.sqrt(mean_squared_error(y_full.values[mask], oof_blend[mask]))
    stn_mae = mean_absolute_error(y_full.values[mask], oof_blend[mask])
    stn_eval.append({'station': stn, 'RMSE': stn_rmse, 'MAE': stn_mae, 'Count': mask.sum()})

stn_eval_df = pd.DataFrame(stn_eval).sort_values('RMSE').reset_index(drop=True)
stn_eval_df.to_csv(os.path.join(base_dir, 'model_plots', 'station_cv_performance.csv'), index=False)

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(stn_eval_df['station'], stn_eval_df['RMSE'], color='#2b5c8f')
ax.set_title('Cross-Validation RMSE Across 12 Beijing Monitoring Stations', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE (µg/m³)', fontsize=11, fontweight='bold')
ax.set_xticklabels(stn_eval_df['station'], rotation=30, ha='right')
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 0.3, f'{y:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_plots', '04_station_rmse_breakdown.png'), bbox_inches='tight')
plt.close()

# Plot 5: Test Set Predictions Distribution and Sample Time-Series
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(final_test_preds, bins=80, kde=True, color='#08519c', ax=axes[0])
axes[0].set_title(f'Test Predictions Distribution (Jan-Feb 2017)\nMean: {np.mean(final_test_preds):.1f}, Median: {np.median(final_test_preds):.1f} µg/m³', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted PM2.5 (µg/m³)', fontsize=11, fontweight='bold')

# Sample time-series trace for one station in test
test_df_copy = test_df.copy()
test_df_copy['PM2.5_pred'] = final_test_preds
aoti_test = test_df_copy[test_df_copy['station'] == 'Aotizhongxin'].sort_values('datetime').head(24 * 14) # first 2 weeks
axes[1].plot(aoti_test['datetime'], aoti_test['PM2.5_pred'], label='Predicted PM2.5', color='#e6550d', linewidth=2)
axes[1].plot(aoti_test['datetime'], aoti_test['PM10'], label='Observed PM10', color='#3182bd', alpha=0.6, linestyle='--')
axes[1].set_title('Two-Week Prediction Timeline (Aotizhongxin Station)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Concentration (µg/m³)', fontsize=11, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_plots', '05_test_predictions_timeseries.png'), bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 6. SUBMISSION GENERATION & VERIFICATION
# ---------------------------------------------------------
print("\n[Step 6/6] Generating Deliverables & Verifying Submission Format...")
# The sample submission had REF_NO and prediction without header
submission_df = pd.DataFrame({
    'REF_NO': test_df['REF_NO'],
    'PM2.5': final_test_preds
})

# Save standard CSV with header
submission_df.to_csv(os.path.join(base_dir, 'submission_with_header.csv'), index=False)
# Save format matching sample_submission (no header, REF_NO and PM2.5)
submission_df.to_csv(os.path.join(base_dir, 'submission.csv'), index=False, header=False)
# Also save excel format matching sample_submission.csv.xlsx
submission_df.to_excel(os.path.join(base_dir, 'submission.xlsx'), index=False, header=False)

print(f"Submission shape: {submission_df.shape}")
print(f"Null values in submission: {submission_df.isnull().sum().to_dict()}")
print(f"First 5 predictions:\n{submission_df.head()}")
print(f"Last 5 predictions:\n{submission_df.tail()}")
print(f"Summary Statistics of Test PM2.5 Predictions:\n{submission_df['PM2.5'].describe()}")

eval_results = {
    'evaluation_metric': 'RMSE',
    'cv_rmse': float(cv_rmse),
    'cv_mae': float(cv_mae),
    'cv_r2': float(cv_r2),
    'per_fold_rmse': [float(x) for x in fold_metrics],
    'oot_2016_rmse_ensemble': float(rmse_ens),
    'oot_2016_rmse_lgb': float(rmse_lgb),
    'oot_2016_rmse_xgb': float(rmse_xgb),
    'oot_2016_rmse_rf': float(rmse_rf),
    'oot_2016_rmse_ridge': float(rmse_ridge),
    'total_runtime_seconds': round(time.time() - total_start, 2)
}
with open(os.path.join(base_dir, 'model_plots', 'evaluation_metrics.json'), 'w') as f:
    json.dump(eval_results, f, indent=4)

print(f"\n Master pipeline finished successfully in {time.time() - total_start:.2f} seconds!")
print("="*70)
