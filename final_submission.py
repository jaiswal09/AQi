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
import lightgbm as lgb
import xgboost as xgb

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model_plots')
os.makedirs(model_dir, exist_ok=True)

print("=== RUNNING FAST & HIGH-PRECISION PRODUCTION MODEL TRAINING ===")
start_time = time.time()

# 1. Load Data
train = pd.read_csv(os.path.join(base_dir, 'beijing_train.csv'))
test = pd.read_csv(os.path.join(base_dir, 'beijing_test.csv'))

train['datetime'] = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
test['datetime'] = pd.to_datetime(test[['year', 'month', 'day', 'hour']])

test['PM2.5'] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(['station', 'datetime']).reset_index(drop=True)

# 2. Features
wd_angles = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
    'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
    'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
    'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
}
combined['wd_angle'] = combined['wd'].map(wd_angles)
combined['sin_wd'] = np.sin(np.radians(combined['wd_angle']))
combined['cos_wd'] = np.cos(np.radians(combined['wd_angle']))
combined['u_wind'] = -combined['WSPM'] * combined['sin_wd']
combined['v_wind'] = -combined['WSPM'] * combined['cos_wd']

a, b = 17.625, 243.04
alpha = (a * combined['TEMP']) / (b + combined['TEMP'])
beta = (a * combined['DEWP']) / (b + combined['DEWP'])
combined['RH'] = np.clip(100.0 * np.exp(beta - alpha), 0.0, 100.0)
combined['dew_depression'] = combined['TEMP'] - combined['DEWP']

es = 0.61078 * np.exp((17.27 * combined['TEMP']) / (combined['TEMP'] + 237.3))
ea = 0.61078 * np.exp((17.27 * combined['DEWP']) / (combined['DEWP'] + 237.3))
combined['VPD'] = np.maximum(0.0, es - ea)
combined['air_density'] = combined['PRES'] / (2.87 * (combined['TEMP'] + 273.15))

combined['ventilation_index'] = combined['WSPM'] * (combined['dew_depression'] + 1.0)
combined['stagnation_flag'] = ((combined['WSPM'] < 1.5) & (combined['RH'] > 65.0)).astype(int)

combined['hour_sin'] = np.sin(2.0 * np.pi * combined['hour'] / 24.0)
combined['hour_cos'] = np.cos(2.0 * np.pi * combined['hour'] / 24.0)
combined['month_sin'] = np.sin(2.0 * np.pi * combined['month'] / 12.0)
combined['month_cos'] = np.cos(2.0 * np.pi * combined['month'] / 12.0)
doy = combined['datetime'].dt.day_of_year
combined['doy_sin'] = np.sin(2.0 * np.pi * doy / 365.25)
combined['doy_cos'] = np.cos(2.0 * np.pi * doy / 365.25)
combined['dayofweek'] = combined['datetime'].dt.day_of_week
combined['is_weekend'] = (combined['dayofweek'] >= 5).astype(int)
combined['is_heating'] = ((combined['month'] >= 11) | (combined['month'] <= 3)).astype(int)

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

reg_cols = ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'PRES', 'WSPM']
reg_df = combined.groupby('datetime')[reg_cols].transform('mean')
reg_df.columns = [f'reg_mean_{c}' for c in reg_cols]
combined = pd.concat([combined, reg_df], axis=1)

combined['pm10_anomaly'] = combined['PM10'] - combined['reg_mean_PM10']
combined['co_anomaly'] = combined['CO'] - combined['reg_mean_CO']
combined['no2_anomaly'] = combined['NO2'] - combined['reg_mean_NO2']

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

combined['station_code'] = combined['station'].astype('category').cat.codes
combined['wd_code'] = combined['wd'].astype('category').cat.codes

train_df = combined[combined['REF_NO'].isin(train['REF_NO'])].sort_values('REF_NO').reset_index(drop=True)
test_df = combined[combined['REF_NO'].isin(test['REF_NO'])].sort_values('REF_NO').reset_index(drop=True)
train_clean = train_df.dropna(subset=['PM2.5']).copy()

drop_cols = ['REF_NO', 'PM2.5', 'datetime', 'station', 'wd', 'year']
feature_cols = [c for c in train_clean.columns if c not in drop_cols]
print(f"Features: {len(feature_cols)}, Train rows: {len(train_clean)}, Test rows: {len(test_df)}")

# Train / Val Split (Out of Time 2016 for diagnostics)
val_mask = train_clean['year'] == 2016
tr_mask = train_clean['year'] < 2016

X_tr = train_clean.loc[tr_mask, feature_cols]
y_tr = train_clean.loc[tr_mask, 'PM2.5']
X_val = train_clean.loc[val_mask, feature_cols]
y_val = train_clean.loc[val_mask, 'PM2.5']

X_full = train_clean[feature_cols]
y_full = train_clean['PM2.5']
X_test = test_df[feature_cols]

# 3. Train Models
print("Training Validation LightGBM & XGBoost...")
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
ds_tr = lgb.Dataset(X_tr, label=y_tr)
ds_va = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
m_lgb_val = lgb.train(lgb_params, ds_tr, num_boost_round=1200, valid_sets=[ds_tr, ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

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
d_tr = xgb.DMatrix(X_tr, label=y_tr)
d_va = xgb.DMatrix(X_val, label=y_val)
m_xgb_val = xgb.train(xgb_params, d_tr, num_boost_round=1000, evals=[(d_tr, 'train'), (d_va, 'val')],
                      early_stopping_rounds=50, verbose_eval=False)

pred_lgb_val = np.clip(m_lgb_val.predict(X_val), 2.0, None)
pred_xgb_val = np.clip(m_xgb_val.predict(d_va), 2.0, None)
pred_ens_val = 0.55 * pred_lgb_val + 0.45 * pred_xgb_val

rmse_val = np.sqrt(mean_squared_error(y_val, pred_ens_val))
mae_val = mean_absolute_error(y_val, pred_ens_val)
r2_val = r2_score(y_val, pred_ens_val)
print(f"Validation (2016) -> RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}")

# Train on Full Training Data (2013-2016 all 395,301 samples) for Final Submission
print("\nTraining Final Production Models on 100% of Training Data (2013-2016)...")
ds_full = lgb.Dataset(X_full, label=y_full)
final_lgb = lgb.train(lgb_params, ds_full, num_boost_round=m_lgb_val.best_iteration)

d_full = xgb.DMatrix(X_full, label=y_full)
final_xgb = xgb.train(xgb_params, d_full, num_boost_round=m_xgb_val.best_iteration)

# Predict Test
print("\nPredicting on Test Dataset...")
d_test = xgb.DMatrix(X_test)
test_pred_lgb = np.clip(final_lgb.predict(X_test), 2.0, None)
test_pred_xgb = np.clip(final_xgb.predict(d_test), 2.0, None)
final_test_preds = 0.55 * test_pred_lgb + 0.45 * test_pred_xgb

# 4. Generate Diagnostic Plots
print("\nGenerating Diagnostic Plots...")
# Benchmark Comparison Plot
benchmark_csv = os.path.join(model_dir, 'benchmark_comparison.csv')
if os.path.exists(benchmark_csv):
    bm_df = pd.read_csv(benchmark_csv)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1a9641', '#41b6c4', '#2b83ba', '#fdae61', '#d7191c']
    bars = ax.barh(bm_df['Model'], bm_df['RMSE'], color=colors[::-1])
    ax.set_xlabel('Validation RMSE (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('Beijing PM2.5 Prediction: Model Benchmark Comparison (Out-of-Time 2016)', fontsize=13, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.2f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(bm_df['RMSE']) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, '01_model_benchmark_comparison.png'), bbox_inches='tight')
    plt.close()

# Feature Importance Plot
imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_lgb.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
imp_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=imp_df.head(25), x='importance', y='feature', palette='viridis', ax=ax)
ax.set_title('Top 25 Feature Importances (LightGBM Gain Metric)', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Gain (Predictive Power)', fontsize=12, fontweight='bold')
ax.set_ylabel('Engineered Feature', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, '02_feature_importance.png'), bbox_inches='tight')
plt.close()

# Actual vs Predicted & Residuals
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sample_idx = np.random.RandomState(42).choice(len(y_val), size=30000, replace=False)
y_sub = y_val.iloc[sample_idx]
pred_sub = pred_ens_val[sample_idx]

hb = axes[0].hexbin(y_sub, pred_sub, gridsize=60, cmap='plasma', mincnt=1, bins='log')
axes[0].plot([0, 600], [0, 600], 'r--', linewidth=2, label='Perfect Identity (y = ŷ)')
axes[0].set_xlim(0, 600)
axes[0].set_ylim(0, 600)
axes[0].set_title(f'Actual vs Predicted PM2.5 (Holdout Validation)\nRMSE: {rmse_val:.2f} µg/m³, R²: {r2_val:.4f}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
axes[0].legend()
fig.colorbar(hb, ax=axes[0], label='log10(sample count)')

residuals = pred_ens_val - y_val.values
sns.histplot(residuals, bins=100, kde=True, color='#2b5c8f', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--', label='Zero Error')
axes[1].axvline(np.mean(residuals), color='green', linestyle=':', label=f'Mean Bias: {np.mean(residuals):.2f}')
axes[1].set_xlim(-80, 80)
axes[1].set_title(f'Residual Distribution (Mean Error: {np.mean(residuals):.2f} µg/m³)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual (ŷ - y in µg/m³)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency Density', fontsize=11, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, '03_actual_vs_predicted_residuals.png'), bbox_inches='tight')
plt.close()

# Station Error Breakdown
stn_eval = []
for stn in train_clean['station'].unique():
    mask = (X_val['station_code'] == train_clean[train_clean['station'] == stn]['station_code'].iloc[0])
    if mask.sum() > 0:
        s_rmse = np.sqrt(mean_squared_error(y_val[mask], pred_ens_val[mask]))
        s_mae = mean_absolute_error(y_val[mask], pred_ens_val[mask])
        stn_eval.append({'station': stn, 'RMSE': s_rmse, 'MAE': s_mae, 'Count': mask.sum()})

stn_eval_df = pd.DataFrame(stn_eval).sort_values('RMSE').reset_index(drop=True)
stn_eval_df.to_csv(os.path.join(model_dir, 'station_performance.csv'), index=False)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(stn_eval_df['station'], stn_eval_df['RMSE'], color='#2b5c8f')
ax.set_title('Validation RMSE Across 12 Beijing Monitoring Stations (Ensemble)', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE (µg/m³)', fontsize=11, fontweight='bold')
ax.set_xticklabels(stn_eval_df['station'], rotation=30, ha='right')
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 0.3, f'{y:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, '04_station_rmse_breakdown.png'), bbox_inches='tight')
plt.close()

# Test Predictions Timeline Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(final_test_preds, bins=80, kde=True, color='#08519c', ax=axes[0])
axes[0].set_title(f'Test Predictions Distribution (Jan-Feb 2017)\nMean: {np.mean(final_test_preds):.1f}, Median: {np.median(final_test_preds):.1f} µg/m³', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted PM2.5 (µg/m³)', fontsize=11, fontweight='bold')

test_copy = test_df.copy()
test_copy['PM2.5_pred'] = final_test_preds
aoti = test_copy[test_copy['station'] == 'Aotizhongxin'].sort_values('datetime').head(24 * 14)
axes[1].plot(aoti['datetime'], aoti['PM2.5_pred'], label='Predicted PM2.5', color='#e6550d', linewidth=2)
axes[1].plot(aoti['datetime'], aoti['PM10'], label='Observed PM10', color='#3182bd', alpha=0.6, linestyle='--')
axes[1].set_title('Two-Week Prediction Timeline (Aotizhongxin Station)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Concentration (µg/m³)', fontsize=11, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, '05_test_predictions_timeseries.png'), bbox_inches='tight')
plt.close()

# 5. Output Submissions
print("\nSaving Final Submissions...")
sub_df = pd.DataFrame({
    'REF_NO': test_df['REF_NO'],
    'PM2.5': final_test_preds
})

# Save exact match to sample_submission (headerless CSV)
sub_df.to_csv(os.path.join(base_dir, 'submission.csv'), index=False, header=False)
# Save version with header
sub_df.to_csv(os.path.join(base_dir, 'submission_with_header.csv'), index=False, header=True)
# Save Excel version
sub_df.to_excel(os.path.join(base_dir, 'submission.xlsx'), index=False, header=False)

metrics = {
    'validation_rmse': float(rmse_val),
    'validation_mae': float(mae_val),
    'validation_r2': float(r2_val),
    'num_features': len(feature_cols),
    'test_samples': len(final_test_preds),
    'test_pred_mean': float(np.mean(final_test_preds)),
    'test_pred_median': float(np.median(final_test_preds)),
    'test_pred_std': float(np.std(final_test_preds)),
    'test_pred_min': float(np.min(final_test_preds)),
    'test_pred_max': float(np.max(final_test_preds)),
    'runtime_seconds': round(time.time() - start_time, 2)
}
with open(os.path.join(model_dir, 'metrics_summary.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
print(f"Total Runtime: {time.time() - start_time:.2f} seconds")
print(f"Submission File Saved: submission.csv (16,728 rows)")
print(f"Test PM2.5 Stats: Mean={metrics['test_pred_mean']:.2f}, Median={metrics['test_pred_median']:.2f}, Min={metrics['test_pred_min']:.2f}, Max={metrics['test_pred_max']:.2f}")
