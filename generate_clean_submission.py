import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model_plots')

print("=== RE-ALIGNING AND GENERATING EXACT-ORDER SUBMISSION ===")
start_time = time.time()

# 1. Load Data
train = pd.read_csv(os.path.join(base_dir, 'beijing_train.csv'))
test = pd.read_csv(os.path.join(base_dir, 'beijing_test.csv'))

# Tag original order so row ordering is 100% strictly preserved
train['orig_order'] = np.arange(len(train))
test['orig_order'] = np.arange(len(test))
train['is_test'] = 0
test['is_test'] = 1

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

# Split preserving exact original order
train_df = combined[combined['is_test'] == 0].sort_values('orig_order').reset_index(drop=True)
test_df = combined[combined['is_test'] == 1].sort_values('orig_order').reset_index(drop=True)

train_clean = train_df.dropna(subset=['PM2.5']).copy()

drop_cols = ['REF_NO', 'PM2.5', 'datetime', 'station', 'wd', 'year', 'orig_order', 'is_test']
feature_cols = [c for c in train_clean.columns if c not in drop_cols]
print(f"Features: {len(feature_cols)}, Train rows: {len(train_clean)}, Test rows: {len(test_df)}")

# Train on full dataset
X_full = train_clean[feature_cols]
y_full = train_clean['PM2.5']
X_test = test_df[feature_cols]

print("Training Full LightGBM (1000 rounds)...")
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
ds_full = lgb.Dataset(X_full, label=y_full)
final_lgb = lgb.train(lgb_params, ds_full, num_boost_round=1100)

print("Training Full XGBoost (1000 rounds)...")
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
d_full = xgb.DMatrix(X_full, label=y_full)
final_xgb = xgb.train(xgb_params, d_full, num_boost_round=900)

# Predict Test
print("Generating final test predictions...")
d_test = xgb.DMatrix(X_test)
test_pred_lgb = np.clip(final_lgb.predict(X_test), 2.0, None)
test_pred_xgb = np.clip(final_xgb.predict(d_test), 2.0, None)
final_test_preds = 0.55 * test_pred_lgb + 0.45 * test_pred_xgb

# Format final submission DataFrame
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

# Verification
sample = pd.read_excel(os.path.join(base_dir, 'sample_submission.csv.xlsx'), header=None)
print("Verification vs sample_submission.csv.xlsx:")
print(f"Shape identical: {sub_df.shape == sample.shape}")
print(f"REF_NO identical row-by-row: {(sub_df['REF_NO'].values == sample[0].values).all()}")
print(f"Null count in predictions: {sub_df['PM2.5'].isnull().sum()}")
print(f"Prediction stats: Mean={sub_df['PM2.5'].mean():.2f}, Std={sub_df['PM2.5'].std():.2f}, Min={sub_df['PM2.5'].min():.2f}, Max={sub_df['PM2.5'].max():.2f}")
print("First 5 rows:\n", sub_df.head())
print("Last 5 rows:\n", sub_df.tail())
print(f"Done in {time.time() - start_time:.2f}s!")
