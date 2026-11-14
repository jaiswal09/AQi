import pandas as pd
import numpy as np
import time
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

print("Starting feature engineering experiment...")
start_time = time.time()

# 1. Load data
train = pd.read_csv('beijing_train.csv')
test = pd.read_csv('beijing_test.csv')

print(f"Loaded train shape: {train.shape}, test shape: {test.shape}")

# Create datetime
train['datetime'] = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
test['datetime'] = pd.to_datetime(test[['year', 'month', 'day', 'hour']])

# Combine temporarily for smooth lag and rolling calculation
test['PM2.5'] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(['station', 'datetime']).reset_index(drop=True)

# 2. Wind Direction Encoding
wd_angles = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}
combined['wd_angle'] = combined['wd'].map(wd_angles)
combined['sin_wd'] = np.sin(np.radians(combined['wd_angle']))
combined['cos_wd'] = np.cos(np.radians(combined['wd_angle']))

# Wind components: zonal (u) and meridional (v)
combined['u_wind'] = -combined['WSPM'] * np.sin(np.radians(combined['wd_angle']))
combined['v_wind'] = -combined['WSPM'] * np.cos(np.radians(combined['wd_angle']))

# 3. Meteorological & Atmospheric Physics
# Relative humidity via Magnus formula
def calc_rh(temp, dewp):
    a, b = 17.625, 243.04
    alpha = (a * temp) / (b + temp)
    beta = (a * dewp) / (b + dewp)
    rh = 100.0 * np.exp(beta - alpha)
    return np.clip(rh, 0, 100)

combined['RH'] = calc_rh(combined['TEMP'], combined['DEWP'])
combined['dew_depression'] = combined['TEMP'] - combined['DEWP']

# Vapor pressure
es = 0.61078 * np.exp((17.27 * combined['TEMP']) / (combined['TEMP'] + 237.3))
ea = 0.61078 * np.exp((17.27 * combined['DEWP']) / (combined['DEWP'] + 237.3))
combined['VPD'] = np.maximum(0, es - ea)

# Air density approximation ~ PRES / (2.87 * (TEMP + 273.15))
combined['air_density'] = combined['PRES'] / (2.87 * (combined['TEMP'] + 273.15))

# 4. Temporal Features
combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24.0)
combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24.0)
combined['month_sin'] = np.sin(2 * np.pi * combined['month'] / 12.0)
combined['month_cos'] = np.cos(2 * np.pi * combined['month'] / 12.0)
doy = combined['datetime'].dt.dayofyear
combined['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
combined['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
combined['dayofweek'] = combined['datetime'].dt.dayofweek
combined['is_weekend'] = (combined['dayofweek'] >= 5).astype(int)
combined['is_heating'] = ((combined['month'] >= 11) | (combined['month'] <= 3)).astype(int)

# 5. Chemical Ratios & Interactions
combined['pm10_co_ratio'] = combined['PM10'] / (combined['CO'] + 1.0)
combined['pm10_so2_ratio'] = combined['PM10'] / (combined['SO2'] + 1.0)
combined['pm10_no2_ratio'] = combined['PM10'] / (combined['NO2'] + 1.0)
combined['so2_no2_ratio'] = combined['SO2'] / (combined['NO2'] + 1.0)
combined['co_no2_ratio'] = combined['CO'] / (combined['NO2'] + 1.0)
combined['total_oxidant'] = combined['O3'] + combined['NO2']
combined['pm10_rh_inter'] = combined['PM10'] * (combined['RH'] / 100.0)
combined['pm10_temp_inter'] = combined['PM10'] * combined['TEMP']
combined['pm10_wspm_inter'] = combined['PM10'] / (combined['WSPM'] + 0.1)

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
# Distance from central Beijing (Tiananmen: 39.9042, 116.4074)
combined['dist_center'] = np.sqrt((combined['lat'] - 39.9042)**2 + (combined['lon'] - 116.4074)**2) * 111.0 # approx km

# 7. Regional City-Wide Synoptic Means (per hour across all stations)
hourly_regional = combined.groupby('datetime')[['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'PRES', 'WSPM']].transform('mean')
hourly_regional.columns = [f'regional_mean_{c}' for c in hourly_regional.columns]
combined = pd.concat([combined, hourly_regional], axis=1)

combined['pm10_anomaly'] = combined['PM10'] - combined['regional_mean_PM10']
combined['co_anomaly'] = combined['CO'] - combined['regional_mean_CO']

# 8. Lag & Rolling Features per Station
lag_cols = ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'WSPM']
for col in lag_cols:
    combined[f'{col}_lag1'] = combined.groupby('station')[col].shift(1)
    combined[f'{col}_lag2'] = combined.groupby('station')[col].shift(2)
    combined[f'{col}_diff1'] = combined[col] - combined[f'{col}_lag1']
    combined[f'{col}_roll_mean_3'] = combined.groupby('station')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    combined[f'{col}_roll_mean_6'] = combined.groupby('station')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
    combined[f'{col}_roll_std_6'] = combined.groupby('station')[col].transform(lambda x: x.rolling(6, min_periods=1).std())
    combined[f'{col}_roll_mean_24'] = combined.groupby('station')[col].transform(lambda x: x.rolling(24, min_periods=1).mean())

# Convert station to category
combined['station'] = combined['station'].astype('category')
combined['wd'] = combined['wd'].astype('category')

# Separate back to train and test
train_fe = combined[combined['REF_NO'].isin(train['REF_NO'])].copy()
test_fe = combined[combined['REF_NO'].isin(test['REF_NO'])].copy()

# Sort back by original index / REF_NO
train_fe = train_fe.sort_values('REF_NO').reset_index(drop=True)
test_fe = test_fe.sort_values('REF_NO').reset_index(drop=True)

# Drop missing target in train
train_fe_clean = train_fe.dropna(subset=['PM2.5']).copy()

print(f"Engineered Train shape: {train_fe_clean.shape}, Test shape: {test_fe.shape}")
print(f"Features count: {train_fe_clean.shape[1]}")

# 9. Out-of-Time Validation: Train on 2013-2015, Validate on 2016
val_mask = train_fe_clean['year'] == 2016
tr_mask = train_fe_clean['year'] < 2016

ignore_cols = ['REF_NO', 'PM2.5', 'datetime', 'year']
feature_cols = [c for c in train_fe_clean.columns if c not in ignore_cols]
print(f"Using {len(feature_cols)} features for modeling.")

X_tr = train_fe_clean.loc[tr_mask, feature_cols]
y_tr = train_fe_clean.loc[tr_mask, 'PM2.5']

X_val = train_fe_clean.loc[val_mask, feature_cols]
y_val = train_fe_clean.loc[val_mask, 'PM2.5']

print(f"Train set: {X_tr.shape}, Val set (2016 full year): {X_val.shape}")

# Evaluate LightGBM
print("\n--- Training LightGBM Baseline ---")
lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'seed': 42,
    'verbose': -1,
    'n_jobs': 4
}

model_lgb = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

y_pred_val = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
val_mae = mean_absolute_error(y_val, y_pred_val)
val_r2 = r2_score(y_val, y_pred_val)

print(f"\n==========================================")
print(f"LightGBM Validation (2016) Performance:")
print(f"RMSE: {val_rmse:.4f}")
print(f"MAE:  {val_mae:.4f}")
print(f"R²:   {val_r2:.4f}")
print(f"==========================================")
print(f"Elapsed time: {time.time() - start_time:.2f} s")
