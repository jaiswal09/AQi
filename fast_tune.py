import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

base_dir = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(base_dir, 'beijing_train.csv'))
test = pd.read_csv(os.path.join(base_dir, 'beijing_test.csv'))

train['datetime'] = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
test['datetime'] = pd.to_datetime(test[['year', 'month', 'day', 'hour']])

test['PM2.5'] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(['station', 'datetime']).reset_index(drop=True)

# Wind features
wd_angles = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}
combined['wd_angle'] = combined['wd'].map(wd_angles)
combined['sin_wd'] = np.sin(np.radians(combined['wd_angle']))
combined['cos_wd'] = np.cos(np.radians(combined['wd_angle']))
combined['u_wind'] = -combined['WSPM'] * combined['sin_wd']
combined['v_wind'] = -combined['WSPM'] * combined['cos_wd']

# Meteorology
a, b = 17.625, 243.04
alpha = (a * combined['TEMP']) / (b + combined['TEMP'])
beta = (a * combined['DEWP']) / (b + combined['DEWP'])
combined['RH'] = np.clip(100.0 * np.exp(beta - alpha), 0, 100)
combined['dew_depression'] = combined['TEMP'] - combined['DEWP']

es = 0.61078 * np.exp((17.27 * combined['TEMP']) / (combined['TEMP'] + 237.3))
ea = 0.61078 * np.exp((17.27 * combined['DEWP']) / (combined['DEWP'] + 237.3))
combined['VPD'] = np.maximum(0, es - ea)
combined['air_density'] = combined['PRES'] / (2.87 * (combined['TEMP'] + 273.15))

# Calendar
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

# Chemical Interactions
combined['pm10_co_ratio'] = combined['PM10'] / (combined['CO'] + 1.0)
combined['pm10_so2_ratio'] = combined['PM10'] / (combined['SO2'] + 1.0)
combined['pm10_no2_ratio'] = combined['PM10'] / (combined['NO2'] + 1.0)
combined['so2_no2_ratio'] = combined['SO2'] / (combined['NO2'] + 1.0)
combined['co_no2_ratio'] = combined['CO'] / (combined['NO2'] + 1.0)
combined['total_oxidant'] = combined['O3'] + combined['NO2']
combined['pm10_rh_inter'] = combined['PM10'] * (combined['RH'] / 100.0)
combined['pm10_temp_inter'] = combined['PM10'] * combined['TEMP']
combined['pm10_wspm_inter'] = combined['PM10'] / (combined['WSPM'] + 0.1)

# Station Metadata
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

# Regional means
reg_cols = ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'PRES', 'WSPM']
reg_df = combined.groupby('datetime')[reg_cols].transform('mean')
reg_df.columns = [f'reg_mean_{c}' for c in reg_cols]
combined = pd.concat([combined, reg_df], axis=1)

combined['pm10_anomaly'] = combined['PM10'] - combined['reg_mean_PM10']
combined['co_anomaly'] = combined['CO'] - combined['reg_mean_CO']

# Lags & rolling
for col in ['PM10', 'CO', 'NO2', 'SO2', 'TEMP', 'WSPM']:
    combined[f'{col}_lag1'] = combined.groupby('station')[col].shift(1)
    combined[f'{col}_lag2'] = combined.groupby('station')[col].shift(2)
    combined[f'{col}_diff1'] = combined[col] - combined[f'{col}_lag1']
    combined[f'{col}_roll_mean_3'] = combined.groupby('station')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    combined[f'{col}_roll_mean_6'] = combined.groupby('station')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
    combined[f'{col}_roll_std_6'] = combined.groupby('station')[col].transform(lambda x: x.rolling(6, min_periods=1).std())
    combined[f'{col}_roll_mean_24'] = combined.groupby('station')[col].transform(lambda x: x.rolling(24, min_periods=1).mean())

combined['station_code'] = combined['station'].astype('category').cat.codes
combined['wd_code'] = combined['wd'].astype('category').cat.codes

train_df = combined[combined['REF_NO'].isin(train['REF_NO'])].sort_values('REF_NO').reset_index(drop=True)
train_df_clean = train_df.dropna(subset=['PM2.5']).copy()

feature_cols = [c for c in train_df_clean.columns if c not in ['REF_NO', 'PM2.5', 'datetime', 'station', 'wd', 'year']]

tr_mask = train_df_clean['year'] < 2016
val_mask = train_df_clean['year'] == 2016

X_tr = train_df_clean.loc[tr_mask, feature_cols]
y_tr = train_df_clean.loc[tr_mask, 'PM2.5']
X_val = train_df_clean.loc[val_mask, feature_cols]
y_val = train_df_clean.loc[val_mask, 'PM2.5']

lgb_tr = lgb.Dataset(X_tr, label=y_tr)
lgb_va = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)

# Test 1: num_leaves=63, lr=0.04
p1 = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.04, 'num_leaves': 63,
      'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'seed': 42, 'verbose': -1, 'n_jobs': 4}
m1 = lgb.train(p1, lgb_tr, num_boost_round=1500, valid_sets=[lgb_tr, lgb_va], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
pred1 = np.clip(m1.predict(X_val), 0, None)
rmse1 = np.sqrt(mean_squared_error(y_val, pred1))
print(f"Num_leaves 63 -> RMSE: {rmse1:.4f}")

# Test 2: num_leaves=95, lr=0.03
p2 = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 95,
      'feature_fraction': 0.75, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'min_child_samples': 40, 'seed': 42, 'verbose': -1, 'n_jobs': 4}
m2 = lgb.train(p2, lgb_tr, num_boost_round=1800, valid_sets=[lgb_tr, lgb_va], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
pred2 = np.clip(m2.predict(X_val), 0, None)
rmse2 = np.sqrt(mean_squared_error(y_val, pred2))
print(f"Num_leaves 95, lr 0.03 -> RMSE: {rmse2:.4f}")
