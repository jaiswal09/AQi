import pandas as pd
import numpy as np

train = pd.read_csv('beijing_train.csv')
test = pd.read_csv('beijing_test.csv')

print('=== TRAIN INFO ===')
print('Train shape:', train.shape)
print('Train columns:', train.columns.tolist())
print('\nTrain missing values:')
print(train.isnull().sum())

print('\n=== TEST INFO ===')
print('Test shape:', test.shape)
print('Test columns:', test.columns.tolist())
print('\nTest missing values:')
print(test.isnull().sum())

print('\n=== TIME SPAN ===')
print('Train years:', sorted(train['year'].unique()))
train_dt = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
print('Train datetime min to max:', train_dt.min(), 'to', train_dt.max())

test_dt = pd.to_datetime(test[['year', 'month', 'day', 'hour']])
print('Test datetime min to max:', test_dt.min(), 'to', test_dt.max())

print('\n=== STATIONS ===')
print('Train stations count:', train['station'].nunique(), train['station'].unique().tolist())
print('Test stations count:', test['station'].nunique(), test['station'].unique().tolist())
print('\nTrain counts per station:\n', train['station'].value_counts())
print('\nTest counts per station:\n', test['station'].value_counts())

print('\n=== TARGET PM2.5 STATS ===')
print(train['PM2.5'].describe())
print('Target null count:', train['PM2.5'].isnull().sum(), f"({train['PM2.5'].isnull().mean()*100:.2f}%)")

print('\n=== NUMERICAL FEATURES STATS ===')
num_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
print('Train stats:\n', train[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
print('Test stats:\n', test[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

print('\n=== CATEGORICAL FEATURES ===')
print('Train wd unique:', train['wd'].unique().tolist())
print('Test wd unique:', test['wd'].unique().tolist())
