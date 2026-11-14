import pandas as pd
import numpy as np

train = pd.read_csv('beijing_train.csv')
test = pd.read_csv('beijing_test.csv')

train['datetime'] = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
test['datetime'] = pd.to_datetime(test[['year', 'month', 'day', 'hour']])

print("=== CHECKING CONTINUITY IN TEST SET ===")
for stn in test['station'].unique()[:3]:
    stn_test = test[test['station'] == stn].sort_values('datetime')
    dt_diff = stn_test['datetime'].diff()
    print(f"Station {stn}: rows = {len(stn_test)}, time range: {stn_test['datetime'].min()} to {stn_test['datetime'].max()}")
    print(f"Diff value counts:\n{dt_diff.value_counts().head(5)}")

print("\n=== ARE TEST ROWS JUST SLICES OR FULL CONTINUOUS TIME? ===")
# Check total hours between 2017-01-01 00:00 and 2017-02-28 23:00
full_hours = pd.date_range('2017-01-01 00:00:00', '2017-02-28 23:00:00', freq='h')
print(f"Total expected hours in Jan-Feb 2017 (non-leap year: 31 + 28 = 59 days * 24 = {len(full_hours)} hours)")
for stn in test['station'].unique()[:5]:
    stn_hours = test[test['station'] == stn]['datetime'].nunique()
    print(f"Station {stn}: unique hours in test = {stn_hours} / {len(full_hours)}")

print("\n=== CORRELATION WITH PM2.5 IN TRAIN ===")
corr = train.select_dtypes(include=[np.number]).corr()['PM2.5'].sort_values(ascending=False)
print(corr)
