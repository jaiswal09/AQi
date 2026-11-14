import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 150

os.makedirs('eda_plots', exist_ok=True)

print("Loading dataset...")
train = pd.read_csv('beijing_train.csv')
test = pd.read_csv('beijing_test.csv')

# Datetime creation
train['datetime'] = pd.to_datetime(train[['year', 'month', 'day', 'hour']])
test['datetime'] = pd.to_datetime(test[['year', 'month', 'day', 'hour']])

# Drop null target rows for target analysis
train_clean = train.dropna(subset=['PM2.5']).copy()

# 1. Distribution of Target Variable (PM2.5)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(train_clean['PM2.5'], bins=100, kde=True, color='#2b5c8f', ax=axes[0])
axes[0].set_title('Distribution of PM2.5 Concentration (Raw)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('PM2.5 (µg/m³)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].axvline(train_clean['PM2.5'].mean(), color='red', linestyle='--', label=f"Mean: {train_clean['PM2.5'].mean():.1f}")
axes[0].axvline(train_clean['PM2.5'].median(), color='green', linestyle='-', label=f"Median: {train_clean['PM2.5'].median():.1f}")
axes[0].axvline(35, color='orange', linestyle=':', label="WHO Annual Guideline (35 µg/m³)")
axes[0].legend(frameon=True)

sns.histplot(np.log1p(train_clean['PM2.5']), bins=80, kde=True, color='#1f77b4', ax=axes[1])
axes[1].set_title('Distribution of log(1 + PM2.5)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('log(1 + PM2.5)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
plt.tight_layout()
plt.savefig('eda_plots/01_pm25_distribution.png', bbox_inches='tight')
plt.close()
print("Saved 01_pm25_distribution.png")

# 2. Correlation Heatmap
fig, ax = plt.subplots(figsize=(11, 9))
num_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
corr = train_clean[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.6, vmax=1.0, mask=mask,
            linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Pearson Correlation Matrix (Air Pollutants & Weather)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_plots/02_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved 02_correlation_heatmap.png")

# 3. PM2.5 vs PM10 and CO Scatter / Hexbin
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
hb1 = axes[0].hexbin(train_clean['PM10'], train_clean['PM2.5'], gridsize=60, cmap='Blues', mincnt=1, bins='log')
axes[0].plot([0, 800], [0, 800], 'r--', label='1:1 Line (PM2.5 = PM10)')
axes[0].set_xlim(0, 800)
axes[0].set_ylim(0, 800)
axes[0].set_title('Relationship: PM2.5 vs PM10 (Log Count)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('PM10 (µg/m³)', fontsize=11)
axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=11)
axes[0].legend()
fig.colorbar(hb1, ax=axes[0], label='log10(count)')

hb2 = axes[1].hexbin(train_clean['CO'], train_clean['PM2.5'], gridsize=60, cmap='Purples', mincnt=1, bins='log')
axes[1].set_xlim(0, 8000)
axes[1].set_ylim(0, 800)
axes[1].set_title('Relationship: PM2.5 vs CO (Log Count)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('CO (µg/m³)', fontsize=11)
axes[1].set_ylabel('PM2.5 (µg/m³)', fontsize=11)
fig.colorbar(hb2, ax=axes[1], label='log10(count)')
plt.tight_layout()
plt.savefig('eda_plots/03_pm25_vs_pm10_co.png', bbox_inches='tight')
plt.close()
print("Saved 03_pm25_vs_pm10_co.png")

# 4. Temporal Trends: Monthly and Diurnal (Hourly) Cycle
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
monthly_avg = train_clean.groupby('month')['PM2.5'].mean().reset_index()
monthly_med = train_clean.groupby('month')['PM2.5'].median().reset_index()
axes[0].plot(monthly_avg['month'], monthly_avg['PM2.5'], marker='o', linewidth=2.5, color='#d95f02', label='Mean')
axes[0].plot(monthly_med['month'], monthly_med['PM2.5'], marker='s', linewidth=2, linestyle='--', color='#7570b3', label='Median')
axes[0].set_title('Seasonal Pattern: Average PM2.5 by Month', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Month', fontsize=11)
axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=11)
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[0].axvspan(11, 12, color='orange', alpha=0.15, label='Heating Season (Nov-Mar)')
axes[0].axvspan(1, 3.5, color='orange', alpha=0.15)
axes[0].legend()

hourly_avg = train_clean.groupby('hour')['PM2.5'].mean().reset_index()
hourly_med = train_clean.groupby('hour')['PM2.5'].median().reset_index()
axes[1].plot(hourly_avg['hour'], hourly_avg['PM2.5'], marker='o', linewidth=2.5, color='#1b9e77', label='Mean')
axes[1].plot(hourly_med['hour'], hourly_med['PM2.5'], marker='s', linewidth=2, linestyle='--', color='#e7298a', label='Median')
axes[1].set_title('Diurnal Pattern: Average PM2.5 by Hour of Day', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Hour of Day (0-23)', fontsize=11)
axes[1].set_ylabel('PM2.5 (µg/m³)', fontsize=11)
axes[1].set_xticks(range(0, 24, 2))
axes[1].legend()
plt.tight_layout()
plt.savefig('eda_plots/04_temporal_trends.png', bbox_inches='tight')
plt.close()
print("Saved 04_temporal_trends.png")

# 5. Spatial Variation Across 12 Monitoring Stations
fig, ax = plt.subplots(figsize=(14, 6))
station_order = train_clean.groupby('station')['PM2.5'].median().sort_values().index
sns.boxplot(data=train_clean, x='station', y='PM2.5', order=station_order, palette='Spectral',
            showfliers=False, ax=ax)
ax.set_title('PM2.5 Distribution Across 12 Beijing Monitoring Stations (Ordered by Median)', fontsize=14, fontweight='bold')
ax.set_xlabel('Station', fontsize=12)
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('eda_plots/05_station_distribution.png', bbox_inches='tight')
plt.close()
print("Saved 05_station_distribution.png")

# 6. Wind Direction and Wind Speed Impact (Pollution Rose)
wd_order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
train_clean_wd = train_clean.dropna(subset=['wd'])
wd_stats = train_clean_wd.groupby('wd').agg(
    mean_pm25=('PM2.5', 'mean'),
    median_pm25=('PM2.5', 'median'),
    mean_wspm=('WSPM', 'mean'),
    count=('PM2.5', 'count')
).reindex(wd_order)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
bars = axes[0].bar(wd_stats.index, wd_stats['mean_pm25'], color=plt.cm.YlOrRd(wd_stats['mean_pm25'] / wd_stats['mean_pm25'].max()))
axes[0].set_title('Mean PM2.5 Concentration by Wind Direction', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Wind Direction (16 Compass Points)', fontsize=11)
axes[0].set_ylabel('Mean PM2.5 (µg/m³)', fontsize=11)
axes[0].set_xticklabels(wd_stats.index, rotation=45)
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.0f}', ha='center', va='bottom', fontsize=9)

axes[1].scatter(train_clean['WSPM'].sample(50000, random_state=42), 
                train_clean['PM2.5'].sample(50000, random_state=42), 
                alpha=0.15, color='#2b5c8f', s=10)
# Rolling median over wind speed bins
wspm_bins = pd.cut(train_clean['WSPM'], bins=np.linspace(0, 10, 21))
wspm_pm25_med = train_clean.groupby(wspm_bins)['PM2.5'].median()
bin_centers = [interval.mid for interval in wspm_pm25_med.index]
axes[1].plot(bin_centers, wspm_pm25_med.values, color='red', linewidth=3, marker='o', label='Median PM2.5')
axes[1].set_title('PM2.5 vs Wind Speed (Dispersal Effect)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Wind Speed (m/s)', fontsize=11)
axes[1].set_ylabel('PM2.5 (µg/m³)', fontsize=11)
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 500)
axes[1].legend()
plt.tight_layout()
plt.savefig('eda_plots/06_wind_effects.png', bbox_inches='tight')
plt.close()
print("Saved 06_wind_effects.png")

# 7. Year-over-Year Trend (Air Pollution Action Plan Progress)
fig, ax = plt.subplots(figsize=(12, 5))
yearly_stn = train_clean.groupby(['year', 'month'])['PM2.5'].mean().unstack(level=0)
for yr in [2013, 2014, 2015, 2016]:
    ax.plot(yearly_stn.index, yearly_stn[yr], marker='o', label=str(yr), linewidth=2)
ax.set_title('Monthly PM2.5 Trend Across Years (2013 - 2016)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Mean PM2.5 (µg/m³)', fontsize=11)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.legend(title='Year')
plt.tight_layout()
plt.savefig('eda_plots/07_yearly_trends.png', bbox_inches='tight')
plt.close()
print("Saved 07_yearly_trends.png")

# 8. PM2.5 / PM10 Ratio Distribution
train_clean['pm_ratio'] = train_clean['PM2.5'] / (train_clean['PM10'] + 1e-5)
valid_ratio = train_clean[train_clean['pm_ratio'] <= 1.5]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(valid_ratio['pm_ratio'], bins=75, kde=True, color='#4575b4', ax=axes[0])
axes[0].set_title('PM2.5 to PM10 Ratio Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('PM2.5 / PM10 Ratio', fontsize=11)
axes[0].axvline(valid_ratio['pm_ratio'].median(), color='red', linestyle='--', 
                label=f"Median Ratio: {valid_ratio['pm_ratio'].median():.2f}")
axes[0].legend()

# Ratio by season
train_clean['season'] = train_clean['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
sns.boxplot(data=train_clean[train_clean['pm_ratio'] <= 1.5], x='season', y='pm_ratio',
            order=season_order, palette='Set2', showfliers=False, ax=axes[1])
axes[1].set_title('PM2.5 / PM10 Ratio by Season', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Season', fontsize=11)
axes[1].set_ylabel('PM2.5 / PM10 Ratio', fontsize=11)
plt.tight_layout()
plt.savefig('eda_plots/08_pm_ratio_seasonal.png', bbox_inches='tight')
plt.close()
print("Saved 08_pm_ratio_seasonal.png")

print("All 8 EDA figures successfully generated!")
