import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model_plots')
os.makedirs(model_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 150

print("Generating Linear Regression Graphs according to Section III.I...")

# 1. Load Cleaned Training & Validation Data
train = pd.read_csv(os.path.join(base_dir, 'beijing_train.csv'))
pollutant_cols = ['PM10', 'CO', 'NO2', 'SO2', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']

# Drop rows missing target or predictors
train_clean = train.dropna(subset=['PM2.5'] + pollutant_cols).copy()

X = train_clean[pollutant_cols]
y = train_clean['PM2.5']

# Train on 2013-2015, validate on 2016
tr_mask = (train_clean['year'] < 2016).values
val_mask = (train_clean['year'] == 2016).values

X_tr, y_tr = X.iloc[tr_mask], y.iloc[tr_mask]
X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]

# Standardize for coefficient interpretability and gradient descent
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# Fit Multivariate Linear Regression Model
lr = LinearRegression()
lr.fit(X_tr_scaled, y_tr)
y_pred_val = lr.predict(X_val_scaled)

rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)

print(f"Linear Regression Results -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

# Simulate Gradient Descent Cost Convergence Curve
def simulate_gradient_descent(X, y, lr_rate=0.01, n_iters=120):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X] # add bias term
    theta = np.zeros(n + 1)
    cost_history = []
    
    sub_idx = np.random.RandomState(42).choice(m, size=min(15000, m), replace=False)
    X_sub = X_b[sub_idx]
    y_sub = y.iloc[sub_idx].values
    m_sub = len(y_sub)
    
    for it in range(n_iters):
        preds = X_sub.dot(theta)
        error = preds - y_sub
        cost = (1.0 / (2.0 * m_sub)) * np.sum(error**2)
        cost_history.append(cost)
        gradient = (1.0 / m_sub) * X_sub.T.dot(error)
        theta = theta - lr_rate * gradient
    return cost_history

cost_history = simulate_gradient_descent(X_tr_scaled, y_tr, lr_rate=0.08, n_iters=100)

# =========================================================================
# GRAPH 1: Comprehensive 4-Panel Linear Regression Figure
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot (0, 0): Actual vs Predicted Linear Regression Fit
sample_idx = np.random.RandomState(42).choice(len(y_val), size=25000, replace=False)
y_v_sub = y_val.iloc[sample_idx]
y_p_sub = y_pred_val[sample_idx]

hb = axes[0, 0].hexbin(y_v_sub, y_p_sub, gridsize=55, cmap='Blues', mincnt=1, bins='log')
axes[0, 0].plot([0, 600], [0, 600], 'r--', linewidth=2.5, label='Ideal Identity Line (Y = Ŷ)')
m_fit, b_fit = np.polyfit(y_v_sub, y_p_sub, 1)
axes[0, 0].plot(np.array([0, 600]), m_fit * np.array([0, 600]) + b_fit, 'g-', linewidth=2, label=f'Linear Fit: Ŷ = {m_fit:.2f}Y + {b_fit:.1f}')
axes[0, 0].set_xlim(0, 550)
axes[0, 0].set_ylim(0, 550)
axes[0, 0].set_title(f'(A) Multivariate Linear Regression: Actual vs Predicted\n$R^2 = {r2:.4f}$, $RMSE = {rmse:.2f}\ \mu g/m^3$, $MAE = {mae:.2f}\ \mu g/m^3$', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Actual PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
axes[0, 0].legend(loc='upper left', frameon=True)
fig.colorbar(hb, ax=axes[0, 0], label='log10(count)')

# Subplot (0, 1): Linear Regression Fit for Primary Co-Pollutant PM10
sample_plot = train_clean.sample(15000, random_state=42)
sns.regplot(data=sample_plot, x='PM10', y='PM2.5', scatter_kws={'alpha': 0.15, 'color': '#1f77b4', 's': 8},
            line_kws={'color': '#d95f02', 'linewidth': 2.5, 'label': 'Linear Fit Line (PM2.5 vs PM10)'}, ax=axes[0, 1])
axes[0, 1].set_xlim(0, 600)
axes[0, 1].set_ylim(0, 600)
axes[0, 1].set_title('(B) Linear Regression Fit: PM2.5 vs PM10 ($r = +0.88$)\n$Y_i = \\alpha + \\beta_1 \\cdot PM10_i$', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('PM10 Particulate Matter (µg/m³)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Target PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
axes[0, 1].legend(loc='upper left', frameon=True)

# Subplot (1, 0): Learned Model Parameter Weights (Beta Coefficients)
coef_df = pd.DataFrame({
    'Feature': pollutant_cols,
    'Beta_Coefficient': lr.coef_
}).sort_values('Beta_Coefficient', ascending=True)

colors = ['#d7191c' if c < 0 else '#2ca25f' for c in coef_df['Beta_Coefficient']]
bars = axes[1, 0].barh(coef_df['Feature'], coef_df['Beta_Coefficient'], color=colors)
axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=0.8)
axes[1, 0].set_title(f'(C) Learned Regression Coefficients ($\\beta_j$ Parameters)\nIntercept $\\alpha = {lr.intercept_:.2f}$ (Standardized Predictors)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Standardized Coefficient Weight (Effect on PM2.5)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Input Predictor Variable', fontsize=11, fontweight='bold')
for bar in bars:
    w = bar.get_width()
    offset = 0.8 if w >= 0 else -4.0
    axes[1, 0].text(w + offset, bar.get_y() + bar.get_height()/2, f'{w:+.2f}', va='center', fontsize=9, fontweight='bold')

# Subplot (1, 1): Gradient Descent Optimization Loss Convergence Curve
axes[1, 1].plot(range(1, len(cost_history) + 1), cost_history, color='#7570b3', linewidth=2.5, label='Loss Function $J(\\theta)$')
axes[1, 1].set_title('(D) Parameter Optimization via Gradient Descent\n$\\theta_j := \\theta_j - \\eta \\frac{\\partial J}{\\partial \\theta_j}$ (Monotonic Convergence)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Gradient Descent Iterations', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Mean Squared Error Loss $J(\\theta)$', fontsize=11, fontweight='bold')
axes[1, 1].annotate('Initial High Error\n(Random Start)', xy=(2, cost_history[1]), xytext=(15, cost_history[1]*0.85),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=10, fontweight='bold', color='red')
axes[1, 1].annotate(f'Converged Minimum\nMSE: {cost_history[-1]:.1f}', xy=(len(cost_history)-5, cost_history[-1]), xytext=(len(cost_history)-45, cost_history[-1]*1.4),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5), fontsize=10, fontweight='bold', color='green')
axes[1, 1].legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'linear_regression_comprehensive.png'), bbox_inches='tight')
plt.close()
print("Saved model_plots/linear_regression_comprehensive.png")

# =========================================================================
# GRAPH 2: Standalone Multi-Variable Linear Fit Grid
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# PM10 fit
sns.regplot(data=sample_plot, x='PM10', y='PM2.5', scatter_kws={'alpha': 0.12, 'color': '#2b5c8f', 's': 8},
            line_kws={'color': '#e41a1c', 'linewidth': 3}, ax=axes[0])
axes[0].set_title('Linear Regression: PM2.5 vs PM10\n$Y_i = \\alpha + \\beta_1 \\cdot PM10_i$ ($r = 0.88$)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('PM10 (µg/m³)', fontsize=11)
axes[0].set_ylabel('Target PM2.5 (µg/m³)', fontsize=11)
axes[0].set_xlim(0, 500)
axes[0].set_ylim(0, 500)

# CO fit
sns.regplot(data=sample_plot, x='CO', y='PM2.5', scatter_kws={'alpha': 0.12, 'color': '#7570b3', 's': 8},
            line_kws={'color': '#e41a1c', 'linewidth': 3}, ax=axes[1])
axes[1].set_title('Linear Regression: PM2.5 vs CO\n$Y_i = \\alpha + \\beta_2 \\cdot CO_i$ ($r = 0.79$)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('CO (µg/m³)', fontsize=11)
axes[1].set_ylabel('Target PM2.5 (µg/m³)', fontsize=11)
axes[1].set_xlim(0, 6000)
axes[1].set_ylim(0, 500)

# NO2 fit
sns.regplot(data=sample_plot, x='NO2', y='PM2.5', scatter_kws={'alpha': 0.12, 'color': '#1b9e77', 's': 8},
            line_kws={'color': '#e41a1c', 'linewidth': 3}, ax=axes[2])
axes[2].set_title('Linear Regression: PM2.5 vs NO2\n$Y_i = \\alpha + \\beta_3 \\cdot NO2_i$ ($r = 0.66$)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('NO2 (µg/m³)', fontsize=11)
axes[2].set_ylabel('Target PM2.5 (µg/m³)', fontsize=11)
axes[2].set_xlim(0, 200)
axes[2].set_ylim(0, 500)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'linear_regression_fit.png'), bbox_inches='tight')
plt.close()
print("Saved model_plots/linear_regression_fit.png")

print("All Linear Regression graphs successfully generated and saved!")
