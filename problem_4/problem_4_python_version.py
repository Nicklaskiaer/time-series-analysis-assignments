# Problem 4: WLS - Local Linear Trend Model (Python version)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------------------------------------
# Read data (replace this with your own "read_data.py" logic if needed)
# Expected columns in Dtrain/Dtest: time, year, total
# time should be datetime-like
# ---------------------------------------------------------------------

# Example:
# If you already have Dtrain and Dtest from another script, comment out the lines above.

# ---------------------------------------------------------------------
# 4.1 Describe the variance-covariance matrix Σ for the local model
# Σ = diag(1 / w), where w = lambda^((N-1), ..., 0)
# ---------------------------------------------------------------------

lambda_ = 0.9

# Ensure chronological order
Dtrain = Dtrain.sort_values("time").reset_index(drop=True)
Dtest = Dtest.sort_values("time").reset_index(drop=True)

N = len(Dtrain)

# Weights from oldest to newest
w = lambda_ ** np.arange(N - 1, -1, -1)  # same as lambda^((N-1):0) in R

# Construct Sigma = diag(1 / w)
Sigma = np.diag(1 / w)

print("N =", N)
print("First 5 lambda-weights:", w[:5])
print("Last 5 lambda-weights:", w[-5:])
print("Sigma shape:", Sigma.shape)
print("Top-left 5x5 block of Sigma:\n", Sigma[:5, :5])
print("Bottom-right 5x5 block of Sigma:\n", Sigma[-5:, -5:])

# ---------------------------------------------------------------------
# 4.2 Plot lambda-weights vs time
# Which time-point has the highest weight?
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(Dtrain["time"], w, marker="o")
plt.xlabel("Time")
plt.ylabel("Lambda-weights")
plt.title("Lambda-weights vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

last_time_point = Dtrain["time"].iloc[-1]
last_weight = w[-1]
print("Last time point:", last_time_point)
print("Last weight:", last_weight)  # should be 1

# ---------------------------------------------------------------------
# 4.3 Sum of lambda-weights
# Corresponding sum in OLS would be N (all weights = 1)
# ---------------------------------------------------------------------

sum_weights = w.sum()
print("Sum of lambda-weights:", sum_weights)
print("Corresponding OLS sum of weights:", N)

# ---------------------------------------------------------------------
# 4.4 Estimate theta_1 and theta_2 for WLS model with lambda = 0.9
# y = total, X = [1, year]
# theta = (X^T Sigma^{-1} X)^(-1) X^T Sigma^{-1} y
# Since Sigma = diag(1/w), Sigma^{-1} = diag(w)
# ---------------------------------------------------------------------

y = Dtrain["total"].to_numpy()
X = np.column_stack([np.ones(N), Dtrain["year"].to_numpy()])

W = np.diag(w)  # Sigma^{-1}

theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
print("theta (manual WLS estimate):")
print(theta)  # [theta_1, theta_2]

# Also fit OLS/WLS with statsmodels for forecasting and intervals
X_train_sm = sm.add_constant(Dtrain["year"])
X_test_sm = sm.add_constant(Dtest["year"])

fit_ols = sm.OLS(Dtrain["total"], X_train_sm).fit()
fit_wls = sm.WLS(Dtrain["total"], X_train_sm, weights=w).fit()

print("\nOLS params:")
print(fit_ols.params)
print("\nWLS params:")
print(fit_wls.params)

# ---------------------------------------------------------------------
# 4.5 Forecast next 12 months (test period) with prediction intervals
# ---------------------------------------------------------------------

# OLS predictions + prediction intervals
pred_ols_sf = fit_ols.get_prediction(X_test_sm).summary_frame(alpha=0.05)

# WLS predictions + prediction intervals
pred_wls_sf = fit_wls.get_prediction(X_test_sm).summary_frame(alpha=0.05)

# statsmodels summary_frame columns typically include:
# mean, mean_se, mean_ci_lower, mean_ci_upper, obs_ci_lower, obs_ci_upper

forecast_table = pd.DataFrame({
    "time": Dtest["time"].values,
    "year": Dtest["year"].values,
    "y_obs": Dtest["total"].values,
    "ols_fit": pred_ols_sf["mean"].values,
    "ols_lwr": pred_ols_sf["obs_ci_lower"].values,
    "ols_upr": pred_ols_sf["obs_ci_upper"].values,
    "wls_fit": pred_wls_sf["mean"].values,
    "wls_lwr": pred_wls_sf["obs_ci_lower"].values,
    "wls_upr": pred_wls_sf["obs_ci_upper"].values,
})

print("\nForecast table (head):")
print(forecast_table.head())

# ---------------------------------------------------------------------
# Plot 1: Last part of train + full test + prediction intervals
# ---------------------------------------------------------------------

x_left = pd.Timestamp("2023-01-01", tz="UTC") if str(Dtrain["time"].dt.tz.iloc[0]) != "None" else pd.Timestamp("2023-01-01")
x_right = Dtest["time"].max() + pd.Timedelta(days=35)

train_win = Dtrain[(Dtrain["time"] >= x_left) & (Dtrain["time"] <= x_right)].copy()

y_all = np.concatenate([
    train_win["total"].to_numpy(),
    Dtest["total"].to_numpy(),
    pred_ols_sf[["mean", "obs_ci_lower", "obs_ci_upper"]].to_numpy().ravel(),
    pred_wls_sf[["mean", "obs_ci_lower", "obs_ci_upper"]].to_numpy().ravel()
])

ylim = [np.nanmin(y_all), np.nanmax(y_all)]
pad = 0.02 * (ylim[1] - ylim[0]) if ylim[1] > ylim[0] else 1.0
ylim = [ylim[0] - pad, ylim[1] + pad]

plt.figure(figsize=(12, 6))

# 3) Training observations
plt.scatter(train_win["time"], train_win["total"], color="black", s=25, label="Train obs")

# 4) Test observations
plt.scatter(Dtest["time"], Dtest["total"], color="red", s=25, label="Test obs")

# 5) OLS forecast + PI
plt.plot(Dtest["time"], pred_ols_sf["mean"], linewidth=2, label="OLS forecast")
plt.plot(Dtest["time"], pred_ols_sf["obs_ci_lower"], linestyle="--", linewidth=1, label="OLS prediction interval")
plt.plot(Dtest["time"], pred_ols_sf["obs_ci_upper"], linestyle="--", linewidth=1)

# 6) WLS forecast + PI
plt.plot(Dtest["time"], pred_wls_sf["mean"], linewidth=2, label="WLS forecast")
plt.plot(Dtest["time"], pred_wls_sf["obs_ci_lower"], linestyle=":", linewidth=1, label="WLS prediction interval")
plt.plot(Dtest["time"], pred_wls_sf["obs_ci_upper"], linestyle=":", linewidth=1)

plt.xlim(x_left, x_right)
plt.ylim(ylim)
plt.xlabel("Time")
plt.ylabel("Total (millions)")
plt.title("OLS vs WLS: 12-month forecast with prediction intervals")
plt.legend(loc="lower right", frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Plot 2: Full history (train + test) with OLS/WLS forecasts and PIs
# ---------------------------------------------------------------------

x_left_full = Dtrain["time"].min()
x_right_full = Dtest["time"].max() + pd.Timedelta(days=35)

y_all_full = np.concatenate([
    Dtrain["total"].to_numpy(),
    Dtest["total"].to_numpy(),
    pred_ols_sf[["mean", "obs_ci_lower", "obs_ci_upper"]].to_numpy().ravel(),
    pred_wls_sf[["mean", "obs_ci_lower", "obs_ci_upper"]].to_numpy().ravel()
])

ylim_full = [np.nanmin(y_all_full), np.nanmax(y_all_full)]
pad_full = 0.02 * (ylim_full[1] - ylim_full[0]) if ylim_full[1] > ylim_full[0] else 1.0
ylim_full = [ylim_full[0] - pad_full, ylim_full[1] + pad_full]

plt.figure(figsize=(12, 6))

# 3) Full training data
plt.scatter(Dtrain["time"], Dtrain["total"], color="black", s=20, label="Train obs")

# 4) Test observations
plt.scatter(Dtest["time"], Dtest["total"], color="red", s=20, label="Test obs")

# 5) OLS forecast + PI
plt.plot(Dtest["time"], pred_ols_sf["mean"], linewidth=2, label="OLS forecast")
plt.plot(Dtest["time"], pred_ols_sf["obs_ci_lower"], linestyle="--", linewidth=1, label="OLS prediction interval")
plt.plot(Dtest["time"], pred_ols_sf["obs_ci_upper"], linestyle="--", linewidth=1)

# 6) WLS forecast + PI
plt.plot(Dtest["time"], pred_wls_sf["mean"], linewidth=2, label="WLS forecast")
plt.plot(Dtest["time"], pred_wls_sf["obs_ci_lower"], linestyle=":", linewidth=1, label="WLS prediction interval")
plt.plot(Dtest["time"], pred_wls_sf["obs_ci_upper"], linestyle=":", linewidth=1)

plt.xlim(x_left_full, x_right_full)
plt.ylim(ylim_full)
plt.xlabel("Time")
plt.ylabel("Total (millions)")
plt.title("OLS vs WLS: Full history with 12-month forecasts")
plt.legend(loc="upper left", frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()