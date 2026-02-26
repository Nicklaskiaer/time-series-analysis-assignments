import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "figures"


def plot_data():
    # --- 1) Read data ---
    df = pd.read_csv("DST_BIL54.csv")
    df["time"] = pd.to_datetime(df["time"] + "-01", format="%Y-%m-%d", utc=True)
    df = df.sort_values("time")

    # --- 2) Create x like in the R script: year + mon/12 (mon is 0..11) ---
    # In pandas: month is 1..12, so (month-1)/12 matches R's mon/12
    df["x"] = df["time"].dt.year + (df["time"].dt.month - 1) / 12.0

    # --- 3) Scale total like in the R script (millions) ---
    df["total"] = pd.to_numeric(df["total"], errors="coerce") / 1e6

    # --- 4) Split train/test like in the R script ---
    teststart = pd.Timestamp("2024-01-01", tz="UTC")
    train = df[df["time"] < teststart].copy()
    test = df[df["time"] >= teststart].copy()

    # --- 5) Plot training data vs x ---
    plt.figure()
    plt.plot(train["x"], train["total"])
    plt.xlabel("x (year + (month-1)/12)")
    plt.ylabel("Total registered motor-driven vehicles (millions)")
    plt.title("Denmark vehicles (training set: 2018-01 to 2023-12)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "1_total_reg_vehicles.png"))

    # --- 6) Quick numeric summary (training only) ---
    start_time = train["time"].iloc[0]
    end_time = train["time"].iloc[-1]
    start_val = train["total"].iloc[0]
    end_val = train["total"].iloc[-1]
    n = len(train)

    abs_change = end_val - start_val
    pct_change = 100.0 * abs_change / start_val
    avg_monthly_change = abs_change / (n - 1)

    overall_mean = train["total"].mean()
    train["month"] = train["time"].dt.month
    month_means = train.groupby("month")["total"].mean()
    month_dev = (month_means - overall_mean).sort_values()

    print("TRAINING PERIOD:", start_time.strftime("%Y-%m"), "to", end_time.strftime("%Y-%m"))
    print(f"START total (M): {start_val:.6g}")
    print(f"END   total (M): {end_val:.6g}")
    print(f"CHANGE (M): {abs_change:.6g}  ({pct_change:.2f}%)")
    print(f"AVG monthly change (M, rough): {avg_monthly_change:.6g}")

    low_month = int(month_dev.index[0])
    high_month = int(month_dev.index[-1])
    print("\nSEASONALITY (very simple):")
    print("Lowest average month:", low_month, "deviation (M):", f"{month_dev.iloc[0]:.6g}")
    print("Highest average month:", high_month, "deviation (M):", f"{month_dev.iloc[-1]:.6g}")

    print("\nMonth deviations (mean(month) - overall_mean), in millions:")
    print(month_dev.round(6).to_string())

    return train, test


def linear_trend_model(train):
    # Expect train to contain columns: 'x' (time variable) and 'total' (Y)
    y_all = train["total"].to_numpy(dtype=float)
    x_all = train["x"].to_numpy(dtype=float)
    N = len(y_all)

    # Time index t = 1,...,N (not strictly needed for X here, but matches the assignment statement)
    # t = np.arange(1, N + 1)

    # Design matrix for ALL points: X = [1, x]
    X_all = np.column_stack([np.ones(N), x_all])

    # --- 2.1: Matrix form for first 3 time points ---
    y3 = y_all[:3]
    x3 = x_all[:3]

    # 1) Symbolic matrix form
    s1 = (
        "Model: y = Xθ + ε\n\n"
        "For t = 1,2,3:\n"
        "[Y1]   [1  x1] [θ1]   [ε1]\n"
        "[Y2] = [1  x2] [θ2] + [ε2]\n"
        "[Y3]   [1  x3]        [ε3]\n"
    )

    # 2) Insert elements (still symbolic, just explicit matrices/vectors)
    s2 = (
        "Vectors/matrices:\n"
        "y = [Y1, Y2, Y3]^T\n"
        "θ = [θ1, θ2]^T\n"
        "ε = [ε1, ε2, ε3]^T\n\n"
        "X = [[1, x1],\n"
        "     [1, x2],\n"
        "     [1, x3]]\n"
    )

    # 3) Insert actual numeric values (max 3 digits; we'll use rounding)
    # Note: totals are large, so rounding to 3 significant digits is more sensible than 3 decimals.
    def sig3(v):
        # 3 significant digits formatting
        return f"{v:.3g}"

    y3_fmt = [sig3(v) for v in y3]
    x3_fmt = [sig3(v) for v in x3]

    s3 = (
        "Numeric (rounded to 3 significant digits):\n"
        f"y = [{y3_fmt[0]}, {y3_fmt[1]}, {y3_fmt[2]}]^T\n\n"
        "X = [[1, " + x3_fmt[0] + "],\n"
        "     [1, " + x3_fmt[1] + "],\n"
        "     [1, " + x3_fmt[2] + "]]\n"
    )

    # Print to console (handy for copying into report)
    print(s1)
    print(s2)
    print(s3)

    # Save as a picture for your report
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    full_text = s1 + "\n" + s2 + "\n" + s3
    plt.text(0.01, 0.99, full_text, va="top", family="monospace")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "2_linear_trend_model_first3.png"), dpi=200)
    plt.show()

    # (Optional, but useful later) Fit θ by OLS on ALL training data:
    # θ_hat = (X^T X)^(-1) X^T y
    theta_hat = np.linalg.lstsq(X_all, y_all, rcond=None)[0]
    y_hat = X_all @ theta_hat
    resid = y_all - y_hat
    sigma2_hat = (resid @ resid) / (N - 2)

    print("\nOLS fit on full training set:")
    print(f"theta_hat = [{theta_hat[0]:.6g}, {theta_hat[1]:.6g}]")
    print(f"sigma^2_hat = {sigma2_hat:.6g}")

    # Return things in case you need them later
    return X_all, y_all, theta_hat, sigma2_hat


def ols_global_linear_trend_model(train, test):
    # --- Build y and X for training ---
    y = train["total"].to_numpy(dtype=float)
    x = train["x"].to_numpy(dtype=float)
    N = len(y)
    X = np.column_stack([np.ones(N), x])  # [1, x]

    # --- 3.1 OLS estimates: theta_hat = (X'X)^(-1) X'y ---
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    theta_hat = XtX_inv @ (X.T @ y)

    # Fitted values and residuals
    y_hat = X @ theta_hat
    resid = y - y_hat

    # Estimate sigma^2 and standard errors of parameters
    df = N - 2
    sigma2_hat = (resid @ resid) / df
    cov_theta = sigma2_hat * XtX_inv
    se_theta = np.sqrt(np.diag(cov_theta))

    # Print parameter estimates + SEs (3.2)
    print("3.1 How estimates were calculated:")
    print("theta_hat = (X^T X)^(-1) X^T y")
    print("sigma^2_hat = sum(resid^2)/(N-2)")
    print("Var(theta_hat) = sigma^2_hat * (X^T X)^(-1)\n")

    print("3.2 OLS estimates and standard errors:")
    print(f"theta1_hat = {theta_hat[0]:.6g}   SE = {se_theta[0]:.6g}")
    print(f"theta2_hat = {theta_hat[1]:.6g}   SE = {se_theta[1]:.6g}")
    print(f"sigma^2_hat = {sigma2_hat:.6g}\n")

    # Plot fitted mean line with observations (3.2)
    plt.figure()
    plt.scatter(train["x"], train["total"], s=12)
    plt.plot(train["x"], y_hat)
    plt.xlabel("x (year + (month-1)/12)")
    plt.ylabel("Total registered vehicles")
    plt.title("Training data + fitted global linear trend (mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3_ols_fitted_mean.png"))
    plt.show()

    # --- 3.3 Forecast test set with prediction intervals ---
    # Build X for test
    x0 = test["x"].to_numpy(dtype=float)
    X0 = np.column_stack([np.ones(len(x0)), x0])
    y_pred = X0 @ theta_hat

    # Prediction interval: y0 ± t * sqrt( sigma^2_hat * (1 + x0'(X'X)^-1 x0) )
    # Use a normal approx for simplicity (≈ 95% => 1.96). If you want exact t, swap z->tcrit.
    z = 1.96
    # Efficient diagonal of X0 (XtX_inv) X0^T:
    # pred_var_mean = np.sum((X0 @ XtX_inv) * X0, axis=1) * sigma2_hat
    pred_var_obs = sigma2_hat * (1.0 + np.sum((X0 @ XtX_inv) * X0, axis=1))
    pred_se_obs = np.sqrt(pred_var_obs)

    lower = y_pred - z * pred_se_obs
    upper = y_pred + z * pred_se_obs

    forecast_table = pd.DataFrame({
        "time": test["time"].dt.strftime("%Y-%m"),
        "x": test["x"],
        "y_pred": y_pred,
        "PI_lower_95": lower,
        "PI_upper_95": upper
    })

    print("3.3 Forecast table (test set) with ~95% prediction intervals:")
    # show nice rounded values
    print(forecast_table.round(2).to_string(index=False))
    print()

    # --- 3.4 Plot fitted + training + forecast + prediction intervals ---
    plt.figure()
    plt.scatter(train["x"], train["total"], s=12, label="Train obs")
    plt.plot(train["x"], y_hat, label="Fitted mean (train)")

    plt.scatter(test["x"], test["total"], s=12, label="Test obs")
    plt.plot(test["x"], y_pred, label="Forecast mean (test)")
    plt.fill_between(test["x"], lower, upper, alpha=0.2, label="~95% Prediction interval")

    plt.xlabel("x (year + (month-1)/12)")
    plt.ylabel("Total registered vehicles")
    plt.title("Global linear trend: fit + 12-month forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3_ols_fit_forecast_pi.png"))
    plt.show()

    # --- 3.5 Comment on forecast (basic, automated hints) ---
    # Compare actual test vs PI coverage (if test has totals)
    if "total" in test.columns:
        y_test = test["total"].to_numpy(dtype=float)
        inside = (y_test >= lower) & (y_test <= upper)
        coverage = inside.mean()
        mae = np.mean(np.abs(y_test - y_pred))
        print("3.5 Forecast comment (quick metrics):")
        print(f"MAE on test = {mae:,.1f}")
        print(f"Fraction of test points inside ~95% PI = {coverage:.2f}")
        print("If many points fall outside intervals or errors are systematic, the forecast is not good.\n")

    # --- 3.6 Residual diagnostics ---
    # Residual vs fitted
    plt.figure()
    plt.scatter(y_hat, resid, s=12)
    plt.axhline(0)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs fitted")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3_residuals_vs_fitted.png"))
    plt.show()

    # Histogram of residuals
    plt.figure()
    plt.hist(resid, bins=20)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3_residual_histogram.png"))
    plt.show()

    # QQ plot (simple, no extra libraries): compare sorted residuals to normal quantiles
    r = np.sort(resid)
    p = (np.arange(1, N + 1) - 0.5) / N
    # normal quantiles using inverse error function approximation via numpy:
    # q = sqrt(2)*erfinv(2p-1)
    # numpy has np.erf but not erfinv in older builds; so we do a normal approx using np.quantile
    # (This fallback is less "textbook", but keeps dependencies minimal.)
    # If your numpy has erfinv: use it. Otherwise skip QQ.
    has_erfinv = hasattr(np, "erfinv")
    if has_erfinv:
        q = np.sqrt(2) * np.erfinv(2 * p - 1)  # type: ignore
        plt.figure()
        plt.scatter(q, r, s=12)
        plt.xlabel("Normal quantiles")
        plt.ylabel("Sorted residuals")
        plt.title("QQ plot (residuals vs Normal)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "3_residual_qq.png"))
        plt.show()
    else:
        print("QQ plot skipped (np.erfinv not available in this numpy build).")

    # Autocorrelation (lag 1) + Durbin-Watson for independence
    lag1 = np.corrcoef(resid[1:], resid[:-1])[0, 1]
    dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)
    print("3.6 Residual checks (numbers):")
    print(f"Lag-1 residual autocorr ≈ {lag1:.3f} (should be near 0 for i.i.d.)")
    print(f"Durbin-Watson ≈ {dw:.3f} (≈2 suggests no autocorrelation)")
    print("Look for: non-random pattern in residual-vs-fitted, heavy tails / skew in histogram/QQ,")
    print("and autocorrelation (lag1 far from 0, DW far from 2).")

    return forecast_table, theta_hat, se_theta, sigma2_hat


def wls_local_linear_trend_model(train, test, lambda_=0.9):
    """
    Problem 4: WLS local linear trend model using exponentially decaying weights.

    Weights are assigned from oldest -> newest as:
        w_t = lambda^(N-1), ..., lambda^1, lambda^0
    so the newest training point gets weight 1.
    """
    # --- Build y and X for training ---
    y = train["total"].to_numpy(dtype=float)
    x = train["x"].to_numpy(dtype=float)
    N = len(y)
    X = np.column_stack([np.ones(N), x])  # [1, x]

    # --- 4.1 Variance-covariance matrix Sigma for local model ---
    # Weights from oldest to newest
    w = lambda_ ** np.arange(N - 1, -1, -1)   # oldest smallest, newest = 1

    # Sigma = diag(1 / w), and Sigma^{-1} = W = diag(w)
    Sigma = np.diag(1.0 / w)
    W = np.diag(w)

    print("\n" + "=" * 70)
    print("4.1 Local model variance-covariance matrix (WLS)")
    print(f"lambda = {lambda_}")
    print(f"N = {N}")
    print("Sigma = diag(1/w), where w = [lambda^(N-1), ..., 1]")
    print("Top-left 5x5 of Sigma:")
    print(np.round(Sigma[:5, :5], 4))
    print("Bottom-right 5x5 of Sigma:")
    print(np.round(Sigma[-5:, -5:], 4))
    print("Global OLS comparison: Sigma_OLS = sigma^2 * I (equal variance / equal weighting).")

    # --- 4.2 Plot lambda-weights vs time ---
    plt.figure()
    plt.plot(train["time"], w, marker="o", markersize=3)
    plt.xlabel("Time")
    plt.ylabel("Lambda-weights")
    plt.title(f"Lambda-weights vs time (lambda={lambda_})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "4_lambda_weights.png"))
    plt.show()

    last_time_point = train["time"].iloc[-1]
    last_weight = w[-1]
    print("\n4.2 Lambda-weights:")
    print("Highest weight is at the newest training time point.")
    print("Newest time point:", last_time_point.strftime("%Y-%m"))
    print(f"Weight at newest point: {last_weight:.6g}")

    # --- 4.3 Sum of weights ---
    sum_weights = np.sum(w)
    print("\n4.3 Sum of lambda-weights:")
    print(f"sum(w) = {sum_weights:.6g}")
    print(f"Corresponding OLS sum of weights = N = {N}")

    # --- 4.4 Manual WLS estimate theta = (X'WX)^(-1) X'Wy ---
    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    theta_wls = XtWX_inv @ (X.T @ W @ y)

    print("\n4.4 WLS parameter estimates (manual matrix formula):")
    print("theta_hat_wls = (X^T W X)^(-1) X^T W y")
    print(f"theta1_hat = {theta_wls[0]:.6g}")
    print(f"theta2_hat = {theta_wls[1]:.6g}")

    # Fitted values and residuals (train)
    y_hat_train = X @ theta_wls
    resid = y - y_hat_train

    # Weighted residual variance estimate (rough)
    # p = 2 parameters
    p = 2
    sigma2_wls_hat = (resid.T @ W @ resid) / (N - p)

    # Approximate covariance for theta under WLS (using sigma^2 * (X'WX)^-1)
    cov_theta_wls = sigma2_wls_hat * XtWX_inv
    se_theta_wls = np.sqrt(np.diag(cov_theta_wls))

    print("\nApprox. WLS standard errors:")
    print(f"SE(theta1_hat) = {se_theta_wls[0]:.6g}")
    print(f"SE(theta2_hat) = {se_theta_wls[1]:.6g}")
    print(f"sigma^2_wls_hat (weighted) = {sigma2_wls_hat:.6g}")

    # --- 4.5 Forecast next 12 months (test period) ---
    x0 = test["x"].to_numpy(dtype=float)
    X0 = np.column_stack([np.ones(len(x0)), x0])
    y_pred = X0 @ theta_wls

    # Prediction interval (approx):
    # Var(y0 - yhat0) ≈ sigma^2 * (1 + x0'(X'WX)^-1 x0)
    z = 1.96
    pred_var_obs = sigma2_wls_hat * (1.0 + np.sum((X0 @ XtWX_inv) * X0, axis=1))
    pred_se_obs = np.sqrt(pred_var_obs)

    lower = y_pred - z * pred_se_obs
    upper = y_pred + z * pred_se_obs

    forecast_table = pd.DataFrame({
        "time": test["time"].dt.strftime("%Y-%m"),
        "x": test["x"],
        "y_obs": test["total"].to_numpy(dtype=float),
        "wls_pred": y_pred,
        "PI_lower_95": lower,
        "PI_upper_95": upper
    })

    print("\n4.5 WLS forecast table (test set) with ~95% prediction intervals:")
    print(forecast_table.round(2).to_string(index=False))
    print()

    # --- Plot: train + fitted (WLS) + test + forecast + PI ---
    plt.figure()
    plt.scatter(train["x"], train["total"], s=12, label="Train obs")
    plt.plot(train["x"], y_hat_train, label=f"WLS fitted mean (train), λ={lambda_}")

    plt.scatter(test["x"], test["total"], s=12, label="Test obs")
    plt.plot(test["x"], y_pred, label="WLS forecast mean (test)")
    plt.fill_between(test["x"], lower, upper, alpha=0.2, label="~95% Prediction interval (WLS)")

    plt.xlabel("x (year + (month-1)/12)")
    plt.ylabel("Total registered vehicles (millions)")
    plt.title(f"Local linear trend (WLS, λ={lambda_}): fit + 12-month forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "4_wls_fit_forecast.png"))
    plt.show()

    # --- Comparison plot: OLS vs WLS on same figure (matches your R idea) ---
    # Reuse your OLS fit manually here for a clean side-by-side plot
    X_ols = X
    XtX = X_ols.T @ X_ols
    XtX_inv = np.linalg.inv(XtX)
    theta_ols = XtX_inv @ (X_ols.T @ y)
    y_hat_ols_train = X_ols @ theta_ols
    y_pred_ols = X0 @ theta_ols

    resid_ols = y - y_hat_ols_train
    sigma2_ols_hat = (resid_ols @ resid_ols) / (N - p)
    pred_var_obs_ols = sigma2_ols_hat * (1.0 + np.sum((X0 @ XtX_inv) * X0, axis=1))
    pred_se_obs_ols = np.sqrt(pred_var_obs_ols)
    lower_ols = y_pred_ols - z * pred_se_obs_ols
    upper_ols = y_pred_ols + z * pred_se_obs_ols

    plt.figure()
    plt.scatter(train["x"], train["total"], s=12, label="Train obs")
    plt.scatter(test["x"], test["total"], s=12, label="Test obs")

    plt.plot(test["x"], y_pred_ols, linewidth=2, label="OLS forecast")
    plt.plot(test["x"], lower_ols, linestyle="--", linewidth=1, label="OLS prediction interval")
    plt.plot(test["x"], upper_ols, linestyle="--", linewidth=1)

    plt.plot(test["x"], y_pred, linewidth=2, label=f"WLS forecast (λ={lambda_})")
    plt.plot(test["x"], lower, linestyle=":", linewidth=1, label="WLS prediction interval")
    plt.plot(test["x"], upper, linestyle=":", linewidth=1)

    plt.xlabel("x (year + (month-1)/12)")
    plt.ylabel("Total registered vehicles (millions)")
    plt.title("OLS vs WLS: 12-month forecast with prediction intervals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "4_ols_vs_wls_forecast.png"))
    plt.show()

    # --- Full history style plot (train + test + forecasts) ---
    plt.figure()
    plt.scatter(train["x"], train["total"], s=12, label="Train obs")
    plt.scatter(test["x"], test["total"], s=12, label="Test obs")

    plt.plot(test["x"], y_pred_ols, linewidth=2, label="OLS forecast")
    plt.plot(test["x"], lower_ols, linestyle="--", linewidth=1)
    plt.plot(test["x"], upper_ols, linestyle="--", linewidth=1)

    plt.plot(test["x"], y_pred, linewidth=2, label=f"WLS forecast (λ={lambda_})")
    plt.plot(test["x"], lower, linestyle=":", linewidth=1)
    plt.plot(test["x"], upper, linestyle=":", linewidth=1)

    plt.xlabel("x (year + (month-1)/12)")
    plt.ylabel("Total registered vehicles (millions)")
    plt.title("OLS vs WLS: Full history with 12-month forecasts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "4_ols_vs_wls_full_history.png"))
    plt.show()

    # Quick test metrics for WLS
    y_test = test["total"].to_numpy(dtype=float)
    inside = (y_test >= lower) & (y_test <= upper)
    coverage = inside.mean()
    mae = np.mean(np.abs(y_test - y_pred))

    print("WLS forecast comment (quick metrics):")
    print(f"MAE on test = {mae:,.6f}")
    print(f"Fraction of test points inside ~95% PI = {coverage:.2f}")

    return {
        "lambda": lambda_,
        "weights": w,
        "Sigma": Sigma,
        "theta_hat_wls": theta_wls,
        "se_theta_wls": se_theta_wls,
        "sigma2_wls_hat": sigma2_wls_hat,
        "forecast_table_wls": forecast_table
    }


def rls_model(train, test):
    """
    Problem 5: Recursive Least Squares (RLS).

    5.1: First two RLS iterations (compute R1, R2 and theta1, theta2).
    5.2: For-loop implementation of RLS for t = 1..3.
    """
    # Ensure training data is in chronological order (same as R: order(Dtrain$time))
    train = train.sort_values("time").reset_index(drop=True)

    y = train["total"].to_numpy(dtype=float)
    x_reg = train["x"].to_numpy(dtype=float)  # regressor (year + (month-1)/12)

    # Regressor vectors x_t = [1; x_t] as column vectors (2,1)
    def x_t_vec(i):
        return np.array([[1.0], [x_reg[i]]])

    # --- 5.1: First two RLS iterations ---
    y1, y2 = y[0], y[1]
    x1 = x_t_vec(0)
    x2 = x_t_vec(1)

    # Initial values given in the assignment
    R0 = np.diag([0.1, 0.1])
    theta0 = np.array([[0.0], [0.0]])

    # Iteration t = 1
    R1 = R0 + x1 @ x1.T
    err1 = y1 - (x1.T @ theta0).item()
    theta1 = theta0 + np.linalg.solve(R1, x1) * err1

    # Iteration t = 2
    R2 = R1 + x2 @ x2.T
    err2 = y2 - (x2.T @ theta1).item()
    theta2 = theta1 + np.linalg.solve(R2, x2) * err2

    print("\n" + "=" * 70)
    print("5.1 First two RLS iterations (R1, R2, theta1, theta2)")
    print("=" * 70)
    print("R1 =\n", R1)
    print("\nR2 =\n", R2)
    print("\n(Optional) theta1 =\n", theta1)
    print("\n(Optional) theta2 =\n", theta2)

    # --- 5.2: For-loop implementation of RLS for t = 1..3 ---
    R = np.diag([0.1, 0.1]).copy()
    theta_hat = np.array([[0.0], [0.0]])

    res_rows = []
    for t in range(3):  # t = 0,1,2 -> first 3 observations
        x_t = x_t_vec(t)
        y_t = y[t]

        # Update information matrix R_t = R_{t-1} + x_t x_t'
        R = R + x_t @ x_t.T

        # Prediction error (innovation): y_t - x_t' theta_{t-1}
        err = float(y_t - (x_t.T @ theta_hat).item())

        # Update parameter estimate: theta_hat_t = theta_hat_{t-1} + R_t^{-1} x_t * err
        theta_hat = theta_hat + np.linalg.solve(R, x_t) * err

        res_rows.append({
            "t": t + 1,
            "time": train["time"].iloc[t],
            "theta1": theta_hat[0, 0],
            "theta2": theta_hat[1, 0],
        })

    res = pd.DataFrame(res_rows)

    print("\n" + "=" * 70)
    print("5.2 RLS for t = 1..3 (theta_hat at each step)")
    print("=" * 70)
    print(res.to_string(index=False))

    print("\nFinal theta_hat at t = 3:")
    print(theta_hat)

    return {
        "R1": R1,
        "R2": R2,
        "theta1": theta1,
        "theta2": theta2,
        "rls_steps": res,
        "theta_hat_t3": theta_hat,
    }


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    # Problem 1: Plot data
    print("Problem 1: Plot data")
    train, test = plot_data()
    print("Problem 2: Linear trend model")
    X_all, y_all, theta_hat, sigma2_hat = linear_trend_model(train)
    print("Problem 3: OLS global linear trend model")
    forecast_table, theta_hat, se_theta, sigma2_hat = ols_global_linear_trend_model(train, test)
    print("Problem 4: WLS local linear trend model")
    wls_results = wls_local_linear_trend_model(train, test, lambda_=0.9)
    print("Problem 5: RLS")
    rls_results = rls_model(train, test)
    