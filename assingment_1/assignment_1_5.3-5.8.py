
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"

def read_and_split(csv_path="DST_BIL54.csv"):
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"] + "-01", format="%Y-%m-%d", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["x"] = df["time"].dt.year + (df["time"].dt.month - 1) / 12.0
    df["total"] = pd.to_numeric(df["total"], errors="coerce") / 1e6  # millions
    teststart = pd.Timestamp("2024-01-01", tz="UTC")
    train = df[df["time"] < teststart].copy().reset_index(drop=True)
    test = df[df["time"] >= teststart].copy().reset_index(drop=True)
    return train, test

def ols_theta(train):
    y = train["total"].to_numpy(dtype=float)
    x = train["x"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), x])
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    return theta

def wls_theta(train, lambda_):
    y = train["total"].to_numpy(dtype=float)
    x = train["x"].to_numpy(dtype=float)
    N = len(y)
    X = np.column_stack([np.ones(N), x])
    w = lambda_ ** np.arange(N - 1, -1, -1)  # oldest -> newest, newest weight = 1
    W = np.diag(w)
    theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    return theta

def rls_run(train, lambda_=1.0, R0_scale=1e-8, theta0=None):
    """
    RLS using information matrix R (same notation as assignment):
      R_t = lambda * R_{t-1} + x_t x_t^T
      theta_t = theta_{t-1} + R_t^{-1} x_t (y_t - x_t^T theta_{t-1})

    With lambda=1 this is RLS without forgetting.
    """
    y = train["total"].to_numpy(dtype=float)
    x = train["x"].to_numpy(dtype=float)
    N = len(y)

    R = np.diag([R0_scale, R0_scale]).astype(float)
    theta = np.zeros((2, 1)) if theta0 is None else np.array(theta0, dtype=float).reshape(2, 1)

    thetas = np.zeros((N, 2))
    for t in range(N):
        xt = np.array([[1.0], [x[t]]])
        if lambda_ == 1.0:
            R = R + xt @ xt.T
        else:
            R = lambda_ * R + xt @ xt.T

        err = y[t] - (xt.T @ theta).item()
        theta = theta + np.linalg.solve(R, xt) * err
        thetas[t, :] = theta.ravel()

    return thetas

def onestep_residuals(train, thetas, burn_in=4):
    """
    epsilon_{t|t-1} = yhat_{t|t-1} - y_t, for t = 2..N
    where yhat_{t|t-1} uses theta_{t-1}.
    We return residuals aligned with time index t (1-based in assignment).
    """
    y = train["total"].to_numpy(dtype=float)
    x = train["x"].to_numpy(dtype=float)
    times = train["time"].to_numpy()

    N = len(y)
    res = []
    res_times = []
    for t in range(1, N):  # predict y[t] using theta at t-1
        theta_prev = thetas[t - 1, :]
        yhat = np.array([1.0, x[t]]) @ theta_prev
        res.append(yhat - y[t])
        res_times.append(times[t])

    res = np.array(res)
    res_times = np.array(res_times)

    # Drop burn-in: residual at index 3 corresponds to t=4, etc.
    return res_times[burn_in:], res[burn_in:]

def rmse_grid(train, lambdas, horizons=range(1, 13), R0_scale=1e-8):
    """
    For each lambda and horizon k:
      epsilon_{t+k|t} = yhat_{t+k|t} - y_{t+k}, with yhat using theta_t.
    RMSE_k(lambda) = sqrt(mean(epsilon^2)) over t=1..N-k.
    """
    y = train["total"].to_numpy(dtype=float)
    x = train["x"].to_numpy(dtype=float)
    N = len(y)

    rmse = np.zeros((len(list(horizons)), len(lambdas)))

    for j, lam in enumerate(lambdas):
        thetas = rls_run(train, lambda_=lam, R0_scale=R0_scale)
        for i, k in enumerate(horizons):
            errs = []
            for t in range(0, N - k):
                theta_t = thetas[t, :]
                yhat = np.array([1.0, x[t + k]]) @ theta_t
                errs.append(yhat - y[t + k])
            rmse[i, j] = np.sqrt(np.mean(np.array(errs) ** 2))

    return rmse

def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def problem_5_outputs(train, test):
    ensure_figures_dir()

    # 5.3: show how R0 affects final theta_N (lambda=1)
    theta_ols = ols_theta(train)
    thetas_bad = rls_run(train, lambda_=1.0, R0_scale=0.1)
    thetas_good = rls_run(train, lambda_=1.0, R0_scale=1e-8)

    print("\n5.3 RLS vs OLS at t=N")
    print("OLS theta =", theta_ols)
    print("RLS theta (R0=0.1 I)   =", thetas_bad[-1])
    print("RLS theta (R0=1e-8 I)  =", thetas_good[-1])

    # 5.4: parameter paths for lambda=0.7 and 0.99
    lam_a, lam_b = 0.7, 0.99
    thetas_a = rls_run(train, lambda_=lam_a, R0_scale=1e-8)
    thetas_b = rls_run(train, lambda_=lam_b, R0_scale=1e-8)

    burn = 4
    times = train["time"]

    plt.figure(figsize=(8, 4))
    plt.plot(times[burn:], thetas_a[burn:, 0], label=f"λ={lam_a}")
    plt.plot(times[burn:], thetas_b[burn:, 0], label=f"λ={lam_b}")
    plt.title("RLS with forgetting: intercept")
    plt.ylabel(r"$\hat\theta_{1,t}$")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_4_theta1_paths.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(times[burn:], thetas_a[burn:, 1], label=f"λ={lam_a}")
    plt.plot(times[burn:], thetas_b[burn:, 1], label=f"λ={lam_b}")
    plt.title("RLS with forgetting: slope")
    plt.ylabel(r"$\hat\theta_{2,t}$")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_4_theta2_paths.png"), dpi=200)
    plt.close()

    # Compare theta_N to WLS for same lambda
    theta_wls_a = wls_theta(train, lam_a)
    theta_wls_b = wls_theta(train, lam_b)
    print("\n5.4 Compare theta_N (RLS) to WLS:")
    print(f"lambda={lam_a}: RLS theta_N={thetas_a[-1]}  vs WLS theta={theta_wls_a}")
    print(f"lambda={lam_b}: RLS theta_N={thetas_b[-1]}  vs WLS theta={theta_wls_b}")

    # 5.5: one-step residual plots
    tA, rA = onestep_residuals(train, thetas_a, burn_in=4)
    tB, rB = onestep_residuals(train, thetas_b, burn_in=4)

    plt.figure(figsize=(8, 4))
    plt.plot(tA, rA)
    plt.axhline(0, linewidth=1)
    plt.title("One-step residuals (RLS, λ=0.7)")
    plt.ylabel(r"$\hat\epsilon_{t|t-1}$")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_5_onestep_residuals_lam07.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(tB, rB)
    plt.axhline(0, linewidth=1)
    plt.title("One-step residuals (RLS, λ=0.99)")
    plt.ylabel(r"$\hat\epsilon_{t|t-1}$")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_5_onestep_residuals_lam099.png"), dpi=200)
    plt.close()

    # 5.6: optimize lambda by horizon
    lambdas = np.round(np.arange(0.50, 1.00, 0.01), 2)  # 0.50..0.99
    horizons = list(range(1, 13))
    rmse = rmse_grid(train, lambdas, horizons=horizons, R0_scale=1e-8)

    plt.figure(figsize=(8, 5))
    for i, k in enumerate(horizons):
        plt.plot(lambdas, rmse[i], label=f"k={k}")
    plt.title("RMSE vs forgetting factor λ (k=1..12)")
    plt.xlabel("λ")
    plt.ylabel("RMSE_k (millions)")
    plt.legend(ncol=3, fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_6_rmse_vs_lambda_allk.png"), dpi=200)
    plt.close()

    opt_idx = rmse.argmin(axis=1)
    opt_lambda = lambdas[opt_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(horizons, opt_lambda, marker="o")
    plt.title("Optimal λ by prediction horizon")
    plt.xlabel("Horizon k (months)")
    plt.ylabel("λ* (minimizes RMSE_k)")
    plt.ylim(0.45, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_6_opt_lambda_by_horizon.png"), dpi=200)
    plt.close()

    print("\n5.6 Optimal lambda by horizon:")
    for k, lam in zip(horizons, opt_lambda):
        print(f"k={k:2d}: lambda*={lam}")

    # 5.7: forecast test set with (a) single lambda and (b) horizon-dependent lambdas
    x_test = test["x"].to_numpy(dtype=float)
    X_test = np.column_stack([np.ones(len(x_test)), x_test])

    # Choose a single lambda (example: lambda=0.9 to compare to WLS in problem 4)
    lam_single = 0.9
    theta_single = rls_run(train, lambda_=lam_single, R0_scale=1e-8)[-1]
    y_pred_single = X_test @ theta_single

    # Horizon-dependent: compute theta_N for each lambda*(k)
    y_pred_h = np.zeros(len(x_test))
    for k in range(1, 13):
        lam_k = opt_lambda[k - 1]
        theta_k = rls_run(train, lambda_=lam_k, R0_scale=1e-8)[-1]
        y_pred_h[k - 1] = np.array([1.0, x_test[k - 1]]) @ theta_k

    # Compare to OLS and WLS(0.9)
    y_pred_ols = X_test @ theta_ols
    theta_wls09 = wls_theta(train, 0.9)
    y_pred_wls = X_test @ theta_wls09

    y_test = test["total"].to_numpy(dtype=float)
    mae = lambda a, b: np.mean(np.abs(a - b))

    print("\n5.7 Test MAE (millions):")
    print("OLS MAE:            ", mae(y_test, y_pred_ols))
    print("WLS (lambda=0.9) MAE", mae(y_test, y_pred_wls))
    print("RLS (lambda=0.9) MAE", mae(y_test, y_pred_single))
    print("RLS horizon-dep. MAE ", mae(y_test, y_pred_h))

    # Plot comparison
    # show last part of train for readability
    mask = train["time"] >= pd.Timestamp("2021-01-01", tz="UTC")

    plt.figure(figsize=(9, 5))
    plt.scatter(train.loc[mask, "x"], train.loc[mask, "total"], s=14, label="Train obs (2021-2023)")
    plt.scatter(test["x"], test["total"], s=18, label="Test obs (2024)")

    plt.plot(test["x"], y_pred_ols, linewidth=2, label="OLS forecast")
    plt.plot(test["x"], y_pred_wls, linewidth=2, label="WLS (λ=0.9) forecast")
    plt.plot(test["x"], y_pred_single, linewidth=2, label="RLS (λ=0.9) forecast")
    plt.plot(test["x"], y_pred_h, linestyle="--", linewidth=1.6, label="RLS horizon-dependent λ_k")

    plt.title("Forecasts for 2024: OLS vs WLS vs RLS")
    plt.xlabel("x (year + (month-1)/12)")
    plt.ylabel("Total vehicles (millions)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_7_forecast_comparison.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    train, test = read_and_split("DST_BIL54.csv")
    problem_5_outputs(train, test)
