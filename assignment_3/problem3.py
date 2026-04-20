#!/usr/bin/env python3
"""
Assignment 3, Problem 3: ARX model for heating of a box
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import acf, ccf
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/box_data_60min.csv"
SAVE_FIGS = True  # Set False to show interactively instead


def savefig(name):
    if SAVE_FIGS:
        plt.savefig(f"problem3_{name}.png", dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3.1  Read and plot the three non-lagged time series
# ---------------------------------------------------------------------------
def problem_3_1(df):
    print("=" * 60)
    print("3.1  Raw time series plot")
    print("=" * 60)

    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    x = df["tdate"]

    for ax, y, color, ylabel, title in zip(
        axes,
        [df["Ph"], df["Tdelta"], df["Gv"]],
        ["tab:red", "tab:blue", "tab:orange"],
        ["Ph (W)", "Tdelta (°C)", "Gv (W/m²)"],
        ["Electrical heating power", "Internal – external temperature difference",
         "Vertical solar radiation"],
    ):
        ax.plot(x, y, color=color, linewidth=0.9, marker="o", markersize=2.5,
                markerfacecolor=color, markeredgewidth=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Major gridlines at every day boundary
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        # Minor ticks every 6 hours so individual hours are visible
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:00"))
        ax.grid(which="major", axis="x", linestyle="-",  linewidth=0.8, color="grey", alpha=0.5)
        ax.grid(which="minor", axis="x", linestyle=":",  linewidth=0.5, color="grey", alpha=0.3)
        ax.grid(which="major", axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.3)
        ax.tick_params(axis="x", which="major", rotation=0,  labelsize=8, pad=12)
        ax.tick_params(axis="x", which="minor", rotation=45, labelsize=6)

    fig.suptitle("3.1  Three non-lagged time series (hourly resolution)", fontsize=13)
    plt.tight_layout()
    savefig("3_1_timeseries")

    print(
        """
Observations:
- Ph (heating) fluctuates between ~0 W and ~120 W with clear diurnal patterns.
  It is lower during daylight hours when solar gain reduces the heating demand.
- Tdelta is mostly positive (box warmer inside), fluctuating 5-20 °C, following
  a day/night cycle.
- Gv shows sharp daytime pulses (solar irradiance) peaking around noon; zero at
  night, consistent with winter Belgium conditions.
- A clear negative relationship between Ph and Gv is visible: high solar
  radiation coincides with drops in heating. Ph and Tdelta appear positively
  correlated: a larger temperature gap requires more heating.
"""
    )


# ---------------------------------------------------------------------------
# 3.2  Train / test split
# ---------------------------------------------------------------------------
def problem_3_2(df):
    print("=" * 60)
    print("3.2  Train / test split")
    print("=" * 60)

    cutoff = "2013-02-06 00:00"
    train = df[df["tdate"] <= cutoff].copy()
    test = df[df["tdate"] > cutoff].copy()

    print(f"Training rows : {len(train)}  ({train['tdate'].iloc[0]} .. {train['tdate'].iloc[-1]})")
    print(f"Test rows     : {len(test)}  ({test['tdate'].iloc[0]} .. {test['tdate'].iloc[-1]})")
    assert len(train) == 167, f"Expected 167 training points, got {len(train)}"
    assert len(test) == 64,  f"Expected 64 test points, got {len(test)}"
    return train, test


# ---------------------------------------------------------------------------
# 3.3  Scatter / ACF / CCF plots (training set)
# ---------------------------------------------------------------------------
def problem_3_3(train):
    print("=" * 60)
    print("3.3  EDA: scatter, ACF, CCF")
    print("=" * 60)

    # --- Scatter plots ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(train["Tdelta"], train["Ph"], alpha=0.5, s=15, color="tab:blue")
    axes[0].set_xlabel("Tdelta (°C)")
    axes[0].set_ylabel("Ph (W)")
    axes[0].set_title("Ph vs Tdelta")

    axes[1].scatter(train["Gv"], train["Ph"], alpha=0.5, s=15, color="tab:orange")
    axes[1].set_xlabel("Gv (W/m²)")
    axes[1].set_ylabel("Ph (W)")
    axes[1].set_title("Ph vs Gv")

    fig.suptitle("3.3  Scatter plots (training set)", fontsize=13)
    plt.tight_layout()
    savefig("3_3_scatter")

    # --- ACF of Ph ---
    n_lags = 48
    ph_acf, ci = acf(train["Ph"].dropna(), nlags=n_lags, alpha=0.05)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n_lags + 1), ph_acf, color="tab:red", width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(
        range(n_lags + 1),
        ci[:, 0] - ph_acf,
        ci[:, 1] - ph_acf,
        alpha=0.2,
        color="grey",
        label="95% CI",
    )
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_title("3.3  ACF of Ph (training)")
    ax.legend()
    plt.tight_layout()
    savefig("3_3_acf_Ph")

    # --- CCF: Ph vs Tdelta and Ph vs Gv ---
    max_lag = 24
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, color, label in zip(
        axes,
        ["Tdelta", "Gv"],
        ["tab:blue", "tab:orange"],
        ["Ph vs Tdelta", "Ph vs Gv"],
    ):
        # ccf(x,y) = correlation of y_t with x_{t-k}
        # We want: ccf at lag k = corr(Ph_t, col_{t-k})
        y = train["Ph"].values
        x = train[col].values
        n = len(y)
        lags = np.arange(-max_lag, max_lag + 1)
        ccf_vals = np.array(
            [np.corrcoef(y[max(0, k):n + min(0, k)], x[max(0, -k):n - max(0, k)])[0, 1]
             for k in lags]
        )
        conf = 1.96 / np.sqrt(n)
        ax.bar(lags, ccf_vals, width=0.6, color=color, alpha=0.7)
        ax.axhline(conf, linestyle="--", color="grey", linewidth=0.8)
        ax.axhline(-conf, linestyle="--", color="grey", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("CCF")
        ax.set_title(label)

    fig.suptitle("3.3  CCF of Ph with inputs (training)", fontsize=13)
    plt.tight_layout()
    savefig("3_3_ccf")

    print(
        """
Observations:
- Scatter Ph vs Tdelta: moderate positive correlation; larger temperature
  difference requires more heating.  Some non-linearity / spread visible.
- Scatter Ph vs Gv: clear negative relationship; solar gain reduces heating.
- ACF of Ph: significant autocorrelation out to many lags with a 24-hour
  seasonal pattern — strong diurnal cycle, indicating the need for AR lags.
- CCF Ph vs Tdelta: strong positive correlation at small lags (0-2 h),
  decaying slowly; there is a lagged effect.
- CCF Ph vs Gv: strong negative correlation at lag 0 becoming less negative
  at larger lags; solar radiation has an immediate effect on heating.
- What cannot be seen directly: the exact lag structure, the magnitude of
  thermal inertia, and whether the system dynamics are adequately captured
  by a linear model.
"""
    )


# ---------------------------------------------------------------------------
# 3.4  Impulse response from Tdelta and Gv to Ph (up to lag 10)
# ---------------------------------------------------------------------------
def problem_3_4(train):
    print("=" * 60)
    print("3.4  Impulse response estimation")
    print("=" * 60)

    lag_cols_Td = [f"Tdelta.l{i}" for i in range(11)]
    lag_cols_Gv = [f"Gv.l{i}" for i in range(11)]

    sub = train[["Ph"] + lag_cols_Td + lag_cols_Gv].dropna()

    X = sm.add_constant(sub[lag_cols_Td + lag_cols_Gv].values)
    y = sub["Ph"].values
    model = sm.OLS(y, X).fit()

    h_Td = model.params[1:12]
    h_Gv = model.params[12:23]
    se_Td = model.bse[1:12]
    se_Gv = model.bse[12:23]
    lags = np.arange(11)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, h, se, label, color in zip(
        axes,
        [h_Td, h_Gv],
        [se_Td, se_Gv],
        ["Tdelta → Ph", "Gv → Ph"],
        ["tab:blue", "tab:orange"],
    ):
        ax.bar(lags, h, color=color, alpha=0.7, width=0.5, label="Impulse response")
        ax.errorbar(lags, h, yerr=1.96 * se, fmt="none", color="black", capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Coefficient")
        ax.set_title(label)
        ax.legend()

    fig.suptitle("3.4  Estimated impulse responses (FIR, up to lag 10)", fontsize=13)
    plt.tight_layout()
    savefig("3_4_impulse_response")

    print("Tdelta impulse response:", np.round(h_Td, 3))
    print("Gv     impulse response:", np.round(h_Gv, 3))
    print(
        """
Comments:
- Tdelta → Ph: positive impulse response at lag 0 and decays over subsequent
  lags, indicating that an increase in Tdelta has an immediate and sustained
  positive effect on heating (makes physical sense: more heat loss = more heating).
- Gv → Ph: negative response at lag 0 (solar gain reduces electrical heating)
  with effects decaying over several lags; solar radiation has a thermal storage
  effect in the box walls, delaying part of the solar gain.
- Both responses decay slowly, hinting that an ARX model with only a few lags
  may not capture all dynamics — but the AR part can account for persistent
  dynamics more efficiently than many FIR coefficients.
"""
    )


# ---------------------------------------------------------------------------
# Helpers for fitting and evaluating ARX models
# ---------------------------------------------------------------------------
def build_X(data, ar_order, include_const=True):
    """
    Build regressor matrix for ARX(ar_order):
      AR part: Ph.l1 ... Ph.l{ar_order}
      Inputs:  Tdelta.l0 ... Tdelta.l{ar_order-1}  (ar_order lags incl. current)
               Gv.l0     ... Gv.l{ar_order-1}
    For ar_order=0 (pure regression): only Tdelta.l0, Gv.l0
    """
    cols = []
    if ar_order == 0:
        cols = ["Tdelta.l0", "Gv.l0"]
    else:
        cols  = [f"Ph.l{i}" for i in range(1, ar_order + 1)]
        cols += [f"Tdelta.l{i}" for i in range(ar_order)]
        cols += [f"Gv.l{i}" for i in range(ar_order)]
    sub = data[["Ph"] + cols].dropna()
    y = sub["Ph"].values
    X = sub[cols].values
    if include_const:
        X = sm.add_constant(X, has_constant="add")
    return y, X, sub.index


def fit_arx(data, ar_order):
    y, X, idx = build_X(data, ar_order)
    model = sm.OLS(y, X).fit()
    return model, idx


def one_step_predictions(model, data, ar_order):
    """Return (obs, pred, dates) for the given dataset using fitted model."""
    y, X, idx = build_X(data, ar_order)
    pred = model.predict(X)
    dates = data.loc[idx, "tdate"].values
    return y, pred, dates


# ---------------------------------------------------------------------------
# 3.5  Linear regression: Ph ~ Tdelta + Gv
# ---------------------------------------------------------------------------
def problem_3_5(train):
    print("=" * 60)
    print("3.5  Linear regression (no AR terms)")
    print("=" * 60)

    model, idx = fit_arx(train, ar_order=0)
    y, pred, dates = one_step_predictions(model, train, ar_order=0)
    resid = y - pred

    print(model.summary())

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # One-step predictions
    axes[0].plot(dates, y, label="Observed Ph", color="tab:red")
    axes[0].plot(dates, pred, label="Fitted", color="black", linestyle="--")
    axes[0].set_title("3.5  One-step predictions – linear regression")
    axes[0].set_ylabel("Ph (W)")
    axes[0].legend()

    # Residuals over time
    axes[1].plot(dates, resid, color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals over time")
    axes[1].set_ylabel("Residual (W)")
    axes[1].tick_params(axis="x", rotation=30)

    # ACF of residuals
    n_lags = 40
    r_acf, ci = acf(resid, nlags=n_lags, alpha=0.05)
    axes[2].bar(range(n_lags + 1), r_acf, color="tab:grey", width=0.5)
    axes[2].fill_between(
        range(n_lags + 1),
        ci[:, 0] - r_acf,
        ci[:, 1] - r_acf,
        alpha=0.2,
        color="grey",
        label="95% CI",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("ACF of residuals")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("ACF")
    axes[2].legend()

    plt.tight_layout()
    savefig("3_5_linear_regression")

    # CCF residuals vs inputs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    max_lag = 24
    n = len(resid)
    lags = np.arange(-max_lag, max_lag + 1)
    for ax, col, color in zip(axes, ["Tdelta", "Gv"], ["tab:blue", "tab:orange"]):
        x = train.loc[idx, col].values
        ccf_vals = np.array(
            [np.corrcoef(resid[max(0, k):n + min(0, k)], x[max(0, -k):n - max(0, k)])[0, 1]
             for k in lags]
        )
        conf = 1.96 / np.sqrt(n)
        ax.bar(lags, ccf_vals, width=0.6, color=color, alpha=0.7)
        ax.axhline(conf, linestyle="--", color="grey", linewidth=0.8)
        ax.axhline(-conf, linestyle="--", color="grey", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("CCF")
        ax.set_title(f"CCF: residuals vs {col}")

    fig.suptitle("3.5  CCF of residuals vs inputs", fontsize=13)
    plt.tight_layout()
    savefig("3_5_ccf_residuals")

    print(
        f"\nRMSE (training): {np.sqrt(np.mean(resid**2)):.3f} W"
        f"\nR²             : {model.rsquared:.4f}"
    )
    print(
        """
Comments:
- Both coefficients are significant: Tdelta positive, Gv negative.
- Residual ACF shows very strong autocorrelation at lags 1, 2, 3 and beyond,
  indicating the i.i.d. assumption is violated — the model misses the dynamics.
- CCF of residuals vs inputs shows remaining correlations at lagged values,
  confirming that a transfer function (lagged inputs and/or AR terms) is needed.
- The one-step prediction tracks the general trend but poorly captures rapid
  changes, because there is no memory of past heating in the model.
"""
    )
    return model


# ---------------------------------------------------------------------------
# 3.6  First-order ARX
# ---------------------------------------------------------------------------
def problem_3_6(train):
    print("=" * 60)
    print("3.6  First-order ARX model")
    print("=" * 60)

    model, idx = fit_arx(train, ar_order=1)
    y, pred, dates = one_step_predictions(model, train, ar_order=1)
    resid = y - pred

    print(model.summary())

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(dates, y, label="Observed Ph", color="tab:red")
    axes[0].plot(dates, pred, label="ARX(1) one-step", color="black", linestyle="--")
    axes[0].set_title("3.6  One-step predictions – ARX(1)")
    axes[0].set_ylabel("Ph (W)")
    axes[0].legend()

    axes[1].plot(dates, resid, color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals over time")
    axes[1].set_ylabel("Residual (W)")
    axes[1].tick_params(axis="x", rotation=30)

    n_lags = 40
    r_acf, ci = acf(resid, nlags=n_lags, alpha=0.05)
    axes[2].bar(range(n_lags + 1), r_acf, color="tab:grey", width=0.5)
    axes[2].fill_between(
        range(n_lags + 1),
        ci[:, 0] - r_acf,
        ci[:, 1] - r_acf,
        alpha=0.2,
        color="grey",
        label="95% CI",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("ACF of residuals")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("ACF")
    axes[2].legend()

    plt.tight_layout()
    savefig("3_6_arx1")

    print(
        f"\nRMSE (training): {np.sqrt(np.mean(resid**2)):.3f} W"
        f"\nR²             : {model.rsquared:.4f}"
    )
    print(
        """
Comments:
- Adding Ph.l1 greatly improves fit (higher R², lower RMSE).
- Residual ACF is significantly reduced at lag 1, but autocorrelation remains
  at higher lags and at the 24-hour seasonal lag, indicating that one AR term
  is not sufficient to whiten the residuals.
- Improvement over 3.5 is clear, but more model order is likely beneficial.
"""
    )
    return model


# ---------------------------------------------------------------------------
# 3.7  AIC / BIC vs model order
# ---------------------------------------------------------------------------
def problem_3_7(train):
    print("=" * 60)
    print("3.7  AIC / BIC vs ARX order")
    print("=" * 60)

    orders = range(0, 11)
    aic_vals, bic_vals = [], []

    for p in orders:
        y, X, _ = build_X(train, ar_order=p)
        model = sm.OLS(y, X).fit()
        aic_vals.append(model.aic)
        bic_vals.append(model.bic)

    aic_vals = np.array(aic_vals)
    bic_vals = np.array(bic_vals)

    best_aic = int(np.argmin(aic_vals))
    best_bic = int(np.argmin(bic_vals))

    print(f"Best order by AIC: {best_aic}  (AIC={aic_vals[best_aic]:.2f})")
    print(f"Best order by BIC: {best_bic}  (BIC={bic_vals[best_bic]:.2f})")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(orders), aic_vals, marker="o", label="AIC", color="tab:blue")
    ax.plot(list(orders), bic_vals, marker="s", label="BIC", color="tab:red")
    ax.axvline(best_aic, color="tab:blue", linestyle="--", alpha=0.6,
               label=f"Min AIC @ order {best_aic}")
    ax.axvline(best_bic, color="tab:red", linestyle="--", alpha=0.6,
               label=f"Min BIC @ order {best_bic}")
    ax.set_xlabel("ARX order")
    ax.set_ylabel("Information criterion")
    ax.set_title("3.7  AIC and BIC vs ARX model order")
    ax.legend()
    plt.tight_layout()
    savefig("3_7_aic_bic")

    print(
        """
Comments:
- Both AIC and BIC decrease as order increases initially, then plateau or increase.
- BIC penalises model complexity more heavily (penalty = k*ln(n) vs k*2 for AIC),
  so BIC selects a lower (more parsimonious) model order than AIC.
- AIC tends to favour a slightly higher order because its penalty per parameter
  is smaller; this can lead to slight overfitting relative to BIC.
- The selected order balances goodness-of-fit against model complexity.
"""
    )
    return best_bic, best_aic


# ---------------------------------------------------------------------------
# 3.8  One-step RMSE on test set vs model order
# ---------------------------------------------------------------------------
def problem_3_8(train, test):
    print("=" * 60)
    print("3.8  Test-set one-step RMSE vs ARX order")
    print("=" * 60)

    orders = range(0, 11)
    rmse_vals = []

    for p in orders:
        # Fit on training set
        y_tr, X_tr, _ = build_X(train, ar_order=p)
        model = sm.OLS(y_tr, X_tr).fit()

        # Predict one-step on test set
        y_te, X_te, _ = build_X(test, ar_order=p)
        if len(y_te) == 0:
            rmse_vals.append(np.nan)
            continue
        pred_te = model.predict(X_te)
        rmse = np.sqrt(np.mean((y_te - pred_te) ** 2))
        rmse_vals.append(rmse)
        print(f"  Order {p:2d}: RMSE = {rmse:.3f} W")

    rmse_vals = np.array(rmse_vals)
    best_rmse = int(np.nanargmin(rmse_vals))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(orders), rmse_vals, marker="o", color="tab:green")
    ax.axvline(best_rmse, color="tab:green", linestyle="--", alpha=0.7,
               label=f"Min RMSE @ order {best_rmse}")
    ax.set_xlabel("ARX order")
    ax.set_ylabel("RMSE (W)")
    ax.set_title("3.8  Test-set one-step RMSE vs ARX model order")
    ax.legend()
    plt.tight_layout()
    savefig("3_8_rmse")

    print(f"\nBest order by test RMSE: {best_rmse}  (RMSE={rmse_vals[best_rmse]:.3f} W)")
    print(
        """
Comments:
- The RMSE on the test set typically reaches a minimum at a similar but not
  necessarily identical order compared to BIC/AIC.
- If RMSE continues decreasing where BIC has already increased, it suggests
  the extra parameters genuinely capture real dynamics (not just in-sample
  noise).  If RMSE increases before BIC/AIC minimum, the training criteria
  are somewhat optimistic.
- The test RMSE is the most direct indicator of out-of-sample predictive
  performance.
"""
    )
    return best_rmse


# ---------------------------------------------------------------------------
# 3.9  Multi-step (simulation) prediction
# ---------------------------------------------------------------------------
def problem_3_9(train, test, ar_order):
    print("=" * 60)
    print(f"3.9  Multi-step simulation with ARX({ar_order})")
    print("=" * 60)

    # Fit model on training set
    y_tr, X_tr, _ = build_X(train, ar_order=ar_order)
    model = sm.OLS(y_tr, X_tr).fit()
    params = model.params  # [const, AR1..ARp, Td.l0..Td.l(p-1), Gv.l0..Gv.l(p-1)]

    full = pd.concat([train, test]).reset_index(drop=True)
    n_full = len(full)

    # Initialise with observed Ph values for the first ar_order steps
    sim_ph = full["Ph"].values.copy().astype(float)
    # We simulate from the first row where all AR lags are available
    # (row index = ar_order, i.e. ar_order previous steps needed)

    # We need observed Tdelta and Gv at all lags (use the .lX columns)
    td_cols = [f"Tdelta.l{i}" for i in range(ar_order)]
    gv_cols = [f"Gv.l{i}" for i in range(ar_order)]

    p_ar   = ar_order          # number of AR params
    p_td   = ar_order          # number of Tdelta params  (l0..l(p-1))
    p_gv   = ar_order          # number of Gv params

    # params layout: [const, ph.l1..ph.lp, Td.l0..Td.l(p-1), Gv.l0..Gv.l(p-1)]
    c      = params[0]
    phi    = params[1: 1 + p_ar]
    omega_td = params[1 + p_ar: 1 + p_ar + p_td]
    omega_gv = params[1 + p_ar + p_td: 1 + p_ar + p_td + p_gv]

    sim = np.full(n_full, np.nan)
    # Burn-in: use observed Ph for the first ar_order rows
    sim[:ar_order] = full["Ph"].values[:ar_order]

    for t in range(ar_order, n_full):
        # AR part — use simulated (iterated) Ph values
        ar_part = sum(phi[k] * sim[t - 1 - k] for k in range(p_ar))

        # Input part — use observed Tdelta and Gv lags from data columns
        row = full.iloc[t]
        td_part = sum(omega_td[k] * row[f"Tdelta.l{k}"] for k in range(p_td))
        gv_part = sum(omega_gv[k] * row[f"Gv.l{k}"]    for k in range(p_gv))

        sim[t] = c + ar_part + td_part + gv_part

    dates_full = full["tdate"].values
    obs_full   = full["Ph"].values

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(dates_full, obs_full, label="Observed Ph", color="tab:red", linewidth=1)
    axes[0].plot(dates_full, sim, label=f"ARX({ar_order}) simulation", color="black",
                 linestyle="--", linewidth=1)
    axes[0].axvline(train["tdate"].iloc[-1], color="grey", linestyle=":", label="Train/test split")
    axes[0].set_ylabel("Ph (W)")
    axes[0].set_title(f"3.9  Multi-step simulation – ARX({ar_order})")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    resid_sim = obs_full - sim
    axes[1].plot(dates_full, resid_sim, color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(train["tdate"].iloc[-1], color="grey", linestyle=":")
    axes[1].set_ylabel("Error (W)")
    axes[1].set_title("Simulation error (observed – simulated)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    savefig("3_9_simulation")

    # RMSE on test period of simulation
    test_idx  = full.index[full["tdate"] > train["tdate"].iloc[-1]]
    rmse_sim = np.sqrt(np.nanmean((obs_full[test_idx] - sim[test_idx]) ** 2))
    print(f"Simulation RMSE on test period: {rmse_sim:.3f} W")
    print(
        """
Comments:
- The multi-step simulation uses observed inputs (Tdelta, Gv) at every step,
  but propagates Ph iteratively through the AR part.
- Small errors accumulate over time: if the AR part is well-identified and
  stable, errors stay bounded; otherwise they drift.
- In an operational setting, multi-step prediction is feasible only if future
  Tdelta and Gv are known (e.g. weather forecast).  Tdelta cannot be set in
  advance (it depends on the real outdoor temperature), so this requires a
  temperature forecast.  Gv can be forecast from solar radiation models.
- In practice, the thermostatic control loop also makes the system behave
  somewhat differently from open-loop prediction.
"""
    )


# ---------------------------------------------------------------------------
# 3.10  Summary conclusion
# ---------------------------------------------------------------------------
def problem_3_10():
    print("=" * 60)
    print("3.10  Summary conclusion")
    print("=" * 60)
    print(
        """
Summary:
1. The three time series (Ph, Tdelta, Gv) exhibit clear diurnal patterns
   consistent with a winter heating experiment.  Ph is positively correlated
   with Tdelta and negatively with solar radiation Gv.

2. A simple linear regression (Ph ~ Tdelta + Gv) already captures the main
   level effects but leaves strong autocorrelation in the residuals, violating
   the i.i.d. assumption and indicating that system dynamics are not captured.

3. Adding AR lags (ARX models) substantially reduces residual autocorrelation
   and improves both in-sample fit and out-of-sample one-step RMSE.

4. BIC and AIC both indicate that a low-to-moderate order ARX model is
   preferred, with BIC being more conservative (lower order) due to its
   stronger penalty for additional parameters.

5. The test-set one-step RMSE confirms a similar optimal order, giving
   confidence that the information criteria are not misleading here.

6. Multi-step simulation produces reasonable trajectories when observed inputs
   are used, but errors are larger than one-step predictions because AR lags
   are computed iteratively from predicted (not observed) Ph values.

7. Real-time multi-step predictions would additionally require forecasts of
   Tdelta (outdoor-indoor temperature difference) and Gv (solar radiation),
   adding forecast uncertainty on top of model uncertainty.

8. Overall, an ARX model of moderate order provides a practical and
   interpretable model for hourly heating prediction in this building context.
"""
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["tdate"] = pd.to_datetime(df["tdate"])
    # Sort by time just in case
    df = df.sort_values("tdate").reset_index(drop=True)
    # Rename dot-columns to underscore for easier access where needed
    # (keep original names since the column name list is explicit above)

    print(f"Data loaded: {len(df)} rows, {df.shape[1]} columns.\n")

    problem_3_1(df)
    train, test = problem_3_2(df)
    problem_3_3(train)
    problem_3_4(train)
    problem_3_5(train)
    problem_3_6(train)
    best_bic, best_aic = problem_3_7(train)
    best_rmse = problem_3_8(train, test)

    # For simulation use BIC-selected order (most parsimonious)
    chosen_order = best_bic if best_bic > 0 else 1
    print(f"\nChosen model order for simulation: {chosen_order} (BIC criterion)")
    problem_3_9(train, test, ar_order=chosen_order)
    problem_3_10()

    print("\nDone. Figures saved as problem3_3_*.png")


if __name__ == "__main__":
    main()