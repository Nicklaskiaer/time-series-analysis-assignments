import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(42)
N = 500  # simulation length
s = 12   # seasonal period
n_lags = 40  # lags to show in ACF/PACF

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build full AR/MA coefficient arrays for multiplicative seasonal ARIMA
# ─────────────────────────────────────────────────────────────────────────────

def make_arma_coeffs(ar_ns=None, ma_ns=None, ar_s=None, ma_s=None, s=12):
    """
    Expand multiplicative seasonal ARMA into a single ARMA representation.

    Parameters
    ----------
    ar_ns : list of float  — non-seasonal AR coefficients [phi_1, phi_2, ...]
    ma_ns : list of float  — non-seasonal MA coefficients [theta_1, theta_2, ...]
    ar_s  : list of float  — seasonal AR coefficients    [Phi_1, Phi_2, ...]
    ma_s  : list of float  — seasonal MA coefficients    [Theta_1, Theta_2, ...]
    s     : int            — seasonal period

    Returns
    -------
    ar_poly, ma_poly  — coefficient arrays in ArmaProcess convention
                         (lag-0 coefficient = 1, AR signs negated)
    """
    ar_ns = ar_ns or []
    ma_ns = ma_ns or []
    ar_s  = ar_s  or []
    ma_s  = ma_s  or []

    # Non-seasonal polynomials: phi(B) = 1 - phi1*B - phi2*B^2 - ...
    p = len(ar_ns)
    phi_ns = np.zeros(p + 1)
    phi_ns[0] = 1.0
    for i, c in enumerate(ar_ns):
        phi_ns[i + 1] = -c          # ArmaProcess uses 1 - phi1*B convention internally

    q = len(ma_ns)
    theta_ns = np.zeros(q + 1)
    theta_ns[0] = 1.0
    for i, c in enumerate(ma_ns):
        theta_ns[i + 1] = c

    # Seasonal polynomials: Phi(B^s) = 1 - Phi1*B^s - ...
    P = len(ar_s)
    phi_s = np.zeros(P * s + 1)
    phi_s[0] = 1.0
    for i, c in enumerate(ar_s):
        phi_s[(i + 1) * s] = -c

    Q = len(ma_s)
    theta_s = np.zeros(Q * s + 1)
    theta_s[0] = 1.0
    for i, c in enumerate(ma_s):
        theta_s[(i + 1) * s] = c

    # Multiply the polynomials: phi(B)*Phi(B^s) and theta(B)*Theta(B^s)
    ar_full = np.polymul(phi_ns[::-1], phi_s[::-1])[::-1]
    ma_full = np.polymul(theta_ns[::-1], theta_s[::-1])[::-1]

    return ar_full, ma_full


# ─────────────────────────────────────────────────────────────────────────────
# Helper: simulate and plot
# ─────────────────────────────────────────────────────────────────────────────

def simulate_and_plot(title, ar_full, ma_full, n=N, lags=n_lags):
    process = ArmaProcess(ar_full, ma_full)
    assert process.isstationary, f"Process {title} is NOT stationary!"
    assert process.isinvertible,  f"Process {title} is NOT invertible!"

    y = process.generate_sample(nsample=n)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Time series
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(y, linewidth=0.8, color='steelblue')
    ax0.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax0.set_title('Simulated series')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('$Y_t$')

    def add_year_lines(ax):
        for k in range(1, lags + 1):
            if k % s == 0:
                ax.axvline(k, color='#e05c2a', linewidth=1.2, linestyle='--', alpha=0.85, zorder=0)
            else:
                ax.axvline(k, color='grey', linewidth=0.5, linestyle=':', alpha=0.4, zorder=0)
        ax.set_xticks(range(1, lags + 1))
        ax.set_xticklabels([k if k % 2 == 0 else '' for k in range(1, lags + 1)], fontsize=6)

    # ACF
    ax1 = fig.add_subplot(gs[1, 0])
    plot_acf(y, lags=lags, ax=ax1, color='steelblue', zero=False)
    ax1.set_title('ACF')
    ax1.set_xlabel('Lag (months)')
    add_year_lines(ax1)

    # PACF
    ax2 = fig.add_subplot(gs[1, 1])
    plot_pacf(y, lags=lags, ax=ax2, method='ywm', color='steelblue', zero=False)
    ax2.set_title('PACF')
    ax2.set_xlabel('Lag (months)')
    add_year_lines(ax2)

    plt.savefig(f"model_{title[:3].strip().replace(' ','_')}.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {title}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  (1,0,0)×(0,0,0)_12   phi1 = 0.6
# Plain AR(1): Y_t = 0.6 Y_{t-1} + eps_t
# ─────────────────────────────────────────────────────────────────────────────
ar, ma = make_arma_coeffs(ar_ns=[0.6])
simulate_and_plot("2.1  AR(1)x(0,0,0)_12  φ₁=0.6", ar, ma)

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  (0,0,0)×(1,0,0)_12   Phi1 = -0.9
# Seasonal AR(1): Y_t = -0.9 Y_{t-12} + eps_t
# ─────────────────────────────────────────────────────────────────────────────
ar, ma = make_arma_coeffs(ar_s=[-0.9])
simulate_and_plot("2.2  (0,0,0)xSAR(1)_12  Φ₁=-0.9", ar, ma)

# ─────────────────────────────────────────────────────────────────────────────
# 2.3  (1,0,0)×(0,0,1)_12   phi1=0.9, Theta1=-0.7
# Expanded: AR polynomial = phi(B), MA polynomial = theta(B)*Theta(B^12)
# MA polynomial: (1 - 0.7*B^12)  — note sign: Theta1=-0.7 => +0.7 in expansion
# ─────────────────────────────────────────────────────────────────────────────
ar, ma = make_arma_coeffs(ar_ns=[0.9], ma_s=[-0.7])
simulate_and_plot("2.3  AR(1)xSMA(1)_12  φ₁=0.9, Θ₁=-0.7", ar, ma)

# ─────────────────────────────────────────────────────────────────────────────
# 2.4  (1,0,0)×(1,0,0)_12   phi1=-0.6, Phi1=-0.8
# Expanded AR: phi(B)*Phi(B^12) = (1+0.6B)(1+0.8B^12)
#            = 1 + 0.6B + 0.8B^12 + 0.48B^13
# ─────────────────────────────────────────────────────────────────────────────
ar, ma = make_arma_coeffs(ar_ns=[-0.6], ar_s=[-0.8])
simulate_and_plot("2.4  AR(1)xSAR(1)_12  φ₁=-0.6, Φ₁=-0.8", ar, ma)

# ─────────────────────────────────────────────────────────────────────────────
# 2.5  (0,0,1)×(0,0,1)_12   theta1=0.4, Theta1=-0.8
# Expanded MA: theta(B)*Theta(B^12) = (1+0.4B)(1-0.8B^12)
#            = 1 + 0.4B - 0.8B^12 - 0.32B^13
# ─────────────────────────────────────────────────────────────────────────────
ar, ma = make_arma_coeffs(ma_ns=[0.4], ma_s=[-0.8])
simulate_and_plot("2.5  MA(1)xSMA(1)_12  θ₁=0.4, Θ₁=-0.8", ar, ma)

# ─────────────────────────────────────────────────────────────────────────────
# 2.6  (0,0,1)×(1,0,0)_12   theta1=-0.4, Phi1=0.7
# AR: Phi(B^12) = 1 - 0.7B^12
# MA: theta(B) = 1 - 0.4B
# ─────────────────────────────────────────────────────────────────────────────
ar, ma = make_arma_coeffs(ar_s=[0.7], ma_ns=[-0.4])
simulate_and_plot("2.6  MA(1)xSAR(1)_12  θ₁=-0.4, Φ₁=0.7", ar, ma)