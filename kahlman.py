import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from pair_gen import find_cointegrated_pairs
from ssd_test import bootstrap_detailed_ssd_test, detailed_ssd_efficiency_test
from utils import generate_signals_from_spread, load_data

# =============================================================================
# Kalman Filter for Dynamic Regression (Dynamic Mean Reversion)
# =============================================================================


def kalman_filter_regression(asset1, asset2, delta=1e-5, R=0.001):
    """
    Estimate time-varying regression parameters for:
         asset2 = alpha + beta * asset1 + noise
    using a Kalman filter with a simple random-walk state model.

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets.
    delta : float
        Process noise variance multiplier.
    R : float
        Observation noise variance.

    Returns:
    --------
    theta : np.array, shape (n, 2)
        Filtered estimates for [alpha, beta] at each time step.
    """
    n = len(asset1)
    theta = np.zeros((n, 2))   # each row: [alpha, beta]
    P = np.zeros((n, 2, 2))      # state covariance matrices

    # Use an initial window (first 20 observations or fewer) for an OLS estimate.
    initial_window = min(20, n)
    X0 = np.column_stack(
        (np.ones(initial_window), asset1.iloc[:initial_window].values))
    y0 = asset2.iloc[:initial_window].values
    theta0 = np.linalg.lstsq(X0, y0, rcond=None)[0]
    theta[0] = theta0
    P[0] = np.eye(2) * 1.0  # initial uncertainty

    # Process noise covariance (assumed constant)
    Q = np.eye(2) * delta

    for t in range(1, n):
        # --- Prediction step ---
        theta_pred = theta[t - 1]
        P_pred = P[t - 1] + Q

        # --- Observation step ---
        # Observation matrix: F = [1, asset1_t]
        F = np.array([1, asset1.iloc[t]])
        yhat = np.dot(F, theta_pred)
        error = asset2.iloc[t] - yhat
        S = np.dot(F, np.dot(P_pred, F)) + R  # Innovation (residual) variance
        K = np.dot(P_pred, F) / S            # Kalman gain

        # --- Update step ---
        theta[t] = theta_pred + K * error
        P[t] = (np.eye(2) - np.outer(K, F)) @ P_pred

    return theta


def compute_portfolio_returns(asset1: pd.Series, asset2: pd.Series, beta_series: pd.Series, signals: pd.Series):
    """
    Compute portfolio returns from the dynamic pairs trading strategy.
    At time t, assume the positions are taken based on the signal generated at t-1.
    The portfolio return is computed as:
         r_portfolio = signal_{t-1} * (r_asset2 - beta_{t-1} * r_asset1)

    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets.
    beta_series : pd.Series
        Time-varying hedge ratio (beta) from the Kalman filter.
    signals : pd.Series
        Trading signals (shifted to avoid lookahead bias).

    Returns:
    --------
    port_returns : pd.Series
        Daily portfolio returns.
    """
    r1 = asset1.pct_change().ffill()
    r2 = asset2.pct_change().ffill()
    # Use previous day's signal and beta to avoid lookahead bias.

    signals_shifted = signals.shift(1).ffill()
    beta_shifted = beta_series.shift(1).ffill()
    port_returns = signals_shifted * (r2 - beta_shifted * r1)
    return port_returns


def filter_ssd_kahlman(data, benchmark, tol=0.5):
    pvalue_matrix, cointegrated_pairs = find_cointegrated_pairs(
        data, significance=0.05)

    filtered_pairs = []
    for a1, a2, p_value in tqdm(cointegrated_pairs):
        asset1 = data[a1]
        asset2 = data[a2]
        # ---- Dynamic Parameter Estimation via Kalman Filter ----
        # Estimate time-varying regression parameters for: asset2 = alpha + beta * asset1 + noise
        theta = kalman_filter_regression(asset1, asset2, delta=1e-5, R=0.001)
        alpha_series = pd.Series(theta[:, 0], index=asset1.index)
        beta_series = pd.Series(theta[:, 1], index=asset1.index)

        # Compute the dynamic spread: spread = asset2 - (alpha + beta * asset1)
        spread = asset2 - (alpha_series + beta_series * asset1)

        # ---- Generate Trading Signals Based on the Spread ----
        signals, zscore = generate_signals_from_spread(
            spread, window=20, threshold=1.0)

        # ---- Compute Strategy (Portfolio) Returns ----
        port_returns = compute_portfolio_returns(
            asset1, asset2, beta_series, signals)

        # For the SSD test, extract candidate returns and benchmark returns.
        # Here, we compute benchmark returns from the benchmark price series.
        candidate_returns = port_returns.dropna().values
        benchmark_returns = benchmark.pct_change().dropna().values
        # Align the lengths if needed.
        T = min(len(candidate_returns), len(benchmark_returns))
        candidate_returns = candidate_returns[:T]
        benchmark_returns = benchmark_returns[:T]

        # ---- Detailed SSD Efficiency Test ----
        status, delta_opt, total_adj, grid, diff, is_ssd_eff = detailed_ssd_efficiency_test(
            candidate_returns, benchmark_returns, num_bins=50, tol=tol
        )
        # print("SSD LP Status:", status)
        # print("Total adjustment needed: {:.6f}".format(total_adj))
        if is_ssd_eff:
            # print("=> The candidate portfolio is (approximately) SSD efficient relative to the benchmark.")
            filtered_pairs.append((a1, a2, p_value))
        else:
            pass
            # print("=> The candidate portfolio is NOT SSD efficient relative to the benchmark.")

    return filtered_pairs
