import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

from pair_gen import find_cointegrated_pairs
from ssd_test import bootstrap_detailed_ssd_test, detailed_ssd_efficiency_test
from utils import generate_signals_from_spread, load_data

# =============================================================================
# 2. Cointegration Pairs Trading Model
# =============================================================================


def compute_hedge_ratio(asset1, asset2):
    """
    Compute the hedge ratio (β) via OLS regression.
    Regress asset2 on asset1.
    """
    X = sm.add_constant(asset1)
    model = sm.OLS(asset2, X).fit()
    beta = model.params.iloc[1]
    return beta


def compute_spread(asset1, asset2, beta):
    """
    Compute the spread: Asset2 - β * Asset1.
    """
    return asset2 - beta * asset1


def compute_portfolio_returns(asset1, asset2, beta, signals):
    """
    Parameters:
    -----------
    asset1, asset2 : pd.Series
        Price series for the two assets.
    beta : np.float
    signals : pd.Series
        Trading signals (shifted to avoid lookahead bias).

    Returns:
    --------
    port_returns : pd.Series
        Daily portfolio returns.
    """
    r1 = asset1.pct_change().fillna(0)
    r2 = asset2.pct_change().fillna(0)
    signals_shifted = signals.shift(1).fillna(0)
    port_returns = signals_shifted * (r2 - beta * r1)
    return port_returns


def filter_ssd_cointegration(data, benchmark, tol=0.5):
    pvalue_matrix, cointegrated_pairs = find_cointegrated_pairs(
        data, significance=0.05)

    filtered_pairs = []

    for a1, a2, p_value in tqdm(cointegrated_pairs):
        asset1 = data[a1]
        asset2 = data[a2]
        # --- Step 1: Cointegration Test ---
        # --- Step 2: Estimate Hedge Ratio and Compute Spread ---
        beta = compute_hedge_ratio(asset1, asset2)
        print("Estimated hedge ratio (β):", beta)
        spread = compute_spread(asset1, asset2, beta)

        # --- Step 3: Generate Trading Signals and Compute Portfolio Returns ---
        signals, zscore = generate_signals_from_spread(
            spread, window=20, threshold=1.0)
        port_returns = compute_portfolio_returns(asset1, asset2, beta, signals)

        # --- Prepare Data for the SSD Test ---
        candidate_returns = port_returns.dropna().values
        benchmark_returns = benchmark.pct_change().dropna().values
        # Align the lengths.
        T = min(len(candidate_returns), len(benchmark_returns))
        candidate_returns = candidate_returns[:T]
        benchmark_returns = benchmark_returns[:T]

        # --- Step 4: Apply the Detailed SSD Efficiency Test ---
        status, delta_opt, total_adj, grid, diff, is_ssd_eff = detailed_ssd_efficiency_test(
            candidate_returns, benchmark_returns, num_bins=50, tol=tol
        )
        # print("\n=== Detailed SSD Test Results ===")
        # print("LP Status:", status)
        # print("Total adjustment needed: {:.6f}".format(total_adj))
        if is_ssd_eff:
            filtered_pairs.append((a1, a2, p_value))

    return filtered_pairs
