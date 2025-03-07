
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from math import sqrt
from scipy.optimize import minimize

from cointegration import compute_hedge_ratio, compute_spread, filter_ssd_cointegration
from kahlman import filter_ssd_kahlman
from utils import compute_stock_portfolio_returns, generate_signals_from_spread, generate_signals_from_stock, load_data


def compute_pair_portfolio_returns(asset1, asset2, beta, signals):
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


# =============================================================================
# 1. Build the Pair Returns DataFrame from Filtered Pairs
# =============================================================================
def build_pair_returns(filtered_pairs, data, window=20, threshold=1.0):
    """
    For each pair (asset1, asset2, _), compute the pair's strategy returns,
    and collect them in a DataFrame.
    """
    pair_returns = {}
    for asset1, asset2, _ in filtered_pairs:
        # Compute hedge ratio via OLS regression.
        beta = compute_hedge_ratio(data[asset1], data[asset2])
        # Compute the spread for the pair.
        spread = compute_spread(data[asset1], data[asset2], beta)
        # Generate trading signals based on the spread.
        signals, _ = generate_signals_from_spread(
            spread, window=window, threshold=threshold)
        # Compute the daily returns for this pair's strategy.
        ret_series = compute_pair_portfolio_returns(
            data[asset1], data[asset2], beta, signals)
        # Save the return series using a combined key.
        pair_returns[f"{asset1}-{asset2}"] = ret_series

    pair_returns_df = pd.DataFrame(pair_returns)
    pair_returns_df.dropna(how='all', inplace=True)
    return pair_returns_df


# =============================================================================
# 2. Compute Optimal Weights (Minimum-Variance Portfolio)
# =============================================================================
def compute_min_variance_weights(returns_df):
    """
    Compute portfolio weights that minimize the portfolio variance based on the
    sample covariance of the strategy returns.

    The optimization is:
        minimize   w^T Sigma w
        subject to sum(w) = 1, w >= 0
    """
    # Align returns: drop any dates with missing values in any pair.
    returns_df = returns_df.dropna()
    cov_mat = returns_df.cov().values
    n = cov_mat.shape[0]

    # Define weight variable.
    w = cp.Variable(n)
    # Objective: minimize portfolio variance.
    objective = cp.Minimize(cp.quad_form(w, cov_mat))
    # Constraints: weights sum to 1, and no shorting (if desired).
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract weights.
    weights = np.array(w.value).flatten()
    # Map the weights back to the asset (pair) names.
    weight_series = pd.Series(weights, index=returns_df.columns)
    return weight_series


def compute_max_sharpe_weights(returns_df):
    """
    Compute portfolio weights that maximize the Sharpe ratio for given strategy returns.
    Constraints: weights sum to 1, no short selling (weights >= 0).

    Parameters:
    returns_df (pd.DataFrame): DataFrame where each column is the return series of a strategy.

    Returns:
    np.ndarray: Optimal weights for each strategy that maximize Sharpe ratio.
    """
    # 1. Calculate expected returns (mu) and covariance matrix (Sigma)
    mu = returns_df.mean().values            # expected return for each strategy
    Sigma = returns_df.cov().values          # covariance matrix of returns
    n = len(mu)                              # number of assets/strategies

    # 2. Define the negative Sharpe ratio function to minimize
    def neg_sharpe_ratio(w):
        # Ensure weights are an array (in case optimizer passes a list)
        w = np.array(w)
        port_return = np.dot(w, mu)
        port_vol = sqrt(w.dot(Sigma).dot(w))
        # If portfolio volatility is zero (edge case), return a large number to penalize
        if port_vol == 0:
            return 1e6  # extremely large penalty
        return -(port_return / port_vol)

    # 3. Set the constraints and bounds
    # weight sum = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # 0 <= w_i <= 1 for all i
    bounds = tuple((0.0, 1.0) for _ in range(n))

    # 4. Initial guess for weights (equal allocation)
    initial_w = np.ones(n) / n

    # 5. Run the optimization
    result = minimize(neg_sharpe_ratio, initial_w, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    # 6. Check for success and return the optimal weights
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


# =============================================================================
# 3. Compute Portfolio Returns and Statistics
# =============================================================================
def compute_portfolio_returns_from_weights(returns_df, weights):
    """
    Given a DataFrame of pair returns and a weight series, compute the daily
    portfolio returns as the weighted sum.
    """
    # Align the dates.
    aligned_returns = returns_df.loc[weights.index].dropna()
    portfolio_returns = aligned_returns.dot(weights)
    return portfolio_returns


def portfolio_statistics(portfolio_returns, target=0.0):
    """
    Calculate portfolio statistics from a series of daily returns.

    Statistics include:
      - Cumulative Return: (product(1 + daily_returns) - 1)
      - Annualized Return: average daily return * 252
      - Annualized Volatility: standard deviation daily returns * sqrt(252)
      - Sharpe Ratio: annualized_return / annualized_volatility (assuming risk-free rate is zero)
      - Downside Deviation: sqrt(mean( (min(0, r - target))^2 ))
        where returns above the target are not penalized.

    Parameters:
    -----------
    portfolio_returns : pd.Series
        Series of daily portfolio returns.
    target : float, optional
        The target return for computing downside deviation (default is 0).

    Returns:
    --------
    stats : dict
        Dictionary containing portfolio statistics.
    """
    # Cumulative return over the period.
    cumulative_return = np.prod(1 + portfolio_returns) - 1

    # Annualized return (assuming 252 trading days per year).
    annualized_return = portfolio_returns.mean() * 252

    # Annualized volatility.
    annualized_vol = portfolio_returns.std() * np.sqrt(252)

    # Sharpe ratio (assume risk-free rate is 0).
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else np.nan

    # Compute downside deviation: only consider returns below the target.
    downside = portfolio_returns[portfolio_returns < target]
    if len(downside) > 0:
        downside_deviation = np.sqrt(np.mean(np.square(downside - target)))
    else:
        downside_deviation = 0.0

    stats = {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Downside Deviation": downside_deviation
    }
    return stats


def print_stats(stats):
    for key, value in stats.items():
        if "Return" in key or "Volatility" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.4f}")


# =============================================================================
# 4. Main Routine: Construct the Portfolio and Test General Stats
# =============================================================================
window = 30
threshold = 1

if __name__ == '__main__':
    train_data = load_data('train_data.csv')
    train_data_wo_benchmark = train_data.drop(
        columns=["^DJI", "^GSPC", "^IXIC"])
    train_benchmark = train_data['^GSPC']
    # filtered_pairs = filter_ssd_cointegration(data_wo_benchmark, benchmark)
    filtered_pairs = filter_ssd_kahlman(
        train_data_wo_benchmark, train_benchmark)
    # print(filtered_pairs)

    # Build a DataFrame of individual pair returns.
    train_pair_returns_df = build_pair_returns(
        filtered_pairs, train_data, window=window, threshold=threshold)

    # Compute optimal weights using minimum-variance optimization.
    weights = compute_max_sharpe_weights(train_pair_returns_df)
    # weights = compute_max_sharpe_weights(train_pair_returns_df)
    # print("Optimized Weights for Pairs Portfolio:")
    # print(weights)
    test_data = load_data('test_data.csv')
    test_data_wo_benchmark = test_data.drop(columns=["^DJI", "^GSPC", "^IXIC"])
    test_pair_returns_df = build_pair_returns(
        filtered_pairs, test_data, window=window, threshold=threshold)
    # Compute the portfolio's daily returns based on the optimized weights.
    portfolio_ret = test_pair_returns_df.dot(weights)

    test_benchmark = test_data['^GSPC']
    bench_signals, _ = generate_signals_from_stock(
        test_benchmark, window=window, threshold=threshold)
    benchmark_ret = compute_stock_portfolio_returns(
        test_benchmark, bench_signals)

    # Compute portfolio statistics.
    port_stats = portfolio_statistics(portfolio_ret)
    print("\nPortfolio Statistics:")
    print_stats(port_stats)

    # Compute benchmark statistics.
    bench_stats = portfolio_statistics(benchmark_ret)
    print("\nBenchmark Statistics:")
    print_stats(bench_stats)

    # Plot the portfolio's cumulative returns.
    plt.figure(figsize=(10, 6))

    cumulative_returns = (1 + portfolio_ret).cumprod()
    plt.plot(cumulative_returns.index, cumulative_returns,
             label="Portfolio Cumulative Returns")

    # Plot the portfolio's cumulative returns.
    cumulative_b_returns = (1 + benchmark_ret).cumprod()
    plt.plot(cumulative_b_returns.index, cumulative_b_returns,
             label="Benchmark Cumulative Returns")

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.title("Optimized Pairs Portfolio Performance")

    plt.legend()
    plt.show()
