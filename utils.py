import pandas as pd


# =============================================================================
# 1. Data Loading
# =============================================================================


def load_data(filename):
    """
    Load CSV data with a datetime index.
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df.dropna(axis=1, inplace=True)
    return df


def generate_signals_from_spread(spread, window=20, threshold=1.0):
    """
    Generate trading signals from the spread time series.
    A rolling window is used to compute the z-score of the spread.

    Parameters:
    -----------
    spread : pd.Series
        The spread time series.
    window : int
        Rolling window size.
    threshold : float
        Z-score threshold for signal generation.

    Returns:
    --------
    signals : pd.Series
        Trading signals: +1 (long spread) when spread is low,
                         -1 (short spread) when spread is high,
                          0 otherwise.
    zscore : pd.Series
        The computed z-scores.
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    signals = pd.Series(index=spread.index, data=0)
    # short the spread (expect reversion downward)
    signals[zscore > threshold] = -1
    # long the spread (expect reversion upward)
    signals[zscore < -threshold] = 1
    return signals, zscore


def compute_stock_portfolio_returns(stock, signals):
    """
    Parameters:
    -----------
    stock : pd.Series
        Price series for the stock.
    signals : pd.Series
        Trading signals (shifted to avoid lookahead bias).

    Returns:
    --------
    port_returns : pd.Series
        Daily portfolio returns.
    """
    # Calculate daily returns of the stock.
    r = stock.pct_change().fillna(0)
    # Shift signals to avoid lookahead bias.
    signals_shifted = signals.shift(1).fillna(0)
    # Compute the portfolio return as the product of the shifted signal and daily returns.
    port_returns = signals_shifted * r
    return port_returns


def generate_signals_from_stock(stock, window=20, threshold=1.0):
    """
    Generate trading signals from a single stock's price series.
    A rolling window is used to compute the z-score of the price.

    Parameters:
    -----------
    stock : pd.Series
        The stock price series.
    window : int
        Rolling window size.
    threshold : float
        Z-score threshold for signal generation.

    Returns:
    --------
    signals : pd.Series
        Trading signals: +1 (long) when price is low relative to its rolling mean,
                         -1 (short) when price is high,
                          0 otherwise.
    zscore : pd.Series
        The computed z-scores.
    """
    rolling_mean = stock.rolling(window=window).mean()
    rolling_std = stock.rolling(window=window).std()
    zscore = (stock - rolling_mean) / rolling_std
    signals = pd.Series(index=stock.index, data=0)
    # Short when price is unusually high (zscore > threshold)
    signals[zscore > threshold] = -1
    # Long when price is unusually low (zscore < -threshold)
    signals[zscore < -threshold] = 1
    return signals, zscore
