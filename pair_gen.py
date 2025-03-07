
import pandas as pd
import itertools
import numpy as np
from statsmodels.tsa.stattools import coint
from tqdm import tqdm


def find_cointegrated_pairs(data, significance=0.05):
    """
    For a DataFrame where each column is an asset's price series,
    this function tests all unique pairs for cointegration and returns
    a matrix of p-values and a list of cointegrated pairs.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing asset price series (each column represents an asset).
    significance : float, optional
        The p-value threshold for cointegration (default is 0.05).

    Returns
    -------
    pvalue_matrix : pd.DataFrame
        A symmetric matrix of p-values from the cointegration tests.
    cointegrated_pairs : list of tuples
        List of tuples in the form (Asset_A, Asset_B, p-value) for pairs that are cointegrated.
    """
    assets = data.columns
    n = len(assets)

    # Create a DataFrame to store the p-values (upper-triangular; lower-triangular remains as ones)
    pvalue_matrix = pd.DataFrame(np.ones((n, n)), index=assets, columns=assets)
    cointegrated_pairs = []

    # Loop over all unique pairs using itertools.combinations
    for asset1, asset2 in tqdm(itertools.combinations(assets, 2), total=(n*(n-1))/2):
        series1 = data[asset1]
        series2 = data[asset2]

        # Perform the Engle-Granger cointegration test
        score, p_value, _ = coint(series1, series2)

        # Store the p-value in the matrix (only upper triangular part)
        pvalue_matrix.loc[asset1, asset2] = p_value

        # If p-value is below the significance threshold, record the pair
        if p_value < significance:
            cointegrated_pairs.append((asset1, asset2, p_value))

    return pvalue_matrix, cointegrated_pairs


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    # Load your market data from a CSV file.
    # The CSV file should have a datetime index and 100 columns (one per asset).
    # For example, the file might be named 'market_data_100_assets.csv'.
    data = pd.read_csv('stock_data.csv',
                       index_col=0, parse_dates=True)

    data.drop(columns=["^DJI", "^GSPC", "^IXIC"], inplace=True)
    data.dropna(axis=1, inplace=True)

    # Optional: Sort the DataFrame by date (if not already sorted)
    data.sort_index(inplace=True)

    # Find cointegrated pairs among the assets
    pvalue_matrix, cointegrated_pairs = find_cointegrated_pairs(
        data, significance=0.05)

    with open("pairs.txt", "w") as f:
        f.write(str(cointegrated_pairs))

    # Display the results
    print("Cointegrated Pairs (with p-value below 0.05):")
    for asset1, asset2, p_value in cointegrated_pairs:
        print(f"{asset1} & {asset2}  -> p-value = {p_value:.4f}")
