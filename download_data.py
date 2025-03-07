import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

# Define the list of tickers
tickers = [
    'IBM', 'BSX', 'KKR', 'QCOM', 'MRK', 'ICE', 'RY', 'ISRG', 'AXP', 'ORCL', 'AMAT', 'MSFT', 'SYK', 'RCL', 'TXN', 'CSCO', 'MCO', 'RDDT', 'PINS', 'BMY', 'TMUS', 'BAC', 'ZTS', 'CRM', 'AMZN', 'WMG', 'OMC', 'MA', 'HLT', 'LULU', 'JNJ', 'ADI', 'IBKR', 'META', 'CB', 'DHR', 'ABNB', 'BKNG', 'NWSA', 'EA', 'MELI', 'BLK', 'FWONA', 'LLY', 'GS', 'PFE', 'TD', 'SCHW', 'AMGN', 'CMCSA', 'FLUT', 'MCK', 'SPGI', 'ADP', 'CPNG', 'GILD', 'TJX', 'BRK-B', 'NOW', 'CMG', 'WBD', 'BX', 'CVS', 'UNH', 'AZO', 'TMO', 'MS', 'MMC', 'AVGO', 'T', 'V', 'ACN', 'AMD', 'BDX', 'DASH', 'NVDA', 'CI', 'CME', 'TTWO', 'ANET', 'YUM', 'SPOT', 'MCD', 'DHI', 'ELV', 'GSAT', 'FOXA', 'LYV', 'TSLA', 'AON', 'INTU', 'ABT', 'ORLY', 'PGR', 'CHTR', 'C', 'HD', 'AAPL', 'MU', 'ABBV', 'UBER', 'VZ', 'WFC', 'LOW', 'PLTR', 'LRCX', 'RBLX', 'GM', 'MDT', 'ZG', 'F', 'KLAC', 'JPM', 'DIS', 'NKE', 'FI', 'NFLX', 'HCA', 'MAR', 'SBUX', 'REGN', 'ADBE', 'ROST', 'GOOG', 'VRTX'
]

benchmarks = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, Nasdaq

# Combine both lists
all_tickers = tickers + benchmarks

# Define the date range
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')

# Download data
data = yf.download(all_tickers, start=start_date,
                   end=end_date, interval='1d')["Close"]

# Save to CSV
data.to_csv("stock_data.csv")

print("Data downloaded and saved to stock_data.csv")


# def download_data(tickers):
#     data = None
#     for attempt in range(5):  # Retry up to 5 times
#         try:
#             data = yf.download(tickers, start=start_date,
#                                end=end_date, interval='1d')['Close']
#             break  # Exit loop if successful
#         except yf.YFRequestError as e:
#             print(f"Error: {e}. Retrying in {2**attempt} seconds...")
#             time.sleep(2**attempt)  # Exponential backoff
#     return data


# # Download data in batches of 5 to avoid rate limits
# batch_size = 25
# data_frames = []
# for i in range(0, len(all_tickers), batch_size):
#     batch = all_tickers[i:i + batch_size]
#     print(f"Downloading batch: {batch}")
#     df = download_data(batch)
#     if df is not None:
#         data_frames.append(df)
#     time.sleep(2)  # Pause to avoid hitting rate limits

# # Concatenate all data
# data = pd.concat(data_frames, axis=1)

# # Save to CSV
# data.to_csv("stock_data.csv")

# print("Data downloaded and saved to stock_data.csv")
