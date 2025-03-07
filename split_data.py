from datetime import datetime, timedelta

# Splitting data into training (n years) and testing (5-n years)
from utils import load_data


def split_data(data, n):
    split_date = (datetime.today() - timedelta(days=(5-n)*365)
                  ).strftime('%Y-%m-%d')
    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]

    train_data.to_csv("train_data.csv")
    test_data.to_csv("test_data.csv")

    print(
        f"Data split into train_data.csv ({n} years) and test_data.csv ({5-n} years)")


data = load_data('stock_data.csv')
# Call the split function with desired training years
split_data(data, n=3)
