import warnings
import riskfolio as rp
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
yf.pdr_override() 
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import datetime



# Tickers of assets
assets = ["ESGU", "EAGG", "ESGE", "ESML", "SUSB", "ESGD", "SHY", "SUSA", "GOVT", "MBB", "SUSC"]
assets.sort()

# Downloading etf_prices for the maximum available period
etf_prices = yf.download(assets)
etf_prices = etf_prices.loc[:,('Adj Close', slice(None))]
etf_prices.columns = assets

# Find the latest start date among all ETFs
latest_start_date = max(etf_prices.index.get_level_values(0).min() for asset in assets)

# Filter etf_prices to start from the latest available start date
etf_prices = etf_prices[etf_prices.index.get_level_values(0) >= latest_start_date]

# Forward fill missing values
etf_prices = etf_prices.ffill()

# Drop remaining rows with NaN values
etf_prices = etf_prices.dropna()

# Calculating returns
rets = etf_prices[assets].pct_change().dropna()

# Monthly rets
monthly_rets = rets.resample('M').agg(lambda x: (x + 1).prod() - 1)



def import_csv_with_date_index(csv_path):
    """
    Import a CSV file, use the first column as the index, and transform it into a date object.

    Parameters:
    - csv_path: Path to the CSV file.

    Returns:
    - DataFrame with the first column as the index and transformed into a date object.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Assuming the first column is the date column
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # Convert the first column to datetime

    # Set the first column as the index
    df.set_index(df.columns[0], inplace=True)

    return df




# Replace 'YOUR_API_KEY' with your actual FRED API key
api_key = '496662ff482d4f6b028d05fa48bfecbb'

def import_fred(tick_):
    # Set the start and end dates for the data
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(2023, 10, 31)   
    # Fetch data from FRED
    data = pdr.get_data_fred(tick_, start_date, end_date)
    return data



def getData(ticker):
    data = yf.download(ticker)["Close"] 
    # Compute daily returns
    data = data.pct_change()    
    # Compute monthly returns
    monthly_rets = data.resample('M').agg(lambda x: (x + 1).prod() - 1)
    return monthly_rets