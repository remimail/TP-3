import yfinance as yf
import pandas as pd

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
