import yfinance as yf
import pandas as pd

# Define a list of ETF tickers
etf_tickers = ["ESGU", "EAGG", "ESGE", "ESML", "SUSB", "ESGD", "SHY", "SUSA", "GOVT", "MBB", "SUSC"]

# Create an empty DataFrame to store the start dates
df_start_dates = pd.DataFrame(index=["Start Date"])

# Loop through each ETF ticker and retrieve the start date
for ticker in etf_tickers:
    etf = yf.Ticker(ticker)
    historical_data = etf.history(period="max")  # Retrieve all available historical data
    start_date = historical_data.index.min()
    df_start_dates[ticker] = [start_date]

# Display the DataFrame with start dates
print(df_start_dates)
