# data.py
# librairies

import yfinance as yf
import pandas as pd
from datetime import datetime

# Function to fetch Microsoft stock data
def fetch_stock_data(ticker='MSFT', start_date=None, end_date=None):
    if start_date is None:
        start_date = '2021-01-01'  # Default to earliest date
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess stock data
def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()

    # Feature engineering (e.g., adding moving averages)
    df['MA_20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average
    df['MA_50'] = df['Close'].rolling(window=50).mean()  # 50-day moving average

    # RSI Calculation
    delta = df['Close'].diff()  # Change in price
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gain over 14 days
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average loss over 14 days
    rs = gain / loss  # Relative Strength
    df['RSI'] = 100 - (100 / (1 + rs))  # RSI formula

    # Normalizing 'Close' price for model input
    df['Normalized_Close'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()

    # Drop initial rows with NaN values in MA columns
    df = df.dropna()

    return df

#  Fetch and preprocess data
if __name__ == "__main__":
    data = fetch_stock_data()
    processed_data = preprocess_data(data)
    print(processed_data.tail())  # Check the processed data