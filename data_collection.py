import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import os

# Create directories for storing data
os.makedirs("data", exist_ok=True)

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    data.to_csv(f"data/{ticker}_raw.csv")
    print("Data saved to data directory.")
    return data

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    """
    print("Adding technical indicators...")
    # Simple Moving Average (SMA)
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    
    # Exponential Moving Average (EMA)
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    
    # Relative Strength Index (RSI)
    df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()

    print("Technical indicators added.")
    return df

def preprocess_data(ticker, start_date, end_date):
    """
    Fetch and preprocess stock data.
    """
    try:
        # Fetch raw data
        df = fetch_stock_data(ticker, start_date, end_date)
        
        # Drop irrelevant columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Save preprocessed data
        output_file = f"data/{ticker}_processed.csv"
        df.to_csv(output_file, index=True)
        print(f"Preprocessed data saved to {output_file}.")
        
        return df
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage
    TICKER = "RELIANCE.NS"  # NSE stock ticker
    START_DATE = "2015-01-01"
    END_DATE = "2024-01-01"
    
    data = preprocess_data(TICKER, START_DATE, END_DATE)
    print(data.head())
