import os
import yfinance as yf
import pandas as pd

# Create a directory for saving data files
os.makedirs("data", exist_ok=True)

def preprocess_data(ticker, start_date, end_date):
    """
    Fetch and preprocess stock data.
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data available for {ticker} in the given date range.")
            return None
        
        print("Data fetched successfully!")
        
        # Save raw data
        raw_file = f"data/{ticker}_raw.csv"
        df.to_csv(raw_file, index=True)
        print(f"Raw data saved to {raw_file}.")
        
        # Preprocessing: Keep only relevant columns and handle missing values
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.dropna(inplace=True)
        
        # Save processed data
        processed_file = f"data/{ticker}_processed.csv"
        df.to_csv(processed_file, index=True)
        print(f"Processed data saved to {processed_file}.")
        
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

if __name__ == "__main__":
    TICKER = "RELIANCE.NS"
    START_DATE = "2015-01-01"
    END_DATE = "2024-01-01"

    preprocess_data(TICKER, START_DATE, END_DATE)
