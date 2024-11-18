import yfinance as yf
import pandas as pd

def load_and_preprocess_data(stock_symbol, start_date, end_date):
    # Download historical stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Feature Engineering
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()

    # RSI Calculation
    delta = stock_data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    stock_data['12_EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['26_EMA'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['12_EMA'] - stock_data['26_EMA']

    # Drop missing values
    stock_data = stock_data.dropna()

    # Target Variable
    stock_data['Target'] = stock_data['Close'].shift(-1)
    stock_data = stock_data.dropna()

    # Features and Target
    X = stock_data[['SMA_50', 'SMA_200', 'EMA_50', 'RSI', 'MACD']]
    y = stock_data['Target']

    # Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return stock_data, X_train, X_test, y_train, y_test
