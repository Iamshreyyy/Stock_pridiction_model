import pandas as pd

def create_features(data):
    # Create additional features (e.g., moving averages)
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Return'] = data['Close'].pct_change()
    
    # Drop NaN values created by rolling
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    data = pd.read_csv('./data/stock_data.csv')
    data = create_features(data)
    print(data.head())