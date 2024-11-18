import matplotlib.pyplot as plt
import yfinance as yf

def plot_predictions(y_test, y_pred, stock_symbol):
    # Fetch stock information using yfinance
    ticker = yf.Ticker(stock_symbol)
    stock_name = ticker.info.get('longName', stock_symbol)  # Default to symbol if name not found
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual", color='blue')
    plt.plot(y_test.index, y_pred, label="Predicted", color='red', alpha=0.7)
    plt.title(f'{stock_name} ({stock_symbol}) Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
