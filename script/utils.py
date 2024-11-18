import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred, stock_symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual", color='blue')
    plt.plot(y_test.index, y_pred, label="Predicted", color='red', alpha=0.7)
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
