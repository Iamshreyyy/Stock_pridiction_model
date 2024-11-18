from data_processing import load_and_preprocess_data
from model_training import train_and_evaluate_model
from utils import plot_predictions

def main():
    stock_symbol = 'MRF.NS'
    start_date = '2010-01-01'
    end_date = '2024-01-01'

    # Step 1: Data processing
    stock_data, X_train, X_test, y_train, y_test = load_and_preprocess_data(stock_symbol, start_date, end_date)

    # Step 2: Train and evaluate model
    model, y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # Step 3: Plot predictions
    plot_predictions(y_test, y_pred, stock_symbol)

if __name__ == "__main__":
    main()
