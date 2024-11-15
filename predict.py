import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for saving plots without displaying them

import matplotlib.pyplot as plt

def load_model(model_file):
    """
    Load a trained model.
    """
    print(f"Loading model from {model_file}...")
    model = joblib.load(model_file)
    print("Model loaded successfully.")
    return model

def predict_next_values(model, last_sequence, steps=10):
    """
    Predict the next `steps` values based on the last sequence of data.
    """
    predictions = []
    for _ in range(steps):
        prediction = model.predict(last_sequence.reshape(1, -1))
        predictions.append(prediction[0])
        # Update the sequence
        last_sequence = np.append(last_sequence[1:], prediction[0])
    return predictions
def plot_predictions(true_values, predicted_values, start_index):
    """
    Plot actual vs. predicted stock prices and save the plot as a file.
    """
    # Convert numpy arrays to pandas Series to use dropna
    true_values = pd.Series(true_values).dropna()
    predicted_values = pd.Series(predicted_values).dropna()

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(true_values)), true_values, label="Actual Prices", color="blue", linestyle="--")
    plt.plot(range(start_index, start_index + len(predicted_values)), predicted_values, label="Predicted Prices", color="red")
    
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Actual vs Predicted Stock Prices")
    plt.legend()
    plt.grid()
    
    # Save or show plot
    plt.savefig("predicted_stock_prices.png")  # Save the plot as an image file
    print("Plot saved as predicted_stock_prices.png")
    # plt.show()  # Uncomment if you want to display the plot



if __name__ == "__main__":
    MODEL_FILE = "models/random_forest_model.pkl"
    DATA_FILE = "data/RELIANCE.NS_processed.csv"
    LOOK_BACK = 60
    FUTURE_STEPS = 10  # Number of days to predict

    # Load model
    model = load_model(MODEL_FILE)

    # Load data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    data = df['Close'].values

    # Use the last LOOK_BACK values for prediction
    last_sequence = data[-LOOK_BACK:]
    predictions = predict_next_values(model, last_sequence, steps=FUTURE_STEPS)

    # Plot actual vs predicted
    plot_predictions(data, predictions, len(data))
