import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
import joblib

# Create directories for saving models
os.makedirs("models", exist_ok=True)

def load_and_prepare_data(file_path, look_back=60):
    """
    Load and prepare data for regression models.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Ensure all values in 'Close' are numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    data = df['Close'].values

    # Create features (X) and labels (y)
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_random_forest(X, y, model_file):
    """
    Train a Random Forest model and save it.
    """
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Save the model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}.")
    return model

if __name__ == "__main__":
    DATA_FILE = "data/RELIANCE.NS_processed.csv"
    MODEL_FILE = "models/random_forest_model.pkl"
    
    # Load and prepare data
    X, y = load_and_prepare_data(DATA_FILE)
    
    # Train the model
    model = train_random_forest(X, y, MODEL_FILE)
