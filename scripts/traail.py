#import Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Step 1: Download Historical Stock Data
stock_symbol = ''  # Example: Apple Inc.
start_date = '2010-01-01'
end_date = '2024-01-01'

# Download historical stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Display the first few rows of the data
print("First few rows of the data:")
print(stock_data.head())

# Step 2: Feature Engineering

# Calculate moving averages
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

# Calculate Exponential Moving Average (EMA)
stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()

# Calculate the Relative Strength Index (RSI)
delta = stock_data['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Calculate Moving Average Convergence Divergence (MACD)
stock_data['12_EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
stock_data['26_EMA'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
stock_data['MACD'] = stock_data['12_EMA'] - stock_data['26_EMA']

# Drop missing values generated by rolling window
stock_data = stock_data.dropna()

# Feature: Target variable is next day's close price
stock_data['Target'] = stock_data['Close'].shift(-1)

# Drop rows where target is NaN
stock_data = stock_data.dropna()

# Display a preview of the engineered data
print("Data with technical indicators:")
print(stock_data.head())

# Step 3: Train-Test Split
# Define features (X) and target variable (y)
X = stock_data[['SMA_50', 'SMA_200', 'EMA_50', 'RSI', 'MACD']]
y = stock_data['Target']

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# Step 4: Model Training
# Initialize and train the model
model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# Step 5: Model Prediction
# Predict the stock prices on the test data
y_pred = model.predict(X_test)

# Step 6: Model Evaluation
# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"\nModel Evaluation Metrics:")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 7: Plotting Predictions vs. Actual Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted", color='red', alpha=0.7)
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()