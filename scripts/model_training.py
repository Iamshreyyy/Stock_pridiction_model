import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the processed data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

# Convert y_train to a 1D array
y_train = y_train.values.ravel()  # Flatten the array

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'data/stock_price_model.pkl')