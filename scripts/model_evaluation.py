import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Load the trained model
model = joblib.load('data/stock_price_model.pkl')

# Make predictions
predictions = model.predict(X_test)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
predictions_df.to_csv('data/predictions.csv', index=False)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save evaluation results
results = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': predictions})
results.to_csv('data/evaluation_results.csv', index=False)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')