import pandas as pd
import matplotlib.pyplot as plt

# Load the predictions
predictions = pd.read_csv('data/predictions.csv')

# Plotting
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(predictions.index, predictions['Actual'], label='Actual Price', color='blue')

# Plot predicted prices
plt.plot(predictions.index, predictions['Predicted'], label='Predicted Price', color='orange')

# Add titles and labels
plt.title('Comparison of Actual vs Predicted Stock Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Show the plot
plt.savefig('data/prediction_comparison.png')  # Save the figure
plt.show()