# app.py
import streamlit as st
from main import run_model
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Import metrics
from utils import plot_predictions  # Import plot_predictions function

def main():
    # Title of the app
    st.title('Stock Price Prediction App')

    # User inputs
    stock_symbol = st.text_input('Enter stock symbol (e.g., AAPL, GOOGL, IOC.NS):', 'IOC.NS')
    start_date = st.date_input('Start date:', pd.to_datetime('2010-01-01'))
    end_date = st.date_input('End date:', pd.to_datetime('2024-01-01'))

    if st.button('Predict'):
        # Run the model
        y_test, y_pred, stock_symbol = run_model(stock_symbol, start_date, end_date)

        # Display results
        st.subheader('Model Evaluation Metrics:')
        st.write(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}')
        st.write(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}')
        st.write(f'R² Score: {r2_score(y_test, y_pred):.4f}')

        # Plot predictions
        st.subheader(f'Predictions for {stock_symbol}')
        plot_predictions(y_test, y_pred, stock_symbol)

if __name__ == "__main__":
    main()