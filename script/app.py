import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import plot_predictions
from main import run_model
import requests

# Load Indian stock list dynamically
@st.cache_data
def load_indian_stocks():
    """
    Load Indian stock data from a local preloaded CSV file.
    """
    try:
        data = pd.read_csv("../data/EQUITY_L.csv")
        data = data[['SYMBOL', 'NAME OF COMPANY']].rename(columns={'SYMBOL': 'Symbol', 'NAME OF COMPANY': 'Name'})
        data['Symbol'] = data['Symbol'] + '.NS'  # Append `.NS` for yfinance compatibility
        return data
    except Exception as e:
        st.error("Failed to load Indian stock data from the local file. Please ensure the file exists.")
        return pd.DataFrame(columns=["Name", "Symbol"])

# Load global stock list (S&P 500)
@st.cache_data
def load_global_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(requests.get(url).content)
    return tables[0][['Security', 'Symbol']].rename(columns={'Security': 'Name'})

def main():
    st.title("Stock Price Prediction App")

    # Load stocks
    indian_stocks = load_indian_stocks()
    global_stocks = load_global_stocks()

    # Combine both datasets
    combined_stocks = pd.concat([indian_stocks, global_stocks], ignore_index=True)

    # Create a dictionary for easy lookup
    stock_dict = {row['Name']: row['Symbol'] for _, row in combined_stocks.iterrows()}

    # Autocomplete dropdown
    stock_name = st.selectbox(
        "Search for a stock by name:",
        options=[""] + list(stock_dict.keys()),
        format_func=lambda x: "Select a stock" if x == "" else x,
    )

    if stock_name:
        stock_symbol = stock_dict[stock_name]
        st.write(f"Selected Stock: {stock_name} ({stock_symbol})")
    else:
        stock_symbol = None

    # Date inputs
    start_date = st.date_input("Start date:", pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End date:", pd.to_datetime("today").date())

    if st.button("Predict") and stock_symbol:
        try:
            # Run the model
            y_test, y_pred, stock_symbol = run_model(stock_symbol, start_date, end_date)

            # Display results
            st.subheader("Model Evaluation Metrics:")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

            # Plot predictions
            st.subheader(f"Predictions for {stock_symbol}")
            plot_predictions(y_test, y_pred, stock_symbol)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
