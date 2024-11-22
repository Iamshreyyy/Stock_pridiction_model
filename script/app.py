# # # import streamlit as st
# # # import pandas as pd
# # # import yfinance as yf
# # # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # # from utils import plot_predictions
# # # from main import run_model
# # # import requests

# # # # Load Indian stock list dynamically
# # # @st.cache_data
# # # def load_indian_stocks():
# # #     """
# # #     Load Indian stock data from a local preloaded CSV file.
# # #     """
# # #     try:
# # #         data = pd.read_csv("../data/EQUITY_L.csv")
# # #         data = data[['SYMBOL', 'NAME OF COMPANY']].rename(columns={'SYMBOL': 'Symbol', 'NAME OF COMPANY': 'Name'})
# # #         data['Symbol'] = data['Symbol'] + '.NS'  # Append `.NS` for yfinance compatibility
# # #         return data
# # #     except Exception as e:
# # #         st.error("Failed to load Indian stock data from the local file. Please ensure the file exists.")
# # #         return pd.DataFrame(columns=["Name", "Symbol"])

# # # # Load global stock list (S&P 500)
# # # @st.cache_data
# # # def load_global_stocks():
# # #     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# # #     tables = pd.read_html(requests.get(url).content)
# # #     return tables[0][['Security', 'Symbol']].rename(columns={'Security': 'Name'})

# # # def main():
# # #     st.title("Stock Price Prediction App")

# # #     # Load stocks
# # #     indian_stocks = load_indian_stocks()
# # #     global_stocks = load_global_stocks()

# # #     # Combine both datasets
# # #     combined_stocks = pd.concat([indian_stocks, global_stocks], ignore_index=True)

# # #     # Create a dictionary for easy lookup
# # #     stock_dict = {row['Name']: row['Symbol'] for _, row in combined_stocks.iterrows()}

# # #     # Autocomplete dropdown
# # #     stock_name = st.selectbox(
# # #         "Search for a stock by name:",
# # #         options=[""] + list(stock_dict.keys()),
# # #         format_func=lambda x: "Select a stock" if x == "" else x,
# # #     )

# # #     if stock_name:
# # #         stock_symbol = stock_dict[stock_name]
# # #         st.write(f"Selected Stock: {stock_name} ({stock_symbol})")
# # #     else:
# # #         stock_symbol = None

# # #     # Date inputs
# # #     start_date = st.date_input("Start date:", pd.to_datetime("2010-01-01"))
# # #     end_date = st.date_input("End date:", pd.to_datetime("today").date())

# # #     if st.button("Predict") and stock_symbol:
# # #         try:
# # #             # Run the model
# # #             y_test, y_pred, stock_symbol = run_model(stock_symbol, start_date, end_date)

# # #             # Display results
# # #             st.subheader("Model Evaluation Metrics:")
# # #             st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
# # #             st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
# # #             st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# # #             # Plot predictions
# # #             st.subheader(f"Predictions for {stock_symbol}")
# # #             plot_predictions(y_test, y_pred, stock_symbol)
# # #         except Exception as e:
# # #             st.error(f"An error occurred: {e}")

# # # if __name__ == "__main__":
# # #     main()
# # import streamlit as st
# # import pandas as pd
# # import yfinance as yf
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from utils import plot_predictions
# # from main import run_model
# # import requests
# # from streamlit_autorefresh import st_autorefresh

# # # Load Indian stock list dynamically
# # @st.cache_data
# # def load_indian_stocks():
# #     """Load Indian stock data from a local preloaded CSV file."""
# #     try:
# #         data = pd.read_csv("../data/EQUITY_L.csv")
# #         data = data[['SYMBOL', 'NAME OF COMPANY']].rename(columns={'SYMBOL': 'Symbol', 'NAME OF COMPANY': 'Name'})
# #         data['Symbol'] = data['Symbol'] + '.NS'  # Append `.NS` for yfinance compatibility
# #         return data
# #     except Exception as e:
# #         st.error("Failed to load Indian stock data from the local file. Please ensure the file exists.")
# #         return pd.DataFrame(columns=["Name", "Symbol"])

# # # Load global stock list (S&P 500)
# # @st.cache_data
# # def load_global_stocks():
# #     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# #     tables = pd.read_html(requests.get(url).content)
# #     return tables[0][['Security', 'Symbol']].rename(columns={'Security': 'Name'})

# # def get_stock_data(symbols):
# #     """Fetch the current stock prices and changes for a list of symbols."""
# #     stock_data = []
# #     for symbol in symbols:
# #         try:
# #             ticker = yf.Ticker(symbol)
# #             data = ticker.history(period="1d")
# #             if not data.empty:
# #                 current_price = data['Close'].iloc[-1]
# #                 previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
# #                 change = current_price - previous_close
# #                 percent_change = (change / previous_close) * 100 if previous_close != 0 else 0
# #                 stock_data.append({
# #                     "symbol": symbol,
# #                     "current_price": current_price,
# #                     "change": change,
# #                     "percent_change": percent_change
# #                 })
# #         except Exception as e:
# #             stock_data.append({"symbol": symbol, "error": str(e)})
# #     return stock_data

# # def display_ticker(stock_data):
# #     """Show a rotating ticker stripe of stock information."""
# #     if stock_data:
# #         ticker_text = " | ".join([f"{stock['symbol']}: ${stock['current_price']:.2f} "
# #                                   f"({stock['change']:+.2f}, {stock['percent_change']:+.2f}%)"
# #                                   for stock in stock_data if "error" not in stock])
# #         st.markdown(f"""
# #         <div style="
# #             white-space: nowrap; 
# #             overflow: hidden; 
# #             width: 100%; 
# #             animation: ticker 15s linear infinite; 
# #             background-color: #f9f9f9; 
# #             padding: 10px;
# #             border-radius: 5px;
# #             font-size: 18px;">
# #             {ticker_text}
# #         </div>
# #         <style>
# #         @keyframes ticker {{
# #             0% {{ transform: translateX(100%); }}
# #             100% {{ transform: translateX(-100%); }}
# #         }}
# #         </style>
# #         """, unsafe_allow_html=True)

# # def main():
# #     st.title("Stock Price Prediction App")

# #     # Load stocks
# #     indian_stocks = load_indian_stocks()
# #     global_stocks = load_global_stocks()

# #     # Combine both datasets
# #     combined_stocks = pd.concat([indian_stocks, global_stocks], ignore_index=True)

# #     # Create a dictionary for easy lookup
# #     stock_dict = {row['Name']: row['Symbol'] for _, row in combined_stocks.iterrows()}

# #     # Create a placeholder for the ticker section
# #     ticker_placeholder = st.empty()

# #     # Autocomplete dropdown for stock selection
# #     stock_name = st.selectbox(
# #         "Search for a stock by name:",
# #         options=[""] + list(stock_dict.keys()),
# #         format_func=lambda x: "Select a stock" if x == "" else x,
# #     )

# #     if stock_name:
# #         stock_symbol = stock_dict[stock_name]
# #         st.write(f"Selected Stock: {stock_name} ({stock_symbol})")
# #     else:
# #         stock_symbol = None

# #     # Date inputs
# #     start_date = st.date_input("Start date:", pd.to_datetime("2010-01-01"))
# #     end_date = st.date_input("End date:", pd.to_datetime("today").date())

# #     # Store prediction in session state
# #     if 'prediction' not in st.session_state:
# #         st.session_state.prediction = None

# #     # Prediction section
# #     with st.container():
# #         if st.button("🚀 Predict") and stock_symbol:
# #             try:
# #                 # Run the model
# #                 y_test, y_pred, stock_symbol = run_model(stock_symbol, start_date, end_date)
# #                 # Store results in session state
# #                 st.session_state.prediction = {
# #                     'y_test': y_test,
# #                     'y_pred': y_pred,
# #                     'stock_symbol': stock_symbol
# #                 }

# #                 # Display model evaluation metrics
# #                 st.subheader("Model Evaluation Metrics:")
# #                 st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
# #                 st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
# #                 st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# #                 # Plot predictions
# #                 st.subheader(f"Predictions for {stock_symbol}")
# #                 plot_predictions(y_test, y_pred, stock_symbol)

# #             except Exception as e:
# #                 st.error(f"An error occurred: {e}")

# #     # Footer
# #     with st.container():
# #         st.markdown("---")
# #         st.write("Developed by [Your Name](https://github.com/your-profile).")

# #     # Auto-refresh for ticker section every 10 seconds
# #     def refresh_ticker():
# #         stock_data = get_stock_data(combined_stocks['Symbol'].tolist())
# #         ticker_placeholder.empty()  # Clear previous content
# #         display_ticker(stock_data)

# #     # Setting up auto-refresh for ticker every 10 seconds
# #     st_autorefresh(interval=10_000, limit=None, key="ticker_refresh")
# #     refresh_ticker()  # Initial load of the ticker data

# # if __name__ == "__main__":
# #     main()
# import streamlit as st
# import pandas as pd
# import yfinance as yf
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from utils import plot_predictions
# from main import run_model
# import requests
# from streamlit_autorefresh import st_autorefresh

# # Load Indian stock list dynamically
# @st.cache_data
# def load_indian_stocks():
#     """Load Indian stock data from a local preloaded CSV file."""
#     try:
#         data = pd.read_csv("../data/EQUITY_L.csv")
#         data = data[['SYMBOL', 'NAME OF COMPANY']].rename(columns={'SYMBOL': 'Symbol', 'NAME OF COMPANY': 'Name'})
#         data['Symbol'] = data['Symbol'] + '.NS'  # Append `.NS` for yfinance compatibility
#         return data
#     except Exception as e:
#         st.error("Failed to load Indian stock data from the local file. Please ensure the file exists.")
#         return pd.DataFrame(columns=["Name", "Symbol"])

# # Load global stock list (S&P 500)
# @st.cache_data
# def load_global_stocks():
#     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#     tables = pd.read_html(requests.get(url).content)
#     return tables[0][['Security', 'Symbol']].rename(columns={'Security': 'Name'})

# def get_stock_data(symbols):
#     """Fetch the current stock prices and changes for a list of symbols."""
#     stock_data = []
#     for symbol in symbols:
#         try:
#             ticker = yf.Ticker(symbol)
#             data = ticker.history(period="1d")
#             if not data.empty:
#                 current_price = data['Close'].iloc[-1]
#                 previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
#                 change = current_price - previous_close
#                 percent_change = (change / previous_close) * 100 if previous_close != 0 else 0
#                 stock_data.append({
#                     "symbol": symbol,
#                     "current_price": current_price,
#                     "change": change,
#                     "percent_change": percent_change
#                 })
#         except Exception as e:
#             stock_data.append({"symbol": symbol, "error": str(e)})
#     return stock_data

# def display_ticker(stock_data):
#     """Show a rotating ticker stripe of stock information."""
#     if stock_data:
#         ticker_text = " | ".join([f"{stock['symbol']}: ${stock['current_price']:.2f} "
#                                   f"({stock['change']:+.2f}, {stock['percent_change']:+.2f}%)"
#                                   for stock in stock_data if "error" not in stock])
#         st.markdown(f"""
#         <div style="
#             white-space: nowrap; 
#             overflow: hidden; 
#             width: 100%; 
#             animation: ticker 15s linear infinite; 
#             background-color: #f9f9f9; 
#             padding: 10px;
#             border-radius: 5px;
#             font-size: 18px;">
#             {ticker_text}
#         </div>
#         <style>
#         @keyframes ticker {{
#             0% {{ transform: translateX(100%); }}
#             100% {{ transform: translateX(-100%); }}
#         }}
#         </style>
#         """, unsafe_allow_html=True)

# def main():
#     st.set_page_config(page_title="Stock Price Prediction App", layout="wide")

#     # Header section
#     with st.container():
#         st.title("📈 Stock Price Prediction App")
#         st.markdown("""
#         This app predicts stock prices using historical data and advanced machine learning models.
#         Enter a stock symbol, select a date range, and click **Predict** to see the results.
#         """)

#     # Load stocks
#     indian_stocks = load_indian_stocks()
#     global_stocks = load_global_stocks()

#     # Combine both datasets
#     combined_stocks = pd.concat([indian_stocks, global_stocks], ignore_index=True)

#     # Create a dictionary for easy lookup
#     stock_dict = {row['Name']: row['Symbol'] for _, row in combined_stocks.iterrows()}

#     # Create a placeholder for the ticker section
#     ticker_placeholder = st.empty()

#     # Autocomplete dropdown for stock selection
#     stock_name = st.selectbox(
#         "Search for a stock by name:",
#         options=[""] + list(stock_dict.keys()),
#         format_func=lambda x: "Select a stock" if x == "" else x,
#     )

#     if stock_name:
#         stock_symbol = stock_dict[stock_name]
#         st.write(f"Selected Stock: {stock_name} ({stock_symbol})")
#     else:
#         stock_symbol = None

#     # Date inputs
#     start_date = st.date_input("Start date:", pd.to_datetime("2010-01-01"))
#     end_date = st.date_input("End date:", pd.to_datetime("today").date())

#     # Store prediction in session state
#     if 'prediction' not in st.session_state:
#         st.session_state.prediction = None

#     # Prediction section
#     with st.container():
#         if st.button("🚀 Predict") and stock_symbol:
#             try:
#                 # Run the model
#                 y_test, y_pred, stock_symbol = run_model(stock_symbol, start_date, end_date)
#                 # Store results in session state
#                 st.session_state.prediction = {
#                     'y_test': y_test,
#                     'y_pred': y_pred,
#                     'stock_symbol': stock_symbol
#                 }

#                 # Display model evaluation metrics
#                 st.subheader("Model Evaluation Metrics:")
#                 st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
#                 st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
#                 st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

#                 # Plot predictions
#                 st.subheader(f"Predictions for {stock_symbol}")
#                 plot_predictions(y_test, y_pred, stock_symbol)

#             except Exception as e:
#                 st.error(f"An error occurred: {e}")

#     # Footer
#     with st.container():
#         st.markdown("---")
#         st.write("Developed by [Your Name](https://github.com/your-profile).")

#     # Auto-refresh for ticker section every 10 seconds
#     def refresh_ticker():
#         stock_data = get_stock_data(combined_stocks['Symbol'].tolist())
#         ticker_placeholder.empty()  # Clear previous content
#         display_ticker(stock_data)

#     # Setting up auto-refresh for ticker every 10 seconds
#     st_autorefresh(interval=10_000, limit=None, key="ticker_refresh")
#     refresh_ticker()  # Initial load of the ticker data

# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import plot_predictions  # Ensure this file exists and contains the function
from main import run_model  # Ensure this file exists and contains the function
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

def get_stock_data(symbols):
    """Fetch the current stock prices and changes for a list of symbols."""
    stock_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - previous_close
                percent_change = (change / previous_close) * 100 if previous_close != 0 else 0
                stock_data.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "change": change,
                    "percent_change": percent_change
                })
        except Exception as e:
            stock_data.append({"symbol": symbol, "error": str(e)})
    return stock_data

def display_ticker(stock_data):
    """Show a rotating ticker stripe of stock information."""
    if stock_data:
        ticker_text = " | ".join([f"{stock['symbol']}: ${stock['current_price']:.2f} "
                                  f"({stock['change']:+.2f}, {stock['percent_change']:+.2f}%)"
                                  for stock in stock_data if "error" not in stock])
        st.markdown(f"""
        <div style="
            white-space: nowrap; 
            overflow: hidden; 
            width: 100%; 
            animation: ticker 15s linear infinite; 
            background-color: #f9f9f9; 
            padding: 10px;
            border-radius: 5px;
            font-size: 18px;">
            {ticker_text}
        </div>
        <style>
        @keyframes ticker {{
            0% {{ transform: translateX(100%); }}
            100% {{ transform: translateX(-100%); }}
        }}
        </style>
        """, unsafe_allow_html=True)

def main():
    # Set up the page configuration
    st.set_page_config(page_title="Stock Price Prediction App", layout="wide")

    # Header section
    with st.container():
        st.title("📈 Stock Price Prediction App")
        st.markdown("""
        This app predicts stock prices using historical data and advanced machine learning models.
        Enter a stock symbol, select a date range, and click **Predict** to see the results.
        """)

    # Define the stock symbols to display
    stock_symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NFLX", "NVDA", "META"]

    # Create a placeholder for the ticker section
    ticker_placeholder = st.empty()

    # Input and prediction sections (remain outside the auto-refresh scope)
    with st.container():
        st.subheader("🔧 User Inputs")
        col1, col2, col3 = st.columns(3)

        with col1:
            stock_symbol = st.text_input(
                'Enter Stock Symbol:',
                placeholder='e.g., AAPL, GOOGL, IOC.NS',
                help="Provide the stock ticker symbol (e.g., AAPL for Apple)."
            )

        with col2:
            start_date = st.date_input(
                "Start Date:",
                value=pd.to_datetime('2010-01-01'),
                help="Select the start date for historical data."
            )

        with col3:
            end_date = st.date_input(
                "End Date:",
                value=pd.to_datetime('2024-01-01'),
                help="Select the end date for historical data."
            )

        st.markdown("---")

    # Prediction and output section (remains outside the auto-refresh scope)
    with st.container():
        if st.button("🚀 Predict"):
            st.subheader("🔍 Model Evaluation and Results")

            try:
                # Run the model and store the results in session state
                if 'prediction' not in st.session_state:
                    y_test, y_pred, stock_symbol = run_model(stock_symbol, start_date, end_date)
                    st.session_state.prediction = {
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'stock_symbol': stock_symbol
                    }

                # Use stored results from session state
                prediction = st.session_state.prediction
                y_test = prediction['y_test']
                y_pred = prediction['y_pred']
                stock_symbol = prediction['stock_symbol']

                # Display evaluation metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.4f}")
                col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.4f}")
                col3.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")

                # Display predictions
                st.subheader(f"📊 Predictions for {stock_symbol}")
                plot_predictions(y_test, y_pred, stock_symbol)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("Click **Predict** to generate predictions.")

    # Footer (remains outside the auto-refresh scope)
    with st.container():
        st.markdown("---")
        st.write("Developed by [Your Name](https://github.com/your-profile).")

    # Auto-refresh only the ticker section every 10 seconds
    def refresh_ticker():
        stock_data = get_stock_data(stock_symbols)
        ticker_placeholder.empty()  # Clear previous content
        display_ticker(stock_data)

    # Setting up auto-refresh for ticker every 10 seconds
    st_autorefresh(interval=10_000, limit=None, key="ticker_refresh")
    refresh_ticker()  # Initial load of the ticker data

if __name__ == "__main__":
    main()


