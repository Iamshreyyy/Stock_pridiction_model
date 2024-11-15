import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load("models/random_forest_model.pkl")

# Title
st.title("Stock Market Prediction")

# Input form
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
look_back = st.sidebar.slider("Look Back Period (Days)", 10, 100, 60)

# Predictions
st.subheader("Prediction")
if st.button("Predict Next Value"):
    # Load data (You need a function to fetch recent data)
    data = pd.read_csv("data/RELIANCE.NS_processed.csv")
    last_sequence = data['Close'].values[-look_back:]
    prediction = model.predict(np.array(last_sequence).reshape(1, -1))
    st.write(f"The predicted next value is: {prediction[0]:.2f}")
