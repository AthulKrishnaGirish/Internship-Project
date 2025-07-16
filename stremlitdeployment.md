import streamlit as st
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

st.title(" Stock Forecasting App (Prophet Model)")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
if st.button("Forecast"):
    df = yf.download(ticker, start="2020-01-01")["Close"].reset_index()
    df.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.write("### Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

