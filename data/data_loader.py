import yfinance as yf
import pandas as pd
import numpy as np

def download_data(tickers,period='2y'):
    #Download historical closing prices for given tickers.
    data = yf.download(tickers, period=period)['Close']
    return data

def calculate_daily_returns(prices):
    #Calculate daily percentage returns from price data.
    returns = prices.pct_change().dropna()
    return returns

def clean_data(data):
    #Remove NaN values and handle missing data.
    data = data.dropna()
    return data

if __name__ == "__main__":
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    print(prices.head())
    print(returns.head())