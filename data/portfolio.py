import numpy as np
import pandas as pd

def build_portfolio(returns, weights):
    #Calculate portfolio daily returns from individual stock returns and weights.
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

def covariance_matrix(returns):
    #Calculate annualized covariance matrix of stock returns.
    cov = returns.cov() * 252
    return cov

if __name__ == "__main__":
    from data_loader import download_data, clean_data, calculate_daily_returns
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    port_returns = build_portfolio(returns, weights)
    cov = covariance_matrix(returns)
    print("Portfolio returns (first 5 days):")
    print(port_returns.head())
    print("Covariance matrix:")
    print(cov)