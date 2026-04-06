import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import download_data, clean_data, calculate_daily_returns
from src.data.portfolio import build_portfolio
from src.risk.historical_var import histo_var, histo_cvar
import numpy as np

tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
prices = download_data(tickers)
prices = clean_data(prices)
returns = calculate_daily_returns(prices)
port_returns = build_portfolio(returns, weights)

var_95 = histo_var(port_returns)
cvar_95 = histo_cvar(port_returns)
print(f"95% Historical VaR: {var_95:.4f} ({var_95*100:.2f}%)")
print(f"95% Historical CVaR: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
print(f"On a 10L portfolio, worst daily loss (95%): Rs {abs(var_95)*1000000:.0f}")
print(f"Average loss on bad days: Rs {abs(cvar_95)*1000000:.0f}")
