import sys
import os
import numpy as np

#Add paths
sys.path.insert(0, 'data')
sys.path.insert(0, 'src/risk')

from data_loader import download_data, clean_data, calculate_daily_returns
from portfolio import build_portfolio, covariance_matrix
from historical_var import histo_var, histo_cvar
from parametric_var import par_var, par_cvar
from monte_carlo_var import mc_var, mc_cvar
from risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown

#Load data once
tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
prices = download_data(tickers)
prices = clean_data(prices)
returns = calculate_daily_returns(prices)
port_returns = build_portfolio(returns, weights)

# Historical VaR tests
def test_historical_var_negative():
    assert histo_var(port_returns) < 0

def test_historical_cvar_worse_than_var():
    assert histo_cvar(port_returns) < histo_var(port_returns)

# Parametric VaR tests
def test_parametric_var_negative():
    assert par_var(port_returns) < 0

def test_parametric_cvar_worse_than_var():
    assert par_cvar(port_returns) < par_var(port_returns)

# Monte Carlo VaR tests
def test_mc_var_negative():
    assert mc_var(returns, weights) < 0

def test_mc_var_close_to_parametric():
    m = mc_var(returns, weights)
    p = par_var(port_returns)
    assert abs(m - p) / abs(p) < 0.3

# Risk metrics tests
def test_sharpe_is_finite():
    s = sharpe_ratio(port_returns)
    assert np.isfinite(s)

def test_sortino_is_finite():
    s = sortino_ratio(port_returns)
    assert np.isfinite(s)

def test_max_drawdown_negative():
    assert max_drawdown(port_returns) < 0

def test_max_drawdown_above_minus_one():
    assert max_drawdown(port_returns) > -1

# Portfolio tests
def test_weights_sum_to_one():
    assert abs(weights.sum() - 1.0) < 0.0001

def test_covariance_matrix_symmetric():
    cov = covariance_matrix(returns)
    assert np.allclose(cov, cov.T)

def test_portfolio_returns_length():
    assert len(port_returns) == len(returns)