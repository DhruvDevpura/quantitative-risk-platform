import sys
import numpy as np

sys.path.insert(0, 'data')
sys.path.insert(0, 'src/optimization')

from data_loader import download_data, clean_data, calculate_daily_returns
from efficient_frontier import portfolio_performance, random_portfolios, max_sharpe_portfolio, min_volatility_portfolio
from black_litterman import market_implied_returns, black_litterman, bl_optimal_weights
from hrp import hrp

tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
prices = download_data(tickers)
prices = clean_data(prices)
returns = calculate_daily_returns(prices)

# Markowitz tests
def test_random_portfolios_shape():
    results, weights = random_portfolios(100, returns.mean(), returns.cov())
    assert results.shape == (100, 3)
    assert len(weights) == 100

def test_weights_sum_to_one():
    _, weights = random_portfolios(100, returns.mean(), returns.cov())
    for w in weights:
        assert abs(np.sum(w) - 1.0) < 0.0001

def test_max_sharpe_beats_min_vol():
    results, weights = random_portfolios(5000, returns.mean(), returns.cov())
    max_s, _ = max_sharpe_portfolio(results, weights)
    min_v, _ = min_volatility_portfolio(results, weights)
    assert max_s[2] >= min_v[2]

# Black-Litterman tests
def test_bl_returns_length():
    cov = returns.cov().values * 252
    market_w = np.array([0.35, 0.20, 0.25, 0.12, 0.08])
    P = np.array([[1, -1, 0, 0, 0], [0, 0, 0, 0, 1]])
    Q = np.array([0.02, 0.03])
    combined = black_litterman(cov, market_w, P, Q)
    assert len(combined) == 5

def test_bl_weights_sum_to_one():
    cov = returns.cov().values * 252
    market_w = np.array([0.35, 0.20, 0.25, 0.12, 0.08])
    P = np.array([[1, -1, 0, 0, 0], [0, 0, 0, 0, 1]])
    Q = np.array([0.02, 0.03])
    combined = black_litterman(cov, market_w, P, Q)
    weights = bl_optimal_weights(combined, cov)
    assert abs(np.sum(weights) - 1.0) < 0.0001

def test_bl_weights_positive():
    cov = returns.cov().values * 252
    market_w = np.array([0.35, 0.20, 0.25, 0.12, 0.08])
    P = np.array([[1, -1, 0, 0, 0], [0, 0, 0, 0, 1]])
    Q = np.array([0.02, 0.03])
    combined = black_litterman(cov, market_w, P, Q)
    weights = bl_optimal_weights(combined, cov)
    assert all(w >= 0 for w in weights)

# HRP tests
def test_hrp_weights_sum_to_one():
    weights = hrp(returns)
    assert abs(weights.sum() - 1.0) < 0.0001

def test_hrp_weights_positive():
    weights = hrp(returns)
    assert all(w > 0 for w in weights)

def test_hrp_weights_length():
    weights = hrp(returns)
    assert len(weights) == 5