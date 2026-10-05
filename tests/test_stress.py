import sys
import numpy as np

sys.path.insert(0, 'data')
sys.path.insert(0, 'src/stress_testing')

from data_loader import download_data, clean_data, calculate_daily_returns
from portfolio import build_portfolio
from stress_test import correlation_shock, volatility_shock, stressed_var
from var_backtest import backtest_var, kupiec_test, christoffersen_test

tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
prices = download_data(tickers)
prices = clean_data(prices)
returns = calculate_daily_returns(prices)
port_returns = build_portfolio(returns, weights)
cov_matrix = returns.cov().values

#Stress test tests
def test_correlation_shock_symmetric():
    shocked = correlation_shock(cov_matrix)
    assert np.allclose(shocked, shocked.T)

def test_correlation_shock_diagonal_unchanged():
    shocked = correlation_shock(cov_matrix)
    assert np.allclose(np.diag(shocked), np.diag(cov_matrix), rtol=0.01)

def test_volatility_shock_scales():
    shocked = volatility_shock(cov_matrix, factor=3.0)
    assert np.allclose(shocked, cov_matrix * 3.0)

def test_stressed_var_worse_than_normal():
    normal = stressed_var(returns, weights, cov_matrix)
    shocked = correlation_shock(cov_matrix, shock_corr=0.9)
    stressed = stressed_var(returns, weights, shocked)
    assert stressed < normal

def test_vol_shock_var_worse_than_normal():
    normal = stressed_var(returns, weights, cov_matrix)
    vol_shocked = volatility_shock(cov_matrix, factor=3.0)
    stressed = stressed_var(returns, weights, vol_shocked)
    assert stressed < normal

#Backtest tests
def test_backtest_breach_count_reasonable():
    breaches, total, _ = backtest_var(port_returns)
    breach_rate = breaches / total
    assert 0.01 < breach_rate < 0.15

def test_backtest_total_days():
    _, total, _ = backtest_var(port_returns)
    assert total > 100

def test_kupiec_accepts_expected_breaches():
    _, passed = kupiec_test(12, 246)
    assert passed == True

def test_kupiec_rejects_too_many_breaches():
    _, passed = kupiec_test(22, 246)
    assert passed == False

def test_kupiec_zero_breaches():
    _, passed = kupiec_test(0, 246)
    assert passed == False

def test_christoffersen_rejects_cluster_in_middle():
    breach_dates = [352 + k for k in range(12)]
    lr_ind, ind_passed, _, _, _ = christoffersen_test(breach_dates, 246)
    assert ind_passed == False

def test_christoffersen_rejects_cluster_at_end():
    breach_dates = [486 + k for k in range(12)]
    lr_ind, ind_passed, _, _, _ = christoffersen_test(breach_dates, 246)
    assert ind_passed == False