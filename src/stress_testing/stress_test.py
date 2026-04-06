import numpy as np

def historical_scenario(returns, weights, scenario_returns):
    #Replay a historical crisis on current portfolio.
    portfolio_scenario = scenario_returns.dot(weights)
    cumulative_loss = (1 + portfolio_scenario).prod() - 1
    worst_day = portfolio_scenario.min()
    return cumulative_loss, worst_day

def correlation_shock(cov_matrix, shock_corr=0.9):
    #Replace correlations with stressed value, keep volatilities same.
    vols = np.sqrt(np.diag(cov_matrix))
    n = len(vols)
    stressed_corr = np.full((n, n), shock_corr)
    np.fill_diagonal(stressed_corr, 1.0)
    stressed_cov = np.outer(vols, vols) * stressed_corr
    return stressed_cov

def volatility_shock(cov_matrix, factor=3.0):
    #Multiply covariance matrix by stress factor.
    return cov_matrix * factor

def stressed_var(returns, weights, stressed_cov, n_simulations=10000, confidence=0.95):
    #Calculate Monte Carlo VaR using stressed covariance matrix.
    mean_returns = returns.mean().values
    L = np.linalg.cholesky(stressed_cov)
    
    portfolio_returns = []
    for i in range(n_simulations):
        Z = np.random.standard_normal(len(weights))
        correlated = mean_returns + L @ Z
        port_return = np.dot(weights, correlated)
        portfolio_returns.append(port_return)
    
    portfolio_returns = np.array(portfolio_returns)
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    return var

import matplotlib.pyplot as plt

def plot_stress_comparison(normal_var, corr_var, vol_var):
    """Bar chart comparing VaR under different stress scenarios."""
    plt.figure(figsize=(10, 6))
    scenarios = ['Normal', 'Correlation\nShock (0.9)', 'Volatility\nShock (3x)']
    values = [abs(normal_var)*100, abs(corr_var)*100, abs(vol_var)*100]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    plt.bar(scenarios, values, color=colors)
    plt.ylabel('VaR (%)')
    plt.title('VaR Under Stress Scenarios (95% Confidence)')
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontweight='bold')
    #plt.savefig('docs/stress_test_comparison.png')
    #plt.show()
    return plt.gcf()

if __name__ == "__main__":
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    sys.path.insert(0, data_dir)
    
    from data_loader import download_data, clean_data, calculate_daily_returns
    import yfinance as yf
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    cov_matrix = returns.cov().values
    
    #Normal VaR
    normal_var = stressed_var(returns, weights, cov_matrix)
    
    #Correlation shock
    shocked_cov = correlation_shock(cov_matrix, shock_corr=0.9)
    corr_var = stressed_var(returns, weights, shocked_cov)
    
    #Volatility shock
    vol_cov = volatility_shock(cov_matrix, factor=3.0)
    vol_var = stressed_var(returns, weights, vol_cov)
    
    #Historical scenario - COVID crash March 2020
    covid_prices = yf.download(tickers, start='2020-03-01', end='2020-03-31', progress=False)['Close']
    covid_returns = covid_prices.pct_change().dropna()
    cum_loss, worst_day = historical_scenario(covid_returns, weights, covid_returns)
    
    print("=== Stress Test Results ===")
    print(f"\nNormal VaR (95%):              {normal_var*100:.2f}%")
    print(f"Correlation Shock VaR (0.9):   {corr_var*100:.2f}%")
    print(f"Volatility Shock VaR (3x):     {vol_var*100:.2f}%")
    print(f"\nCOVID March 2020 Replay:")
    print(f"  Cumulative Loss:  {cum_loss*100:.2f}%")
    print(f"  Worst Single Day: {worst_day*100:.2f}%")
    print(f"  On 10L portfolio: Rs {abs(cum_loss)*1000000:.0f} lost")
    plot_stress_comparison(normal_var, corr_var, vol_var)