import numpy as np

def mc_var(returns , weights , no_sim = 10000 , confidence = 0.95):
    #Calculate Monte Carlo VaR using correlated simulations.
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    L = np.linalg.cholesky(cov_matrix)

    portfolio_returns  = []

    for i in range(no_sim):
        Z = np.random.standard_normal(len(weights))
        correlated_returns = mean_returns + L @ Z
        port_return = np.dot(weights,correlated_returns)
        portfolio_returns.append(port_return)

    portfolio_returns = np.array(portfolio_returns)
    var = np.percentile(portfolio_returns,(1-confidence)*100)
    return var

def mc_cvar(returns, weights, n_simulations=10000, confidence=0.95):
    #Calculate Monte Carlo CVaR.
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    L = np.linalg.cholesky(cov_matrix)
    
    portfolio_returns = []
    for i in range(n_simulations):
        Z = np.random.standard_normal(len(weights))
        correlated_returns = mean_returns + L @ Z
        port_return = np.dot(weights, correlated_returns)
        portfolio_returns.append(port_return)
    
    portfolio_returns = np.array(portfolio_returns)
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return cvar

if __name__ == "__main__":
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    sys.path.insert(0, data_dir)
    
    from data_loader import download_data, clean_data, calculate_daily_returns
    from portfolio import build_portfolio
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    
    var_95 = mc_var(returns, weights)
    cvar_95 = mc_cvar(returns, weights)
    print(f"95% Monte Carlo VaR: {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"95% Monte Carlo CVaR: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"On a 10L portfolio, worst daily loss (95%): Rs {abs(var_95)*1000000:.0f}")
    print(f"Average loss on bad days: Rs {abs(cvar_95)*1000000:.0f}")