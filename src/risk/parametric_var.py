import numpy as np

def par_var(returns, confidence=0.95):
    #Calculate Parametric (Variance-Covariance) VaR assuming normal distribution.
    from scipy.stats import norm
    mean = returns.mean()
    std = returns.std()
    z_score = norm.ppf(1 - confidence)
    var = mean + z_score * std
    return var

def par_cvar(returns, confidence=0.95):
    #Calculate Parametric CVaR assuming normal distribution.
    from scipy.stats import norm
    mean = returns.mean()
    std = returns.std()
    z_score = norm.ppf(1 - confidence)
    var = mean + z_score * std
    pdf_z = norm.pdf(z_score)
    cvar = mean - std * (pdf_z / (1 - confidence))
    return cvar

if __name__ == "__main__":
    import sys
    import os
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
    port_returns = build_portfolio(returns, weights)
    
    var_95 = par_var(port_returns)
    cvar_95 =par_cvar(port_returns)
    print(f"95% Parametric VaR: {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"95% Parametric CVaR: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"On a 10L portfolio, worst daily loss (95%): Rs {abs(var_95)*1000000:.0f}")
    print(f"Average loss on bad days: Rs {abs(cvar_95)*1000000:.0f}")