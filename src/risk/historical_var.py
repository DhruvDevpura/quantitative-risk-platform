import numpy as np

def histo_var(returns,confidence=0.95):
    #Calculate Historical Var.
    var = np.percentile(returns, (1-confidence)*100)
    return var

def histo_cvar(returns,confidence=0.95):
    #Calculate Historical Conditional Var (Expected shortfall).
    var = histo_var(returns,confidence)
    cvar = returns[returns <= var].mean()
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
    
    var_95 = histo_var(port_returns)
    cvar_95 = histo_cvar(port_returns)
    print(f"95% Historical VaR: {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"95% Historical CVaR: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"On a 10L portfolio, worst daily loss (95%): Rs {abs(var_95)*1000000:.0f}")
    print(f"Average loss on bad days: Rs {abs(cvar_95)*1000000:.0f}")