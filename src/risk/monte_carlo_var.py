import numpy as np

def estimate_nu(port_returns):
    #Fit t-degrees of dreedom. Return (mle,moments)
    from scipy.stats import t as student_t,kurtosis
    nu_mle = student_t.fit(port_returns)[0] 
    k = kurtosis(port_returns)
    nu_mom = 4 + 6/k if k>0 else np.nan
    return nu_mle,nu_mom

def mc_var(returns , weights , n_sim = 10000 , confidence = 0.95 , dist="normal",nu=None):
    #Calculate Monte Carlo VaR using correlated simulations.
    port = simulate_returns(returns,weights,n_sim = n_sim,dist=dist,nu=nu)
    return np.percentile(port,(1-confidence)*100)

def mc_cvar(returns, weights, n_sim=10000, confidence=0.95, dist="normal",nu=None):
    #Calculate Monte Carlo CVaR.
    port = simulate_returns(returns,weights,n_sim=n_sim,dist=dist,nu=nu)
    var = np.percentile(port,(1-confidence)*100)
    return port[port <= var].mean()


def simulate_returns(returns,weights,n_sim=10000,dist="normal",nu = None):
    #Simulate n portfolio returns.
    #dist = "normal" - multivariate normal , cov = sigma
    #dist = "t" - multivariate t with nu df, cov = sigma 
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    if dist=="normal":
        L = np.linalg.cholesky(cov_matrix)
        Z = np.random.standard_normal((n_sim,len(weights)))

        asset_draws = mean_returns + Z@L.T
        portfolio_returns = asset_draws@weights
        return portfolio_returns
    
    elif dist=="t":
        if nu is None or nu<=2:
            raise ValueError(f"t simulation needs nu>2 (variance undefined) got {nu}")
        
        L = np.linalg.cholesky(cov_matrix) * np.sqrt((nu-2)/nu)
        Z = np.random.standard_normal((n_sim,len(weights)))
        W = np.random.chisquare(nu,size=(n_sim,1))

        asset_draws = mean_returns + (Z@L.T) * np.sqrt(nu/W)
        portfolio_returns = asset_draws@weights
        return portfolio_returns
    
    else:
        raise ValueError(f"Unknown dist: {dist}")



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
    port_returns = build_portfolio(returns,weights)
    nu_mle,nu_mom = estimate_nu(port_returns)
    nu = nu_mle

    print(f"nu: MLE {nu_mle:.2f} | moments {nu_mom:.2f} | using {nu:.2f}\n")

    var_95 = mc_var(returns, weights)
    var_99 = mc_var(returns, weights,confidence=0.99)
    var_95_t = mc_var(returns,weights,dist="t",nu=nu)
    var_99_t = mc_var(returns,weights,confidence=0.99,dist="t",nu=nu)
    cvar_95 = mc_cvar(returns,weights)
    cvar_95_t = mc_cvar(returns,weights,confidence=0.95,dist="t",nu=nu)

    print(f"95% Monte Carlo VaR normal : {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"99% Monte Carlo VaR normal : {var_99:.4f} ({var_99*100:.2f}%)")
    print(f"95% Monte Carlo VaR t : {var_95_t:.4f} ({var_95_t*100:.2f}%)")
    print(f"99% Monte Carlo VaR t : {var_99_t:.4f} ({var_99_t*100:.2f}%)")
    print(f"95% Monte Carlo CVaR normal: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"95% Monte Carlo CVaR t: {cvar_95_t:.4f} ({cvar_95_t*100:.2f}%)")

    print(f"On a 10L portfolio, worst daily loss (95% and normal): Rs {abs(var_95)*1000000:.0f}")
    print(f"On a 10L portfolio, worst daily loss (95% and t): Rs {abs(var_95_t)*1000000:.0f}")

    print(f"Average loss on bad days(normal): Rs {abs(cvar_95)*1000000:.0f}")
    print(f"Average loss on bad days(t): Rs {abs(cvar_95_t)*1000000:.0f}")