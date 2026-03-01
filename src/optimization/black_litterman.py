import numpy as np

def market_implied_returns(cov_matrix, market_weights, delta = 2.5):
    #Calculate equilibrium returns implied by market cap weights.
    pi = delta * cov_matrix.dot(market_weights)
    return pi

def black_litterman(cov_matrix, market_weights, P, Q, tau = 0.5 , delta = 2.5):
    #Calculate BlackLitterman Implied returns
    pi = market_implied_returns(cov_matrix, market_weights, delta)

    tau_cov = tau * cov_matrix
    omega = tau * P.dot(cov_matrix).dot(P.T)

    precision_market = np.linalg.inv(tau_cov)
    precision_views = P.T.dot(np.linalg.inv(omega)).dot(P)

    combined_precision = precision_market + precision_views
    combined_returns = np.linalg.inv(combined_precision).dot(precision_market.dot(pi) + P.T.dot(np.linalg.inv(omega)).dot(Q))

    return combined_returns

def bl_optimal_weights(combined_returns, cov_matrix, delta=2.5):
    #Calculate optimal weights from BL returns.
    weights = np.linalg.inv(delta * cov_matrix).dot(combined_returns)
    weights = np.maximum(weights, 0)  # no short selling
    weights = weights / np.sum(weights)
    return weights

if __name__ == "__main__":
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    sys.path.insert(0, data_dir)
    
    from data_loader import download_data, clean_data, calculate_daily_returns
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    cov_matrix = returns.cov().values * 252

    #Market cap weights (approximate, based on real market caps)
    market_weights = np.array([0.35, 0.20, 0.25, 0.12, 0.08])
    
    #Your views:
    #View 1: Reliance outperforms TCS by 2%
    #View 2: ITC returns 3%
    P = np.array([
        [1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    Q = np.array([0.02, 0.03])
    
    combined_returns = black_litterman(cov_matrix, market_weights, P, Q)
    weights = bl_optimal_weights(combined_returns, cov_matrix)
    
    print("=== Black-Litterman Results ===")
    print("\nMarket Implied Returns:")
    pi = market_implied_returns(cov_matrix, market_weights)
    for t, r in zip(tickers, pi):
        print(f"  {t}: {r*100:.2f}%")
    
    print("\nBL Combined Returns:")
    for t, r in zip(tickers, combined_returns):
        print(f"  {t}: {r*100:.2f}%")
    
    print("\nOptimal Weights:")
    for t, w in zip(tickers, weights):
        print(f"  {t}: {w*100:.1f}%")