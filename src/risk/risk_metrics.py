import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.06):
    #Calculate annualized Sharpe Ratio.
    mean = returns.mean() * 252
    std = returns.std() * (252**(0.5))
    sharpe = (mean - risk_free_rate) / std
    return sharpe
    
def sortino_ratio(returns, risk_free_rate=0.06):
    #Calculate annualized Sortino Ratio.#
    mean = returns.mean() * 252
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * (252**0.5)
    sortino = (mean - risk_free_rate) / downside_std
    return sortino
    
def max_drawdown(returns):
    #Calculate Maximum Drawdown.
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    return max_dd
    

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
    port_returns = build_portfolio(returns, weights)

    print(f"Sharpe Ratio: {sharpe_ratio(port_returns):.4f}")
    print(f"Sortino Ratio: {sortino_ratio(port_returns):.4f}")
    print(f"Max Drawdown: {max_drawdown(port_returns)*100:.2f}%")