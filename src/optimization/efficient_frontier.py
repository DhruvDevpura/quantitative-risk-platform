import numpy as np
import matplotlib.pyplot as plt

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.06):
    #Calculate annualized return, volatility, and Sharpe for given weights.
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252 , weights)))
    sharpe = (returns - risk_free_rate) / volatility
    return returns , volatility , sharpe

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.06):
    #Generate random portfolios and calculate their performance.
    results = np.zeros((num_portfolios,3))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)

        ret , vol , sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[i] = [ret,vol,sharpe]
        weights_record.append(weights)
    
    return results , weights_record

def max_sharpe_portfolio(results , weights_record):
    #Find the portfolio with maximum Sharpe ratio.
    max_idx = np.argmax(results[:,2])
    return results[max_idx], weights_record[max_idx]

def min_volatility_portfolio(results, weights_record):
    #Find the portfolio with minimum volatility.
    min_idx = np.argmin(results[:, 1])
    return results[min_idx], weights_record[min_idx]

import matplotlib.pyplot as plt

def plot_efficient_frontier(results, max_sharpe, min_vol):
    """Plot the efficient frontier with optimal portfolios marked."""
    plt.figure(figsize=(12, 6))
    plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe[1], max_sharpe[0], c='red', marker='*', s=300, label='Max Sharpe')
    plt.scatter(min_vol[1], min_vol[0], c='blue', marker='*', s=300, label='Min Volatility')
    plt.title('Efficient Frontier')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.legend()
    #plt.savefig('docs/efficient_frontier.png')
    #plt.show()
    return plt.gcf()

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
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    results, weights_record = random_portfolios(10000, mean_returns, cov_matrix)
    max_s, max_w = max_sharpe_portfolio(results, weights_record)
    min_v, min_w = min_volatility_portfolio(results, weights_record)
    
    print("=== Max Sharpe Portfolio ===")
    print(f"Return: {max_s[0]*100:.2f}%  Vol: {max_s[1]*100:.2f}%  Sharpe: {max_s[2]:.4f}")
    for t, w in zip(tickers, max_w):
        print(f"  {t}: {w*100:.1f}%")
    
    print("\n=== Min Volatility Portfolio ===")
    print(f"Return: {min_v[0]*100:.2f}%  Vol: {min_v[1]*100:.2f}%  Sharpe: {min_v[2]:.4f}")
    for t, w in zip(tickers, min_w):
        print(f"  {t}: {w*100:.1f}%")
    
    plot_efficient_frontier(results, max_s, min_v)