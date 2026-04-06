import numpy as np
import matplotlib.pyplot as plt

def plot_weight_comparison(tickers, markowitz_w, bl_w, hrp_w):
    #Plot bar chart comparing weights across all 3 methods.
    x = np.arange(len(tickers))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, markowitz_w * 100, width, label='Markowitz', color='#e74c3c')
    plt.bar(x, bl_w * 100, width, label='Black-Litterman', color='#3498db')
    plt.bar(x + width, hrp_w * 100, width, label='HRP', color='#2ecc71')
    
    plt.xlabel('Stocks')
    plt.ylabel('Weight (%)')
    plt.title('Portfolio Optimization: Weight Comparison')
    plt.xticks(x, [t.replace('.NS', '') for t in tickers])
    plt.legend()
    #plt.savefig('docs/optimization_comparison.png')
    #plt.show()
    return plt.gcf()

if __name__ == "__main__":
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    sys.path.insert(0, data_dir)
    sys.path.insert(0, current_dir)
    
    from data_loader import download_data, clean_data, calculate_daily_returns
    from efficient_frontier import random_portfolios, max_sharpe_portfolio
    from black_litterman import black_litterman, bl_optimal_weights, market_implied_returns
    from hrp import hrp
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    #Markowitz
    results, weights_record = random_portfolios(10000, mean_returns, cov_matrix)
    _, markowitz_w = max_sharpe_portfolio(results, weights_record)
    markowitz_w = np.array(markowitz_w)
    
    #Black-Litterman
    market_weights = np.array([0.35, 0.20, 0.25, 0.12, 0.08])
    P = np.array([[1, -1, 0, 0, 0], [0, 0, 0, 0, 1]])
    Q = np.array([0.02, 0.03])
    cov_annual = cov_matrix.values * 252
    combined = black_litterman(cov_annual, market_weights, P, Q)
    bl_w = bl_optimal_weights(combined, cov_annual)
    
    #HRP
    hrp_weights = hrp(returns)
    hrp_w = hrp_weights.values
    
    print("=== Weight Comparison ===")
    print(f"{'Stock':<15} {'Markowitz':>10} {'BL':>10} {'HRP':>10}")
    for i, t in enumerate(tickers):
        print(f"{t:<15} {markowitz_w[i]*100:>9.1f}% {bl_w[i]*100:>9.1f}% {hrp_w[i]*100:>9.1f}%")
    
    plot_weight_comparison(tickers, markowitz_w, bl_w, hrp_w)