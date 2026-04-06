import numpy as np
from scipy.stats import chi2

def backtest_var(returns, confidence=0.95, window=252):
    #Rolling window VaR backtest using historical method.
    breaches = 0
    total = 0
    breach_dates = []
    
    for i in range(window, len(returns)):
        historical_window = returns[i-window:i]
        var = np.percentile(historical_window, (1 - confidence) * 100)
        actual = returns.iloc[i]
        
        if actual < var:
            breaches += 1
            breach_dates.append(i)
        total += 1
    
    return breaches, total, breach_dates

def kupiec_test(breaches, total, confidence=0.95):
    #Kupiec's POF test for VaR model accuracy.
    p = 1 - confidence  #expected breach rate
    p_hat = breaches / total  #actual breach rate
    
    if p_hat == 0 or p_hat == 1:
        return 0, True
    
    lr = -2 * (np.log((1-p)**(total-breaches) * p**breaches) - 
               np.log((1-p_hat)**(total-breaches) * p_hat**breaches))
    
    critical = chi2.ppf(0.95, df=1)
    passed = lr < critical
    
    return lr, passed

import matplotlib.pyplot as plt

def plot_backtest(returns, breach_dates, window=252):
    """Plot portfolio returns with VaR breaches highlighted."""
    plt.figure(figsize=(14, 6))
    tested_returns = returns.iloc[window:]
    plt.plot(tested_returns.values, alpha=0.7, color='steelblue', linewidth=0.8)
    
    breach_idx = [b - window for b in breach_dates]
    plt.scatter(breach_idx, tested_returns.iloc[breach_idx].values, 
                color='red', s=30, zorder=5, label=f'VaR Breaches ({len(breach_dates)})')
    
    plt.title('VaR Backtest: Portfolio Returns with Breach Points')
    plt.xlabel('Trading Days')
    plt.ylabel('Daily Return')
    plt.legend()
    #plt.savefig('docs/var_backtest.png')
    #plt.show()
    return plt.gcf()

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
    
    breaches, total, breach_dates = backtest_var(port_returns)
    lr, passed = kupiec_test(breaches, total)
    expected = int(total * 0.05)
    
    print("=== VaR Backtest Results ===")
    print(f"Total days tested: {total}")
    print(f"Expected breaches (5%): {expected}")
    print(f"Actual breaches: {breaches} ({breaches/total*100:.1f}%)")
    print(f"Kupiec LR statistic: {lr:.4f}")
    print(f"Model status: {'PASS' if passed else 'FAIL'}")
    plot_backtest(port_returns, breach_dates)