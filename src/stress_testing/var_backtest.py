import numpy as np
from scipy.stats import chi2

def backtest_var(returns, confidence=0.95, window=252):
    #Rolling window VaR backtest using historical method.
    if(len(returns)<window):
        raise ValueError(f"Backtest needs more than {window} obs for the rolling window, got {len(returns)}")
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

def christoffersen_test(breach_dates, total,window=252, confidence=0.95):
    #Christoffesen independence and conditional coverage test
    hits = np.zeros(total,dtype=int)
    for b in breach_dates:
        hits[b - window] = 1
    n00 = n01 = n10 = n11 = 0
    for i in range(1,total):
        prev,curr = hits[i-1],hits[i]
        if prev==0 and curr==0:
            n00 += 1
        elif prev==0 and curr==1:
            n01 += 1
        elif prev==1 and curr==0:
            n10 += 1
        else:
            n11 += 1
    
    pi01 = n01/(n00+n01) if (n00+n01) > 0 else 0.0
    pi11 = n11/(n10+n11) if (n10+n11) > 0 else 0.0
    pi   = (n01+n11)/(n00+n01+n10+n11)

    log_l_null = np.log((1-pi)**(n00+n10) * pi**(n01+n11))
    log_l_alt = np.log((1-pi01)**n00 * pi01**n01 * (1-pi11)**n01 * pi11**n11)
    lr_ind = -2*(log_l_null - log_l_alt)
    ind_passed = lr_ind < chi2.ppf(confidence,df=1)

    breaches = int(hits.sum())
    lr_uc,_ = kupiec_test(breaches,total,confidence)
    lr_cc = lr_uc + lr_ind
    cc_passed = lr_cc < chi2.ppf(confidence,df=1)

    transitions = {"n00": n00, "n01": n01, "n10": n10, "n11": n11,"pi01": pi01, "pi11": pi11, "pi": pi}
    
    return lr_ind,ind_passed,lr_cc,cc_passed,transitions
    


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

    lr_ind, ind_passed, lr_cc, cc_passed, tr = christoffersen_test(breach_dates, total)
    print(f"\nTransitions: n00={tr['n00']} n01={tr['n01']} n10={tr['n10']} n11={tr['n11']}")
    print(f"pi01 = {tr['pi01']:.4f}   pi11 = {tr['pi11']:.4f}")
    print(f"Christoffersen LR_ind: {lr_ind:.4f}  ({'PASS' if ind_passed else 'FAIL'})")
    print(f"Conditional coverage LR_cc: {lr_cc:.4f}  ({'PASS' if cc_passed else 'FAIL'})")