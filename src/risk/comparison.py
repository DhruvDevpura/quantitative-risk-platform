import numpy as np
import matplotlib.pyplot as plt

def plot_var_comparison(port_returns, hist_var, param_var, mc_var):
    """Plot histogram of portfolio returns with VaR lines."""
    plt.figure(figsize=(12, 6))
    plt.hist(port_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(hist_var, color='red', linestyle='--', linewidth=2, label=f'Historical VaR: {hist_var*100:.2f}%')
    plt.axvline(param_var, color='green', linestyle='--', linewidth=2, label=f'Parametric VaR: {param_var*100:.2f}%')
    plt.axvline(mc_var, color='orange', linestyle='--', linewidth=2, label=f'Monte Carlo VaR: {mc_var*100:.2f}%')
    plt.title('Portfolio Returns Distribution with VaR Estimates')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('var_comparison.png')
    plt.show()

if __name__ == "__main__":
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    sys.path.insert(0, data_dir)
    sys.path.insert(0, current_dir)
    
    from data_loader import download_data, clean_data, calculate_daily_returns
    from portfolio import build_portfolio
    from historical_var import histo_var, histo_cvar
    from parametric_var import par_var, par_cvar
    from monte_carlo_var import mc_var, mc_cvar
    from risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    prices = download_data(tickers)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    port_returns = build_portfolio(returns, weights)
    
    h_var = histo_var(port_returns)
    p_var = par_var(port_returns)
    m_var = mc_var(returns, weights)
    
    print("=== VaR Comparison (95% Confidence) ===")
    print(f"Historical VaR:   {h_var*100:.2f}%")
    print(f"Parametric VaR:   {p_var*100:.2f}%")
    print(f"Monte Carlo VaR:  {m_var*100:.2f}%")
    print(f"\nSharpe Ratio:  {sharpe_ratio(port_returns):.4f}")
    print(f"Sortino Ratio: {sortino_ratio(port_returns):.4f}")
    print(f"Max Drawdown:  {max_drawdown(port_returns)*100:.2f}%")
    
    plot_var_comparison(port_returns, h_var, p_var, m_var)