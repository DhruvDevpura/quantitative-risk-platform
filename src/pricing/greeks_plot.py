import numpy as np
import matplotlib.pyplot as plt
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from black_scholes import delta_call, gamma, vega

def plot_greeks():
    """Plot how Greeks change as stock price moves."""
    S_range = np.linspace(50, 150, 100)
    K, T, r, sigma = 100, 1, 0.05, 0.2
    
    deltas = [delta_call(S, K, T, r, sigma) for S in S_range]
    gammas = [gamma(S, K, T, r, sigma) for S in S_range]
    vegas = [vega(S, K, T, r, sigma) for S in S_range]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].plot(S_range, deltas, color='#e74c3c', linewidth=2)
    axes[0].set_title('Delta vs Stock Price')
    axes[0].set_xlabel('Stock Price')
    axes[0].set_ylabel('Delta')
    axes[0].axvline(K, color='gray', linestyle='--', alpha=0.5, label='Strike')
    axes[0].legend()
    
    axes[1].plot(S_range, gammas, color='#3498db', linewidth=2)
    axes[1].set_title('Gamma vs Stock Price')
    axes[1].set_xlabel('Stock Price')
    axes[1].set_ylabel('Gamma')
    axes[1].axvline(K, color='gray', linestyle='--', alpha=0.5, label='Strike')
    axes[1].legend()
    
    axes[2].plot(S_range, vegas, color='#2ecc71', linewidth=2)
    axes[2].set_title('Vega vs Stock Price')
    axes[2].set_xlabel('Stock Price')
    axes[2].set_ylabel('Vega')
    axes[2].axvline(K, color='gray', linestyle='--', alpha=0.5, label='Strike')
    axes[2].legend()
    
    plt.suptitle('Option Greeks Sensitivity (K=100, T=1yr, σ=20%)', fontsize=13)
    plt.tight_layout()
    plt.savefig('docs/greeks_sensitivity.png')
    plt.show()

if __name__ == "__main__":
    plot_greeks()