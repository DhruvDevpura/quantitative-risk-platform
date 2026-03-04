# Quantitative Risk Analytics Platform

A risk management platform I built from scratch in Python. It covers derivatives pricing, VaR modeling, portfolio optimization, and stress testing — all applied on a real portfolio of 5 Indian stocks (Reliance, TCS, HDFC Bank, Infosys, ITC) using 2 years of daily market data.

The goal was to understand how risk is actually measured and managed, not just the theory but the implementation.

> **Note:** Interactive Streamlit dashboard and final integration tests are coming soon.

## Key Results

| Metric | Value |
|--------|-------|
| Historical VaR (95%) | -1.30% |
| Parametric VaR (95%) | -1.37% |
| Monte Carlo VaR (95%) | -1.39% |
| COVID March 2020 Stress Loss | -20.30% |
| VaR Backtest (Kupiec Test) | PASS |

On a ₹10L portfolio, the worst expected daily loss is around ₹13,000 under normal conditions. Under a COVID-like crash, that jumps to ₹2L+ in a single month.

## What's Inside

**Phase 1 — Pricing Engine:** Black-Scholes pricing, Greeks (Delta, Gamma, Vega, Theta, Rho) for calls and puts, Monte Carlo simulation, bond pricing with YTM and duration.

**Phase 2 — Value-at-Risk:** Three approaches — Historical, Parametric, and Monte Carlo VaR using Cholesky decomposition for correlated simulations. Also includes CVaR, Sharpe ratio, Sortino ratio, and max drawdown.

**Phase 3 — Portfolio Optimization:** Markowitz efficient frontier, Black-Litterman model with investor views, and Hierarchical Risk Parity (HRP). Compared all three to show how each handles allocation differently.

**Phase 4 — Stress Testing & Backtesting:** Correlation shocks, volatility shocks, COVID scenario replay. Rolling window VaR backtest validated with Kupiec's likelihood ratio test.

## Visualizations

![VaR Comparison](docs/var_comparison.png)
![Efficient Frontier](docs/efficient_frontier.png)
![Optimization Comparison](docs/optimization_comparison.png)
![Stress Tests](docs/stress_test_comparison.png)
![VaR Backtest](docs/var_backtest.png)
![Greeks](docs/greeks_sensitivity.png)

## Project Structure
```
├── data/
│   ├── data_loader.py
│   └── portfolio.py
├── src/
│   ├── pricing/
│   ├── risk/
│   ├── optimization/
│   └── stress_testing/
├── tests/
├── docs/
└── README.md
```

## How to Run
```bash
pip install numpy scipy pandas matplotlib yfinance

# run any module directly
python3 src/risk/historical_var.py
python3 src/optimization/efficient_frontier.py
python3 src/stress_testing/stress_test.py

# run all tests
python3 -m pytest tests/ -v
```

## Built With

Python, NumPy, SciPy, Pandas, Matplotlib, yfinance