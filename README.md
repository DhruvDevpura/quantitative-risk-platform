# Quantitative Risk Analytics Platform

**Live Demo:** [quantriskplatform.streamlit.app](https://quantriskplatform.streamlit.app)

A risk management platform built from scratch in Python. Covers derivatives pricing, VaR modeling, portfolio optimization, and stress testing — applied on a real portfolio of Indian stocks using live market data from Yahoo Finance.

Built to understand how risk is actually measured and managed at firms like banks and asset managers — not just the theory but the full implementation.

## Live Dashboard
```bash
pip install numpy scipy pandas matplotlib yfinance streamlit
streamlit run dashboard.py
```

The dashboard connects all 4 phases — select any NSE tickers, adjust confidence level, and explore risk metrics, optimization methods, and stress scenarios interactively.

## Key Results

*Portfolio: 5 NSE large-caps (30/25/20/15/10). Rolling 2-year window, as of 2026-09-04. Prices dividend-adjusted.*

| Metric | Normal | Student-t (ν≈7.6) |
|--------|--------|-------------------|
| Monte Carlo VaR (95%) | -1.57% | -1.54% |
| Monte Carlo VaR (99%) | -2.20% | -2.39% |
| Monte Carlo CVaR (95%) | -1.96% | -2.07% |
| Monte Carlo CVaR (99%) | -2.51% | -2.96% |

| Metric | Value |
|--------|-------|
| Historical VaR (95%) | -1.56% |
| Historical CVaR (95%)| -2.07% |
| Parametric VaR (95%) | -1.57% |
| COVID March 2020 Replay | -20.30% |
| Historical VaR Backtest — Kupiec, 2y window | **FAIL** — 22 breaches vs 12 expected (8.9%), LR 6.42 |
| Historical VaR Backtest — Kupiec, 5y window | PASS — 55 breaches vs 49 expected (5.6%), LR 0.66 |

Degrees of freedom are estimated from the data (MLE 7.56, method-of-moments 7.50). The two distributions are variance-matched, so the difference is tail shape alone. They cross at **96.61% confidence** — below that the normal is more conservative, above it the t is. Basel measures market-risk VaR at 99% and FRTB at 97.5% Expected Shortfall, both above the crossover: on this portfolio the normal assumption understates 99% Expected Shortfall by 18% (₹25,090 vs ₹29,570 on ₹10L).
Historical CVaR at 95% (−2.07%) matches the t model (−2.07%) rather than the normal (−1.96%). That is supportive since the historical method assumes no distribution — though with roughly 25 observations in the tail, a two-year sample cannot separate them decisively.

The 2-year result uses the 30/25/20/15/10 weights; the 5-year result uses equal weights from the dashboard, so the two are not strictly comparable. What is comparable is the direction: unconditional coverage passes over five years and fails over the most recent two, meaning the breaches are not uniformly distributed in time. Kupiec cannot detect that — it counts breaches without regard to when they occur. Christoffersen's independence test is the correct instrument and is not yet implemented.
## What's Inside

**Phase 1 — Pricing Engine:** Black-Scholes pricing, Greeks (Delta, Gamma, Vega, Theta, Rho) for calls and puts, Monte Carlo simulation, bond pricing with YTM and duration.

**Phase 2 — Value at Risk:** Historical, Parametric, and Monte Carlo VaR using Cholesky decomposition for correlated simulations. CVaR, Sharpe, Sortino, and max drawdown.

**Phase 3 — Portfolio Optimization:** Markowitz efficient frontier, Black-Litterman model with investor views, and Hierarchical Risk Parity. All three compared side by side.

**Phase 4 — Stress Testing:** Correlation shocks, volatility shocks, COVID scenario replay. Rolling window VaR backtest validated with Kupiec's likelihood ratio test.

## Visualizations

![VaR Comparison](docs/var_comparison.png)
![Efficient Frontier](docs/efficient_frontier.png)
![Optimization Comparison](docs/optimization_comparison.png)
![Stress Tests](docs/stress_test_comparison.png)
![VaR Backtest](docs/var_backtest.png)
![Greeks](docs/greeks_sensitivity.png)

## Project Structure
```
├── dashboard.py
├── data/
│   ├── data_loader.py
│   └── portfolio.py
├── src/
│   ├── pricing/
│   ├── risk/
│   ├── optimization/
│   └── stress_testing/
├── tests/
└── docs/
```

## How to Run
```bash
pip install numpy scipy pandas matplotlib yfinance streamlit

# Run the dashboard
streamlit run dashboard.py

# Run individual modules
python3 src/risk/historical_var.py
python3 src/optimization/efficient_frontier.py

# Run all tests
python3 -m pytest tests/ -v
```

## Built With

Python · NumPy · SciPy · Pandas · Matplotlib · yfinance · Streamlit