import sys
sys.path.insert(0, '.')

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from data.data_loader import download_data, clean_data, calculate_daily_returns
from data.portfolio import build_portfolio, covariance_matrix

from src.risk.historical_var import histo_var, histo_cvar
from src.risk.parametric_var import par_var, par_cvar
from src.risk.monte_carlo_var import mc_var, mc_cvar
from src.risk.risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown
from src.risk.comparison import plot_var_comparison

from src.optimization.efficient_frontier import random_portfolios, max_sharpe_portfolio, min_volatility_portfolio, plot_efficient_frontier
from src.optimization.black_litterman import black_litterman, bl_optimal_weights
from src.optimization.hrp import hrp
from src.optimization.optimization_comparison import plot_weight_comparison

from src.stress_testing.stress_test import historical_scenario, correlation_shock, volatility_shock, stressed_var, plot_stress_comparison
from src.stress_testing.var_backtest import backtest_var, kupiec_test, plot_backtest

from src.pricing.black_scholes import bs_call_price, bs_put_price, delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put
from src.pricing.black_scholes import bs_call_price, bs_put_price
from src.pricing.bond_pricing import bond_price, yield_to_maturity, duration
from src.pricing.monte_carlo import mc_call_price, mc_put_price

st.set_page_config(layout="wide", page_title="Quant Risk Platform")

@st.cache_data
def load_data(tickers,period):
    prices = download_data(list(tickers),period=period)
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    n = len(tickers)
    weights = np.ones(n) / n
    port_returns = build_portfolio(returns,weights)
    cov = covariance_matrix(returns)
    mean_returns = returns.mean()

    return {
        "returns": returns,
        "port_returns": port_returns,
        "cov": cov,
        "mean_returns": mean_returns,
        "weights": weights,
        "tickers": list(tickers)
    }

@st.cache_data
def load_crisis_data(tickers, start, end):
    prices = yf.download(list(tickers), start=start, end=end, progress=False)['Close']
    prices = clean_data(prices)
    returns = calculate_daily_returns(prices)
    return returns

st.title("Quantitative Risk Analytics Platform")
st.markdown("Built on real NSE market data · Black-Scholes · VaR · Portfolio Optimization · Stress Testing")

st.sidebar.title("Configuration")

DEFAULT_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']

ALL_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS',
    'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
    'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS',
    'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS',
    'TATAMOTORS.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS',
    'ADANIENT.NS', 'ADANIPORTS.NS',
    'ONGC.NS', 'BPCL.NS', 'NTPC.NS', 'POWERGRID.NS',
    'LT.NS', 'ULTRACEMCO.NS'
]

selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=ALL_TICKERS,
    default=DEFAULT_TICKERS
)

period = st.sidebar.selectbox(
    "Data Period",
    options=["1y", "2y", "5y"],
    index=2
)

confidence = st.sidebar.slider(
    "Confidence Level",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01
)

st.sidebar.info(f"Weights: equal across {len(selected_tickers)} tickers")

data = load_data(tuple(selected_tickers), period)

returns = data["returns"]
port_returns = data["port_returns"]
cov = data["cov"]
mean_returns = data["mean_returns"]
weights = data["weights"]
tickers = data["tickers"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Risk Analysis",
    "Portfolio Optimizer",
    "Stress Testing",
    "Options & Bonds"
])

with tab1:
    st.header("Risk Analysis")
    col1, col2 = st.columns(2)
    with col1:
        n_sim = st.slider("MC Simulations", 1000, 50000, 10000, step=1000)
    with col2:
        st.metric("Confidence Level", f"{confidence:.0%}")
        st.caption("Change in sidebar")
    
    h_var = histo_var(port_returns, confidence)
    h_cvar = histo_cvar(port_returns, confidence)
    p_var = par_var(port_returns, confidence)
    p_cvar = par_cvar(port_returns, confidence)
    m_var = mc_var(returns, weights, n_sim, confidence)
    m_cvar = mc_cvar(returns, weights, n_sim, confidence)

    fig = plot_var_comparison(port_returns, h_var, p_var, m_var)
    st.pyplot(fig)
    plt.close()

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Value at Risk")
        st.metric("Historical VaR", f"{h_var*100:.2f}%")
        st.metric("Parametric VaR", f"{p_var*100:.2f}%")
        st.metric("Monte Carlo VaR", f"{m_var*100:.2f}%")

    with col2:
        st.subheader("CVaR (Expected Shortfall)")
        st.metric("Historical CVaR", f"{h_cvar*100:.2f}%")
        st.metric("Parametric CVaR", f"{p_cvar*100:.2f}%")
        st.metric("Monte Carlo CVaR", f"{m_cvar*100:.2f}%")

    with col3:
        st.subheader("Risk Metrics")
        st.metric("Sharpe Ratio", f"{sharpe_ratio(port_returns):.4f}")
        st.metric("Sortino Ratio", f"{sortino_ratio(port_returns):.4f}")
        st.metric("Max Drawdown", f"{max_drawdown(port_returns)*100:.2f}%")

with tab2:
    st.header("Portfolio Optimizer")
    method = st.selectbox("Optimization Method",["Markowitz", "Black-Litterman", "HRP"])

    if method == "Markowitz":
        n_portfolios = st.slider("Simulated Portfolios", 1000, 10000, 5000, step=500)
        results, weights_record = random_portfolios(n_portfolios, mean_returns, cov)
        max_s, max_w = max_sharpe_portfolio(results, weights_record)
        min_v, min_w = min_volatility_portfolio(results, weights_record)
        fig = plot_efficient_frontier(results, max_s, min_v)
        st.pyplot(fig)
        plt.close()
        st.subheader("Max Sharpe Portfolio Weights")
        weight_df = pd.DataFrame({
            "Ticker": [t.replace('.NS','') for t in tickers],
            "Weight": [f"{w*100:.1f}%" for w in max_w]
        })
        st.dataframe(weight_df, use_container_width=True)

    elif method == "Black-Litterman":
        n = len(tickers)
        market_weights = np.ones(n) / n
        P = np.array([[1, -1, 0, 0, 0, *[0]*(n-5)][:n]])  # RELIANCE outperforms TCS
        Q = np.array([0.02])
        cov_annual = cov.values * 252
        combined = black_litterman(cov_annual, market_weights, P, Q)
        bl_w = bl_optimal_weights(combined, cov_annual)
        st.subheader("Black-Litterman Weights")
        st.caption(f"View: {tickers[0].replace('.NS','')} outperforms market by 2%")
        weight_df = pd.DataFrame({
            "Ticker": [t.replace('.NS','') for t in tickers],
            "Weight": [f"{w*100:.1f}%" for w in bl_w]
        })
        st.dataframe(weight_df, use_container_width=True)
        chard_data = pd.DataFrame({"Weight": bl_w},index = [t.replace('.NS','') for t in tickers])
        st.bar_chart(chard_data)

    elif method == "HRP":
        hrp_weights = hrp(returns)
        hrp_w = hrp_weights.values
        st.subheader("HRP Weights")
        weight_df = pd.DataFrame({
            "Ticker": [t.replace('.NS','') for t in tickers],
            "Weight": [f"{w*100:.1f}%" for w in hrp_w]})
        
        st.dataframe(weight_df, use_container_width=True)
        chard_data = pd.DataFrame({"Weight": hrp_w},index = [t.replace('.NS','') for t in tickers])
        st.bar_chart(chard_data)

with tab3:
    st.header("Stress Testing")

    st.subheader("Parametric Stress Tests")
    col1, col2 = st.columns(2)
    with col1:
        shock_corr = st.slider("Correlation Shock", 0.5, 0.99, 0.9, step=0.05)
    with col2:
        vol_factor = st.slider("Volatility Shock Factor", 1.0, 5.0, 3.0, step=0.5)

    normal_var = stressed_var(returns, weights, cov.values)
    corr_cov = correlation_shock(cov.values, shock_corr)
    corr_var = stressed_var(returns, weights, corr_cov)
    vol_cov = volatility_shock(cov.values, vol_factor)
    vol_var = stressed_var(returns, weights, vol_cov)

    fig = plot_stress_comparison(normal_var, corr_var, vol_var)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("Historical Scenario: COVID Crash (March 2020)")
    crisis_returns = load_crisis_data(tuple(tickers), '2020-03-01', '2020-03-31')

    if crisis_returns.empty:
        st.error("Could not load crisis data.")
    else:
        cum_loss, worst_day = historical_scenario(crisis_returns, weights, crisis_returns)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cumulative Loss", f"{cum_loss*100:.2f}%")
        with col2:
            st.metric("Worst Single Day", f"{worst_day*100:.2f}%")
        with col3:
            st.metric("Impact on ₹10L", f"₹{abs(cum_loss)*1000000:,.0f}")

    st.divider()
    st.subheader("VaR Backtest & Kupiec Test")
    breaches, total, breach_dates = backtest_var(port_returns, confidence)
    lr, passed = kupiec_test(breaches, total, confidence)
    expected = int(total * (1 - confidence))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Days Tested", total)
    with col2:
        st.metric("Expected Breaches", expected)
    with col3:
        st.metric("Actual Breaches", breaches)
    with col4:
        st.metric("Kupiec LR Stat", f"{lr:.4f}")

    if passed:
        st.success("✅ Model PASSED Kupiec Test — breach rate is statistically acceptable")
    else:
        st.error("❌ Model FAILED Kupiec Test — too many VaR breaches")

    fig = plot_backtest(port_returns, breach_dates)
    st.pyplot(fig)
    plt.close()

with tab4:
    st.header("Options & Bonds")

    option_tab, bond_tab = st.tabs(["Options", "Bonds"])

    with option_tab:
        st.subheader("Black-Scholes Option Pricer")
        col1, col2 = st.columns(2)
        with col1:
            S = st.slider("Spot Price (S)", 50, 500, 100)
            K = st.slider("Strike Price (K)", 50, 500, 100)
            T = st.slider("Time to Maturity (T, years)", 0.1, 3.0, 1.0, step=0.1)
        with col2:
            r = st.slider("Risk-Free Rate (r)", 0.01, 0.15, 0.06, step=0.01)
            sigma = st.slider("Volatility (σ)", 0.05, 0.80, 0.20, step=0.01)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Call Option")
            st.metric("BS Call Price", f"₹{bs_call_price(S, K, T, r, sigma):.4f}")
            st.metric("MC Call Price", f"₹{mc_call_price(S, K, T, r, sigma):.4f}")
        with col2:
            st.subheader("Put Option")
            st.metric("BS Put Price", f"₹{bs_put_price(S, K, T, r, sigma):.4f}")
            st.metric("MC Put Price", f"₹{mc_put_price(S, K, T, r, sigma):.4f}")

        st.divider()
        st.subheader("Greeks")
        greeks_df = pd.DataFrame({
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Call": [
                f"{delta_call(S, K, T, r, sigma):.4f}",
                f"{gamma(S, K, T, r, sigma):.4f}",
                f"{vega(S, K, T, r, sigma):.4f}",
                f"{theta_call(S, K, T, r, sigma):.4f}",
                f"{rho_call(S, K, T, r, sigma):.4f}"
            ],
            "Put": [
                f"{delta_put(S, K, T, r, sigma):.4f}",
                f"{gamma(S, K, T, r, sigma):.4f}",
                f"{vega(S, K, T, r, sigma):.4f}",
                f"{theta_put(S, K, T, r, sigma):.4f}",
                f"{rho_put(S, K, T, r, sigma):.4f}"
            ]
        })
        st.dataframe(greeks_df, use_container_width=True)

        st.divider()
        st.subheader("Greeks Sensitivity")
        S_range = np.linspace(50, 150, 200)
        deltas = [delta_call(s, K, T, r, sigma) for s in S_range]
        gammas = [gamma(s, K, T, r, sigma) for s in S_range]
        vegas = [vega(s, K, T, r, sigma) for s in S_range]

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

        plt.suptitle(f'Greeks Sensitivity (K={K}, T={T}yr, σ={sigma:.0%})', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with bond_tab:
        st.subheader("Bond Pricing")
        col1, col2 = st.columns(2)
        with col1:
            face_value = st.slider("Face Value (₹)", 100, 10000, 1000, step=100)
            coupon_rate = st.slider("Coupon Rate", 0.01, 0.20, 0.08, step=0.01)
            market_price = st.slider("Market Price (₹)", 100, 10000, 950, step=50)
        with col2:
            bond_T = st.slider("Maturity (years)", 1, 30, 10)
            frequency = st.selectbox("Coupon Frequency", [1, 2, 4], index=1)

        r_bond = 0.06
        bp = bond_price(face_value, coupon_rate, r_bond, bond_T, frequency)
        ytm = yield_to_maturity(face_value, coupon_rate, market_price, bond_T, frequency)
        dur = duration(face_value, coupon_rate, r_bond, bond_T, frequency)

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bond Price", f"₹{bp:.2f}")
        with col2:
            st.metric("Yield to Maturity", f"{ytm*100:.4f}%")
        with col3:
            st.metric("Duration", f"{dur:.4f} years")