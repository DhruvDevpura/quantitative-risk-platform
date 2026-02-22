import numpy as np
from scipy.stats import norm
from black_scholes import bs_call_price, bs_put_price

S0 = 42 #stock price
r = 0.1 #risk-free intrest rate
K = 40 #strike price
T = 0.5 #time in years
sigma = 0.2 #volatility of stock

def mc_call_price(S, K, T, r, sigma):
    #Calculate European call option price using Monte Carlo simulation.
    Z = np.random.standard_normal(10000)
    S_T = S * np.exp((r - 0.5*sigma**2)*T + sigma*((T**(0.5))*Z))
    payoff = np.maximum(S_T - K,0)
    price = np.exp(-r*T)*np.mean(payoff)
    return price

def mc_put_price(S, K, T, r, sigma):
    #Calculate European put option price using Monte Carlo simulation.
    Z = np.random.standard_normal(10000)
    S_T = S * np.exp((r - 0.5*sigma**2)*T + sigma*((T**(0.5))*Z))
    payoff = np.maximum(K - S_T,0)
    price = np.exp(-r*T)*np.mean(payoff)
    return price

if __name__ == "__main__":
    print("MC Call Price: ", mc_call_price(S0, K, T, r, sigma))
    print("MC Put Price: ", mc_put_price(S0, K, T, r, sigma))
    print("BS Call Price: ", bs_call_price(S0, K, T, r, sigma))
    print("BS Put Price: ", bs_put_price(S0, K, T, r, sigma))
