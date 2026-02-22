import numpy as np
from scipy.stats import norm

S0 = 42 #stock price
r = 0.1 #risk-free intrest rate
K = 40 #strike price
T = 0.5 #time in years
sigma = 0.2 #volatility of stock

def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r*T + ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    d2 = (np.log(S/K) + (r*T - ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    return d1, d2

def bs_call_price(S, K, T, r, sigma):
    d1,d2 = d1_d2(S, K, T, r, sigma)
    c = (S*(norm.cdf(d1))) - (K*(np.exp((-1)*r*T))*(norm.cdf(d2)))
    return c

def bs_put_price(S, K, T, r, sigma):
    d1,d2 = d1_d2(S, K, T, r, sigma)
    p = (K*(np.exp(-r*T))*(norm.cdf(-d2))) - (S*(norm.cdf(-d1)))
    return p

def put_call_parity(c,p,S,K,T,r):
    lhs = c + (K*(np.exp((-r)*T))) 
    rhs = p + S
    if(abs(lhs - rhs) < 0.000001):
        return "Put Call Parity Holds"
    else:
        return "P-C Parity doesnt hold --> Arbitrage oppurtunity!"

def delta(S, K, T, r, sigma):
    d1,d2 = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def gamma(S, K, T, r, sigma):
    d1,d2 = d1_d2(S, K, T, r, sigma)
    gma = (norm.pdf(d1))/(S*sigma*(T**(0.5)))
    return gma

def vega(S, K, T, r, sigma):
    d1,d2 = d1_d2(S, K, T, r, sigma)
    vg = S * norm.pdf(d1) * (T**(0.5))
    return vg

def theta(S, K, T, r, sigma):
    d1,d2 = d1_d2(S, K, T, r, sigma)
    th = -(S * norm.pdf(d1) * sigma) / (2*(T**(0.5))) - r*K*np.exp(-r*T)*norm.cdf(d2)
    return th

def rho(S, K, T, r, sigma):
    d1,d2 = d1_d2(S, K, T, r, sigma)
    rh = K * T * (np.exp(-r*T)) * norm.cdf(d2)
    return rh

if __name__ == "__main__":
    c = bs_call_price(S0,K,T,r,sigma)
    p = bs_put_price(S0,K,T,r,sigma)
    print("Call price: ", c)
    print("Put price: ", p)
    print(put_call_parity(c,p,S0,K,T,r))
    print("Delta: ", delta(S0, K, T, r, sigma))
    print("Gamma: ", gamma(S0, K, T, r, sigma))
    print("Vega: ", vega(S0, K, T, r, sigma))
    print("Theta: ", theta(S0, K, T, r, sigma))
    print("Rho: ", rho(S0, K, T, r, sigma))
