import numpy as np
from scipy.stats import norm

S0 = 42 #stock price
r = 0.1 #risk-free intrest rate
K = 40 #strike price
T = 0.5 #time in years
sigma = 0.2 #volatility of stock

def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r*T + ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    d2 = (np.log(S/K) + (r*T - ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    c = (S*(norm.cdf(d1))) - (K*(np.exp((-1)*r*T))*(norm.cdf(d2)))
    return c

def bs_put_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r*T + ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    d2 = (np.log(S/K) + (r*T - ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    c = -(S*(norm.cdf(-d1))) + (K*(np.exp((-1)*r*T))*(norm.cdf(-d2)))
    return c

c = bs_call_price(S0,K,T,r,sigma)
p = bs_put_price(S0,K,T,r,sigma)

def put_call_parity(c,p,S,K,T,r):
    lhs = c + (K*(np.exp((-r)*T))) 
    rhs = p + S
    if(abs(lhs - rhs) < 0.000001):
        return "Put Call Parity Holds"
    else:
        return "P-C Parity doesnt hold --> Arbitrage oppurtunity!"

print("Call price: ",c)
print("Put price: ",p)
print(put_call_parity(c,p,S0,K,T,r))
