import numpy as np
from scipy.stats import norm

def d1_d2(S, K, T, r, sigma):
    #Calculate d1 and d2 parameters for Black-Scholes formula.
    d1 = (np.log(S/K) + (r*T + ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    d2 = (np.log(S/K) + (r*T - ((sigma**2)*(T/2))))/(sigma*(T**(0.5)))
    return d1, d2

def bs_call_price(S, K, T, r, sigma):
    #Calculate European call option price using Black-Scholes formula.
    d1,d2 = d1_d2(S, K, T, r, sigma)
    c = (S*(norm.cdf(d1))) - (K*(np.exp((-1)*r*T))*(norm.cdf(d2)))
    return c

def bs_put_price(S, K, T, r, sigma):
    #Calculate European put option price using Black-Scholes formula.
    d1,d2 = d1_d2(S, K, T, r, sigma)
    p = (K*(np.exp(-r*T))*(norm.cdf(-d2))) - (S*(norm.cdf(-d1)))
    return p

def put_call_parity(c,p,S,K,T,r):
    #Check if put-call parity holds for given call and put prices.
    lhs = c + (K*(np.exp((-r)*T))) 
    rhs = p + S
    if(abs(lhs - rhs) < 0.000001):
        return "Put Call Parity Holds"
    else:
        return "P-C Parity doesnt hold --> Arbitrage oppurtunity!"

def delta_call(S, K, T, r, sigma):
    #Calculate call option Delta - sensitivity to stock price change.
    d1,d2 = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def gamma(S, K, T, r, sigma):
    #Calculate option Gamma - rate of change of Delta.
    d1,d2 = d1_d2(S, K, T, r, sigma)
    gma = (norm.pdf(d1))/(S*sigma*(T**(0.5)))
    return gma

def vega(S, K, T, r, sigma):
    #Calculate option Vega - sensitivity to volatility change.
    d1,d2 = d1_d2(S, K, T, r, sigma)
    vg = S * norm.pdf(d1) * (T**(0.5))
    return vg

def theta_call(S, K, T, r, sigma):
    #Calculate call option Theta - time decay per unit time.
    d1,d2 = d1_d2(S, K, T, r, sigma)
    th = -(S * norm.pdf(d1) * sigma) / (2*(T**(0.5))) - r*K*np.exp(-r*T)*norm.cdf(d2)
    return th

def rho_call(S, K, T, r, sigma):
    #Calculate call option Rho - sensitivity to interest rate change.
    d1,d2 = d1_d2(S, K, T, r, sigma)
    rh = K * T * (np.exp(-r*T)) * norm.cdf(d2)
    return rh

def delta_put(S, K, T, r, sigma):
    #Calculate put option Delta.
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

def theta_put(S, K, T, r, sigma):
    #Calculate put option Theta.
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return -(S * norm.pdf(d1) * sigma) / (2*(T**(0.5))) + r*K*np.exp(-r*T)*norm.cdf(-d2)

def rho_put(S, K, T, r, sigma):
    #Calculate put option Rho.
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r*T) * norm.cdf(-d2)

if __name__ == "__main__":
    S0 = 42 #stock price
    r = 0.1 #risk-free intrest rate
    K = 40 #strike price
    T = 0.5 #time in years
    sigma = 0.2 #volatility of stock
    c = bs_call_price(S0,K,T,r,sigma)
    p = bs_put_price(S0,K,T,r,sigma)
    print("Call price: ", c)
    print("Put price: ", p)
    print(put_call_parity(c,p,S0,K,T,r))
    print("Delta: ", delta_call(S0, K, T, r, sigma))
    print("Gamma: ", gamma(S0, K, T, r, sigma))
    print("Vega: ", vega(S0, K, T, r, sigma))
    print("Theta: ", theta_call(S0, K, T, r, sigma))
    print("Rho: ", rho_call(S0, K, T, r, sigma))
    print("Delta: ", delta_put(S0, K, T, r, sigma))
    print("Theta: ", theta_put(S0, K, T, r, sigma))
    print("Rho: ", rho_put(S0, K, T, r, sigma))
