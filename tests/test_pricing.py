import numpy as np
import sys
sys.path.append('src/pricing')
from black_scholes import *
from monte_carlo import *
from bond_pricing import *

#Black-Scholes tests
def test_bs_call_price():
    assert abs(bs_call_price(42, 40, 0.5, 0.1, 0.2) - 4.76) < 0.01

def test_bs_put_price():
    assert abs(bs_put_price(42, 40, 0.5, 0.1, 0.2) - 0.81) < 0.01

def test_put_call_parity():
    c = bs_call_price(42, 40, 0.5, 0.1, 0.2)
    p = bs_put_price(42, 40, 0.5, 0.1, 0.2)
    lhs = c + 40 * np.exp(-0.1 * 0.5)
    rhs = p + 42
    assert abs(lhs - rhs) < 0.0001

def test_deep_itm_call():
    c = bs_call_price(100, 50, 0.5, 0.1, 0.2)
    assert c > 100 - 50 * np.exp(-0.1 * 0.5) - 0.01

def test_deep_otm_call():
    c = bs_call_price(50, 100, 0.5, 0.1, 0.2)
    assert c < 0.5

def test_vega_positive():
    v = vega(42, 40, 0.5, 0.1, 0.2)
    assert v > 0

def test_delta_range():
    d = delta_call(42, 40, 0.5, 0.1, 0.2)
    assert 0 < d < 1

def test_theta_negative():
    t = theta_call(42, 40, 0.5, 0.1, 0.2)
    assert t < 0

def test_delta_put_negative():
    d = delta_put(42, 40, 0.5, 0.1, 0.2)
    assert -1 < d < 0

def test_theta_put_negative():
    t = theta_put(42, 40, 0.5, 0.1, 0.2)
    assert t < 0

def test_rho_call_positive():
    r = rho_call(42, 40, 0.5, 0.1, 0.2)
    assert r > 0

def test_rho_put_negative():
    r = rho_put(42, 40, 0.5, 0.1, 0.2)
    assert r < 0

#Monte Carlo tests
def test_mc_call_close_to_bs():
    mc = mc_call_price(42, 40, 0.5, 0.1, 0.2)
    bs = bs_call_price(42, 40, 0.5, 0.1, 0.2)
    assert abs(mc - bs) / bs < 0.05

def test_mc_put_close_to_bs():
    mc = mc_put_price(42, 40, 0.5, 0.1, 0.2)
    bs = bs_put_price(42, 40, 0.5, 0.1, 0.2)
    assert abs(mc - bs) / bs < 0.05

#Bond pricing tests
def test_bond_price():
    assert abs(bond_price(1000, 0.06, 0.05, 5, 2) - 1043.76) < 0.01

def test_ytm():
    assert abs(yield_to_maturity(1000, 0.06, 950, 5, 2) - 0.072) < 0.002

def test_par_bond():
    p = bond_price(1000, 0.06, 0.06, 5, 2)
    assert abs(p - 1000) < 0.01