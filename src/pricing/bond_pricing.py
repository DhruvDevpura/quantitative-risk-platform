face_value = 1000
coupon_rate = 0.06
r = 0.05 #discount rate
T = 5 #years
frequency = 2 #semi-annual
market_price = 950

def bond_price(face_value, coupon_rate, r, T, frequency):
    price = 0
    coupon_payment_per_period = face_value * coupon_rate / frequency
    total_periods = T * frequency
    for i in range(total_periods):
        price += (coupon_payment_per_period/(1+(r/frequency))**(i+1))
    price += (face_value)/((1+(r/frequency))**(total_periods))
    return price

def yield_to_maturity(face_value, coupon_rate, market_price, T, frequency):
    r = coupon_rate  # initial guess
    for i in range(1000):
        f = bond_price(face_value, coupon_rate, r, T, frequency) - market_price
        f_prime = (bond_price(face_value, coupon_rate, r + 0.0001, T, frequency) - bond_price(face_value, coupon_rate, r, T, frequency)) / 0.0001
        r = r - f / f_prime
        if abs(f) < 0.000001:
            break
    return r

bp = bond_price(face_value, coupon_rate, r, T, frequency)
ytm = yield_to_maturity(face_value, coupon_rate, market_price, T, frequency)
print(bp)
print(ytm)