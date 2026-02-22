face_value = 1000
coupon_rate = 0.06
r = 0.05 #discount rate
T = 5 #years
frequency = 2 #semi-annual
market_price = 950

def bond_price(face_value, coupon_rate, r, T, frequency):
    #Calculate bond price using discounted cash flow method.
    price = 0
    coupon_payment_per_period = face_value * coupon_rate / frequency
    total_periods = int(T * frequency)
    for i in range(total_periods):
        price += coupon_payment_per_period / (1 + r/frequency)**(i+1)
    price += (face_value)/((1+(r/frequency))**(total_periods))
    return price

def yield_to_maturity(face_value, coupon_rate, market_price, T, frequency):
    #Calculate yield to maturity using Newton-Raphson method.
    r = coupon_rate  # initial guess
    for i in range(1000):
        f = bond_price(face_value, coupon_rate, r, T, frequency) - market_price
        f_prime = (bond_price(face_value, coupon_rate, r + 0.0001, T, frequency) - bond_price(face_value, coupon_rate, r, T, frequency)) / 0.0001
        r = r - f / f_prime
        if abs(f) < 0.000001:
            break
    return r

def duration(face_value, coupon_rate, r, T, frequency):
    #Calculate Macaulay duration - weighted average time of cash flows.
    price = bond_price(face_value, coupon_rate, r, T, frequency)
    coupon = face_value * coupon_rate / frequency
    total_periods = int(T * frequency)
    dur = 0
    for i in range(1, total_periods + 1):
        t = i / frequency  # time in years
        pv = coupon / (1 + r/frequency)**i  # present value of this coupon
        dur += t * pv
    dur += T * face_value / (1 + r/frequency)**total_periods
    dur = dur / price
    return dur

if __name__ == "__main__":
    bp = bond_price(face_value, coupon_rate, r, T, frequency)
    ytm = yield_to_maturity(face_value, coupon_rate, market_price, T, frequency)
    dur = duration(face_value, coupon_rate, r, T, frequency)
    print("Bond Price: ", bp)
    print("YTM: ", ytm)
    print("Duration: ", dur)