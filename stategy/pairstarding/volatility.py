#隐含波动率模型
#Vega（ν）：衡量标的资产价格波动率变动时，期权价格的变化幅度，
#是用来衡量期货价格的波动率的变化对期权价值的影响。
#Vega，指期权费（P）变化与标的汇率波动性（Volatility）变化的敏感性。

#从理论上讲，要获得隐含波动率的大小并不困难。由于期权定价模型(如BS模型)
#给出了期权价格与五个基本参数(标的股价、执行价格、利率、到期时间、波动率)
#之间的定量关系，只要将其中前4个基本参数及期权的实际市场价格作为已知量代
#入定价公式，就可以从中解出惟一的未知量，其大小就是隐含波动率。
# Black-Scholes-Merton (1973)
#
# Valuation of European call options in Black-Scholes-Merton model
# incl. Vega function and inplied volatility estimation
# bsm_function.py
#

# Analytical Black-Scholes-Merton (BSM) Formula

def bsm_call_value(S0, K, T, r, sigma):
""" Valuation of European call option in BSM model.
Analytical formula.

Parameters
==========
S0 : float
initial stock/index level
K : float
strick price
T : float
maturity date (in year fractions)
r : float
constant risk-free short rate
sigma : float
volatility factor in diffusion term

Returns
=======
values : float
present value of the European call option
"""
from math import log, sqrt, exp
from scipy import stats

S0 = float(S0)
d1 = (log(S0 / K) + (r + 0.5 *sigma ** 2) * T) / (sigma * sqrt(T))
d2 = (log(S0 / K) + (r - 0.5 *sigma ** 2) * T) / (sigma * sqrt(T))
value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
- K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
# stats.norm.cdf -> cumulative distribution function
# for normal distribution
return value

# Vega function

def bsm_vega(S0, K, T, r, sigma):
""" Vega of European option in BSM model.

Parameters
==========
S0 : float
initial stock/index level
K : float
strick price
T : float
maturity date (in year fractions)
r : float
constant risk-free short rate
sigma : float
volatility factor in diffusion term

Returns
=======
Vega : float
partial derivation of BSM formula with respect
to sigma, i.e. Vega
"""
from math import log, sqrt
from scipy import stats

S0 = float(S0)
d1 = (log(S0 / K) + (r + 0.5 *sigma ** 2) * T) / (sigma * sqrt(T))
vega = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(T)
return vega

# implied volatility function

def bsm_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
""" Implied volatility of European call option in BSM model.

Parameters
==========
S0 : float
initial stock/index level
K : float
strick price
T : float
maturity date (in year fractions)
r : float
constant risk-free short rate
sigma_est : float
estimate of impl. volatility
it : integer


Returns
=======
Vega : float
partial derivation of BSM formula with respect
to sigma, i.e. Vega
"""
for i in range(it):
sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)
/bsm_vega(S0, K, T, r, sigma_est))
return sigma_est





#中国波指iVIX,000188.SH
#已实现波动率（RV）


#vanna表示vega对标的价格变化的敏感度
#volga表示vega对波动率变化的敏感度

#VV模型套利

#SABR波动率模型	  	