﻿#jupyter notebook --no-browser --port 6061 --ip=192.168.56.102
#https://pypi.tuna.tsinghua.edu.cn/simple/
#source code url https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False

## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000

return_vec = np.random.randn(n_assets, n_obs)

plt.plot(return_vec.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

print (rand_weights(n_assets))


def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    #if sigma > 2:
        #return random_portfolio(returns)
    return mu, sigma

n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(return_vec) 
    for _ in range(n_portfolios)
])

plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

weights, returns, risks = optimal_portfolio(return_vec)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')

#print (weights)

###Backtesting on real market data

#zipline包安装有问题，该段代码已经无法运行
######################datareader行情数据只能一只股票一只股票取###############################
##############################################################################################
###############################################################################################
############################################################################
#from zipline.utils.factory import load_bars_from_yahoo
end = pd.Timestamp.utcnow()
start = end - 2500 * pd.tseries.offsets.BDay()

#data = load_bars_from_yahoo(stocks=['IBM', 'GLD', 'XOM', 'AAPL', 
#                                    'MSFT', 'TLT', 'SHY'],
#                            start=start, end=end)
import pandas_datareader.data as web
stocks=['IBM', 'GLD', 'XOM', 'AAPL','MSFT', 'TLT', 'SHY']
stocks = ','.join(stocks)
data = web.DataReader(stocks, "yahoo",start,end)  
    
#data['Open','AAPL'].head()
#######################################################################
######################################################################
######################################################################
########################################################################



#######################python3.5 datereader  有问题###############################################
########################################################################
from pandas_datareader import data as dreader

symbols = ['GOOG', 'AAPL','GLD', 'XOM']

pnls = {i:dreader.DataReader(i,'yahoo','2016-01-01','2016-09-01') for i in symbols}


# plot them
plt.plot(pnls['GOOG']['Adj Close'], label='GOOG')
plt.plot(pnls['AAPL']['Adj Close'], label='AAPL')
plt.plot(pnls['GLD']['Adj Close'], label='GLD')
plt.plot(pnls['XOM']['Adj Close'], label='XOM')

# generate a legend box
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
       ncol=4, mode="expand", borderaxespad=0.)
 
# annotate an important value
#plt.annotate("Important value", (55,20), xycoords='data',
#         xytext=(5, 38),
#         arrowprops=dict(arrowstyle='->'))
plt.show()

######
GOOG_data=pnls['GOOG']['Adj Close'].diff()/pnls['GOOG']['Adj Close']
AAPL_data=pnls['AAPL']['Adj Close'].diff()/pnls['AAPL']['Adj Close']
GLD_data=pnls['GLD']['Adj Close'].diff()/pnls['GLD']['Adj Close']
XOM_data=pnls['XOM']['Adj Close'].diff()/pnls['XOM']['Adj Close']
total_data=pd.DataFrame([GOOG_data,AAPL_data,GLD_data,XOM_data])
total_data.index=['GOOG','AAPL','GLD','XOM']
total_data=total_data.dropna(axis=1,how='all') 

plt.plot(total_data.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')

n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(total_data) 
    for _ in range(n_portfolios)
])


plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')



#######################realdata-optimal_portfolio#####################

weights, returns, risks = optimal_portfolio(total_data)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')

print (weights)
###################################################
##############################################################
###############################################################
###############################################################
#####################################################



################pnls['GOOG']['Adj Close'].plot(figsize=(8,5))

def download_ohlc(sector_tickers, start, end):
    sector_ohlc = {}
    for sector, tickers in sector_tickers.iteritems():
        print ('Downloading data from Yahoo for %s sector' % sector)
        data = web.DataReader(tickers, 'yahoo', start, end)
        for item in ['Open', 'High', 'Low']:
            data[item] = data[item] * data['Adj Close'] / data['Close']
        data.rename(items={'Open': 'open', 'High': 'high', 'Low': 'low',
                           'Adj Close': 'close', 'Volume': 'volume'},
                    inplace=True)
        data.drop(['Close'], inplace=True)
        sector_ohlc[sector] = data
    print ('Finished downloading data')
    return sector_ohlc

#######################################################################
##########################################################################

data['Adj Close'].plot(figsize=(8,5))
plt.ylabel('price in $')    

#####################################
#####################生成模拟真实盘口数据######################
######
GOOG_data_close=pnls['GOOG']['Adj Close']
AAPL_data_close=pnls['AAPL']['Adj Close']
GLD_data_close=pnls['GLD']['Adj Close']
XOM_data_close=pnls['XOM']['Adj Close']
total_data_close=pd.DataFrame([GOOG_data_close,AAPL_data_close,GLD_data_close,XOM_data_close])
total_data_close.index=['GOOG','AAPL','GLD','XOM']
total_data_close=total_data_close.dropna(axis=1,how='all') 
total_data_close.plot(figsize=(8,5))

#############################################
###################################################

import zipline
from zipline.api import (history, 
                         set_slippage, 
                         slippage,
                         set_commission, 
                         commission, 
                         order_target_percent)

from zipline import TradingAlgorithm


def initialize(context):
    '''
    Called once at the very beginning of a backtest (and live trading). 
    Use this method to set up any bookkeeping variables.
    
    The context object is passed to all the other methods in your algorithm.

    Parameters

    context: An initialized and empty Python dictionary that has been 
             augmented so that properties can be accessed using dot 
             notation as well as the traditional bracket notation.
    
    Returns None
    '''
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    context.tick = 0
    
def handle_data(context, data):
    '''
    Called when a market event occurs for any of the algorithm's 
    securities. 

    Parameters

    data: A dictionary keyed by security id containing the current 
          state of the securities in the algo's universe.

    context: The same context object from the initialize function.
             Stores the up to date portfolio as well as any state 
             variables defined.

    Returns None
    '''
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    if context.tick < 100:
        return
    # Get rolling window of past prices and compute returns
    prices = history(100, '1d', 'price').dropna()
    returns = prices.pct_change().dropna()
    try:
        # Perform Markowitz-style portfolio optimization
        weights, _, _ = optimal_portfolio(returns.T)
        # Rebalance portfolio accordingly
        for stock, weight in zip(prices.columns, weights):
            order_target_percent(stock, weight)
    except ValueError as e:
        # Sometimes this error is thrown
        # ValueError: Rank(A) < p or Rank([P; A; G]) < n
        pass
        
# Instantinate algorithm        
algo = TradingAlgorithm(initialize=initialize, 
                        handle_data=handle_data)
# Run algorithm
results = algo.run(total_data.T)
results.portfolio_value.plot()                