#jupyter notebook --no-browser --port 6061 --ip=192.168.56.102


import tushare as ts
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns
import scipy.optimize as sco
import sys

sys.path.append('/home/huangtuo/gitdoc/markowitz-portfolio-optimization')
from markowitz import MarkowitzOpt

#import cvxopt as opt
#from cvxopt import blas, solvers

#tushare升级需要用api进行调用
ts.set_token('fbe098e754f69ea09a7bd0c144a00754e93aab1911508cb408c5cb21')
pro = ts.pro_api()



def output_data(security,source,begin_date,end_date,column): 
	  if source=='tushare':
	     fm=pro.daily(ts_code=security, start_date=begin_date, end_date=end_date)
	     fm.index=fm['trade_date']
	     fm=pd.DataFrame(fm['close'])
	     fm=fm.rename(columns={'close':security})
	  return(fm)

#initialize date	  
begin_date='20190101'
end_date='20191231'	
interest_rate = 0								# Fixed interest rate
min_return = 0.003								# Minimum desired return

  
convertible_bond_code=(['300059.sz','000001.sz','000783.sz','300335.sz'])
symbols= convertible_bond_code
column= "close"	
#outdata 初始化 生成
outdata=output_data(convertible_bond_code[0],'tushare',begin_date,end_date,column)	
for i in range(1,len(symbols)): 
   outdata = outdata.join(output_data(symbols[i],'tushare',begin_date,end_date,column))

####[datetime.strptime(x,'%Y%m%d') for x in outdata.index]
outdata.index=[datetime.strptime(x,'%Y%m%d') for x in outdata.index]

#reset by index order
outdata=outdata.sort_index()

plt.figure(figsize=(14, 7))
for c in outdata.columns.values:
    plt.plot(outdata.index, outdata[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')

returns = outdata.pct_change()

plt.figure(figsize=(14, 7))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')

##############################################

# Variable Initialization

start_date = '2019-01-04'
index = shift_returns.index
start_index = index.get_loc(start_date)
end_date = index[-1]
end_index = index.get_loc(end_date)
date_index_iter = start_index
convertible_bond_code=(['300059.sz','000001.sz','000783.sz','300335.sz'])
StockList=convertible_bond_code
StockList.append('InterestRate')
distribution = pd.DataFrame(index=StockList)
loop_returns = pd.Series(index=index)
# Start Value
total_value = 1.0
loop_returns[index[date_index_iter]] = total_value

#########
table = outdata
returns = outdata.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178
##########MarkowitzOpt_new function #######
###################################################
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
    

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
    
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

############main optimisation function ##########
def display_ef_with_selected(table, mean_returns, cov_matrix, risk_free_rate):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    

    return max_sharpe_allocation

##############################################

original_returns = outdata.pct_change()
while date_index_iter + shift < end_index:
  date = index[date_index_iter]
  index_returns=original_returns[date_index_iter-1:date_index_iter+1]
  index_returns.insert(4,'InterestRate',[0,0])
  table = outdata[date_index_iter-1:date_index_iter+1]
  table.insert(4,'InterestRate',[1,1])	
  returns = index_returns
  mean_returns = returns.mean()
  cov_matrix = returns.cov()
  portfolio_alloc = display_ef_with_selected(table,mean_returns, cov_matrix, risk_free_rate)
  portfolio_alloc=portfolio_alloc.values[0,]/100
  distribution[date.strftime('%Y-%m-%d')] = portfolio_alloc
  
  # Calculating portfolio return
  date2 = index[date_index_iter+shift]
  temp1 = outdata.ix[date2]/outdata.ix[date]
  temp1.ix[StockList[-1]] = interest_rate+1
  temp2 = pd.Series(np.array(portfolio_alloc.ravel()).reshape(len(portfolio_alloc)),index=StockList)
  total_value = np.sum(total_value*temp2*temp1)
  # Increment Date
  date_index_iter += shift
  loop_returns[index[date_index_iter]] = total_value

# Remove dates that there are no trades from returns
loop_returns = loop_returns[np.isfinite(loop_returns)]


# Plot portfolio allocation of last 10 periods
ax = distribution.T.ix[-10:].plot(kind='bar',stacked=True)
plt.ylim([0,1])
plt.xlabel('Date')
plt.ylabel('distribution')
plt.title('distribution vs. Time')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('allocation.png')

# Plot stock prices and shifted returns
fig, axes = plt.subplots(nrows=2,ncols=1)
outdata.plot(ax=axes[0])
shift_returns.plot(ax=axes[1])
axes[0].set_title('Stock Prices')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[1].set_title(str(shift)+ ' Day Shift returns')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('returns ' + str(shift) + ' Days Apart')
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('stocks.png', pad_inches=1)
fig.tight_layout()

# Plot portfolio returns vs. time
plt.figure()
loop_returns.plot()
plt.xlabel('Date')
plt.ylabel('Portolio returns')
plt.title('Portfolio returns vs. Time')
# plt.savefig('returns.png')
