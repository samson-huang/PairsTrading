#jupyter notebook --no-browser --port 6061 --ip=192.168.56.102


import tushare as ts
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns
import scipy.optimize as sco
import sys

#sys.path.append('/home/huangtuo/gitdoc/markowitz-portfolio-optimization')
#from markowitz import MarkowitzOpt

#import cvxopt as opt
#from cvxopt import blas, solvers

#tushare升级需要用api进行调用
ts.set_token('**********************')
pro = ts.pro_api()



def output_data_fund(security,source,begin_date,end_date,column): 
	  if source=='tushare':
	     fm=pro.fund_nav(ts_code=security, start_date=begin_date, end_date=end_date)
	     fm=fm.drop_duplicates(subset=['nav_date'], keep='last', inplace=False)
	     fm.index=fm['nav_date']
	     fm=pd.DataFrame(fm['adj_nav'])
	     fm=fm.rename(columns={'adj_nav':security})
	  return(fm)

#initialize date	  
begin_date='20190101'
end_date='20191231'
interest_rate = 0								# Fixed interest rate
min_return = 0.003								# Minimum desired return

  
convertible_bond_code=(['163817.OF','229002.OF','006395.OF','006381.OF'])
symbols= convertible_bond_code
column= "adj_nav"	
#outdata 初始化 生成
outdata=output_data_fund(convertible_bond_code[0],'tushare',begin_date,end_date,column)	
for i in range(1,len(symbols)):
   frist_outdata=output_data_fund(symbols[i],'tushare',begin_date,end_date,column)
   outdata = outdata.join(frist_outdata)

####[datetime.strptime(x,'%Y%m%d') for x in outdata.index]
outdata.index=[datetime.strptime(x,'%Y%m%d') for x in outdata.index]

#fill nan value
outdata=outdata.fillna(method='ffill')
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
shift =1
returns = outdata.pct_change()
index = returns.index
start_index = index.get_loc(start_date)
end_date = index[-1]
end_index = index.get_loc(end_date)
date_index_iter = start_index

StockList=convertible_bond_code
StockList.append('InterestRate')
distribution = pd.DataFrame(index=StockList)
loop_returns = pd.Series(index=index)
# Start Value
total_value = 1.0
loop_returns[index[date_index_iter]] = total_value

#########
table = outdata
original_returns = outdata.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178
##########MarkowitzOpt_new function #######
###################################################
#####20200324
def portfolio_annualised_performance_new(weights, mean_returns, cov_matrix,sum_day):
    returns = np.sum(mean_returns*weights )*100 
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(sum_day)
    return std, returns
    
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
    
    an_vol = np.std(returns) 
    an_rt = mean_returns
    

    return max_sharpe_allocation

##############################################


while date_index_iter + shift <= end_index:
  date = index[date_index_iter]
  #index_returns=original_returns[date_index_iter-1:date_index_iter+1]
  index_returns=original_returns[1:date_index_iter+1]
  index_returns.insert(original_returns.shape[1],'InterestRate',np.zeros(date_index_iter))
  #table = outdata[date_index_iter-1:date_index_iter+1]
  table = outdata[1:date_index_iter+1]
  table.insert(original_returns.shape[1],'InterestRate',np.ones(date_index_iter))	
  returns = index_returns
 ###############################
    ##################-20200324
  end_price   = np.array(table.tail(1))
  start_price = np.array(table.head(1))
  an_rt=(end_price-start_price)/start_price
  an_rt=pd.DataFrame(an_rt)
  an_rt.columns=table.columns
  an_rt=an_rt.loc[0,]
    ######################   
 ################################# 
  mean_returns = an_rt
  cov_matrix = returns.cov()
  portfolio_alloc = display_ef_with_selected(table,mean_returns, cov_matrix, risk_free_rate)
  portfolio_alloc=portfolio_alloc.values[0,]/100
  distribution[date.strftime('%Y-%m-%d')] = portfolio_alloc
  
  # Calculating portfolio return
  date2 = index[date_index_iter+shift]
  temp1 = outdata.loc[date2]/outdata.loc[date]
  temp1.loc[StockList[-1]] = interest_rate+1
  temp2 = pd.Series(np.array(portfolio_alloc.ravel()).reshape(len(portfolio_alloc)),index=StockList)
  total_value = np.sum(loop_returns[index[date_index_iter]]*temp2*temp1)
  # Increment Date
  date_index_iter += shift
  loop_returns[index[date_index_iter]] = total_value

# Remove dates that there are no trades from returns
loop_returns = loop_returns[np.isfinite(loop_returns)]


# Plot portfolio allocation of last 10 periods
ax = distribution.T.loc[-10:].plot(kind='bar',stacked=True)
plt.ylim([0,1])
plt.xlabel('Date')
plt.ylabel('distribution')
plt.title('distribution vs. Time')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('allocation.png')

# Plot stock prices and shifted returns
fig, axes = plt.subplots(nrows=2,ncols=1)
outdata.plot(ax=axes[0])
original_returns.plot(ax=axes[1])
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


"""
################################################
def display_ef_with_selected_old(table, mean_returns, cov_matrix, risk_free_rate,sum_day):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance_new(max_sharpe['x'], mean_returns, cov_matrix,sum_day)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance_new(min_vol['x'], mean_returns, cov_matrix,sum_day)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns)*100 
    #an_rt = mean_returns
    ##################-20200324
    end_price   = np.array(table.tail(1))
    start_price = np.array(table.head(1))
    an_rt=(end_price-start_price)/end_price
    an_rt=pd.DataFrame(an_rt)
    an_rt.columns=table.columns
    an_rt=an_rt.ix[0,]*100
    ######################
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,2))
    print ("Annualised Volatility:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    print ("-"*80)
    print ("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(table.columns):
        print (txt,":","annuaised return",round(an_rt.ix[i],2),", annualised volatility:",round(an_vol[i],2))
    print ("-"*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)
##################
def random_portfolios_new(num_portfolios, mean_returns, cov_matrix, risk_free_rate,sum_day):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(5)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance_new(weights, mean_returns, cov_matrix,sum_day)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
    
def display_calculated_ef_with_random_old(table,mean_returns, cov_matrix, num_portfolios, risk_free_rate,sum_day):
    results, _ = random_portfolios_new(num_portfolios,mean_returns, cov_matrix, risk_free_rate,sum_day)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance_new(max_sharpe['x'], mean_returns, cov_matrix,sum_day)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance_new(min_vol['x'], mean_returns, cov_matrix,sum_day)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,2))
    print ("Annualised Volatility:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)

#############


#############
date_index_iter=243
date = index[date_index_iter]
#index_returns=original_returns[date_index_iter-1:date_index_iter+1]
index_returns=original_returns[1:date_index_iter+1]
index_returns.insert(original_returns.shape[1],'InterestRate',np.zeros(date_index_iter))
#table = outdata[date_index_iter-1:date_index_iter+1]
table = outdata[1:date_index_iter+1]
table.insert(original_returns.shape[1],'InterestRate',np.ones(date_index_iter))	
returns = index_returns
 #  ##############################
      ##################-20200324
end_price   = np.array(table.tail(1))
start_price = np.array(table.head(1))
an_rt=(end_price-start_price)/table.head(1)
an_rt=pd.DataFrame(an_rt)
an_rt.columns=table.columns
an_rt=an_rt.ix[0,]
      ######################   
 #  ################################ 
mean_returns = an_rt
cov_matrix = index_returns.cov()
display_ef_with_selected_old(table,mean_returns, cov_matrix, risk_free_rate,date_index_iter)
display_calculated_ef_with_random_old(table,mean_returns, cov_matrix, num_portfolios, risk_free_rate,date_index_iter)
##################
def random_portfolios_new(num_portfolios, mean_returns, cov_matrix, risk_free_rate,sum_day):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(5)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance_new(weights, mean_returns, cov_matrix,sum_day)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
    
def display_calculated_ef_with_random_old(table,mean_returns, cov_matrix, num_portfolios, risk_free_rate,sum_day):
    results, _ = random_portfolios_new(num_portfolios,mean_returns, cov_matrix, risk_free_rate,sum_day)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance_new(max_sharpe['x'], mean_returns, cov_matrix,sum_day)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance_new(min_vol['x'], mean_returns, cov_matrix,sum_day)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,2))
    print ("Annualised Volatility:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
"""