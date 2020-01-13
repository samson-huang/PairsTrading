#jupyter notebook --no-browser --port 6061 --ip=192.168.56.102


import tushare as ts
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


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
	  
	  
convertible_bond_code=(['300059.sz','000001.sz'])
symbols= convertible_bond_code
column= "close"	
#outdata 初始化 生成
outdata=output_data(convertible_bond_code[0],'tushare',"20190101","20190131",column)	
for i in range(1,len(symbols)): 
   outdata = outdata.join(output_data(symbols[i],'tushare',"20190101","20190131",column))

####[datetime.strptime(x,'%Y%m%d') for x in outdata.index]
outdata.index=[datetime.strptime(x,'%Y%m%d') for x in outdata.index]

# Specify number of days to shift
shift = 5
# Compute returns over the time period specified by shift
shift_returns = outdata/outdata.shift(shift) - 1


# Specify filter "length"
filter_len = shift
# Compute mean and variance
shift_returns_mean = shift_returns.ewm(span=filter_len).mean()
shift_returns_var = shift_returns.ewm(span=filter_len).var()

# Compute covariances
StockList=convertible_bond_code
NumStocks = len(convertible_bond_code)
covariance = pd.DataFrame()
for FirstStock in np.arange(NumStocks-1):
    for SecondStock in np.arange(FirstStock+1,NumStocks):
        ColumnTitle = StockList[FirstStock] + '-' + StockList[SecondStock]
        covariance[ColumnTitle] = shift_returns[StockList[FirstStock]].ewm(span=filter_len).cov(shift_returns[StockList[SecondStock]])
