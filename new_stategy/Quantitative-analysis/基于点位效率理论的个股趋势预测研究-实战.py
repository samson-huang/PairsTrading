# coding: utf-8
#import tushare as ts
import sys 
sys.path.append("C://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//") 
import foundation_tushare 
import json
from Hugos_tools.Approximation import (Approximation, Mask_dir_peak_valley,
                                          Except_dir, Mask_status_peak_valley,
                                          Relative_values)
from Hugos_tools.performance import Strategy_performance
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Union, Callable, Any)

import datetime as dt
import empyrical as ep
import numpy as np
import pandas as pd
import talib
import scipy.stats as st
from IPython.display import display

from sklearn.pipeline import Pipeline

from jqdatasdk import (auth, get_price, get_trade_days)

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 使用ts
# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
my_ts  = foundation_tushare.TuShare(setting['token'], max_retry=60)


start = '20050101'
end = '20220421'
'''
创业板指
set123=my_ts.index_basic(market='SZSE')
#set123[set123.name.str.contains('创业板')]
399006.SZ
'''
index_name='399006.SZ'
index_df = my_ts.query('index_daily', ts_code=index_name, 
start_date=start, end_date=end,fields='trade_date,close,high,low,open')    
mid_data=index_df
mid_data.index = pd.to_datetime(mid_data.trade_date)
del mid_data['trade_date']
mid_data.sort_index(inplace=True)  # 排序

status_frame: pd.DataFrame = foundation_tushare.get_clf_wave(mid_data, 2, 'c', True)
dir_frame: pd.DataFrame = foundation_tushare.get_clf_wave(mid_data, 2, 'c', False)

rv = Relative_values('dir')
rv_df:pd.DataFrame = rv.fit_transform(dir_frame)

test_rv_df:pd.DataFrame = rv_df[['close','relative_time','relative_price']].copy()
for i in range(1,25):

    test_rv_df[i] = test_rv_df['close'].pct_change(i).shift(-i)
    
#drop_tmp = test_rv_df.dropna(subset=['relative_price'])
#relative_price，relative_time图隐藏
#drop_tmp[['close', 'relative_price', 'relative_time']].plot(figsize=(18, 12),
#                                                            subplots=True);
 
#朴素贝叶斯与逻辑回归
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib # 用于模型导出

train_df = test_rv_df.loc[:'2022-01-01'].dropna()

test_df = test_rv_df.loc['2022-01-01':]

x_test = test_df[['relative_time','relative_price']]

tscv = TimeSeriesSplit(n_splits=5,max_train_size=180)

nb = GaussianNB()

lr = LogisticRegression()
for i,(train_index, test_index) in enumerate(tscv.split(train_df)):

    x_train = train_df.iloc[train_index][['relative_time','relative_price']]
    y_train = train_df.iloc[train_index][1]
    y_sign = np.where(y_train > 0.,1,0)
    lr.fit(x_train,y_sign)
    nb.fit(x_train,y_sign)


df = pd.DataFrame()
next_ret = test_rv_df['close'].pct_change().shift(-1)
next_ret1 = test_rv_df['close'].pct_change()
df['GaussianNB'] = next_ret.loc[test_df.index] * nb.predict(x_test)
df['LogisticRegression'] = next_ret.loc[test_df.index] * lr.predict(x_test)
#ep.cum_returns(df).plot(figsize=(18,6))
#ep.cum_returns(next_ret1.loc[x_test.index]).plot(color='darkgray',label='HS300')
#plt.legend();  
df1 = pd.DataFrame()
test1=test_rv_df['close']*0+1
df1['GaussianNB'] = test1.loc[test_df.index] * nb.predict(x_test)
df1['LogisticRegression'] = test1.loc[test_df.index] * lr.predict(x_test)
df1.columns=['GaussianNB_MARK','LogisticRegression_MARK']

test123=next_ret1*100
test123.columns = ['pct_chg'] 
test4=pd.merge(test123,df1,how='inner', left_index=True, right_index=True)
test4.columns =['pct_chg','GaussianNB_MARK','LogisticRegression_MARK']
#summary(test4)

test4=pd.merge(test4,df,how='inner', left_index=True, right_index=True)
foundation_tushare.foundation_summary(test4)
test4.tail()

#ep.cum_returns(df).plot(figsize=(18,6))
#ep.cum_returns(next_ret1.loc[x_test.index]).plot(color='darkgray',label=index_name)
#plt.legend();  