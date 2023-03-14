import sys
#
#path_local = "G://GitHub"
path_local = "C://Users//huangtuo//Documents//GitHub"
sys.path.append(path_local+"//PairsTrading//new_stategy//factor_analysis//")

from my_lib.data_download.data_io import DataReader
from my_lib.factor_evaluate.factor_evaluate import factor_stats
import pandas as pd
import numpy as np


#
import tushare as ts
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.tears import create_full_tear_sheet

import json
import os
import pickle
import warnings
#
def calc_factor():
	#这里是计算因子的，把因子处理成特定的（di,ii)格式。
    close_df = DataReader.read_dailyMkt('close')
    return close_df.pct_change(20)

#计算因子
factor_df = calc_factor()
factor_df.tail(5)


#股票池
univ_a = DataReader.read_IdxWeight('399300.SZ')#沪深300
univ_a = univ_a.where(pd.isnull(univ_a),1)
univ_a.tail(5)


#st股、停牌、涨跌停
ST_valid = DataReader.read_ST_valid()
suspend_valid = DataReader.read_suspend_valid()
limit_valid = DataReader.read_limit_valid()
forb_days = ST_valid*suspend_valid*limit_valid
forb_days.tail(5)

#每日收益率矩阵
rtn_df = DataReader.read_dailyRtn()
rtn_df.tail(5)



#实际沪深300个股收益数据
factor_df_hs300=factor_df*univ_a
factor_df_hs300=factor_df_hs300.dropna(axis=1,how='all')


#准备沪深300个股close数据
test_close = DataReader.read_dailyMkt('close')
close_hs300=test_close*univ_a
close_hs300=close_hs300.dropna(axis=1,how='all')
close_hs300.head(2)


#转换城alphalens-example方式
test_df=factor_df_hs300
test_df=test_df.reset_index()
test_df=test_df.melt(id_vars=['trade_date'],var_name='ts_code',value_name='pct_chg')

assets = test_df.set_index([test_df['trade_date'], test_df['ts_code']], drop=True)
assets = assets.drop(['trade_date','ts_code'],axis=1)

assets=assets.dropna(axis=0,how='all')

assets.tail()


######################

# 我们是使用pct_chg因子数据预测收盘价，因此需要偏移1天，但是这里有2只股票，所以是shift(2)
ret = get_clean_factor_and_forward_returns(assets[['pct_chg']].shift(2), close)
create_full_tear_sheet(ret, long_short=False)


##################
test123=rtn_df.reset_index()
#################


