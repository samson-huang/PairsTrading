import pandas as pd
import tushare as ts
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.tears import create_full_tear_sheet

import json
import os
import pickle
import sys
import warnings
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api()

df = pro.daily(ts_code='000001.SZ,600982.SH', start_date='20200101', end_date='20200122')
df.index = pd.to_datetime(df['trade_date'])
df.index.name = None
df.sort_index(inplace=True)
df.head()

# 多索引的因子列，第一个索引为日期，第二个索引为股票代码
assets = df.set_index([df.index, df['ts_code']], drop=True)
assets.tail()

# column为股票代码，index为日期，值为收盘价
close = df.pivot_table(index='trade_date', columns='ts_code', values='close')
close.index = pd.to_datetime(close.index)
close.tail()

# 我们是使用pct_chg因子数据预测收盘价，因此需要偏移1天，但是这里有2只股票，所以是shift(2)
ret = get_clean_factor_and_forward_returns(assets[['pct_chg']].shift(2), close)
create_full_tear_sheet(ret, long_short=False)



##################
test123=rtn_df.reset_index()
#################
