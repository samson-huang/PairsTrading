
import datetime
import json
import os
import pickle
import sys
import warnings
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
import tushare as ts

warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api()


stock_list = ['300750.XSHE','600000.XSHG','300601.XSHE'] #假设这里面都是港股股票
start = '2020-01-01'
end = '2020-12-31'
dir_name = 'C:/Users/huangtuo/.qlib/qlib_data/fund_data'

with open('C://temp//upload//codefundsecname.json') as file:
    code2secname = json.loads(file.read())

stk_set =  list(code2secname.keys())
for index_code in stk_set:
    if index_code[:2] in ('00','')
def make_dataset(stock_pool:list, start:str, end:str):
    fields = ['volume','money','open','high','low','close','factor']
    for code in stock_pool:
        df = get_price(code,
               start_date=start,
               end_date=end,
               frequency='daily',
               fields=fields,
               skip_paused=False,
               fq='pre')
        df['vwap'] = df['money']/df['volume'] #pre-fq-factor
        df['change'] = df['close'].pct_change()
        df.insert(0,'date',[''.join(str(x)[0:10].split('-')) for x in list(df.index)])
        df.insert(0,'stock_code',[''.join(code.split('.'))] * len(df))
        file_name = dir_name + '/' + ''.join(code.split('.')) + '.csv'
        df.to_csv(file_name,header = True, index = False)
make_dataset(stock_list,start,end)