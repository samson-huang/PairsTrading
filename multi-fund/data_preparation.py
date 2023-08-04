
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


stock_list = ['518880.SH'] #假设这里面都是港股股票
start = '20210101'
end = '20230731'
dir_name = 'C:/Users/huangtuo/.qlib/qlib_data/fund_data/csv'

with open('C://temp//upload//codefundsecname.json') as file:
    code2secname = json.loads(file.read())

fields = ['ts_code', 'trade_date','open','high','low','close','pre_close','pct_chg','vol','amount']


stk_set =  list(code2secname.keys())

def make_dataset(stock_pool:list, start:str, end:str):
    for index_code in stock_pool:
        if index_code[:2] in ('00','39'):
            #make_dataset(stock_list, start, end)
           df = pro.index_daily(ts_code=index_code, start_date=start,
                                end_date=end, fields=fields)
        if index_code[:2] in ('51', '15'):
           df = pro.fund_daily(ts_code=index_code, start_date=start,
                                end_date=end, fields=fields)
        df['factor']=1
        df['vwap'] = df['amount'] / df['vol']
        df = df.rename(columns={'pct_chg': 'change', 'vol': 'volume'})
        file_name = dir_name + '/' + index_code + '.csv'
        df.to_csv(file_name,header = True, index = False)
make_dataset(stk_set, start, end)


        #df['ts_code'] = df['ts_code'].apply(lambda x: x.split('.')[1] + x.split('.')[0])
        #file_name = dir_name + '/' + code.split('.')[1] + code.split('.')[0] + '.csv'

#全量替换数据
C:\qlib-main\scripts
python dump_bin.py dump_all --csv_path C:\Users\huangtuo\.qlib\qlib_data\fund_data\csv --qlib_dir C:\Users\huangtuo\.qlib\qlib_data\fund_data --symbol_field_name ts_code  --date_field_name trade_date  --include_fields open,high,low,close,pre_close,change,volume,amount,factor,vwap

#增量更新数据

python dump_bin.py dump_update --csv_path C:\Users\huangtuo\.qlib\qlib_data\fund_data\csv --qlib_dir C:\Users\huangtuo\.qlib\qlib_data\fund_data --symbol_field_name ts_code  --date_field_name trade_date  --include_fields open,high,low,close,pre_close,change,pct_chg,vol,amount



