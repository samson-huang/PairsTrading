import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
from send_mail_tool import *
from technical_analysis_patterns import (rolling_patterns2pool,plot_patterns_chart)
from typing import (List, Tuple, Dict, Callable, Union)
from tqdm.notebook import tqdm
import json
#from jqdatasdk import (auth,get_price,get_trade_days,finance,query,get_industries)

import pandas as pd
import numpy as np
import empyrical as ep


import seaborn as sns
import matplotlib as mpl
import mplfinance as mpf
import matplotlib.pyplot as plt
import datetime

test_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//close.pkl')
test_pre_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//pre_close.pkl')
test_high = pd.read_pickle('C://temp//fund_data//base_data//mkt//high.pkl')
test_low = pd.read_pickle('C://temp//fund_data//base_data//mkt//low.pkl')
test_amount = pd.read_pickle('C://temp//fund_data//base_data//mkt//amount.pkl')
test_open = pd.read_pickle('C://temp//fund_data//base_data//mkt//open.pkl')

if __name__ == '__main__':
   # list_sh     上证380   上证180       上证50      沪深300     科创50
   list_sh = ( '000009.SH','000010.SH', '000016.SH', '000300.SH', '000688.SH',
              # 中证1000     中证100   中证500	   中证800
              '000852.SH', '000903.SH', '000905.SH', '000906.SH')
   # 深圳指数   深证成指    中小板指	   创业板指     深证100
   list_sz = ('399001.SZ', '399005.SZ', '399006.SZ', '399330.SZ')

   list_1 = list_sh + list_sz
   list_1 = list(list_1)
   local_datetime = datetime.datetime.now().strftime('%Y%m%d')

   for index_code in list_1:
      #index_code = '000300.SH'

      local_url='C://temp//upload//'+local_datetime+'_'+index_code.replace('.', '')+'_detail.jpg'
      #'close,pre_close,high,low,amount'
      #fields=['trade_date','open', 'close', 'low', 'high']

      dfs = [test_open[index_code],test_close[index_code],test_low[index_code],test_high[index_code]]
      result = pd.concat(dfs,axis=1)
      result.columns = ['open', 'close', 'low', 'high']

      data1=result[-100:]
      data1.index = pd.to_datetime(data1.index)
      data1.sort_index(inplace=True)
      patterns_record1 = rolling_patterns2pool(data1['close'],n=35)

      plot_patterns_chart(data1,patterns_record1,True,False,local_url.replace('detail', 'overall'))
      plt.title(index_code.replace('.', ''))
      plot_patterns_chart(data1,patterns_record1,True,True,local_url);


   # 邮件发送
   local_url_mail = 'C://temp//upload//' + local_datetime
   recer = ["tianfangfang1105@126.com","huangtuo02@163.com", ]
   send_fundmail=send_mail_tool(_recer=recer,local_url=local_url_mail).action_send()