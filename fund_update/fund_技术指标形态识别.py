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

import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
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

test_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//close.pkl')
test_pre_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//pre_close.pkl')
test_high = pd.read_pickle('C://temp//fund_data//base_data//mkt//high.pkl')
test_low = pd.read_pickle('C://temp//fund_data//base_data//mkt//low.pkl')
test_amount = pd.read_pickle('C://temp//fund_data//base_data//mkt//amount.pkl')
test_open = pd.read_pickle('C://temp//fund_data//base_data//mkt//open.pkl')

index_code = '000300.SH'
#'close,pre_close,high,low,amount'
#fields=['trade_date','open', 'close', 'low', 'high']
dfs = [test_open[index_code],test_close[index_code],test_low[index_code],test_high[index_code]]
result = pd.concat(dfs,axis=1)
result.columns = ['open', 'close', 'low', 'high']

data1=result[-365:]
data1.index = pd.to_datetime(data1.index)
data1.sort_index(inplace=True)
patterns_record1 = rolling_patterns2pool(data1['close'],n=35)
plot_patterns_chart(data1,patterns_record1,True,False)
plt.title('沪深300')
plot_patterns_chart(data1,patterns_record1,True,True);


