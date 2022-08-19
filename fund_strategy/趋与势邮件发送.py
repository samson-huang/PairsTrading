from typing import (Tuple,List,Callable,Union,Dict)

import pandas as pd
import numpy as np
import empyrical as ep
from collections import (defaultdict,namedtuple)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import json
import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
from trend_model_tool import *
import mpl_finance as mpf

from matplotlib import ticker
from matplotlib.pylab import date2num

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

index_code = '000300.SH'
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

   with open('C://temp//upload//codefundsecname.json') as file:
      code2secname = json.loads(file.read())

   for index_code in list_1:
        txt_url ='C://temp//upload//'+ local_datetime + '_trend//readme.txt'
        local_url = 'C://temp//upload//'+ local_datetime + '_trend//' + local_datetime + '_' + code2secname[index_code] + '_trend.jpg'
        dfs = [test_open[index_code], test_close[index_code], test_low[index_code], test_high[index_code]]
        result = pd.concat(dfs, axis=1)
        result.columns = ['open', 'close', 'low', 'high']

        price = result[-500:]
        price.index = pd.to_datetime(price.index)
        price.sort_index(inplace=True)
        price=price.dropna(axis=0,how='any')

        pd.DataFrame.plot.ochl = plot_ochl

        score = price['close'].rolling(60).apply(calc_trend_score,raw=False)
        lower_bound = score.rolling(20).apply(lambda x: x.quantile(0.05),raw=False)
        upper_bound = score.rolling(20).apply(lambda x: x.quantile(0.85),raw=False)
        df = pd.concat((score,upper_bound,lower_bound),axis=1)
        df.columns = ['score','upper_bound','lower_bound']

        flag = get_hold_flag(df)
        next_ret = price['close'].pct_change().shift(-1)
        algorithms_ret = flag * next_ret.loc[flag.index]

        algorithms_cum = ep.cum_returns(algorithms_ret)

        fig,axes = plt.subplots(2,figsize=(18,12))

        axes[0].set_title('指数')
        (price['close']/price['close'][0]).plot(ax=axes[0])
        axes[1].set_title('净值')
        flag.plot(ax=axes[0],secondary_y=True,ls='--',color='darkgray')
        algorithms_cum.plot(ax=axes[1]);
        fig.savefig(local_url)
        #######################################
        #评测模型

        test123=price['close'].pct_change()*100
        test123.columns = ['pct_chg']
        new_flag=flag.to_frame()
        test4=pd.merge(test123,new_flag,how='inner', left_index=True, right_index=True)
        test4.columns=['pct_chg','trend_MARK']
        test4=test4.dropna(axis=0,how='any')
        writing_text=summary(test4)
        with open(txt_url, "a") as file:
            file.write("\n"+"#############"+local_datetime + '_' + code2secname[index_code]+"##############")
            file.write("\n"+new_flag.tail(20).T)
            file.write("\n" + writing_text.T)
            file.write("\n" + "###########################")

