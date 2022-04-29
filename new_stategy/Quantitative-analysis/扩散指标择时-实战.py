# coding: utf-8
import talib
import pandas as pd
import numpy as np

import itertools
import operator # 运算符代替库
#from jqdata import *

import seaborn as sns
from pylab import mpl

import matplotlib.pyplot as plt
import datetime as datetime

plt.rcParams['font.family'] = 'serif'

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn')

import sys 
sys.path.append("G://GitHub//PairsTrading//new_stategy//foundation_tools//") 
import foundation_tushare 
from Creat_RSRS import (RSRS,rolling_apply)  # 自定义信号生成
import json

# 使用ts
#test123 = my_pro.index_basic(market='SSE')
#test123[test123.name.str.contains('上证50')]
#000016.SH 上证50	  
#
# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
my_pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)





###################################################################################
# 回测框架
# 获取信号进行回测
class DiffusionIndicatorBackTest(object):
    '''
    close_df:index-date,columns-codes
    mkt_cap_df:index-date,columns-codes
    singal_ser:pd.Series或者func 在method参数为ls时 singal_ser需要为func
    cal_func:平滑N1,N2的计算方式
    N,N1,N2:如研报描述
    method:long多头,short空头,ls多空
    ==================
    return dict
        fast_ma:pd.Series
        slow_ma:pd.Series
        flag:np.array 1为多头 0为无持仓 -1为空头
        algorithm_returns:策略收益率
        algorithm_cum:策略净值
    '''

    def __init__(self, symbol: str, singal_ser: pd.Series, start_date: str,
                 end_date: str, cal_func, N1: int, N2: int, method: str):

        self.symbol = symbol
        self.singal_func = singal_ser
        self.start_date = start_date
        self.end_date = end_date
        self.cal_func = cal_func
        self.N1 = N1
        self.N2 = N2
        self.method = method
        
        self.singal = pd.Series()
        self.index_price = pd.Series()
        self.algorithm_returns = pd.Series()  # 策略收益
        self.algorithm_cum = pd.Series()  # 策略净值

        self.flag = np.array([])  # 持仓标记
        self.fast_ma = pd.Series()
        self.slow_ma = pd.Series()

    def backtest(self):

        # 获取指数下期收益率用于回测
        offset_end = TdaysOffset(self.end_date, 1)

        # 数据获取
        '''
        close_df = get_price(
            self.symbol,
            self.start_date,
            offset_end,
            fields=['close', 'pre_close'],
            panel=False)
        '''
        close_df = foundation_tushare.distributed_query(my_pro.daily,self.symbol,
             self.start_date,self.end_date,
             'trade_date,ts_code,close,pre_close',4500)
        close_df.index=close_df['trade_date']
        close_df=close_df.sort_index()
        # 计算收益率
        pct_chg = close_df['close'] / close_df['pre_close'] - 1
        next_ret = pct_chg.shift(-1)
        next_ret = next_ret.loc[self.start_date:self.end_date]

        # 计算指数净值
        self.index_price = close_df['close']
        
        # 多空
        if self.method == 'ls':
            
            self._CalLs(next_ret) # 多空的计算处理
            

        else:
            self._checksingalfunc()
            self._GetFlag()  # 获取持仓标记

            self.CalAlgorithmCum(next_ret)  # 获取策略净值
    
    
    # 计算多空
    def _CalLs(self,next_ret):
        
        n = len(next_ret)
        flag = np.zeros((2,n))
        ret = np.zeros(n)
        
        for i,md in enumerate(['long','short']):
            
            self.method = md
            self._checksingalfunc()     # 获取信号
            self._GetFlag() # 获取持仓标记
            self.CalAlgorithmCum(next_ret)
        
            ret += self.algorithm_returns # 多空收益
            flag[i] = self.flag
            
        self.flag = flag
        self.algorithm_returns = ret
        self.algorithm_cum = (1 + self.algorithm_returns).cumprod() # 计算累计收益率
        
        self.method = 'ls' # 将其该会原始状态
        
    # 检查singal_ser格式
    def _checksingalfunc(self):
        
        if hasattr(self.singal_func, '__call__') :
            
            self.singal = self.singal_func(self.method)
            
        elif self.method and self.method != 'ls':
            
            self.singal = self.singal_func
            
        else:
            
            raise ValueError('多空方法下singal_ser需要传入计算方法而非pd.Series')
            

    # 标记持仓
    def _GetFlag(self) -> np.array:

        #fast_ma = self.singal_ser.rolling(self.N1).mean()
        #slow_ma = fast_ma.rolling(self.N2).mean()

        fast_ma = self.cal_func(self.singal, self.N1)
        slow_ma = self.cal_func(fast_ma, self.N2)

        self.fast_ma = fast_ma.loc[self.start_date:self.end_date]
        self.slow_ma = slow_ma.loc[self.start_date:self.end_date]

        if self.method == 'long':

            self.flag = np.where(self.fast_ma > self.slow_ma, 1, 0)

        elif self.method == 'short':

            self.flag = np.where(self.fast_ma > self.slow_ma, -1, 0)

        else:

            raise ValueError('多空参数仅能为:long,short')

    # 计算收益率等风险指标
    def CalAlgorithmCum(self, next_ret: pd.Series):
    
        algorithm_returns = self.flag * next_ret
        algorithm_cum = (1 + algorithm_returns).cumprod()

        self.algorithm_returns = algorithm_returns
        self.algorithm_cum = algorithm_cum

        
    # 获取收益指标
    def GetRisk(self) -> dict:

        # 年化收益率
        annual_algo_return = pow(self.algorithm_cum[-1] / self.algorithm_cum[0],
                                 250 / len(self.algorithm_cum)) - 1

        # 策略波动率
        algorithm_volatility = self.algorithm_returns.std() * np.sqrt(250)

        # 夏普
        sharpe = (annual_algo_return - 0.04) / algorithm_volatility

        # 最大回撤点，最大回撤
        md_p, md_r = self._GetMaxDrawdown()
        md_date_range = '{}-{}'.format(
            self.algorithm_cum.index[md_p[1]].strftime('%Y%m%d'),
            self.algorithm_cum.index[md_p[0]].strftime('%Y%m%d'))
                
        # 交易次数，胜率
        trade, winratio = self.TradeCount()

        return {
            '交易次数': trade,
            '胜率': winratio,
            '累计净值': round(self.algorithm_cum[-1], 4),
            '年化收益率': round(annual_algo_return, 4),
            '年化波动率': round(algorithm_volatility, 4),
            '夏普': round(sharpe, 4),
            '最大回撤': round(md_r, 4),
            '回撤时间': md_date_range,
            '收益回撤比': round(annual_algo_return / md_r, 4)
        }

    # 获取最大回撤
    def _GetMaxDrawdown(self) -> tuple:
        '''
        algorithm_cum
        ===========
        return 最大回撤位置,最大回撤
        '''
        arr = self.algorithm_cum.values
        i = np.argmax((np.maximum.accumulate(arr) - arr) /
                      np.maximum.accumulate(arr))  # end of the period
        j = np.argmax(arr[:i])  # start of period
        return ([i, j], (1 - arr[i] / arr[j]))  # 回撤点，回撤比率
    
    # 获取交易次数、胜率
    def TradeCount(self):
        
        if self.method == 'ls':
            
            num = 0
            w_num = 0
            
            for flag in self.flag:
                
                open_num,win, wincount = self._GetWinCount(flag)
                num += open_num
                w_num += win
                
            return num, w_num / num # 交易次数，胜率
        
        else:
            
            open_num,win, wincount = self._GetWinCount(self.flag)
            
            return open_num,wincount
        
    # TradeCound的底层计算
    def _GetWinCount(self,flag):
        '''
        统计
        '''  
            
        flag = np.abs(flag)
        # 交易次数 1为开仓 -1为平仓
        trade_num = flag - np.insert(flag[:-1], 0, 0)

        # 交易次数
        open_num = sum(trade_num[trade_num == 1])

        # 统计开仓收益
        temp_df = pd.DataFrame({
            'flag': flag,
            'algorithm_returns': self.algorithm_returns
        })

        temp_df['mark'] = (temp_df['flag'] != temp_df['flag'].shift(1))
        temp_df['mark'] = temp_df['mark'].cumsum()

        # 开仓至平仓的持有收益
        tradecumsumratio = temp_df.query('flag==1').groupby(
            'mark')['algorithm_returns'].sum()
        win = len(tradecumsumratio[tradecumsumratio > 0])

        wincount = round(win / open_num, 4)

        return open_num,win, wincount  # 交易次数，盈利次数,胜率


###################################################################################

# 画信号图
def IndicatorPlot(index_price: pd.Series, fast_ma: pd.Series,
                  slow_ma: pd.Series, flag: np.array):
    '''
    index_price:index为date value为指数价格
    fast_ma:index为date value为快指标
    slow_ma:index为date value为慢指标
    ==========
    return 图表
    '''
    plt.rcParams['font.family'] = 'serif'

    # 标记点
    idx = list(range(len(flag)))
    marks = np.where(flag == 1, idx, 0)
    marks = marks[marks != 0].tolist()

    # 画图
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('持仓标记')

    index_price.plot(markevery=marks, marker='o', mfc='r', ms=2)
    plt.legend()

    ax2 = fig.add_subplot(212)
    ax2.set_title('信号')
    fast_ma.plot()
    slow_ma.plot()
    plt.legend(['fast_ma', 'slow_ma'])
    plt.show()


# 画净值
def CumPlot(index_price: pd.Series, algorithm_cum: pd.Series, title: str):
    '''
    index_price:index为date value为指数价格
    algorithm_cum:index为date value为净值
    title:图表标题
    ==========
    return 图表
    '''
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(18, 8))
    plt.title(title)

    if isinstance(algorithm_cum, list):
        for cum in algorithm_cum:

            cum.plot(
                markevery=GetMaxDrawdown(cum),
                marker='^',
                mec='black',
                mew=2,
                mfc='g',
                ms=7,
                label=cum.name)
        (index_price / index_price[0]).plot(
            ls='--', color='SteelBlue', alpha=0.6, label='benchmark_return')
        plt.legend()

    else:

        algorithm_cum.plot(
            markevery=GetMaxDrawdown(algorithm_cum),
            marker='^',
            mec='black',
            mew=2,
            mfc='g',
            ms=7)

        (index_price / index_price[0]).plot(
            ls='--', color='SteelBlue', alpha=0.6)
        plt.legend(['algorithm_return', 'benchmark_return'])


###################################################################################

def GetMaxDrawdown(algorithm_cum) -> list:
    '''
        algorithm_cum
        ===========
        return 最大回撤位置
        '''
    arr = algorithm_cum.values
    i = np.argmax((np.maximum.accumulate(arr) - arr) /
                  np.maximum.accumulate(arr))  # end of the period
    j = np.argmax(arr[:i])  # start of period
    return [i, j]  # 回撤点


def TdaysOffset(end_date: str, count: int) -> int:
    '''
    end_date:为基准日期
    count:为正则后推，负为前推
    -----------
    return datetime.date
    '''
    df = my_pro.trade_cal(exchange='SSE')
    test_2=df[df['cal_date']==end_date].index[0]
    if count > 0:
        df1=df[test_2:]
        df1=df1[df1['is_open']==1]
        df1=df1.reset_index(drop=True)
        return df1[df1.index==count]['cal_date'].values[0]
    elif count < 0:
        df1=df[:test_2+1]
        df1=df1[df1['is_open']==1]
        df1=df1.reset_index(drop=True)
        count_df1=len(df1)+count
        return df1[df1.index==count_df1]['cal_date'].values[0]
    else:

        raise ValueError('别闹！')            
    '''   
    trade_date = get_trade_days(end_date=end_date, count=1)[0]

    if count > 0:
        # 将end_date转为交易日

        trade_cal = get_all_trade_days().tolist()

        trade_idx = trade_cal.index(trade_date)

        return trade_cal[trade_idx + count]

    elif count < 0:

        return get_trade_days(end_date=trade_date, count=abs(count))[0]
    
    else:

        raise ValueError('别闹！')
    '''    
'''
import datetime
from dateutil.parser import parse
# ts的日历需要处理一下才会返回成交日列表
## 减少ts调用 改用jq的数据....
def query_trade_dates(start_date: str, end_date: str) -> list:
    start_date = parse(start_date).strftime('%Y%m%d')
    end_date = parse(end_date).strftime('%Y%m%d')
    ############################
    df = my_pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    dates = df.query('is_open==1')['cal_date'].values.tolist()
    return  dates
test123=query_trade_dates(start_date, end_date)
'''
###################################################################################

# 获取MA扩散指标信号
def CreatMaSingal(index_symbol: str, start_date: str, end_date: str, N, N1,
                  weight_method: str, method: str,close_df1) -> pd.Series:
    '''
    N,N1的作用只是去前序计算期
    method:short空头,long多头
    weight_method:avg等权,mktcap市值加权
    '''
    # 数据计算前序期
    begin_date = TdaysOffset(
        start_date,
        -(N + N1),
    )

    # 成分股使用end_date时点成分股
    #security_list = get_index_stocks(index_symbol, date=end_date)000016.SH
    #security_list =my_pro.index_weight(index_code=index_symbol, trade_date=end_date)
    #security_list =my_pro.index_weight(index_code='000016.SH', start_date='20220331',end_date='20220331')
    ## 前复权数据
    #security_list_temp=security_list['con_code'].to_list()
    #st = ','.join([str(s) for s in security_list_temp])
    #close_df = my_pro.daily(ts_code=st, start_date=begin_date,
    # end_date=end_date)
    '''
    close_df = foundation_tushare.distributed_query(my_pro.daily,st,
                                   start_date,
                                   end_date,'trade_date,ts_code,close',4500) 
    #close_df = get_price(
    #    security_list, begin_date, end_date, fields='close', panel=False)
    close_df.to_csv('C:\\temp\\close_df.csv')
    '''

    close_df = close_df1
    close_df = pd.pivot_table(
        close_df, index='trade_date', columns='ts_code', values='close')

    # 计算均线
    ma_df = close_df.rolling(N).mean()
    action = {'long': operator.gt, 'short': operator.lt}

    if weight_method == 'avg':

        return (action[method](close_df, ma_df)).sum(axis=1)

    elif weight_method == 'mktcap':

        ## 获取流通市值
        mak_cap_df = GetValuation(
            st, start_date=begin_date, end_date=end_date)
        '''   
        mak_cap_df = pd.pivot_table(
            mak_cap_df,
            index='day',
            columns='code',
            values='circulating_market_cap')
        '''     
        mak_cap_df.to_csv('C:\\temp\\mak_cap_df.csv')
        mak_cap_df = pd.pivot_table(
            mak_cap_df,
            index='trade_date',
            columns='ts_code',
            values='circ_mv')
        
        weights = mak_cap_df.apply(lambda x: x / x.sum(), axis=1).fillna(0)
        return (action[method](close_df, ma_df) * weights).sum(axis=1)

# 获取ROC扩散指标信号
def CreatROCSingal(index_symbol: str, start_date: str, end_date: str, N, N1,
                  weight_method: str, method: str) -> pd.Series:
    '''
    N,N1的作用只是去前序计算期
    method:short空头,long多头
    weight_method:avg等权,mktcap市值加权
    '''
    # 数据计算前序期
    begin_date = TdaysOffset(
        start_date,
        -(N + N1),
    )

    # 成分股使用end_date时点成分股
    #security_list = get_index_stocks(index_symbol, date=end_date)
    security_list =my_pro.index_weight(index_code=index_symbol, trade_date=end_date)
    
    ## 前复权数据
    '''
    close_df = get_price(
        security_list, begin_date, end_date, fields='close', panel=False)
    close_df = pd.pivot_table(
        close_df, index='time', columns='code', values='close')
    '''
    '''
    close_df = foundation_tushare.distributed_query(my_pro.daily,st,
                                   start_date,
                                   end_date,'trade_date,ts_code,close',4500)
    '''
    mak_cap_df.to_csv('C:\\temp\\mak_cap_df.csv')
    close_df = pd.pivot_table(
        close_df, index='trade_date', columns='ts_code', values='close') 
    #######                                           
    # 计算roc
    pct_chg = close_df.pct_change(N)
    action = {'long': operator.gt, 'short': operator.lt}

    if weight_method == 'avg':

        return (action[method](pct_chg, 0)).sum(axis=1)

    elif weight_method == 'mktcap':

        ## 获取流通市值
        mak_cap_df = GetValuation(
            security_list, start_date=begin_date, end_date=end_date)
        mak_cap_df = pd.pivot_table(
            mak_cap_df,
            index='trade_date',
            columns='ts_code',
            values='circulating_market_cap')

        weights = mak_cap_df.apply(lambda x: x / x.sum(), axis=1).fillna(0)
        return (action[method](pct_chg, 0) * weights).sum(axis=1)

# 获取KDJ扩散指标信号
def CreatKDJSingal(index_symbol: str, start_date: str, end_date: str, N, N1,
                  weight_method: str, method: str) -> pd.Series:
    '''
    N,N1的作用只是去前序计算期
    method:short空头,long多头
    weight_method:avg等权,mktcap市值加权
    '''
    # 数据计算前序期
    begin_date = TdaysOffset(
        start_date,
        -(N + N1),
    )

    # 成分股使用end_date时点成分股
    security_list = get_index_stocks(index_symbol, date=end_date)

    ## 前复权数据
    close_df = get_price(
        security_list, begin_date, end_date, fields=['close','high','low'], panel=False)
    
    # 计算获取kd值
    kd = close_df.groupby('code',group_keys=False).apply(cal_kd,fastk_period=N)
    df = pd.concat([close_df[['time','code']],kd],axis=1)

    # 获取k,d
    slowk = pd.pivot_table(df,index='time',columns='code',values='k')
    slowd = pd.pivot_table(df,index='time',columns='code',values='d')
    
    action = {'long': operator.gt, 'short': operator.lt}

    if weight_method == 'avg':

        return (action[method](slowk, slowd)).sum(axis=1)

    elif weight_method == 'mktcap':

        ## 获取流通市值
        mak_cap_df = GetValuation(
            security_list, start_date=begin_date, end_date=end_date)
        
        mak_cap_df = pd.pivot_table(
            mak_cap_df,
            index='day',
            columns='code',
            values='circulating_market_cap')

        weights = mak_cap_df.apply(lambda x: x / x.sum(), axis=1).fillna(0)
        return (action[method](slowk, slowd) * weights).sum(axis=1)
    
# 计算KD
def cal_kd(df,fastk_period:int):
    
    k,d = talib.STOCH(df['high'],
                      df['low'],
                      df['close'],
                      fastk_period=fastk_period, 
                      slowk_period=3, 
                      slowd_period=3)
    
    df = pd.concat([k,d],axis=1)
    df.columns = ['k','d']
    return df

# 获取RSI扩散指标信号
def CreatRSISingal(index_symbol: str, start_date: str, end_date: str, N, N1,
                  weight_method: str, method: str) -> pd.Series:
    '''
    N,N1的作用只是去前序计算期
    method:short空头,long多头
    weight_method:avg等权,mktcap市值加权
    '''
    # 数据计算前序期
    begin_date = TdaysOffset(
        start_date,
        -(N + N1),
    )

    # 成分股使用end_date时点成分股
    security_list = get_index_stocks(index_symbol, date=end_date)

    ## 前复权数据
    close_df = get_price(
        security_list, begin_date, end_date, fields='close', panel=False)
    
    # 计算获取RSI
    close_df['rsi'] = close_df.groupby('code')['close'].transform(talib.RSI,timeperiod=N)
    rsi = pd.pivot_table(close_df,index='time',columns='code',values='rsi')
    
    action = {'long': operator.gt, 'short': operator.lt}
    limit = {'long':70,'short':30}
    
    if weight_method == 'avg':
    
        return (action[method](rsi, limit[method])).sum(axis=1)

    elif weight_method == 'mktcap':

        ## 获取流通市值
        mak_cap_df = GetValuation(
            security_list, start_date=begin_date, end_date=end_date)
        
        mak_cap_df = pd.pivot_table(
            mak_cap_df,
            index='day',
            columns='code',
            values='circulating_market_cap')

        weights = mak_cap_df.apply(lambda x: x / x.sum(), axis=1).fillna(0)
        
        return (action[method](rsi, limit[method]) * weights).sum(axis=1)
    
# 突破N日高点
def CreatStageHighSingal(index_symbol: str, start_date: str, end_date: str, N, N1,
                  weight_method: str, method: str) -> pd.Series:
    '''
    N,N1的作用只是去前序计算期
    method:short空头,long多头
    weight_method:avg等权,mktcap市值加权
    '''
    # 数据计算前序期
    begin_date = TdaysOffset(
        start_date,
        -(N + N1),
    )

    # 成分股使用end_date时点成分股
    security_list = get_index_stocks(index_symbol, date=end_date)

    ## 前复权数据
    data = get_price(
        security_list, begin_date, end_date, fields=['close','pre_close'], panel=False)
    
    close_df = pd.pivot_table(
        data, index='time', columns='code', values='close')
    
    pre_close = pd.pivot_table(
        data, index='time', columns='code', values='pre_close')
    
    roll_max = pre_close.rolling(N).max()
    
    # 计算均线
    action = {'long': operator.gt, 'short': operator.lt}

    if weight_method == 'avg':

        return (action[method](close_df, roll_max)).sum(axis=1)

    elif weight_method == 'mktcap':

        ## 获取流通市值
        mak_cap_df = GetValuation(
            security_list, start_date=begin_date, end_date=end_date)
        mak_cap_df = pd.pivot_table(
            mak_cap_df,
            index='day',
            columns='code',
            values='circulating_market_cap')

        weights = mak_cap_df.apply(lambda x: x / x.sum(), axis=1).fillna(0)
        return (action[method](close_df, roll_max) * weights).sum(axis=1)
  
# 市值数据获取
def GetValuation(symbol: list, start_date: str, end_date: str) -> pd.DataFrame:

    #dates = get_trade_days(start_date, end_date)
    #dates = GetEveryDay(start_date, end_date)
    '''    
    limit = 10000
    n_symbols = len(symbol)
    n_days = len(dates)

    if n_symbols * n_days > limit:

        n = limit // n_symbols
        df_list = []
        i = 0
        pos1, pos2 = n * i, n * (i + 1) - 1

        while pos2 < n_days:

            df = get_valuation(
                symbol,
                start_date=dates[pos1],
                end_date=dates[pos2],
                fields='circulating_market_cap')


                                   
            df_list.append(df)
            i += 1
            pos1, pos2 = n * i, n * (i + 1) - 1
        if pos1 < n_days:
            df = get_valuation(
                symbol,
                start_date=dates[pos1],
                end_date=dates[-1],
                fields='circulating_market_cap')
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
    else:
        df = get_valuation(
            symbol,
            start_date=start_date,
            end_date=end_date,
            fields='circulating_market_cap')
    '''            
    df = foundation_tushare.distributed_query(
         my_pro.daily_basic,symbol,
         start_date=start_date,
         end_date=end_date,
         fields='trade_date,ts_code,circ_mv')
    return df

####################################################################
def GetEveryDay(begin_date,end_date):
#获得两日期间的日期列表
    global date_list
    date_list = []
    begin_date = datetime.strptime(begin_date,"%Y%m%d")
    end_date = datetime.strptime(end_date,"%Y%m%d")
    while begin_date <= end_date:
          date_str = begin_date.strftime("%Y%m%d")
          date_list.append(date_str)
          begin_date += datetime.timedelta(days=1)
    #print('日期列表已形成')
    return date_list
###################################################################################

# 网格寻参大法
def GetGridRiskReport(N: int, singal_ser: pd.Series,N1S:int=20,N2S:int=10):

    risk_list = []
    for j in range(20, N, 10):
        for i in range(10, j, 5):

            bt = DiffusionIndicatorBackTest(
                #symbol=index_symbol,
                symbol=st,
                singal_ser=singal_ser,
                start_date=start_date,
                end_date=end_date,
                cal_func=talib.MA,
                N1=j,
                N2=i,
                method='long')

            bt.backtest()

            risk = pd.Series(bt.GetRisk())
            risk.name = '{}_{}_{}'.format(N, j, i)
            risk_list.append(risk)

    report_df = pd.DataFrame(risk_list)
    report_df.index.names = ['N|N1|N2']

    return report_df


# 目标指数
index_symbol = '000300.SH'

# 复现时间设定
#start_date, end_date = '20022401', '20220425'
# 复现时间设定
start_date, end_date = '20070101', '20200430'

security_list =my_pro.index_weight(index_code=index_symbol, trade_date=end_date)
security_list_temp=security_list['con_code'].to_list()
st = ','.join([str(s) for s in security_list_temp])
close_df1 = foundation_tushare.distributed_query(my_pro.daily,st,
                                                start_date, end_date,
                                                'trade_date,ts_code,close,pre_close', 4500)

# 计算等权信号
ma_avg_long = CreatMaSingal(index_symbol,start_date,end_date,160,150,'avg','long',close_df1)

# 计算加权信号
ma_mktcap_long = CreatMaSingal(index_symbol,start_date,end_date,160,150,'mktcap','long',close_df1)

# 网格寻参
report_df = GetGridRiskReport(160,ma_avg_long)
# 查看夏普胜率最高的前5组
report_df.sort_values(['夏普','胜率'],ascending=False).head()

# 等权多头
ma = DiffusionIndicatorBackTest(
    #symbol=index_symbol,
    symbol=st,
    singal_ser=ma_avg_long,
    start_date=start_date,
    end_date=end_date,
    cal_func=talib.MA,
    N1=150,
    N2=25,
    method='long')

ma.backtest()  # 回测

index_price = getattr(ma,'index_price')
fast_ma = getattr(ma,'fast_ma')
slow_ma = getattr(ma,'slow_ma')
flag = getattr(ma,'flag')
ma_avg_long = getattr(ma,'algorithm_cum')
ma_avg_long.name = '等权多头'

IndicatorPlot(index_price,fast_ma,slow_ma,flag)
CumPlot(index_price,ma_avg_long,'等权多头收益')

# 查看回测情况
pd.DataFrame(ma.GetRisk(),index=['160_150_25'])

#多空情况
def Singal(method):
    return CreatMaSingal(index_symbol, start_date, end_date, 160, 150, 'avg', method)


risk_list = []
cum_list = []
flag_list = []

for label, method in zip(['等权多头', '等权空头', '等权多空'], ['long', 'short', 'ls']):
    print(label)
    # 加权多头
    ma = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=Singal,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=150,
        N2=25,
        method=method)

    ma.backtest()  # 回测

    cum = getattr(ma, 'algorithm_cum')
    cum.name = label

    rept = pd.Series(ma.GetRisk())
    rept.name = label

    flag_list.append(getattr(ma, 'flag'))
    risk_list.append(rept)
    cum_list.append(cum)


pd.DataFrame(risk_list)

CumPlot(getattr(ma,'index_price'), cum_list, '等权MA扩散指标纯多、多空、纯空净值曲线')

#流通市值加权处理
# 网格寻参
report_df = GetGridRiskReport(160,ma_mktcap_long)
# 查看夏普胜率最高的前5组
report_df.sort_values(['夏普','胜率'],ascending=False).head()

# 加权多头
ma1 = DiffusionIndicatorBackTest(
    symbol=index_symbol,
    singal_ser=ma_mktcap_long,
    start_date=start_date,
    end_date=end_date,
    cal_func=talib.MA,
    N1=80,
    N2=30,
    method='long')

ma1.backtest()  # 回测

index_price = getattr(ma1,'index_price')
fast_ma = getattr(ma1,'fast_ma')
slow_ma = getattr(ma1,'slow_ma')
flag = getattr(ma1,'flag')
ma_mktcap_long = getattr(ma1,'algorithm_cum')
ma_mktcap_long.name = '加权多头'

IndicatorPlot(index_price,fast_ma,slow_ma,flag)
CumPlot(index_price,ma_mktcap_long,'加权多头收益')

# 查看回测情况
pd.DataFrame(ma1.GetRisk(),index=['160_80_30'])

#多空情况
def Singal(method):
    return CreatMaSingal(index_symbol, start_date, end_date, 160, 150, 'mktcap', method)


risk_list = []
cum_list = []
flag_list = []

for label, method in zip(['流通市值加权多头', '流通市值加权空头', '流通市值加权多空'], ['long', 'short', 'ls']):
    print(label)
    # 加权多头
    ma = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=Singal,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=80,
        N2=30,
        method=method)

    ma.backtest()  # 回测

    cum = getattr(ma, 'algorithm_cum')
    cum.name = label

    rept = pd.Series(ma.GetRisk())
    rept.name = label

    flag_list.append(getattr(ma, 'flag'))
    risk_list.append(rept)
    cum_list.append(cum)


pd.DataFrame(risk_list)

CumPlot(getattr(ma,'index_price'), cum_list, '流通市值加权MA扩散指标纯多、多空、纯空净值曲线')


##################################################################################
#ROC指标
'''
ROC(Price Rate of Change)又称变动率指标，是以今天的收盘价比较其 N 天前的收盘 价的差除以 N 天前的收盘（本质其实是 N 日 K 线的涨幅）
进行计算的。是另一种 常用的构建扩散指标的技术指标。 判断个股多空头状态方法：对成分股价格计算其 N 日 ROC，当 ROC 值大于 0 时认 
为个股处于多头状态，当 ROC 值小于 0 时认为个股处于空头状态。 这里我们使用沪深300指数计算扩散指标，并且使用移动平均的方法进行两次平滑。 
一共有三个参数，N，N1，N2，计算 N 日 ROC，第一次平滑计算 N1 日移动平均线， 第二次平滑计算 N2 日移动平均线。参数的范围：
N [60，250），单位为天，每隔 10 天取值；N1 [20，N），每取一个 N 值，N1 取值 20到 N 之间的数，每隔 10 天取值；N2 [10，N1），
每隔5 天取值。其他的处理方式和使用MA指标时的方式一样。
'''
#数据获取
# 目标指数
index_symbol = '000300.XSHG'

# 复现时间设定
start_date, end_date = '2007-01-01', '2020-04-30'
# 计算等权信号
#roc_avg_long = CreatROCSingal(index_symbol,start_date,end_date,100,90,'avg','long')

# 计算加权信号
roc_mktcap_long = CreatMaSingal(index_symbol,start_date,end_date,100,90,'mktcap','long')

#等权处理
# 网格寻参
report_df = GetGridRiskReport(100,roc_avg_long)
# 查看夏普胜率最高的前5组
report_df.sort_values(['夏普','胜率'],ascending=False).head()

#根据上面夏普及胜率优化结果N1=90,N2=30
# 等权多头
roc = DiffusionIndicatorBackTest(
    symbol=index_symbol,
    singal_ser=roc_avg_long,
    start_date=start_date,
    end_date=end_date,
    cal_func=talib.MA,
    N1=90,
    N2=30,
    method='long')

roc.backtest()  # 回测

index_price = getattr(roc,'index_price')
fast_ma = getattr(roc,'fast_ma')
slow_ma = getattr(roc,'slow_ma')
flag = getattr(roc,'flag')
roc_avg_long = getattr(roc,'algorithm_cum')
roc_avg_long.name = '等权多头'

IndicatorPlot(index_price,fast_ma,slow_ma,flag)
CumPlot(index_price,roc_avg_long,'等权多头收益')

# 查看回测情况
pd.DataFrame(ma.GetRisk(),index=['100_90_30'])

#多空情况
def Singal(method):
    return CreatROCSingal(index_symbol, start_date, end_date, 100, 90, 'avg', method)


risk_list = []
cum_list = []
flag_list = []

for label, method in zip(['等权多头', '等权空头', '等权多空'], ['long', 'short', 'ls']):
    print(label)
    # 加权多头
    roc = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=Singal,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=90,
        N2=30,
        method=method)

    roc.backtest()  # 回测

    cum = getattr(roc, 'algorithm_cum')
    cum.name = label

    rept = pd.Series(roc.GetRisk())
    rept.name = label

    flag_list.append(getattr(roc, 'flag'))
    risk_list.append(rept)
    cum_list.append(cum)


pd.DataFrame(risk_list)

CumPlot(getattr(roc,'index_price'), cum_list, '等权权ROC扩散指标纯多、多空、纯空净值曲线')

#流通市值加权处理
# 网格寻参
report_df = GetGridRiskReport(100,roc_mktcap_long)
# 查看夏普胜率最高的前5组
report_df.sort_values(['夏普','胜率'],ascending=False).head()


# 等权多头
roc = DiffusionIndicatorBackTest(
    symbol=index_symbol,
    singal_ser=roc_mktcap_long,
    start_date=start_date,
    end_date=end_date,
    cal_func=talib.MA,
    N1=90,
    N2=80,
    method='long')

roc.backtest()  # 回测

index_price = getattr(roc,'index_price')
fast_ma = getattr(roc,'fast_ma')
slow_ma = getattr(roc,'slow_ma')
flag = getattr(roc,'flag')
roc_avg_long = getattr(roc,'algorithm_cum')
roc_avg_long.name = '等权多头'

IndicatorPlot(index_price,fast_ma,slow_ma,flag)
CumPlot(index_price,roc_avg_long,'等权多头收益')

# 查看回测情况
pd.DataFrame(roc.GetRisk(),index=['100_90_80'])

#多空情况
def Singal(method):
    return CreatROCSingal(index_symbol, start_date, end_date, 100, 90, 'mktcap', method)


risk_list = []
cum_list = []
flag_list = []

for label, method in zip(['流通市值加权多头', '流通市值加权空头', '流通市值加权多空'], ['long', 'short', 'ls']):
    print(label)
    # 加权多头
    roc = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=Singal,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=90,
        N2=80,
        method=method)

    roc.backtest()  # 回测

    cum = getattr(roc, 'algorithm_cum')
    cum.name = label

    rept = pd.Series(roc.GetRisk())
    rept.name = label

    flag_list.append(getattr(roc, 'flag'))
    risk_list.append(rept)
    cum_list.append(cum)


pd.DataFrame(risk_list)

CumPlot(getattr(roc,'index_price'), cum_list, '流通市值加权ROC扩散指标纯多、多空、纯空净值曲线')


##################################################################################################
#KDJ扩散指标

#数据获取

# 目标指数
index_symbol = '000300.XSHG'

# 复现时间设定
start_date, end_date = '2007-01-01', '2020-04-30'
# 计算等权信号
kdj_avg_long = CreatKDJSingal(index_symbol,start_date,end_date,270,240,'avg','long')

# 计算加权信号
kdj_mktcap_long = CreatKDJSingal(index_symbol,start_date,end_date,270,240,'mktcap','long')

# 网格寻参
report_df = GetGridRiskReport(270,kdj_mktcap_long,80,40)
# 查看夏普胜率最高的前5组
report_df.sort_values(['夏普','胜率'],ascending=False).head()

kdj = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=kdj_mktcap_long,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=240,
        N2=195,
        method='long')

kdj.backtest()  # 回测


index_price = getattr(kdj,'index_price')
fast_ma = getattr(kdj,'fast_ma')
slow_ma = getattr(kdj,'slow_ma')
flag = getattr(kdj,'flag')
kdj_avg_long = getattr(kdj,'algorithm_cum')

IndicatorPlot(index_price,fast_ma,slow_ma,flag)
CumPlot(index_price,kdj_avg_long,'流通市值加权多头收益')


# 查看回测情况
pd.DataFrame(kdj.GetRisk(),index=['270_240_195'])

#多空情况
def Singal(method):
    return CreatKDJSingal(index_symbol, start_date, end_date, 270, 240, 'mktcap', method)


risk_list = []
cum_list = []
flag_list = []

for label, method in zip(['流通市值加权多头', '流通市值加权空头', '流通市值加权多空'], ['long', 'short', 'ls']):
    print(label)
    # 加权多头
    roc = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=Singal,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=240,
        N2=195,
        method=method)

    roc.backtest()  # 回测

    cum = getattr(roc, 'algorithm_cum')
    cum.name = label

    rept = pd.Series(roc.GetRisk())
    rept.name = label

    flag_list.append(getattr(roc, 'flag'))
    risk_list.append(rept)
    cum_list.append(cum)


pd.DataFrame(risk_list)

CumPlot(getattr(roc,'index_price'), cum_list, '流通市值加权KDJ扩散指标纯多、多空、纯空净值曲线')


#######################################################################################
#RSI扩散指标
# 目标指数
index_symbol = '000300.XSHG'

# 复现时间设定
start_date, end_date = '2007-01-01', '2020-04-30'
# 计算加权信号
rsi_mktcap_long = CreatRSISingal(index_symbol,start_date,end_date,130,100,'mktcap','long')

# 网格寻参
report_df = GetGridRiskReport(130,rsi_mktcap_long,20,10)
# 查看夏普胜率最高的前5组
report_df.sort_values(['夏普','胜率'],ascending=False).head()


rsi = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=rsi_mktcap_long,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=110,
        N2=60,
        method='long')

rsi.backtest()  # 回测

index_price = getattr(rsi,'index_price')
fast_ma = getattr(rsi,'fast_ma')
slow_ma = getattr(rsi,'slow_ma')
flag = getattr(rsi,'flag')
rsi_avg_long = getattr(rsi,'algorithm_cum')

IndicatorPlot(index_price,fast_ma,slow_ma,flag)
CumPlot(index_price,rsi_avg_long,'流通市值加权多头收益')


# 查看回测情况
pd.DataFrame(kdj.GetRisk(),index=['130_110_65'])


def Singal(method):
    return CreatKDJSingal(index_symbol, start_date, end_date, 130, 110, 'mktcap', method)


risk_list = []
cum_list = []
flag_list = []

for label, method in zip(['流通市值加权多头', '流通市值加权空头', '流通市值加权多空'], ['long', 'short', 'ls']):
    print(label)
    # 加权多头
    rsi = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=Singal,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=110,
        N2=65,
        method=method)

    rsi.backtest()  # 回测

    cum = getattr(rsi, 'algorithm_cum')
    cum.name = label

    rept = pd.Series(roc.GetRisk())
    rept.name = label

    flag_list.append(getattr(rsi, 'flag'))
    risk_list.append(rept)
    cum_list.append(cum)


pd.DataFrame(risk_list)


CumPlot(getattr(rsi,'index_price'), cum_list, '流通市值加权RSI扩散指标纯多、多空、纯空净值曲线')

####################################################
####N日突破前期最高价

#数据获取
# 目标指数
index_symbol = '000300.XSHG'

# 复现时间设定
start_date, end_date = '2007-01-01', '2020-04-30'

df_list = []

for i in range(40, 251, 10):
    high_mktcap_long = CreatStageHighSingal(index_symbol, start_date, end_date, i, 90, 'mktcap', 'long')

    # 网格寻参
    df_list.append(GetGridRiskReport(i, high_mktcap_long, 20, 10))

report_df = pd.concat(df_list)

# 查看夏普胜率最高的前5组
report_df.sort_values(['夏普', '胜率'], ascending=False).head()

# 计算加权信号
high_mktcap_long = CreatStageHighSingal(index_symbol,start_date,end_date,230,220,'mktcap','long')

stage_high = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=high_mktcap_long,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=220,
        N2=15,
        method='long')

stage_high.backtest()  # 回测

index_price = getattr(stage_high,'index_price')
fast_ma = getattr(stage_high,'fast_ma')
slow_ma = getattr(stage_high,'slow_ma')
flag = getattr(stage_high,'flag')
rsi_avg_long = getattr(stage_high,'algorithm_cum')

IndicatorPlot(index_price,fast_ma,slow_ma,flag)
CumPlot(index_price,rsi_avg_long,'流通市值加权多头收益')

# 查看回测情况
pd.DataFrame(stage_high.GetRisk(),index=['130_110_65'])

#多空
def Singal(method):
    return CreatStageHighSingal(index_symbol, start_date, end_date, 220, 15, 'mktcap', method)


risk_list = []
cum_list = []
flag_list = []

for label, method in zip(['流通市值加权多头', '流通市值加权空头', '流通市值加权多空'], ['long', 'short', 'ls']):
    print(label)
    # 加权多头
    bt = DiffusionIndicatorBackTest(
        symbol=index_symbol,
        singal_ser=Singal,
        start_date=start_date,
        end_date=end_date,
        cal_func=talib.MA,
        N1=220,
        N2=15,
        method=method)

    bt.backtest()  # 回测

    cum = getattr(bt, 'algorithm_cum')
    cum.name = label

    rept = pd.Series(bt.GetRisk())
    rept.name = label

    flag_list.append(getattr(bt, 'flag'))
    risk_list.append(rept)
    cum_list.append(cum)


####

pd.DataFrame(risk_list)

CumPlot(getattr(stage_high,'index_price'), cum_list, '流通市值加权RSI扩散指标纯多、多空、纯空净值曲线')
















