# coding: utf-8
# 引入库
import tushare as ts
# 标记交易时点
import talib
import numpy as np
import pandas as pd

import statsmodels.api as sm  # 线性回归

import pyfolio as pf  # 组合分析工具


import itertools  # 迭代器工具


# 画图
import matplotlib.pyplot as plt
import seaborn as sns


# 设置字体 用来正常显示中文标签
plt.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('seaborn')

import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
import foundation_tushare
from Creat_RSRS import (RSRS,rolling_apply)  # 自定义信号生成
import json

# 使用ts
# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
#pro = ts.pro_api('your token')

# 数据获取及回测用函数

def query_index_data(ts_code: str, start: str, end: str, fields: str) -> pd.DataFrame:
    '''获取指数行情数据'''

    df = pro.index_daily(ts_code=ts_code, start_date=start,
                         end_date=end, fields='ts_code,trade_date,' + fields)

    df['trade_date'] = pd.to_datetime(df['trade_date'])

    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)

    return df


# 持仓标记不对称取值
def add_flag(signal_ser: pd.Series, S: float, s: float) -> pd.DataFrame:
    '''
    开平仓标记
        1-open/hold 0-close
        signal_ser:index-date values-RSRS
        S:阈值
    '''

    flag = pd.Series(np.zeros(len(signal_ser)), index=signal_ser.index)

    pre_day = signal_ser.index[0]

    for trade, signal in signal_ser.items():

        if signal > S:

            flag[trade] = 1

        elif signal < s:

            flag[trade] = 0

        else:

            flag[trade] = flag[pre_day]

        pre_day = trade

    return flag


def creat_algorithm_returns(signal_df: pd.DataFrame, benchmark_ser: pd.Series, S: float, s:float) -> tuple:
    '''生成策略收益表'''

    flag_df = signal_df.apply(lambda x: add_flag(x, S, s))

    log_ret = np.log(benchmark_ser / benchmark_ser.shift(1))  # 获取对数收益率

    next_ret = log_ret.shift(-1)  # 获取next_ret

    # 策略收益
    algorithm_ret = flag_df.apply(lambda x: x * next_ret)

    # 使用pyfolio分析格式化index
    algorithm_ret = algorithm_ret.tz_localize('UTC')
    algorithm_ret = algorithm_ret.dropna()

    benchmark = log_ret.tz_localize('UTC').reindex(algorithm_ret.index)

    return algorithm_ret, benchmark


def view_nav(algorithm_ret: pd.DataFrame, benchmark_ser: pd.Series):
    '''画净值图'''

    plt.rcParams['font.family'] = 'Microsoft JhengHei'
    # 策略净值
    algorithm_cum = (1 + algorithm_ret).cumprod()

    benchmark = (1 + benchmark_ser).cumprod()

    benchmark = benchmark.reindex(algorithm_cum.index)

    algorithm_cum.plot(figsize=(18, 8))  # 画图
    benchmark.plot(label='benchmark', ls='--', color='black')
    plt.legend()


def view_signal(close_ser: pd.Series, signal_ser: pd.Series):
    '''查看信号与指数的关系'''

    plt.rcParams['font.family'] = 'Microsoft JhengHei'
    close_ser = close_ser.reindex(signal_ser.index)
    plt.figure(figsize=(18, 8))
    close_ser.plot(color='Crimson')
    plt.ylabel('收盘价')
    plt.legend(['close'], loc='upper left')

    plt.twinx()
    signal_ser.plot()
    plt.ylabel('信号')
    plt.legend([signal_ser.name], loc='upper right')

# LR指标
def cala_LR(close: pd.Series) -> pd.Series:
    '''
    close：index-date value-close
    '''
    periods = list(range(10, 250, 10))
    ma = pd.concat([close.rolling(i).mean() for i in periods], axis=1)
    ma.columns = periods
    ma = ma.dropna()

    return ma.apply(lambda x: np.mean(np.where(close.loc[x.name] > x, 1, 0)), axis=1)


class RSRS_improve2(RSRS):

    # 重写方法 加入ls过滤器
    def get_RSRS(self, df: pd.DataFrame, LR_ser: pd.Series, N: int, M: int, method: str) -> pd.DataFrame:
        '''
        计算各类RSRS

            df:index-date columns-|close|high|low|money|pre_close|
            N:计算RSRS
            M:修正标准分所需参数
            method:选择 ols 或 wls 回归
        '''
        selects = {'ols': (df, lambda x: self._cala_ols(x, 'low', 'high'), N),
                   'wls': (df, lambda x: self._cala_wls(x, 'low', 'high', 'money'), N)}

        ret_quantile = LR_ser.rolling(M).apply(
            lambda x: x.rank(pct=True)[-1], raw=False)

        rsrs_df = rolling_apply(*selects[method])  # 计算RSRS

        res_df = (rsrs_df.pipe(self.cala_RSRS_z, M)
                  .pipe(self.cala_revise_RSRS)
                  .pipe(self.cala_negative_revise_RSRS)
                  .pipe(self.cala_passivation_RSRS, ret_quantile))

        return res_df.drop(columns='R_2').iloc[M:]

test_open = pd.read_pickle('C://temp//fund_data//base_data//mkt//open.pkl')
test_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//close.pkl')
test_pre_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//pre_close.pkl')
test_high = pd.read_pickle('C://temp//fund_data//base_data//mkt//high.pkl')
test_low = pd.read_pickle('C://temp//fund_data//base_data//mkt//low.pkl')
test_amount = pd.read_pickle('C://temp//fund_data//base_data//mkt//amount.pkl')

index_code = '399006.SZ'
#'close,pre_close,high,low,amount'
dfs = [test_close[index_code],test_pre_close[index_code],test_high[index_code],test_low[index_code],test_amount[index_code]]
result = pd.concat(dfs,axis=1)
result.columns = ['close','pre_close','high','low','amount']

result.index=pd.to_datetime(result.index)
result.sort_index(inplace=True)

close_df=result

price_df=close_df[['close']]
# 指标计算
LR = cala_LR(price_df['close'])


rsrs = RSRS_improve2()  # 调用RSRS计算类
signal_df = rsrs.get_RSRS(close_df, (1 - LR), 10, 60, 'ols')  # 获取各RSRS信号

signal_df.tail()

#获取最后一天收盘买卖信号

flag_df=signal_df.apply(lambda x: add_flag(x, 0.7, -0.7))
flag_df.tail()
##########################################
algorithm_ret_ver2, benchmark = creat_algorithm_returns(
    signal_df, close_df['close'], 0.7, -0.7)
view_nav(algorithm_ret_ver2[-200:], benchmark[-200:])