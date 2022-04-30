# 引入库
#from jqdata import *

import pickle
import itertools  # 迭代器工具
import numpy as np
import pandas as pd
import prettytable as pt
import scipy.stats as st
import statsmodels.api as sm
# granger 因果检验
from statsmodels.tsa.stattools import grangercausalitytests


import calendar  # 日历
import datetime as dt
from tqdm import *
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from IPython.core.display import HTML

# 画图
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as mg # 不规则子图
import matplotlib.dates as mdate
import seaborn as sns

# 设置字体 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']
#mpl.rcParams['font.family'] = 'serif'
# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('seaborn')

import sys
sys.path.append("G://GitHub//PairsTrading//new_stategy//foundation_tools//")
import foundation_tushare
from Creat_RSRS import (RSRS,rolling_apply)  # 自定义信号生成
import json

# 使用ts
# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)

# 数据准备
start_date = '20050101'
end_date = '20171231'

# 观察期
begin_date = '20060101'
watch_date = '20161231'

# 数据获取
index_name='000300.SH'
HS300 = pro.query('index_daily', ts_code=index_name,
start_date=start_date, end_date=end_date,fields='trade_date,open,close,pre_close,high,low')
HS300.index = pd.to_datetime(HS300.trade_date)
del HS300['trade_date']
HS300.sort_index(inplace=True)  # 排序
std_window = 22
mean_window = 10

# 计算收益率
ret = HS300['close'] / HS300['pre_close'] - 1
# 收益率标准差
ret_std = ret.rolling(std_window).std().dropna()
ret_mean = ret_std.rolling(mean_window).mean().dropna()


# 计算振幅
amplitude = (HS300['high'] - HS300['low']) / HS300['pre_close']

# 计算振幅标准差
amplitude_std = amplitude.rolling(std_window).std().dropna()
amplitude_mean = amplitude_std.rolling(mean_window).mean().dropna()

# 统一观察窗口
amplitude_std = amplitude_std.loc[begin_date:watch_date]
amplitude_mean = amplitude_mean.loc[begin_date:watch_date]
ret_std = ret_std.loc[begin_date:watch_date]
ret_mean = ret_mean.loc[begin_date:watch_date]


# 获取granger因果检验结果的p值显示
def grangercausalitytests_pvalue(ret: pd.DataFrame, singal: pd.DataFrame, title: str):
    result = grangercausalitytests(
        np.c_[ret.reindex(singal.index), singal], maxlag=31, verbose=False)
    p_value = []
    for i, items_value in result.items():
        p_value.append(items_value[0]['params_ftest'][1])

    mpl.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(18, 6))
    plt.title(title)
    plt.bar(range(len(p_value)), p_value, width=0.4)
    plt.xticks(range(len(p_value)), np.arange(1, 32, 1))
    plt.axhline(0.5, ls='--', color='black', alpha=0.5, label='p值0.05显著水平')
    plt.legend()
    plt.show()


# 检验信号与滞后期收益率的相关系数
def show_corrocef(close_df: pd.DataFrame, singal: pd.DataFrame, title: str):
    period = np.arange(1, 32, 1)  # 滞后周期间隔

    temp = []  # 储存数据

    for i in period:
        # 收益未来收益与信号的相关系数
        lag_ret = close_df['close'].pct_change(i).shift(-i)
        temp.append(
            np.corrcoef(lag_ret.reindex(singal.index), singal)[0][1])

    mpl.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(18, 6))
    plt.title(title)
    plt.bar(range(len(temp)), temp, width=0.4)
    plt.xticks(range(len(temp)), period)
    plt.show()

show_corrocef(HS300,ret_mean,"收益率标准差均值与未来收益的关系")

show_corrocef(HS300,amplitude_std,"振幅标准差均值与未来收益的关系")

#granger因果检验
#收益率标准差均值与收益率进行granger因果检验发现p值均大于显著性水平0.05,
# 无法拒绝原假设收益率标准差均值无法对收益率进行较好的预测。

grangercausalitytests_pvalue(ret,ret_mean,'收益率波动率均值与收益率的granger检验的p值')

grangercausalitytests_pvalue(ret,amplitude_mean,'振幅波动率均值与收益率的granger检验的p值')


#波动率分解：上行波动与下行波动
'''
振幅剪刀差具体定义为：

上行波动 = (HIGH - OPEN)/OPEN

下行波动 = (OPEN - LOW)/OPEN

剪刀差 = 上行波动 - 下行波动

收益率剪刀差具体定义为:

上行波动率 = np.where(ret > 0 ,std ,0)

下行波动率 = np.where(ret < 0 ,std ,0)

剪刀差 = 上行波动 - 下行波动
'''
# 振幅剪刀差
Upward_volatility = HS300['high'] / HS300['open'] - 1
Downside_volatility = 1-HS300['low'] / HS300['open']

diff_vol = Upward_volatility - Downside_volatility
#查看数据分布

fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(1,3,1)
ax1 = sns.distplot(Upward_volatility.dropna(),label='上行波动率')
ax1 = sns.distplot(Downside_volatility.dropna(),label='下行波动率')
plt.legend()

ax2 = fig.add_subplot(1,3,2)
ax2 = sns.distplot(Upward_volatility.dropna(),color='g',label='上行波动率')
plt.legend()

ax3 = fig.add_subplot(1,3,3)
ax3 = sns.distplot(Downside_volatility.dropna(),label='下行波动率')
plt.legend()


show_corrocef(HS300,diff_vol.loc[begin_date:watch_date],"振幅剪刀差与未来收益的关系")

# 振幅剪刀差
Upward_volatility = np.where(ret.loc[begin_date:watch_date] > 0 ,ret_std ,0)
Downside_volatility = np.where(ret.loc[begin_date:watch_date] < 0 ,ret_std ,0)

diff_vol = Upward_volatility - Downside_volatility

fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(1,3,1)
ax1 = sns.distplot(Upward_volatility,label='上行波动率')
ax1 = sns.distplot(Downside_volatility,label='下行波动率')
plt.legend()

ax2 = fig.add_subplot(1,3,2)
ax2 = sns.distplot(Upward_volatility,color='g',label='上行波动率')
plt.legend()

ax3 = fig.add_subplot(1,3,3)
ax3 = sns.distplot(Downside_volatility,label='下行波动率')
plt.legend()

show_corrocef(HS300,pd.Series(diff_vol,ret_std.index),"收益率标准差刀差与未来收益的关系")

#基于单向波动差构建择时策略

'''
因此我们可以基于上面这个逻辑构建针对指数的择时策略，当前一天的上行波动率减去下行波动率的差值趋势
（为增加稳定性，采用60日移动均值）为正时就看多，反之则看空。 回测区间为2006年至2016年看下图净值感觉还不错，
中间回撤的幅度也有点大，超额收益也主要是来自 08 年。其它年份获取超额收益较少。
(granger检验与研报不同 但简单回测结果确差不多...)
'''

# 振幅剪刀差
Upward_volatility = HS300['high'] / HS300['open'] - 1
Downside_volatility = 1-HS300['low'] / HS300['open']

diff_vol = Upward_volatility - Downside_volatility
diff_ma = diff_vol.rolling(60).mean()

'''
strategy 1:
单向波动差移动平均为正：买入
单向波动差移动平均为负：卖出
'''
flag = np.where(diff_ma.loc[begin_date:watch_date] > 0, 1, 0)
slice_ser = ret.shift(-1).loc[begin_date:watch_date]
slice_benchmark = HS300.loc[begin_date:watch_date, 'close']

strategy_ret = flag * slice_ser
strategy_cum = (1 + strategy_ret).cumprod()
benchmark = slice_benchmark / slice_benchmark[0]

mpl.rcParams['font.family'] = 'serif'
plt.figure(figsize=(18, 8))
plt.title('波动剪刀差策略与沪深 300净值')
strategy_cum.plot()
benchmark.plot(color='r', ls='--', alpha=0.5)
plt.legend(['strategy1', 'HS300'])
#####################################################################

excess_ret = strategy_ret - ret.loc[begin_date:watch_date]
excess_cum = (1 + excess_ret).cumprod()

show_excess = excess_cum.groupby(
    pd.Grouper(freq='Y')).apply(lambda x: pow(x[-1] / x[0], 244 / len(x)) - 1)
plt.figure(figsize=(10,6))
plt.title('相对强弱RPS值与指数历史趋势')

plt.bar(range(len(show_excess)),show_excess.values)
plt.xticks(range(len(show_excess)),['%s年'%x.strftime('%Y') for x in show_excess.index])
plt.show()


####################################################################
'''
strategy 1:
单向波动差移动平均为正：买入
单向波动差移动平均为负：卖出
'''
diff_ma10 = diff_vol.rolling(10).mean()

flag_1 = np.where(diff_ma10.loc[begin_date:watch_date] > 0, 1, 0)
slice_ser = ret.shift(-1).loc[begin_date:watch_date]

strategy_ret1 = flag_1 * slice_ser
strategy_cum1 = (1 + strategy_ret1).cumprod()

plt.figure(figsize=(18, 8))
plt.title('10天与60天移动平均波动差值策略净值对比')
strategy_cum.plot()
strategy_cum1.plot()
benchmark.plot(color='r', ls='--', alpha=0.5)
plt.legend(['strategy_ma60','strategy_ma10', 'HS300'])
#############################################################################
#相对强弱 RPS指标
'''
"强者恒强、弱者恒弱"常为市场所证实。个股或市场的强弱表现其本身就 是基本面、资金面、投资者情绪等多种因素的综合作用下的体现。通常市场 强势与否，可以用市场相对强弱 RPS 指标来表示。

计算 RPS 值 RPS_1=(当前涨跌幅-MIN(250 交易日涨幅))/(MAX(250 交易日涨幅)-MIN(250 交易日涨幅)) （注：其值在 0%到 100%区间内）
然后取 10 个交易日移动平均值：RPS=MA(RPS_1)
**根据尝试发现RPS的定义应该是**:

RPS_1 = (当前收盘价 - min(过去250日收盘价))/(max(过去250日收盘价）-min(过去250日收盘价))

RPS = RPS_1的10日移动平均值

不是用涨跌幅....坑爹啊！
'''
# 计算RPS
def GetRPS(df: pd.DataFrame,period:int)->pd.Series:

    rps = (df['close'] - df['close'].rolling(250,min_periods=0).min()) / (
        df['close'].rolling(250,min_periods=0).max() - df['close'].rolling(250,min_periods=0).min())
    return rps.rolling(period,min_periods = 0).mean().loc[begin_date:watch_date]

#跟报告对比貌似就是用收盘价
# 计算rps
rps = GetRPS(HS300,10).dropna()

plt.figure(figsize=(18,8))
plt.title('相对强弱RPS值与指数历史趋势')
slice_benchmark.plot(color='black',alpha=0.8)
plt.ylabel('HS300收盘价')
plt.legend(['HS300'],loc=2)

plt.twinx()
rps.plot(color='r',alpha=0.6)
plt.ylabel('RPS')
plt.legend(['RPS'],loc=1)

#择时策略2
"""
strategy 2
RPS上穿80%，买入
RPS下穿80%，卖出
"""
pre_rps = rps.shift(1)

flag = []
for trade_date, rps_values in rps.items():

    last = rps_values
    per = pre_rps.loc[trade_date]

    if (last > 0.8) and (per < 0.8):

        flag.append(1)

    elif (last < 0.8) and (per > 0.8):

        flag.append(0)

    else:
        try:
            last_flag = flag[-1]
        except IndexError:
            last_flag = 0

        flag.append(last_flag)

strategy2_ret = flag * slice_ser.shift(-1).loc[rps.index]
strategy2_cum = (1 + strategy2_ret).cumprod()

plt.figure(figsize=(18, 8))
plt.title('相对强弱RPS值择时策略效果')
strategy2_cum.plot()
benchmark.loc[strategy2_cum.index].plot(color='r', ls='--', alpha=0.5)
legend(['RPS', 'HS300'])

#相对强弱 RPS下波动率差值策略
def Sensitivity_analysis(close_df: pd.DataFrame, period: int):
    rps = GetRPS(close_df, period).dropna()
    ret = close_df['close'] / close_df['pre_close'] - 1

    up = np.where(ret.loc[rps.index] > 0, rps, 0)
    down = np.where(ret.loc[rps.index] <= 0, rps, 0)

    diff_ = pd.Series(up - down, rps.index)
    diff_ = diff_.rolling(period, min_periods=0).mean()

    flag = np.where(diff_ > 0, 1, 0)

    strategy_ret = flag * ret.shift(-1).loc[rps.index]
    strategy_cum = (1 + strategy_ret).cumprod()

    Total_Annualized_Returns = pow(strategy_cum[-2] / strategy_cum[0],
                                   250 / len(strategy_cum)) - 1

    return strategy_cum, Total_Annualized_Returns

period = np.arange(1,61,1)
temp = {}
for i in tqdm(period,desc='寻找参数'):
    _,Total_Annualized_Returns = Sensitivity_analysis(HS300,i)
    temp[i] = Total_Annualized_Returns

print('收益最优的移动平均参数为:%s'%max(temp,key=temp.get))

strategy_cum,Total_Annualized_Returns = Sensitivity_analysis(HS300,13)

plt.figure(figsize=(18, 8))
plt.title('相对强弱 RPS下波动率差值策略与HS300净值')
strategy_cum.plot()
benchmark.loc[strategy_cum.index].plot(color='r', ls='--', alpha=0.5)
plt.legend(['RPS', 'HS300'])







