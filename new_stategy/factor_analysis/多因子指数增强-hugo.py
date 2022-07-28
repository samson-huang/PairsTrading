#from jqdata import *
from jqdatasdk  import (get_factor_values,
                      calc_factors,
                      Factor)

import talib
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.optimize import minimize

from tqdm import tqdm_notebook
from dateutil.parser import parse

import seaborn as sns
import matplotlib.pyplot as plt

import json
import tushare as ts
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api(timeout=5)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.style.use('seaborn')


#1.1因子构造
#部分因子无法直接使用聚宽的因子库获取故这里手动构造模型所需因子，
#此部分因子在上述模型所需的表中的因子计算方式项中标记有Factor

class EPTTM(Factor):
    name = 'EPTTM'
    max_window = 1
    dependencies = ['pe_ratio']

    def calc(self, data):
        return (1 / data['pe_ratio']).iloc[0]


class SPTTM(Factor):
    name = 'SPTTM'
    max_window = 1
    dependencies = ['ps_ratio']

    def calc(self, data):
        return (1 / data['ps_ratio']).iloc[0]


class SUE0(Factor):
    '''含漂移项'''

    name = 'SUE0'
    max_window = 1

    global fields

    fields = [f'net_profit_{i}' if i != 0 else 'net_profit' for i in range(9)]

    dependencies = fields

    def calc(self, data):
        # 数据结构为 columns为 net_profit至net_profit_8
        df = pd.concat([v.T for v in data.values()], axis=1)
        df.columns = fields
        df.fillna(0, inplace=True)

        # 漂移项可以根据过去两年盈利同比变化Q{i,t} - Q{i,t-4}的均值估计
        # 数据结构为array
        tmp = df.iloc[:, 1:5].values - df.iloc[:, 5:].values

        C = np.mean(tmp, axis=1)  # 漂移项 array

        epsilon = np.std(tmp, axis=1)  # 残差项epsilon array

        Q = df.iloc[:, 4] + C + epsilon  # 带漂移项的季节性随机游走模型

        return (df.iloc[:, 0] - Q) / epsilon


class SUR0(Factor):
    '''含漂移项'''

    name = 'SUR0'
    max_window = 1

    global fields

    fields = [f'operating_revenue_{i}' if i !=
                                          0 else 'operating_revenue' for i in range(9)]

    dependencies = fields

    def calc(self, data):
        # 数据结构为 columns为 net_profit至net_profit_8
        df = pd.concat([v.T for v in data.values()], axis=1)
        df.columns = fields
        df.fillna(0, inplace=True)

        # 漂移项可以根据过去两年盈利同比变化Q{i,t} - Q{i,t-4}的均值估计
        # 数据结构为array
        tmp = df.iloc[:, 1:5].values - df.iloc[:, 5:].values

        C = np.mean(tmp, axis=1)  # 漂移项 array

        epsilon = np.std(tmp, axis=1)  # 残差项epsilon array

        Q = df.iloc[:, 4] + C + epsilon  # 带漂移项的季节性随机游走模型

        return (df.iloc[:, 0] - Q) / epsilon


class DELTAROE(Factor):
    '''单季度净资产收益率-去年同期单季度净资产收益率'''

    name = 'DELTAROE'
    max_window = 1
    dependencies = ['roe', 'roe_4']

    def calc(self, data):
        return (data['roe'] - data['roe_4']).iloc[0]


class DELTAROA(Factor):
    '''单季度总资产收益率-去年同期单季度中资产收益率'''

    name = 'DELTAROA'
    max_window = 1
    dependencies = ['roa', 'roa_4']

    def calc(self, data):
        return (data['roa'] - data['roa_4']).iloc[0]


class ILLIQ(Factor):
    name = 'ILLIQ'
    max_window = 21
    dependencies = ['close', 'money']

    def calc(self, data):
        abs_ret = np.abs(data['close'].pct_change().shift(1).iloc[1:])

        return (abs_ret / data['money'].iloc[1:]).mean()


class ATR1M(Factor):
    '''过去20个交易日日内真实波幅均值'''
    name = 'ATR1M'
    max_window = 22
    dependencies = ['close', 'high', 'low']

    def calc(self, data):
        HIGH = data['high'].shift(1).iloc[1:]
        LOW = data['low'].shift(1).iloc[1:]
        CLOSE = data['close'].shift(1).iloc[1:]

        tmp = np.maximum(HIGH - LOW, np.abs(CLOSE.shift(1) - HIGH))
        TR = np.maximum(tmp, np.abs(CLOSE.shift(1) - LOW))

        return TR.iloc[-20:].mean()


class ATR3M(Factor):
    '''过去60个交易日日内真实波幅均值'''
    name = 'ATR3M'
    max_window = 62
    dependencies = ['close', 'high', 'low']

    def calc(self, data):
        HIGH = data['high'].shift(1).iloc[1:]
        LOW = data['low'].shift(1).iloc[1:]
        CLOSE = data['close'].shift(1).iloc[1:]

        tmp = np.maximum(HIGH - LOW, np.abs(CLOSE.shift(1) - HIGH))
        TR = np.maximum(tmp, np.abs(CLOSE.shift(1) - LOW))

        return TR.iloc[-60:].mean()











