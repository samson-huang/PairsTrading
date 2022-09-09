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

######################################### 筛选成分股 ################################################

class FilterStocks(object):
    '''
    获取某日的成分股股票
    1. 过滤st
    2. 过滤上市不足N个月
    3. 过滤当月交易不超过N日的股票
    ---------------
    输入参数：
        index_symbol:指数代码,A约等于全市场,800是设置的HS300+ZZ500
        watch_date:日期
        N:上市不足N月
        active_day:过滤交易不足N日的股票
    '''

    def __init__(self, index_symbol: str, watch_date: str, N: int = 3, active_day: int = 15):

        self.__index_symbol = index_symbol
        self.__watch_date = parse(watch_date).date()
        self.__N = N  # 过滤上市不足N月股票
        self.__active_day = active_day  # 交易日期

#####################################  获取并过滤成分股 ##############################

    # 获取股票池
    @property
    def Get_Stocks(self) -> list:
        '''
        bar_datetime:datetime.date
        '''

        if self.__index_symbol == 'A':

            stockList = get_index_stocks('000002.XSHG', date=self.__watch_date) + get_index_stocks(
                '399107.XSHE', date=self.__watch_date)

        else:
            stockList = get_index_stocks(
                self.__index_symbol, date=self.__watch_date)

        # 过滤ST
        st_data = get_extras(
            'is_st', stockList, end_date=self.__watch_date, count=1).iloc[0]

        stockList = st_data[st_data == False].index.tolist()

        # 剔除停牌、新股及退市股票
        stockList = self.delect_stop(stockList, self.__watch_date, self.__N)

        # 近15日均有交易的股票
        active_stock = self.delect_pause(
            stockList, self.__watch_date, self.__active_day)

        return active_stock

    # 去除上市距beginDate不足 3 个月的股票
    @staticmethod
    def delect_stop(stocks: list, beginDate: datetime.date,
                    n: int = 30 * 3) -> list:

        return [
            code for code in stocks
            if get_security_info(code).start_date < (beginDate -
                                                     datetime.timedelta(days=n))
        ]

    # 近15日内有交易
    @staticmethod
    def delect_pause(stocks: list, beginDate: datetime.date, n: int = 15) -> list:

        beginDate = get_trade_days(end_date=beginDate, count=1)[
            0].strftime('%Y-%m-%d')

        # 获取过去22日的交易数据
        df = get_price(
            stocks, end_date=beginDate, count=22, fields='paused', panel=False)

        # 当日交易
        t_trade = df.query('paused==0 and time==@beginDate')[
            'code'].values.tolist()

        # 当日交易 且 15日都有交易记录
        total_num = df[df['code'].isin(t_trade)].groupby('code')[
            'paused'].sum()

        return total_num[total_num < n].index.tolist()


def get_factor(func, index_symbol: str, start: str, end: str, freq: str = 'ME') -> pd.DataFrame:
    '''
    因子获取
    ---------
        func:为因子获取函数
        index_symbol:成分股代码
        freq:日期频率
    '''

    periods = GetTradePeriod(start, end, freq)

    factor_dic = {}
    for d in tqdm_notebook(periods):
        securities = FilterStocks(
            index_symbol, d.strftime('%Y-%m-%d'), N=12).Get_Stocks
        factor_dic[d] = func(securities, d)

    factor_df = pd.concat(factor_dic)
    factor_df.index.names = ['date', 'code']

    return factor_df


# 获取年末季末时点

def GetTradePeriod(start_date: str, end_date: str, freq: str = 'ME') -> list:
    '''
    start_date/end_date:str YYYY-MM-DD
    freq:M月，Q季,Y年 默认ME E代表期末 S代表期初
    ================
    return  list[datetime.date]
    '''
    days = pd.Index(pd.to_datetime(get_trade_days(start_date, end_date)))
    idx_df = days.to_frame()

    if freq[-1] == 'E':
        day_range = idx_df.resample(freq[0]).last()
    else:
        day_range = idx_df.resample(freq[0]).first()

    day_range = day_range[0].dt.date

    return day_range.dropna().values.tolist()


def query_model1_factor(securities: list, watch_date: str) -> pd.DataFrame:
    '''获取天风证券 指数增强模型'''

    import warnings
    warnings.filterwarnings("ignore")

    fields = ['natural_log_of_market_cap', 'book_to_price_ratio',
              'ROC20', 'ROC60',
              'net_profit_growth_rate', 'operating_revenue_growth_rate',
              'total_profit_growth_rate', 'roe_ttm',
              'roa_ttm', 'VOL20',
              'VOL60']

    part_a = get_factor_values(
        securities, fields, start_date=watch_date, end_date=watch_date)
    part_a = dict2frame(part_a)

    # 自定义因子
    fields = [EPTTM(), SPTTM(), SUE0(), SUR0(), DELTAROE(), DELTAROA(),
              ILLIQ(), ATR1M(), ATR3M()]

    part_b = calc_factors(securities, fields,
                          start_date=watch_date, end_date=watch_date)
    part_b = dict2frame(part_b)

    # 辅助项
    part_c = IndusrtyMktcap(securities, watch_date)

    factor_df = pd.concat([part_a, part_b, part_c], axis=1)

    return factor_df


def dict2frame(dic: dict) -> pd.DataFrame:
    '''将data的dict格式转为df'''

    tmp_v = [v.T for v in dic.values()]
    name = [k.upper() for k in dic.keys()]

    df = pd.concat(tmp_v, axis=1)
    df.columns = name

    return df


def IndusrtyMktcap(securities: list, watch_date: str) -> pd.DataFrame:
    '''增加辅助 行业及市值'''

    # indusrty_dict = get_industry(securities, watch_date)

    # indusrty_ser = pd.Series({k: v.get('sw_l1', {'industry_code': np.nan})[
    #                        'industry_code'] for k, v in indusrty_dict.items()})

    # indusrty_ser.name = 'INDUSTRY_CODE'

    industry_ser = get_stock_ind(securities, watch_date)

    mkt_cap = get_valuation(securities, end_date=watch_date,
                            fields='market_cap', count=1).set_index('code')['market_cap']

    return pd.concat([industry_ser, mkt_cap], axis=1)


def get_stock_ind(securities: list, watch_date: str, level: str = 'sw_l1', method: str = 'industry_code') -> pd.Series:
    '''
    获取行业
    --------
        securities:股票列表
        watch_date:查询日期
        level:查询股票所属行业级别
        method:返回行业名称or代码
    '''

    indusrty_dict = get_industry(securities, watch_date)

    indusrty_ser = pd.Series({k: v.get('sw_l1', {method: np.nan})[
        method] for k, v in indusrty_dict.items()})

    indusrty_ser.name = method.upper()

    return indusrty_ser

# 设置时间范围
START_DATE = '2010-01-01'
END_DATE = '2020-09-30'

# 因子获取
factors = get_factor(query_model1_factor,'000300.XSHG',START_DATE,END_DATE)
# 因子储存
factors.to_csv('../../Data/index_enhancement.csv')







