############################
#20230710 重载多因子指数增强
#######################
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

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.style.use('seaborn')

from pprint import pprint
from pathlib import Path
import pandas as pd
import qlib


import tushare as ts
import warnings
import json
warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api()

#暂时不从qlib获取数据
#qlib.init()

MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "tutorial_exp"



# 数据读取
#factors4 = pd.read_csv('c:\\temp\\factors4_20230711.csv',index_col=[0,1],parse_dates=[0],dtype={'INDUSTRY_CODE':str})

#对称正交化
#factors5 = factors4.groupby(level='date').apply(lowdin_orthogonal)
#factors5 = factors5.reset_index(level=0, drop=True)
#factors5.info()


def get_weighs(symbol: str, start: str, end: str, method: str = 'cons') -> pd.DataFrame:
    '''
    获取月度指数成份权重
    --------
        mehtod:ind 输出 行业权重
               cons 输出 成份股权重
    '''
    periods = GetTradePeriod(start, end, 'ME')

    ser_dic = {}

    if method == 'ind':
        '''
        for d in periods:
            # 获取当日成份及权重
            index_w = get_index_weights(symbol, date=d)
            # 获取行业
            index_w['ind'] = get_stock_ind(index_w.index.tolist(), d)
            # 计算行业所占权重
            weight = index_w.groupby('ind')['weight'].sum() / 100

            ser_dic[d] = weight

        ser = pd.concat(ser_dic, names=['date', 'industry']).reset_index()
        ser['date'] = pd.to_datetime(ser['date'])
        return ser.set_index(['date', 'industry'])
        '''
        elif method == 'cons':

        # df = pd.concat([get_index_weights(symbol, date=d) for d in periods])
        # 查询到对应日期，且有权重数据，返回
        # pandas.DataFrame， code(股票代码)，display_name(股票名称), date(日期),
        # weight(权重)；

        df = pd.concat([pro.index_weight(index_code=symbol, trade_date=d.strftime('%Y%m%d'))
                        for d in periods])
        df = df[['con_code', 'trade_date', 'weight']]
        df = df.rename(columns={'trade_date': 'date', 'con_code': 'code'})
        # df.drop(columns='display_name', inplace=True)
        df.set_index('date', append=True, inplace=True)
        df = df.swaplevel()
        df['weight'] = df['weight'] / 100
        df = df.reset_index()
        # 将'date'列转换为datetime类型
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        # 设置日期的格式
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df = df.set_index(['date', 'code'])
        return df[['weight']]


def get_group(ser: pd.Series, N: int = 3, ascend: bool = True) -> pd.Series:
    '''默认分三组 升序'''
    ranks = ser.rank(ascending=ascend)
    label = ['G' + str(i) for i in range(1, N + 1)]

    return pd.cut(ranks, bins=N, labels=label)

# 获取年末季末时点

def GetTradePeriod(start_date: str, end_date: str, freq: str = 'ME') -> list:
    '''
    start_date/end_date:str YYYY-MM-DD
    freq:M月，Q季,Y年 默认ME E代表期末 S代表期初
    ================
    return  list[datetime.date]
    '''

    #days = pd.Index(pd.to_datetime(get_trade_days(start_date, end_date)))

    trade_cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
    trade_cal = trade_cal['cal_date']
    days = pd.Index(pd.to_datetime(trade_cal))
    #获取指定日期范围内的所有交易日, 返回 [numpy.ndarray], 包含指定的 start_date
    # 和 end_date, 默认返回至 datatime.date.today() 的所有交易日
    idx_df = days.to_frame()

    if freq[-1] == 'E':
        day_range = idx_df.resample(freq[0]).last()
    else:
        day_range = idx_df.resample(freq[0]).first()

    day_range = day_range['cal_date'].dt.date

    return day_range.dropna().values.tolist()


def stratified_sampling(symbol: str, START_DATE: str, END_DATE: str, factors: pd.DataFrame) -> pd.DataFrame:
    factors_ = factors.copy()
    ind_weight = get_weighs(symbol, START_DATE, END_DATE)

    # 市值等量分三组
    k1 = [pd.Grouper(level='date'),
          pd.Grouper(key='INDUSTRY_CODE')]

    factors_['GROUP'] = factors_.groupby(
        k1)['market_cap'].apply(lambda x: get_group(x, 3))
        # 获取每组得分最大的
    k2 = [pd.Grouper(level='date'),
          pd.Grouper(key='INDUSTRY_CODE'),
          pd.Grouper(key='GROUP')]

    industry_kfold_stock = factors_.groupby(
        k2)['SCORE'].apply(lambda x: x.idxmax()[1])

    # 格式调整
    industry_kfold_stock = industry_kfold_stock.reset_index()
    industry_kfold_stock = industry_kfold_stock.set_index(['date', 'SCORE'])
    industry_kfold_stock.index.names = ['date', 'code']

    # 加入权重
    #industry_kfold_stock['weight'] = ind_weight['weight']
    # 按照两个索引列合并20230714
    industry_kfold_stock = industry_kfold_stock.merge(ind_weight['weight'], left_index=True, right_index=True)


    # 令权重加总为1
    industry_kfold_stock['w'] = industry_kfold_stock.groupby(
        level='date')['weight'].transform(lambda x: x / x.sum())

    industry_kfold_stock['NEXT_RET'] = factors['NEXT_RET']

    return industry_kfold_stock

# 获取分层数据
# 数据读取
factors5 = pd.read_csv('c:\\temp\\factors5_20230713.csv',index_col=[0,1],parse_dates=[0],dtype={'INDUSTRY_CODE':str})
factors5.index = factors5.index.set_levels(['2023-07-03'], level=0)
START_DATE = '20230703'
END_DATE = '20230703'
symbol = '399300.SZ'
test123 = get_weighs(symbol, START_DATE, END_DATE)
result_df = stratified_sampling('399300.SZ', START_DATE, END_DATE, factors5[[
                                'INDUSTRY_CODE', 'market_cap', 'SCORE', 'NEXT_RET']])