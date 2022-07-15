# 引入库
from typing import Dict, List

import empyrical as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append("C://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")

from data_service import get_index_daily, get_sales_depart_billboard
from scr import HMA, calc_netbuy, get_exchange_set
from scr.bt_func import (analysis_rets, analysis_trade, get_backtesting,
                         netbuy_cross)
from scr.plotting import (plot_distribution, plot_indicator,
                          plot_qunatile_signal)
from scr.tushare_api import TuShare

#import json
my_ts = TuShare()


# 数据获取
billboard_df: pd.DataFrame = get_sales_depart_billboard(
    '2013-01-01', '2022-02-02')

# 数据储存
billboard_df.to_csv('c:/data/billboard.csv', encoding='utf-8')

# 读取本地文件
billboard_df: pd.DataFrame = pd.read_csv('c:/data/billboard.csv',
                                         encoding='utf-8',
                                         index_col=[0],
                                         parse_dates=['trade_date'])

# 获取沪深300数据
hs300: pd.DataFrame = get_index_daily(code='000300.SH',
                                      start_date='20130101',
                                      end_date='20220222')
hs300.set_index('trade_date', inplace=True)

# 查看数据结构
hs300.head()

# 查看数据结构
billboard_df.head()

# 席位划分
exchange_set: Dict = get_exchange_set(billboard_df)

# 画分布
size: int = len(exchange_set)

for name, cond in exchange_set.items():
    is_netbuy_s: pd.Series = calc_netbuy(billboard_df[cond], hs300['amount'])
    plot_distribution(is_netbuy_s, hs300['close'], 10, f'{name}信号分布情况')


# 数据准备
def prepare_bt_data(ohlc: pd.DataFrame,
                    billboard: pd.DataFrame,
                    exchange_set: Dict = None,
                    fast_period: int = 30,
                    slow_period: int = 100) -> pd.DataFrame:
    """准备回测用数据

    Args:
        ohlc (pd.DataFrame): index-date OHLC及amount
        billboard (pd.DataFrfame): 龙虎榜数据
        exchange_set (Dict, optional): 席位划分字典,None时全量. Defaults to None.
        fast_period (int, optional): 短周期均线. Defaults to 30.
        slow_period (int, optional): 长周期均线. Defaults to 100.

    Returns:
        pd.DataFrame
    """
    if exchange_set is None:

        is_netbuy_s: pd.Series = calc_netbuy(billboard, ohlc['amount'])

    else:

        is_netbuy_s: pd.Series = calc_netbuy(billboard[exchange_set],
                                             ohlc['amount'])

    # 计算HMA信号
    is_netbuy_s_s: pd.Series = HMA(is_netbuy_s, fast_period)
    is_netbuy_s_l: pd.Series = HMA(is_netbuy_s, slow_period)

    is_netbuy_s_s.name = 'fast'
    is_netbuy_s_l.name = 'slow'

    bt_data: pd.DataFrame = pd.concat((ohlc, is_netbuy_s_s, is_netbuy_s_l),
                                      axis=1)
    return bt_data

# 获取数据
bt_data:pd.DataFrame = prepare_bt_data(hs300,billboard_df,exchange_set['机构席位'],30,100)

# 回测
bt_result = get_backtesting(bt_data, 'hs300', netbuy_cross)

analysis_rets(bt_data['close'], bt_result.result)

analysis_trade(bt_data['close'],bt_result.result)