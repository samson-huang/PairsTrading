
import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

from qlib.data import D
import talib
from typing import List, Tuple, Dict
# 配置数据
#train_period = ("2019-01-01", "2021-12-31")
#valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2019-01-01", "2024-05-14")

market = "filter_fund"
benchmark = "SH000300"

# 获取test时段的行情原始数据
stockpool: List = D.instruments(market=market)
raw_data: pd.DataFrame = D.features(
    stockpool,
    fields=["$open", "$high", "$low", "$close", "$volume"],
    start_time=test_period[0],
    end_time=test_period[1],
)

#for col in ['$open', '$high', '$low', '$close', '$volume']:
#    raw_data[col] = raw_data[col].fillna(method='ffill')
raw_data = raw_data.dropna(how='all', axis=0)

# 替换列名中的特殊字符
raw_data.columns = [col.replace('$', '') for col in raw_data.columns]
df=raw_data

# 计算 RSI 指标
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 应用 calculate_rsi 函数
df['RSI'] = df.groupby('instrument')['close'].apply(calculate_rsi)
# 生成买入卖出信号列
df['RSI_signal'] = 0  # 默认为持有状态

# 生成买入卖出信号
df.loc[df['RSI'] < 30, 'RSI_signal'] = 1  # 买入信号
df.loc[df['RSI'] > 70, 'RSI_signal'] = -1  # 卖出信号
####MACD指标
# 计算 MACD 指标
df[['MACD', 'MACD_Signal', 'MACD_Hist']] = df.groupby(level='instrument')['close'].apply(lambda x: pd.DataFrame(talib.MACD(x, fastperiod=12, slowperiod=26, signalperiod=9)).T)

# 生成买入和卖出信号
df['MACD_signal'] = 0
df.loc[(df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) < df['MACD_Signal'].shift(1)) & (df['MACD_Hist'] > 0), 'MACD_signal'] = 1
df.loc[(df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) > df['MACD_Signal'].shift(1)) & (df['MACD_Hist'] < 0), 'MACD_signal'] = -1

import numpy as np
#计算 CCI
# 计算典型价格
df['TP'] = (df['high'] + df['low'] + df['close']) / 3

# 定义计算移动平均函数
def calculate_sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

# 定义计算平均绝对偏差函数
def calculate_mad(series, window):
    return series.rolling(window=window, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

#
N = 14  # 周期
df['SMA_TP'] = calculate_sma(df['TP'], N)
df['MAD_TP'] = calculate_mad(df['TP'], N)
df['CCI'] = (df['TP'] - df['SMA_TP']) / (0.015 * df['MAD_TP'])

# 生成买入卖出信号列
df['CCI_signal'] = 0  # 默认为持有状态

# 生成买入卖出信号
df.loc[df['CCI'] > 100, 'CCI_signal'] = 1  # 买入信号
df.loc[df['CCI'] < -100, 'CCI_signal'] = -1  # 卖出信号

# 计算 EMA 指标
df['EMA_12'] = df.groupby(level='instrument')['close'].apply(lambda x: talib.EMA(x, timeperiod=12))
df['EMA_26'] = df.groupby(level='instrument')['close'].apply(lambda x: talib.EMA(x, timeperiod=26))

# 生成 EMA 信号
df['EMA_signal'] = 0
df.loc[(df['EMA_12'] > df['EMA_26']) & (df['EMA_12'].shift(1) <= df['EMA_26'].shift(1)), 'EMA_signal'] = 1
df.loc[(df['EMA_12'] < df['EMA_26']) & (df['EMA_12'].shift(1) >= df['EMA_26'].shift(1)), 'EMA_signal'] = -1

# 计算 ROC 指标
df['ROC'] = df.groupby(level='instrument')['close'].apply(lambda x: talib.ROC(x, timeperiod=12))

# 生成买入卖出信号
df['ROC_signal'] = 0

# 买入信号: ROC由负转正
df.loc[(df['ROC'] > 0) & (df['ROC'].shift(1) <= 0), 'ROC_signal'] = 1

# 卖出信号: ROC由正转负
df.loc[(df['ROC'] < 0) & (df['ROC'].shift(1) >= 0), 'ROC_signal'] = -1


## 计算 NVI 指标
def calculate_nvi(group):
    close = group['close']
    volume = group['volume']
    nvi = [1000]  # 初始 NVI 值

    for i in range(1, len(group)):
        if volume[i] < volume[i - 1]:
            nvi.append(nvi[-1] * (1 + (close[i] - close[i - 1]) / close[i - 1]))
        else:
            nvi.append(nvi[-1])

    return pd.Series(nvi, index=group.index)

# 应用 calculate_nvi 函数
df['NVI'] = df.groupby('instrument', group_keys=False).apply(calculate_nvi)

# 生成买入卖出信号
df['NVI_signal'] = 0

# 买入信号: NVI创新高
df.loc[df['NVI'] == df['NVI'].rolling(window=len(df), min_periods=1).max(), 'NVI_signal'] = 1

# 卖出信号: NVI创新低
df.loc[df['NVI'] == df['NVI'].rolling(window=len(df), min_periods=1).min(), 'NVI_signal'] = -1


# 计算 PVI 指标
def calculate_pvi(group):
    close = group['close']
    volume = group['volume']
    pvi = [1000]  # 初始 PVI 值

    for i in range(1, len(group)):
        if volume[i] > volume[i - 1]:
            pvi.append(pvi[-1] * (1 + (close[i] - close[i - 1]) / close[i - 1]))
        else:
            pvi.append(pvi[-1])

    return pd.Series(pvi, index=group.index)

# 应用 calculate_pvi 函数
df['PVI'] = df.groupby('instrument').apply(calculate_pvi).reset_index(level=0, drop=True)

# 生成买入卖出信号列
df['PVI_signal'] = 0  # 默认为持有状态

# 买入信号: PVI 创新高
df.loc[df['PVI'] == df['PVI'].rolling(window=len(df), min_periods=1).max(), 'PVI_signal'] = 1

# 卖出信号: PVI 创新低
df.loc[df['PVI'] == df['PVI'].rolling(window=len(df), min_periods=1).min(), 'PVI_signal'] = -1



##################################################################


####################################################################
signal_cols = [col for col in df.columns if '_signal' in col]
new_df = df[signal_cols]
totel_exp=new_df.agg('sum', axis=1)
basically_cols=['open','high','low','close','volume']
totel_exp = totel_exp.to_frame()
totel_exp.columns = ['score']
data_1: pd.DataFrame = pd.merge(
    df[basically_cols], totel_exp, how="inner", left_index=True, right_index=True
)

# 将列名a改为A
data_1.index = data_1.index.rename({'instrument':'code'})
data_1 = data_1.reset_index('code')
data_1['rank'] = data_1['score']
ranked_data = data_1
#ranked_data["datetime"] = pd.to_datetime(ranked_data["datetime"])

##################################################################
###################################################################
import sys
import os
local_path = os.getcwd()
local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import LowRankStrategy_new
from typing import List, Tuple


###################################################
    bt_result = get_backtesting(
        ranked_data,
        name="code",
        strategy=LowRankStrategy_new,
        mulit_add_data=True,
        feedsfunc=AddSignalData,
        strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False, 'show_log': False},
        begin_dt=test_period[0],
        end_dt=test_period[1],
    )
    trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
    TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()

    OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()

###########################
    trader_df = pd.DataFrame(trade_logger)
    orders_df = pd.DataFrame(OrderAnalyzer)
##############################
    benchmark_old = ["SH000300"]
    #data, benchmark = get_backtest_data(ranked_data, test_period[0], test_period[1], market, benchmark_old)
    benchmark: pd.DataFrame = D.features(
        benchmark_old,
        fields=["$close"],
        start_time=test_period[0],
        end_time=test_period[1],
    ).reset_index(level=0, drop=True)
    benchmark_ret: pd.Series = benchmark['$close'].pct_change()

    algorithm_returns: pd.Series = pd.Series(
        bt_result.result[0].analyzers._TimeReturn.get_analysis()
    )
    report = analysis_rets(algorithm_returns, bt_result.result, benchmark['$close'].pct_change(), use_widgets=True)

    from plotly.offline import iplot
    from plotly.offline import init_notebook_mode

    init_notebook_mode()
    for chart in report:
        iplot(chart)


