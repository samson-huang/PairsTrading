import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/cn_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime
from qlib.data import D
import talib
from typing import List, Tuple, Dict

test_period = ("2005-01-01", "2024-05-15")

market = "csi300"
#benchmark = "SZ16070"
benchmark = "SH000300"
# 获取test时段的行情原始数据
stockpool: List = D.instruments(market=market)
raw_data: pd.DataFrame = D.features(
    stockpool,
    fields=["$open", "$high", "$low", "$close", "$volume"],
    start_time=test_period[0],
    end_time=test_period[1],
)

#raw_data.loc[[( 'SH600000','2005-04-08')], :]
#raw_data.loc[[( 'SH600000')], :]

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def knn_stock_prediction(df, n_neighbors=5, split_percentage=0.95):
    """
    使用 K-Nearest Neighbors (KNN) 算法进行股票价格预测。

    参数:
    df (pandas.DataFrame): 包含股票数据的 DataFrame
    n_neighbors (int): KNN 算法中的 n_neighbors 参数
    split_percentage (float): 训练集和测试集的分割比例

    返回:
    df (pandas.DataFrame): 包含预测信号的 DataFrame

    """
    df = df.dropna()
    # df = df[['Open', 'High', 'Low', 'Close']]
    # df['Open'] = df['Open'].astype(np.float64)
    # df['High'] = df['High'].astype(np.float64)
    # df['Low'] = df['Low'].astype(np.float64)
    # df['Close'] = df['Close'].astype(np.float64)

    instruments = df.index.get_level_values('instrument').unique()
    predicted_signals = []

    for instrument in instruments:
        instrument_df = df.loc[instrument]

        # 特征工程
        instrument_df['Open-Close'] = instrument_df['open'] - instrument_df['close']
        instrument_df['High-Low'] = instrument_df['high'] - instrument_df['low']
        X = instrument_df[['Open-Close', 'High-Low']]

        # 目标变量
        Y = np.where(instrument_df['close'].shift(-1) > instrument_df['close'], 1, -1)

        # 数据集拆分
        split = int(split_percentage * len(instrument_df))
        X_train = X[:split]
        Y_train = Y[:split]
        X_test = X[split:]
        Y_test = Y[split:]

        # 训练 KNN 模型
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, Y_train)

        # 计算准确率
        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))
        print(
            f'Instrument: {instrument}, Train_data Accuracy: {accuracy_train:.2f}, Test_data Accuracy: {accuracy_test:.2f}')

        # 生成预测信号
        instrument_df['Predicted_Signal'] = knn.predict(X)
        instrument_df['Predicted_Signal'] = instrument_df['Predicted_Signal'].where(
            instrument_df['Predicted_Signal'] != -1, 0)
        predicted_signals.append(instrument_df['Predicted_Signal'])

    # 合并预测信号
    df['Predicted_Signal'] = pd.concat(predicted_signals, keys=instruments).reorder_levels(['instrument', 'datetime'])

    return df


# 替换列名中的特殊字符
raw_data.columns = [col.replace('$', '') for col in raw_data.columns]
test=knn_stock_prediction(raw_data)

test.index = test.index.rename({'instrument':'code'})
test = test.rename(columns={'Predicted_Signal': 'rank'})
test = test.reset_index('code')


benchmark_old = ["SH000300"]
# data, benchmark = get_backtest_data(ranked_data, test_period[0], test_period[1], market, benchmark_old)
benchmark: pd.DataFrame = D.features(
    benchmark_old,
    fields=["$close"],
    start_time=test_period[0],
    end_time=test_period[1],
).reset_index(level=0, drop=True)
benchmark_ret: pd.Series = benchmark['$close'].pct_change()

split = int(0.95 * len(benchmark))

benchmark[split:].head()
#2023-05-23

start_date = '2023-05-23'
end_date = '2024-05-15'

mask = (test.index.get_level_values('datetime') >= start_date) & (test.index.get_level_values('datetime') <= end_date)
result = test.loc[mask]

# 将列名a改为A
result = result.rename(columns={'Predicted_Signal': 'rank'})
result.index = result.index.rename({'instrument':'code'})
result = result.reset_index('code')


import sys
import os
local_path = os.getcwd()
local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import LowRankStrategy_new


bt_result = get_backtesting(
    result,
    name="code",
    strategy=LowRankStrategy_new,
    mulit_add_data=True,
    feedsfunc=AddSignalData,
    strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False, 'show_log': False},
    begin_dt='2023-05-23',
    end_dt=test_period[1],
)

trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()

OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()

trader_df = pd.DataFrame(trade_logger)
orders_df = pd.DataFrame(OrderAnalyzer)

algorithm_returns: pd.Series = pd.Series(
    bt_result.result[0].analyzers._TimeReturn.get_analysis()
)
benchmark_new = benchmark[split:]
report = analysis_rets(algorithm_returns, bt_result.result, benchmark_new['$close'].pct_change(), use_widgets=True)

from plotly.offline import iplot
from plotly.offline import init_notebook_mode

init_notebook_mode()
for chart in report:
    iplot(chart)



mask = (test.index.get_level_values('datetime') >= start_date)