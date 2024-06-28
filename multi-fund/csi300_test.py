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

