import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime

from qlib.data import D
import talib
from typing import List, Tuple, Dict


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def knn_stock_prediction_new(df, n_neighbors=5, split_date=None):
    """
    使用 K-Nearest Neighbors (KNN) 算法进行股票价格预测。

    参数:
    df (pandas.DataFrame): 包含股票数据的 DataFrame
    n_neighbors (int): KNN 算法中的 n_neighbors 参数
    split_date (datetime or str): 用于分割数据集的日期

    返回:
    df (pandas.DataFrame): 包含预测信号的 DataFrame
    """
    df = df.dropna()
    instruments = df.index.get_level_values('instrument').unique()
    predicted_signals = []

    for instrument in instruments:
        instrument_df = df.loc[instrument]

        # 获取股票最早交易日期
        start_date = instrument_df.index.min()

        # 如果指定的分割日期早于股票开始交易日期，则使用默认值 0.95 进行划分
        if split_date is not None and pd.Timestamp(split_date) < start_date:
            print(f"For instrument {instrument}, split date is earlier than start date. Using default split.")
            split = int(0.95 * len(instrument_df))
            instrument_df['Open-Close'] = instrument_df['open'] - instrument_df['close']
            instrument_df['High-Low'] = instrument_df['high'] - instrument_df['low']
            X = instrument_df[['Open-Close', 'High-Low']]
            Y = np.where(instrument_df['close'].shift(-1) > instrument_df['close'], 1, -1)
            X_train = X[:split]
            Y_train = Y[:split]
            X_test = X[split:]
            Y_test = Y[split:]
            first_date = instrument_df.index[split]
        else:
            # 特征工程
            instrument_df['Open-Close'] = instrument_df['open'] - instrument_df['close']
            instrument_df['High-Low'] = instrument_df['high'] - instrument_df['low']
            X = instrument_df[['Open-Close', 'High-Low']]

            # 目标变量
            Y = np.where(instrument_df['close'].shift(-1) > instrument_df['close'], 1, -1)

            if split_date is not None:
                split_date_timestamp = pd.Timestamp(split_date)
                mask = instrument_df.index < split_date_timestamp
                X_train = X[mask]
                Y_train = Y[mask]
                X_test = X[~mask]
                Y_test = Y[~mask]
                first_date = instrument_df.index[~mask][0]
            else:
                split = int(0.95 * len(instrument_df))
                X_train = X[:split]
                Y_train = Y[:split]
                X_test = X[split:]
                Y_test = Y[split:]
                first_date = instrument_df.index[split]

        # 检查数据形状和类型
        if len(X_train.shape)!= 2:
            X_train = X_train.values.reshape(-1, 2)
        if len(Y_train.shape)!= 1:
            Y_train = Y_train.values.reshape(-1)

        # 检查 X_train 的数据类型并转换
        if X_train.select_dtypes(include=np.number).dtypes[0]!= np.float64:
            X_train = X_train.astype(np.float64)

        # 检查 Y_train 的数据类型并转换
        if Y_train.dtype!= np.int64:
            Y_train = Y_train.astype(np.int64)

        # 检查缺失值和无穷值
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            X_train = np.nan_to_num(X_train)
        if np.any(np.isnan(Y_train)) or np.any(np.isinf(Y_train)):
            Y_train = np.nan_to_num(Y_train)

        # 训练 KNN 模型
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, Y_train)

        # 计算准确率
        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))
        print(f'Instrument: {instrument}, Train_data Accuracy: {accuracy_train:.2f}, Test_data Accuracy: {accuracy_test:.2f}')

        # 生成预测信号
        if X_test is not None:
            instrument_df.loc[X_test.index, 'Predicted_Signal'] = knn.predict(X_test)
        else:
            instrument_df['Predicted_Signal'] = knn.predict(X_train)
        predicted_signals.append(instrument_df['Predicted_Signal'])

        # 打印 instrument 名称和第一个日期（如果有）
        if first_date is not None:
            print(f'Instrument: {instrument}, First date of X[split:]: {first_date}')
        else:
            print(f'Instrument: {instrument}, No split performed.')

    # 合并预测信号
    df['Predicted_Signal'] = pd.concat(predicted_signals, keys=instruments).reorder_levels(['instrument', 'datetime'])

    return df

if __name__ == '__main__':
    test_period = ("2005-01-01", "2024-11-04")

    market = "all _fund"
    # benchmark = "SZ16070"
    benchmark = "SH000300"



    # 获取test时段的行情原始数据
    stockpool: List = D.instruments(market=market)
    raw_data: pd.DataFrame = D.features(
        stockpool,
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=test_period[0],
        end_time=test_period[1],
    )
   # 替换列名中的特殊字符
    raw_data.columns = [col.replace('$', '') for col in raw_data.columns]
    test=knn_stock_prediction_new(raw_data,split_date='2024-07-01')
    test = test.rename(columns={'Predicted_Signal': 'rank'})
    test.index = test.index.rename({'instrument': 'code'})
    test = test.reset_index('code')
    test.to_csv("c:\\temp\\test_20240705.csv")
   #test=knn_stock_prediction(raw_data)