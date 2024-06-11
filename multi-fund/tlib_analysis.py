import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

from qlib.data import D
from typing import List, Tuple, Dict
# 配置数据
#train_period = ("2019-01-01", "2021-12-31")
#valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2021-01-01", "2024-05-14")

market = "filter_fund"
benchmark = "SZ160706"
'''
from qlib.contrib.data.handler import Alpha158
dh = Alpha158(instruments='filter_fund',
              start_time=test_period[0],
              end_time=test_period[1],
              infer_processors={}
              )
#按dataframe 生成Alpha158，因子例子
test1=dh.fetch()
test1.head(2)

POOLS: List = D.list_instruments(D.instruments(market), as_list=True)

# 未来期收益
next_ret: pd.DataFrame = D.features(POOLS, fields=["Ref($close,-1)/$close-1"],start_time=test_period[0], end_time=test_period[1], freq='day')
next_ret.columns = ["next_ret"]
next_ret: pd.DataFrame = next_ret.swaplevel()
next_ret.sort_index(inplace=True)

# 基准
bench: pd.DataFrame = D.features([benchmark], fields=["$close/Ref($close,1)-1"],start_time=test_period[0], end_time=test_period[1])
bench: pd.Series = bench.droplevel(level=0).iloc[:, 0]
'''
##自己计算技术指标#######
##########################################
# 获取test时段的行情原始数据
stockpool: List = D.instruments(market=market)
raw_data: pd.DataFrame = D.features(
    stockpool,
    fields=["$open", "$high", "$low", "$close", "$volume"],
    start_time=test_period[0],
    end_time=test_period[1],
)

raw_data: pd.DataFrame = raw_data.reset_index(level=1).rename(
    columns={"instrument": "code"}
)
import talib
df=raw_data
####MACD指标
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
# 买入信号
df['buy_MACD_signal'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) < df['MACD_Signal'].shift(1)) & (df['MACD_Hist'] > 0)
# 卖出信号
df['sell_MACD_signal'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) > df['MACD_Signal'].shift(1)) & (df['MACD_Hist'] < 0)

####EMA指标
df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)

df['EMA_signal'] = 0
signals['EMA_signal'][(df['EMA_12'] > df['EMA_26']) & (df['EMA_12'].shift(1) <= df['EMA_26'].shift(1))] = 1
signals['EMA_signal'][(df['EMA_12'] < df['EMA_26']) & (df['EMA_12'].shift(1) >= df['EMA_26'].shift(1))] = -1


def generate_bollinger_bands_signals(df, close_col='close', timeperiod=20, nbdevup=2, nbdevdn=2, signal_col='signal'):
    """
    使用Bollinger Bands指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    timeperiod (int): Bollinger Bands的时间周期, 默认为20
    nbdevup (int): 上轨标准差个数, 默认为2
    nbdevdn (int): 下轨标准差个数, 默认为2
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 Bollinger Bands 指标
    df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df[close_col], timeperiod=timeperiod,
                                                                   nbdevup=nbdevup, nbdevdn=nbdevdn)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: 收盘价突破上轨
    df.loc[(df[close_col] > df['BB_UPPER']) & (df[close_col].shift(1) <= df['BB_UPPER'].shift(1)), signal_col] = 1

    # 卖出信号: 收盘价跌破下轨
    df.loc[(df[close_col] < df['BB_LOWER']) & (df[close_col].shift(1) >= df['BB_LOWER'].shift(1)), signal_col] = -1
    return df


def generate_keltner_channel_signals(df, close_col='close', high_col='high', low_col='low', timeperiod=20,
                                     atr_multiplier=1.5, signal_col='signal'):
    """
    使用Keltner Channel指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    high_col (str): 最高价所在列的名称, 默认为'high'
    low_col (str): 最低价所在列的名称, 默认为'low'
    timeperiod (int): Keltner Channel的时间周期, 默认为20
    atr_multiplier (float): ATR的倍数, 默认为1.5
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 Keltner Channel 指标
    df['KC_MIDDLE'] = talib.EMA(df[close_col], timeperiod=timeperiod)
    df['KC_UPPER'] = df['KC_MIDDLE'] + atr_multiplier * talib.ATR(df[high_col], df[low_col], df[close_col],
                                                                  timeperiod=timeperiod)
    df['KC_LOWER'] = df['KC_MIDDLE'] - atr_multiplier * talib.ATR(df[high_col], df[low_col], df[close_col],
                                                                  timeperiod=timeperiod)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: 收盘价突破上轨
    df.loc[(df[close_col] > df['KC_UPPER']) & (df[close_col].shift(1) <= df['KC_UPPER'].shift(1)), signal_col] = 1

    # 卖出信号: 收盘价跌破下轨
    df.loc[(df[close_col] < df['KC_LOWER']) & (df[close_col].shift(1) >= df['KC_LOWER'].shift(1)), signal_col] = -1

    return df

# 假设我们有一个名为 'df' 的DataFrame,包含股票的收盘价 'close', 最高价 'high', 最低价 'low'
df = generate_keltner_channel_signals(df, signal_col='keltner_signal')

#计算KDJ指标并生成买入卖出信号的函数
def generate_kdj_signals(df, close_col='close', low_col='low', high_col='high', fastk_period=9, slowk_period=3,
                         slowd_period=3, signal_col='signal'):
    """
    使用KDJ指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    low_col (str): 最低价所在列的名称, 默认为'low'
    high_col (str): 最高价所在列的名称, 默认为'high'
    fastk_period (int): KDJ快线周期, 默认为9
    slowk_period (int): KDJ慢线K周期, 默认为3
    slowd_period (int): KDJ慢线D周期, 默认为3
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 KDJ 指标
    df['K'], df['D'] = talib.STOCH(df[high_col], df[low_col], df[close_col],
                                   fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0,
                                   slowd_period=slowd_period, slowd_matype=0)
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: K线由下向上穿过D线
    df.loc[(df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)), signal_col] = 1

    # 卖出信号: K线由上向下穿过D线
    df.loc[(df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)), signal_col] = -1

    return df

# 假设我们有一个名为 'df' 的DataFrame,包含股票的收盘价 'close', 最高价 'high', 最低价 'low'
#df = generate_stochastic_signals(df, signal_col='my_signal')
def generate_stochastic_signals(df, close_col='close', low_col='low', high_col='high', fastk_period=14, slowk_period=3,
                                slowd_period=3, signal_col='signal'):
    """
    使用Stochastic指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    low_col (str): 最低价所在列的名称, 默认为'low'
    high_col (str): 最高价所在列的名称, 默认为'high'
    fastk_period (int): Stochastic快线周期, 默认为14
    slowk_period (int): Stochastic慢线K周期, 默认为3
    slowd_period (int): Stochastic慢线D周期, 默认为3
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 Stochastic 指标
    df['%K'], df['%D'] = talib.STOCH(df[high_col], df[low_col], df[close_col],
                                     fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0,
                                     slowd_period=slowd_period, slowd_matype=0)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: %K线由下向上穿过%D线
    df.loc[(df['%K'] > df['%D']) & (df['%K'].shift(1) <= df['%D'].shift(1)), signal_col] = 1

    # 卖出信号: %K线由上向下穿过%D线
    df.loc[(df['%K'] < df['%D']) & (df['%K'].shift(1) >= df['%D'].shift(1)), signal_col] = -1

    return df

#威廉指标(William's %R)
# 假设我们有一个名为 'df' 的DataFrame,包含股票的收盘价 'close', 最高价 'high', 最低价 'low'
#df = generate_williams_r_signals(df, signal_col='my_signal')
def generate_williams_r_signals(df, close_col='close', low_col='low', high_col='high', timeperiod=14,
                                signal_col='signal'):
    """
    使用William's %R指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    low_col (str): 最低价所在列的名称, 默认为'low'
    high_col (str): 最高价所在列的名称, 默认为'high'
    timeperiod (int): William's %R的计算周期, 默认为14
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 William's %R 指标
    df['%R'] = talib.WILLR(df[high_col], df[low_col], df[close_col], timeperiod=timeperiod)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: %R由高向低穿过-80
    df.loc[(df['%R'] < -80) & (df['%R'].shift(1) >= -80), signal_col] = 1

    # 卖出信号: %R由低向高穿过-20
    df.loc[(df['%R'] > -20) & (df['%R'].shift(1) <= -20), signal_col] = -1

    # 计算策略收益
    df['positions'] = df[signal_col].shift(1)
    df['strategy'] = (df['positions']) * df[close_col]

    return df

# 假设我们有一个名为 'df' 的DataFrame,包含股票的收盘价 'close'
#df = generate_roc_signals(df, signal_col='my_signal')

def generate_roc_signals(df, close_col='close', timeperiod=12, signal_col='signal'):
    """
    使用ROC指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    timeperiod (int): ROC的计算周期, 默认为12
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 ROC 指标
    df['ROC'] = talib.ROC(df[close_col], timeperiod=timeperiod)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: ROC由负转正
    df.loc[(df['ROC'] > 0) & (df['ROC'].shift(1) <= 0), signal_col] = 1

    # 卖出信号: ROC由正转负
    df.loc[(df['ROC'] < 0) & (df['ROC'].shift(1) >= 0), signal_col] = -1

    return df

# 假设我们有一个名为 'df' 的DataFrame,包含股票的收盘价 'close'
#df = generate_mom_signals(df, signal_col='my_signal')

def generate_mom_signals(df, close_col='close', timeperiod=12, signal_col='signal'):
    """
    使用MOM指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    timeperiod (int): MOM的计算周期, 默认为12
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 MOM 指标
    df['MOM'] = talib.MOM(df[close_col], timeperiod=timeperiod)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: MOM由负转正
    df.loc[(df['MOM'] > 0) & (df['MOM'].shift(1) <= 0), signal_col] = 1

    # 卖出信号: MOM由正转负
    df.loc[(df['MOM'] < 0) & (df['MOM'].shift(1) >= 0), signal_col] = -1

    return df

# 假设我们有一个名为 'df' 的DataFrame,包含股票的收盘价 'close' 和成交量 'volume'
#df = generate_obv_signals(df, signal_col='my_signal')
def generate_obv_signals(df, close_col='close', volume_col='volume', signal_col='signal'):
    """
    使用OBV指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格和成交量数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    volume_col (str): 成交量所在列的名称, 默认为'volume'
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 OBV 指标
    df['OBV'] = talib.OBV(df[close_col], df[volume_col])

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: OBV创新高且价格也创新高
    df.loc[(df['OBV'] > df['OBV'].shift(1)) & (df[close_col] > df[close_col].shift(1)), signal_col] = 1

    # 卖出信号: OBV创新低且价格也创新低
    df.loc[(df['OBV'] < df['OBV'].shift(1)) & (df[close_col] < df[close_col].shift(1)), signal_col] = -1

    return df


def generate_mfi_signals(df, close_col='close', low_col='low', high_col='high', volume_col='volume', timeperiod=14,
                         signal_col='signal'):
    """
    使用Money Flow Index(MFI)指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格和成交量数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    low_col (str): 最低价所在列的名称, 默认为'low'
    high_col (str): 最高价所在列的名称, 默认为'high'
    volume_col (str): 成交量所在列的名称, 默认为'volume'
    timeperiod (int): MFI的计算周期, 默认为14
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 Money Flow Index (MFI) 指标
    df['Typical Price'] = (df[high_col] + df[low_col] + df[close_col]) / 3
    df['Money Flow'] = df['Typical Price'] * df[volume_col]
    positive_flow = df['Money Flow'].where(df['Money Flow'] > df['Money Flow'].shift(1), 0)
    negative_flow = -df['Money Flow'].where(df['Money Flow'] < df['Money Flow'].shift(1), 0)
    df['MFI'] = 100 - 100 / (1 + positive_flow.rolling(timeperiod).sum() / negative_flow.rolling(timeperiod).sum())

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: MFI由超卖(<20)向上突破
    df.loc[(df['MFI'] > 20) & (df['MFI'].shift(1) <= 20), signal_col] = 1

    # 卖出信号: MFI由超买(>80)向下突破
    df.loc[(df['MFI'] < 80) & (df['MFI'].shift(1) >= 80), signal_col] = -1

    return df


def generate_volumeroc_signals(df, volume_col='volume', timeperiod=12, signal_col='signal'):
    """
    使用VolumeROC指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票成交量数据的DataFrame
    volume_col (str): 成交量所在列的名称, 默认为'volume'
    timeperiod (int): VolumeROC的计算周期, 默认为12
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 VolumeROC 指标
    df['VolumeROC'] = talib.ROC(df[volume_col], timeperiod=timeperiod)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: VolumeROC由负转正
    df.loc[(df['VolumeROC'] > 0) & (df['VolumeROC'].shift(1) <= 0), signal_col] = 1

    # 卖出信号: VolumeROC由正转负
    df.loc[(df['VolumeROC'] < 0) & (df['VolumeROC'].shift(1) >= 0), signal_col] = -1

    return df


def generate_nvi_signals(df, close_col='close', volume_col='volume', signal_col='signal'):
    """
    使用NVI(Negative Volume Index)指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格和成交量数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    volume_col (str): 成交量所在列的名称, 默认为'volume'
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 NVI 指标
    df['NVI'] = 1000
    for i in range(1, len(df)):
        if df[volume_col].iloc[i] < df[volume_col].iloc[i - 1]:
            df['NVI'].iloc[i] = df['NVI'].iloc[i - 1] + (df[close_col].iloc[i] - df[close_col].iloc[i - 1]) * \
                                df['NVI'].iloc[i - 1] / df[close_col].iloc[i - 1]
        else:
            df['NVI'].iloc[i] = df['NVI'].iloc[i - 1]

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: NVI创新高
    df.loc[df['NVI'] == df['NVI'].rolling(window=len(df), min_periods=1).max(), signal_col] = 1

    # 卖出信号: NVI创新低
    df.loc[df['NVI'] == df['NVI'].rolling(window=len(df), min_periods=1).min(), signal_col] = -1

    return df


def generate_pvi_signals(df, close_col='close', volume_col='volume', signal_col='signal'):
    """
    使用PVI(Positive Volume Index)指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票价格和成交量数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    volume_col (str): 成交量所在列的名称, 默认为'volume'
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 PVI 指标
    df['PVI'] = 1000
    for i in range(1, len(df)):
        if df[volume_col].iloc[i] > df[volume_col].iloc[i - 1]:
            df['PVI'].iloc[i] = df['PVI'].iloc[i - 1] + (df[close_col].iloc[i] - df[close_col].iloc[i - 1]) * \
                                df['PVI'].iloc[i - 1] / df[close_col].iloc[i - 1]
        else:
            df['PVI'].iloc[i] = df['PVI'].iloc[i - 1]

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: PVI创新高
    df.loc[df['PVI'] == df['PVI'].rolling(window=len(df), min_periods=1).max(), signal_col] = 1

    # 卖出信号: PVI创新低
    df.loc[df['PVI'] == df['PVI'].rolling(window=len(df), min_periods=1).min(), signal_col] = -1

    return df


def generate_ema_signals(df, close_col='close', timeperiod=12, signal_col='signal'):
    """
    使用EMA(Exponential Moving Average)指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票收盘价数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    timeperiod (int): EMA的计算周期, 默认为12
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 EMA 指标
    df['EMA'] = talib.EMA(df[close_col], timeperiod=timeperiod)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: 收盘价由下往上穿越EMA
    df.loc[(df[close_col] > df['EMA']) & (df[close_col].shift(1) <= df['EMA'].shift(1)), signal_col] = 1

    # 卖出信号: 收盘价由上往下穿越EMA
    df.loc[(df[close_col] < df['EMA']) & (df[close_col].shift(1) >= df['EMA'].shift(1)), signal_col] = -1

    return df


def generate_tema_signals(df, close_col='close', timeperiod=12, signal_col='signal'):
    """
    使用TEMA(Triple Exponential Moving Average)指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票收盘价数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    timeperiod (int): TEMA的计算周期, 默认为12
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 TEMA 指标
    df['TEMA'] = talib.TEMA(df[close_col], timeperiod=timeperiod)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: 收盘价由下往上穿越TEMA
    df.loc[(df[close_col] > df['TEMA']) & (df[close_col].shift(1) <= df['TEMA'].shift(1)), signal_col] = 1

    # 卖出信号: 收盘价由上往下穿越TEMA
    df.loc[(df[close_col] < df['TEMA']) & (df[close_col].shift(1) >= df['TEMA'].shift(1)), signal_col] = -1

    return df


def generate_kama_signals(df, close_col='close', fastlength=2, slowlength=30, signal_col='signal'):
    """
    使用KAMA(Kaufman's Adaptive Moving Average)指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票收盘价数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    fastlength (int): KAMA快速周期, 默认为2
    slowlength (int): KAMA慢速周期, 默认为30
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 KAMA 指标
    df['KAMA'] = talib.KAMA(df[close_col], fastlength, slowlength)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: 收盘价由下往上穿越KAMA
    df.loc[(df[close_col] > df['KAMA']) & (df[close_col].shift(1) <= df['KAMA'].shift(1)), signal_col] = 1

    # 卖出信号: 收盘价由上往下穿越KAMA
    df.loc[(df[close_col] < df['KAMA']) & (df[close_col].shift(1) >= df['KAMA'].shift(1)), signal_col] = -1

    return df


def generate_bbands_signals(df, close_col='close', timeperiod=20, nbdevup=2, nbdevdn=2, signal_col='signal'):
    """
    使用Bollinger Bands指标生成股票买入和卖出信号

    参数:
    df (pandas.DataFrame): 包含股票收盘价数据的DataFrame
    close_col (str): 收盘价所在列的名称, 默认为'close'
    timeperiod (int): 布林带计算周期, 默认为20
    nbdevup (float): 上轨偏差系数, 默认为2
    nbdevdn (float): 下轨偏差系数, 默认为2
    signal_col (str): 存储买卖信号的列名, 默认为'signal'

    返回值:
    pandas.DataFrame: 包含买卖信号的DataFrame
    """

    # 计算 Bollinger Bands 指标
    df['BBAND_UPPER'], df['BBAND_MIDDLE'], df['BBAND_LOWER'] = talib.BBANDS(
        df[close_col], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)

    # 生成买入卖出信号
    df[signal_col] = 0

    # 买入信号: 收盘价由下往上穿越下轨
    df.loc[(df[close_col] > df['BBAND_LOWER']) & (df[close_col].shift(1) <= df['BBAND_LOWER'].shift(1)), signal_col] = 1

    # 卖出信号: 收盘价由上往下穿越上轨
    df.loc[
        (df[close_col] < df['BBAND_UPPER']) & (df[close_col].shift(1) >= df['BBAND_UPPER'].shift(1)), signal_col] = -1

    return df


