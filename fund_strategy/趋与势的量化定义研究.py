from typing import (Tuple,List,Callable,Union,Dict)

import pandas as pd
import numpy as np
import empyrical as ep
from collections import (defaultdict,namedtuple)
from jqdatasdk import *


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 标准化趋势
class Normalize_Trend(object):
    
    '''
    标准化价格位移
    
    注意:位移向量比状态变化向量多一个初始单元0
    '''
    
    def __init__(self,close_ser: pd.Series) -> None:

        if not isinstance(close_ser, pd.Series):

            raise ValueError('输入参数类型必须为pd.Series')

        self.close_ser = close_ser

    def normalize_monotone(self) -> pd.Series:
        '''单调性标准化'''

        sign = self.close_ser.pct_change().apply(np.sign)
        sign = sign.cumsum().fillna(0)

        return sign

    def normalize_movingaverage(self, window: int = 5) -> pd.Series:
        '''5周期均线的标准化'''

        close_ser = self.close_ser
        size = len(close_ser)

        if size < window:

            raise ValueError('输入数据长度小于窗口期')

        ma = close_ser.rolling(window).mean()
        sign = (close_ser - ma).apply(np.sign).iloc[window - 2:]
        sign = sign.cumsum().fillna(0)

        return sign

    def normalize_compound(self, window: int = 5):

        close_ser = self.close_ser

        size = len(close_ser)

        if size < window:

            raise ValueError('输入数据长度小于窗口期')

        sign_monotone = close_ser.pct_change().apply(np.sign)

        ma = close_ser.rolling(window).mean()
        sign_ma = (close_ser - ma).apply(np.sign)

        # @jqz1226
        # 可以按照4种情形分别分析：
        # 1. 前一个交易日收盘价位于均线之下，当前收盘价站上均线，状态记为1；分析：当前sign_ma = 1，
        # 收盘价能从均线下跃到均线上，必然是由于价格上涨，故sign_monotone = 1, 于是 (1+1)/2 = 1
        # 2. 前一个交易日收盘价位于均线之上，当前收盘价跌破均线，状态记为-1；分析：当前sign_ma=-1，
        # 收盘价能从均线上掉到均线下，必然是由于价格下跌，故sign_monotone = -1, 于是((-1)+(-1))/2 = -1
        # 3. 3a) 前一个交易日收盘价位于均线之上，当前收盘价位于均线之上，当前收盘价大于或等于前一个交易日收盘价，
        # 状态记为1；分析：当前sign_ma = 1，收盘价上升，sign_monotone = 1, 于是 (1+1)/2 = 1
        # 3b) 前一个交易日收盘价位于均线之上，当前收盘价位于均线之上，当前收盘价小于前一个交易日收盘价，
        # 状态记为0；分析：当前sign_ma = 1，收盘价下降，sign_monotone = -1, 于是 ((1)+(-1))/2 = 0
        # 4. 4a) 前一个交易日收盘价位于均线之下，当前收盘价位于均线之下，当前收盘价大于前一个交易日收盘价，
        # 状态记为0，分析：当前sign_ma = -1，收盘价上升，sign_monotone = 1, 于是 (-1+1)/2 = 0
        # 4b) 前一个交易日收盘价位于均线之下，当前收盘价位于均线之下，当前收盘价小于或等于前一个交易日收盘价，
        # 状态记为-1。分析：当前sign_ma = -1，收盘价下降，sign_monotone = -1, 于是 ((-1)+(-1))/2 = -1

        sign_compound = (sign_monotone + sign_ma) / 2  # 简单平均
        sign_compound = sign_compound.iloc[window - 2:].cumsum().fillna(0)

        return sign_compound

class Tren_Score(object):
    '''
    根据标准化后的价格数据计算趋势得分
    ------
    输入参数：
        normalize_trend_ser:pd.Series index-date values-标准化后的价格数据

    方法：
        评分方法均有两种计算模式区别是划分波段的方法不同
        分别是opposite/absolute 即【相对波段划分】和【绝对波段划分】

        calc_trend_score:计算“趋势”得分
            score Dict
                - trend_score 势得分
                - act_score 趋得分
            - point_frame Dict 标记表格
            - point_mask Dict 标记点
        calc_absolute_score:计算混合模式得分
    '''
    def __init__(self, normalize_trend_ser: pd.Series) -> None:

        if not isinstance(normalize_trend_ser, pd.Series):

            raise ValueError('输入参数类型必须为pd.Series')

        self.normalize_trend_ser = normalize_trend_ser

        # 储存标记点表格
        self.point_frame:Dict[pd.DataFrame] = defaultdict(pd.DataFrame)
        self.score_record = namedtuple('ScoreRecord','trend_score,act_score')
        self.score:Dict = defaultdict(namedtuple)

        # 储存标记点标记
        self.point_mask:Dict[List] =  defaultdict(list)

        self.func_dic: Dict = {
            'opposite': self._get_opposite_piont,
            'absolute': self._get_absolute_point
        }

    def calc_trend_score(self, method: str) -> float:
        '''势'''

        func: Callable = self.func_dic[method]

        # 趋势极值点得标记
        cond:pd.Series = func()

        # 势得分
        trend_score = np.square(self.normalize_trend_ser[cond].diff()).sum()
        # 趋得分
        act_score = self.normalize_trend_ser.diff().sum()

        self.score[method] = self.score_record(trend_score=trend_score,
                                               act_score=act_score)
        
        self.point_frame[method] = self.normalize_trend_ser[cond]

        self.point_mask[method] = cond

    def calc_absolute_score(self) -> float:

        '''势的终极定义'''

        opposite = self.calc_trend_score('opposite')
        absolute = self.calc_trend_score('absolute')

        N = len(self.normalize_trend_ser)

        return max(opposite, absolute) / (N ** (3 / 2))

    def _get_opposite_piont(self) -> List:
        '''
        获取相对拐点的位置
        ------
        return np.array([True,..False,...True])
            True表示为拐点，False表示不是
        '''
        ser = self.normalize_trend_ser
        flag_ser = pd.Series(index=ser.index, dtype=ser.index.dtype)

        dif = ser.diff().fillna(method='bfill')

        for idx, i in dif.items():

            try:
                previous_i
            except NameError:

                previous_idx = idx
                previous_i = i
                flag_ser[idx] = True
                continue

            if i != previous_i:

                flag_ser[previous_idx] = True
            else:
                flag_ser[previous_idx] = False

            previous_idx = idx
            previous_i = i

        flag_ser.iloc[0] = True
        flag_ser.iloc[-1] = True

        # 拐点索引

        return flag_ser.values.tolist()

    def _get_absolute_point(self) -> List:
        '''
        获取绝对拐点的位置
        ------
        return np.array([True,..False,...True])
            True表示为拐点，False表示不是
        '''
        arr = self.normalize_trend_ser.values
        size = len(arr)
        
        # TODO:不知道我是不是没理解研报算法
        # 如果使用下面算法找最大最小 在[0,-1,-1,0,1,0,-1,-1,-2]这种情况下
        # 最大值会被标记在下标为8的元素上

        # distances = np.abs(arr.reshape(-1, 1) - np.tile(arr, (size, 1)))

        # d_arr = np.tril(distances)[:, 0]
        # # 获取最大/小值
        # ind_max = np.argmax(d_arr)
        # ind_min = np.argmin(d_arr)
    
        # # 最大/小值索引下标
        # idx_max = np.argwhere(d_arr == ind_max).reshape(1, -1)[0]
        # idx_min = np.argwhere(d_arr == ind_min).reshape(1, -1)[0]

        ind_max = np.max(arr)
        ind_min = np.min(arr)

        idx_max = np.argwhere(arr == ind_max).reshape(1, -1)[0]
        idx_min = np.argwhere(arr == ind_min).reshape(1, -1)[0]
        point = np.append(idx_min, idx_max)
        point = np.append(point, [0, size - 1])
        point = np.unique(point)
        cond = [True if i in point else False for i in range(size)]

        return cond
        


import sys
#ys.path.append('../..')
# Hugos_tools.plot_finance import candlestick_ochl
import mpl_finance as mpf

from matplotlib import ticker
from matplotlib.pylab import date2num

# 画K线
def plot_ochl(data_df,title:str=None,ax=None):
    
    if ax is None:
        
        fig,ax = plt.subplots()
        
    #data = self._data.copy()
    data = data_df.copy()
    date_tickers = pd.to_datetime(data.index)
    date_tickers = data.index
    
    data['dates'] = np.arange(len(data))
    ax.xaxis_date()
    ax.set_xlim(1, len(data))  # 高版本mpl不需要这个..这个看不到下标为0得K线
    def format_date(x,pos):
        
        if x<0 or x>len(date_tickers)-1:
            
            return ''
        
        return date_tickers[int(x)].strftime('%Y-%m-%d')
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    mpf.candlestick_ochl(
                    ax=ax,
                    quotes=data[['dates', 'open', 'close', 'high', 'low']].values,
                    width=0.7,
                    colorup='r',
                    colordown='g',
                    alpha=0.7)
    ax.set_title(title)
    
    plt.xticks(rotation=30)
    return ax

pd.DataFrame.plot.ochl = plot_ochl



import sys 
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")

import foundation_tushare
import json
from datetime import datetime

# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
my_pro = foundation_tushare.TuShare(setting['token'], max_retry=60)



index_code = '399006.SZ'
test_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//close.pkl')
test_pre_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//pre_close.pkl')
test_high = pd.read_pickle('C://temp//fund_data//base_data//mkt//high.pkl')
test_low = pd.read_pickle('C://temp//fund_data//base_data//mkt//low.pkl')
test_amount = pd.read_pickle('C://temp//fund_data//base_data//mkt//amount.pkl')
test_open = pd.read_pickle('C://temp//fund_data//base_data//mkt//open.pkl')

dfs = [test_open[index_code], test_close[index_code], test_low[index_code], test_high[index_code]]
result = pd.concat(dfs, axis=1)
result.columns = ['open', 'close', 'low', 'high']

price = result
price.index = pd.to_datetime(price.index)
price.sort_index(inplace=True)
price=price.dropna(axis=0,how='any')

#price = my_pro.index_daily(ts_code='399006.SZ')
#price.index=price['trade_date']
#price=pd.DataFrame(price[['close','open','high','low']])

#price.index=[datetime.strptime(x,'%Y%m%d') for x in price.index]
#price=price.sort_index()


normalize = Normalize_Trend(price['close'])


fig,axes = plt.subplots(1,2,figsize=(18,5))

sign1 = normalize.normalize_monotone()

plot_ochl(price,ax=axes[0])
sign1.plot(marker='o',ax=axes[1])




fig,axes = plt.subplots(1,2,figsize=(18,5))

sign3 = normalize.normalize_compound(5)

plot_ochl(price,ax=axes[0])
sign3.plot(marker='o',ax=axes[1])


test_value = [0,1,1,2,1,2,1,1,0,0,1,2]
test_ser = pd.Series(data=test_value,index=range(len(test_value)))

test_ser.plot(marker='o');


test_arr = np.array([[0,0,0],
                     [1,1,1],
                     [2,2,2],
                     [3,3,3],
                     [4,4,4],
                     [5,5,5],
                     [6,6,4],
                     [7,7,5],
                     [8,6,6],
                     [9,7,7],
                     [8,8,8]])

df1 = pd.DataFrame(test_arr,columns='A,B,C'.split(','))


def get_act_trend_score(ser: pd.Series) -> Tuple:

    trend_score = Tren_Score(ser)
    trend_score.calc_trend_score('opposite')
    return (trend_score.score['opposite'].act_score,
            trend_score.score['opposite'].trend_score)

test_arr = np.array([[0,0,0],
                     [1,1,1],
                     [2,2,2],
                     [3,3,3],
                     [4,4,4],
                     [5,5,5],
                     [6,6,4],
                     [7,7,5],
                     [8,6,6],
                     [9,7,7],
                     [8,8,8]])

df1 = pd.DataFrame(test_arr,columns='A,B,C'.split(','))

# df1.plot(subplots=True,layout=(1,3),marker='o');
score_df1 = df1.apply(lambda x:get_act_trend_score(x)).T

fif,axes = plt.subplots(1,3,figsize=(18,6))

for ax,(col_name,score_ser),(_,ser_v) in zip(axes,score_df1.T.items(),df1.items()):
    
    a,b = score_ser # 趋,势
    ser_v.plot(ax=ax,marker='o',title=f'{col_name}:<{a},{b}>')

#对量化趋势得分进行应用
#检查不同行情下得分段情况
hs300_temp = my_pro.index_daily(ts_code='000300.SH')
hs300 = hs300_temp
hs300.index=hs300['trade_date']
hs300=pd.DataFrame(hs300[['close','open','high','low']])
hs300.index=pd.to_datetime(hs300.index)
hs300=hs300.sort_index()

slice_price = hs300.loc['2020-01-01':'2021-10-01','close']

weekly_bar = slice_price.resample('W').last()

normalize = Normalize_Trend(weekly_bar)
normalize_trend_ser = normalize.normalize_compound(5)

trend_score = Tren_Score(normalize_trend_ser)
trend_score.calc_trend_score('absolute')

fig, axes = plt.subplots(2, figsize=(18, 12))

#plot.ochl(hs300.loc['2020-01-01':'2021-10-01'],ax=axes[0],title='沪深300')
plot_ochl(hs300.loc['2020-01-01':'2021-10-01'],ax=axes[0],title='沪深300')
a = trend_score.score['absolute'].trend_score
b = trend_score.score['absolute'].act_score

axes[1].set_title(f'标准化后走势<势={a},趋={b}>')
normalize_trend_ser.plot(ax=axes[1], marker='o',ms=4)
trend_score.point_frame['absolute'].plot(
    ax=axes[1], marker='o', ls='--', color='darkgray');

#构造信号用于回测
def calc_trend_score(ser: pd.Series) -> float:
    # 转为周度
    # ser = ser.resample('W').last().copy()
    normalize = Normalize_Trend(ser)

    # 此时是周级别得均线
    normalize_trend_ser = normalize.normalize_compound(5)

    trend_score = Tren_Score(normalize_trend_ser)

    trend_score.calc_trend_score('absolute')

    return trend_score.score['absolute'].trend_score

score = hs300['close'].rolling(60).apply(calc_trend_score,raw=False)
lower_bound = score.rolling(20).apply(lambda x: x.quantile(0.05),raw=False)
upper_bound = score.rolling(20).apply(lambda x: x.quantile(0.85),raw=False)

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
plot_ochl(hs300,ax=axes[0],title='走势')

axes[1].set_title('信号')
score.plot(ax=axes[1], color='darkgray', label='trend_score')
upper_bound.plot(ax=axes[1], ls='--', label='upper')
lower_bound.plot(ax=axes[1], ls='--', label='lower')

plt.legend();

#大于上轨开仓至信号上传下轨平仓

def get_hold_flag(df: pd.DataFrame) -> pd.Series:
    '''
    标记持仓
    ------
    输入参数：
        df:index-date columns-score|lower_bound|upper_bound
    ------
    return index-date 1-持仓;0-空仓
    '''
    flag = pd.Series(index=df.index, data=np.zeros(len(df)))

    for trade, row in df.iterrows():

        sign = row['score']
        lower = row['lower_bound']
        upper = row['upper_bound']
        try:
            previous_score
        except NameError:
            previous_score = sign
            previous_lower = lower
            previous_upper = upper
            order_flag = 0
            continue

        if previous_score > previous_lower and sign <= lower:

            flag[trade] = 0
            order_flag = 0

        elif previous_score < previous_upper and sign >= upper:
            flag[trade] = 1
            order_flag = 1

        else:

            flag[trade] = order_flag

        previous_score = sign
        previous_lower = lower
        previous_upper = upper

    return flag

df = pd.concat((score,upper_bound,lower_bound),axis=1)
df.columns = ['score','upper_bound','lower_bound']

flag = get_hold_flag(df)
next_ret = hs300['close'].pct_change().shift(-1)
algorithms_ret = flag * next_ret.loc[flag.index]

algorithms_cum = ep.cum_returns(algorithms_ret)


test123456=hs300['close']/hs300['close'][0]
fig,axes = plt.subplots(2,figsize=(18,12))

axes[0].set_title('指数')
test123456[-200:].plot(ax=axes[0])
axes[1].set_title('净值')
flag[-200:].plot(ax=axes[0],secondary_y=True,ls='--',color='darkgray')
algorithms_cum[-200:].plot(ax=axes[1]);


##########################################################################################
def creat_algorithm_returns(flag_df: pd.Series, benchmark_ser: pd.Series) -> tuple:
    '''生成策略收益表'''


    log_ret = np.log(benchmark_ser / benchmark_ser.shift(1))  # 获取对数收益率

    next_ret = log_ret.shift(-1)  # 获取next_ret

    # 策略收益
    algorithm_ret = flag_df.apply(lambda x: x * next_ret)

    # 使用pyfolio分析格式化index
    algorithm_ret = algorithm_ret.tz_localize('UTC')
    algorithm_ret = algorithm_ret.dropna()

    benchmark = log_ret.tz_localize('UTC').reindex(algorithm_ret.index)

    return algorithm_ret, benchmark


def view_nav(algorithm_ret: pd.DataFrame, benchmark_ser: pd.Series):
    '''画净值图'''

    plt.rcParams['font.family'] = 'Microsoft JhengHei'
    # 策略净值
    algorithm_cum = (1 + algorithm_ret).cumprod()

    benchmark = (1 + benchmark_ser).cumprod()

    benchmark = benchmark.reindex(algorithm_cum.index)

    algorithm_cum.plot(figsize=(18, 8))  # 画图
    benchmark.plot(label='benchmark', ls='--', color='black')
    plt.legend()

################################################################################
######################画图看效果#############################
algorithm_cum = (1 + algorithms_cum).cumprod()

benchmark = (1 + benchmark_ser).cumprod()

benchmark = benchmark.reindex(algorithm_cum.index)
benchmark.plot(figsize=(18, 8))  # 画图
test.plot(label='benchmark', ls='--', color='black')
plt.legend()