# 引入库
import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
from technical_analysis_patterns import (rolling_patterns2pool,plot_patterns_chart)
from typing import (List, Tuple, Dict, Callable, Union)
from tqdm.notebook import tqdm
import json
#from jqdatasdk import (auth,get_price,get_trade_days,finance,query,get_industries)

import pandas as pd
import numpy as np
import empyrical as ep


import seaborn as sns
import matplotlib as mpl
import mplfinance as mpf
import matplotlib.pyplot as plt

setting = json.load(open('C:\config\config.json'))
import tushare as ts
pro = ts.pro_api(setting['token'])

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

#auth('','')


#data1 = get_price('000009.SH', start_date='2021-01-21', end_date='2021-12-31',
#                 fields=['open', 'close', 'low', 'high'], panel=False)
#data1 = pro.index_daily(ts_code='399006.SZ', start_date='20220101', end_date='20220727',fields=['trade_date','open', 'close', 'low', 'high'])

data1 = pro.daily(ts_code='000009.SZ', start_date='20210121', end_date='20211231',fields=['trade_date','open', 'close', 'low', 'high'])
data1.index = pd.to_datetime(data1.trade_date)
del data1['trade_date']
data1.sort_index(inplace=True)
patterns_record1 = rolling_patterns2pool(data1['close'],n=35)
plot_patterns_chart(data1,patterns_record1,True,False)
plt.title('中国宝安')
plot_patterns_chart(data1,patterns_record1,True,True);



#申万一级行业形态识别情况
def patterns_res2json(dic: Dict) -> str:
    """将结果转为json

    Args:
        dic (Dict): 结果字典
    Returns:
        str
    """
    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""

        if isinstance(obj, np.datetime64):
            return pd.to_datetime(obj).strftime('%Y-%m-%d')

        if isinstance(obj, np.ndarray):
            return list(map(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'), obj.tolist()))

    return json.dumps(dic, default=json_serial, ensure_ascii=False)


def pretreatment_events(factor: pd.DataFrame, returns: pd.DataFrame, before: int, after: int) -> pd.DataFrame:
    """预处理事件,将其拉到同一时间

    Args:
        factor (pd.DataFrame): MuliIndex level0-date level1-asset
        returns (pd.DataFrame): index-datetime columns-asset
        before (int): 事件前N日
        after (int): 事件后N日

    Returns:
        pd.DataFrame: [description]
    """
    all_returns = []
    for timestamp, df in factor.groupby(level='date'):

        equities = df.index.get_level_values('asset')

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1,
                           len(returns.index))

        equities_slice = set(equities)

        series = returns.loc[returns.index[starting_index:ending_index],
                             equities_slice]
        series.index = range(starting_index - day_zero_index,
                             ending_index - day_zero_index)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)


def get_event_cumreturns(pretreatment_events: pd.DataFrame) -> pd.DataFrame:
    """以事件当日为基准的累计收益计算

    Args:
        pretreatment_events (pd.DataFrame): index-事件前后日 columns-asset

    Returns:
        pd.DataFrame
    """
    df = pd.DataFrame(index=pretreatment_events.index,
                      columns=pretreatment_events.columns)

    df.loc[:0] = pretreatment_events.loc[:0] / pretreatment_events.loc[0] - 1
    df.loc[1:] = pretreatment_events.loc[0:] / pretreatment_events.loc[0] - 1

    return df


def get_industry_price(codes: Union[str, List], start: str, end: str) -> pd.DataFrame:
    """获取行业指数日度数据. 限制获取条数Limit=4000

    Args:
        codes (Union[str,List]): 行业指数代码
        start (str): 起始日
        end (str): 结束日

    Returns:
        pd.DataFrame: 日度数据
    """
    def query_func(code: str, start: str, end: str) -> pd.Series:

        return finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == code,
                                                                       finance.SW1_DAILY_PRICE.date >= start,
                                                                       finance.SW1_DAILY_PRICE.date <= end))

    if isinstance(codes, str):
        codes = [codes]

    return pd.concat((query_func(code, start, end) for code in codes))


def calc_events_ret(ser: pd.Series, pricing: pd.DataFrame, before: int = 3, end: int = 10, group: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """ 计算形态识别前累计收益率情况

    Args:
        ser (pd.Series): _description_
        pricing (pd.DataFrame): 价格数据 index-date columns-指数
        before (int, optional): 识别前N日. Defaults to 3.
        end (int, optional): 识别后N日. Defaults to 10.
        group (bool, optional): 是否分组. Defaults to True.

    Returns:
        Union[pd.Series, pd.DataFrame]
    """
    events = pretreatment_events(ser, pricing, before,  end)

    rets = get_event_cumreturns(events)

    if group:

        return rets.mean(axis=1)

    else:
        return rets


def get_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """计算胜率

    Args:
        df (pd.DataFrame): index-days columns

    Returns:
        pd.DataFrame
    """
    return df.apply(lambda x: np.sum(np.where(x > 0, 1, 0)) / x.count(), axis=1)

def get_pl(df:pd.DataFrame)->pd.DataFrame:

    """计算盈亏比

    Returns:
        pd.DataFrame
    """
    return df.apply(lambda x:x[x>0].mean() / x[x<0].mean(),axis=1)

def plot_events_ret(ser: pd.Series, title: str = '', ax=None):
    """绘制事件收益率图

    Args:
        ser (pd.Series): 收益率序列
        ax (_type_, optional):Defaults to None.

    Returns:
        ax
    """
    if ax is None:
        fig, ax = plt.figure(figsize=(18, 4))

    line_ax = ser.plot(ax=ax, marker='o', title=title)

    line_ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, pos: '%.2f%%' % (x * 100)))

    line_ax.set_xlabel('天')
    line_ax.set_ylabel('平均收益率')
    ax.axvline(0, ls='--', color='black')

    return ax



# 获取申万一级行业列表
indstries_frame = get_industries(name='sw_l1', date=None)

#industry_price = get_industry_price(indstries_frame.index.tolist(),'2014-01-01','2022-02-18')
# 数据储存
industry_price.to_csv('sw_lv1.csv')

# 读取申万一级行业数据
industry_price = pd.read_csv('sw_lv1.csv', index_col=[
                             'name', 'date'], parse_dates=True).drop(columns='Unnamed: 0')

industry_price.head()

# time 2:14:46
## 形态识别数量受窗口期 及 更新字典的窗口 大小影响
dic = {}  # 储存形态识别结果

for name, df in tqdm(industry_price.groupby(level='name')):

    if len(df) > 120:
        dic[name] = rolling_patterns2pool(df.loc[name, 'close'], 35, reset_window=120)._asdict()

# 将结果储存为json
res_json =patterns_res2json(dic)

# 数据储存
with open('C:\\temp\\res_json20220603.json','w',encoding='utf-8') as file:
    json.dump(res_json,file)

# 读取 形态识别后的文件
with open('res_json.json','r',encoding='utf-8') as file:

    res_json = json.load(file)


def query_trade_dates(start_date: str, end_date: str) -> list:
    df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    dates = df.query('is_open==1')['cal_date'].values.tolist()
    return  dates

# 获取交易日历
trade_calendar = query_trade_dates('20130601','20220221')

idx = pd.to_datetime(trade_calendar)


###########################
row_data = []  # 获取形态识别时的时点

res_dic = json.loads(res_json)  # json转为字典

for code, res1 in res_dic.items():
    for pattern_name, point_tuple in res1['patterns'].items():

        for p1, p2 in point_tuple:
            watch_date = idx.get_loc(p2) + 3  # 模拟三天后识别形态

            row_data.append([code, pattern_name, idx[watch_date]])

# 转为frame格式
stats_df = pd.DataFrame(row_data, columns=['指数', '形态', '时间'])

stats_df['value'] = 1

factor_df = pd.pivot_table(stats_df, index=['时间', '指数'], columns=['形态'], values='value')

factor_df = factor_df.sort_index()

factor_df.index.names = ['date', 'asset']

pricing = pd.pivot_table(industry_price.reset_index(),
                         index='date', columns='name', values='close')

#形态识别后前后平均收益情况
# TODO:收益减去指数自身 评价其超额情况 否则无法真实评价
group_ret = factor_df.groupby(level=0, axis=1).apply(
    lambda x: calc_events_ret(x.dropna(), pricing))

size = group_ret.shape[1]
fig, axes = plt.subplots(size, figsize=(18, 4 * size))

axes = axes.flatten()
for ax, (name, ser) in zip(axes, group_ret.items()):
    plot_events_ret(ser, name, ax)

plt.subplots_adjust(hspace=0.4)

# 计算胜率
evet_ret = factor_df.groupby(level=0, axis=1).apply(
    lambda x: calc_events_ret(x.dropna(), pricing, group=False))

grouped = evet_ret.groupby(level=[0, 1], axis=1)
# 计算胜率
win_ratio = grouped.apply(get_win_rate).loc[[3, 5, 10]].T.swaplevel().sort_index()
# 计算盈亏比
pl_df = grouped.apply(get_pl).loc[[3, 5, 10]].T.swaplevel().sort_index()

win_ratio.columns = pd.MultiIndex.from_tuples([('胜率',3),('胜率',5),('胜率',10)])
pl_df.columns = pd.MultiIndex.from_tuples([('盈亏比',3),('盈亏比',5),('盈亏比',10)])

pattern_count = factor_df.groupby(level=1).sum().stack()
stats = pd.concat((win_ratio,pl_df),axis=1)
stats[('识别次数','All')] = pattern_count

stats