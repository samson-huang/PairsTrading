# 初始引入
import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")

from BuildPeriodicDate import *
import talib
import numpy as np
import pandas as pd
import empyrical as ep
import scipy.stats as st
import statsmodels.api as sm

import alphalens as al
from alphalens import plotting
import alphalens.performance as perf

#from jqdata import *
#from jqfactor import (Factor, calc_factors)

from functools import reduce
from tqdm import tqdm_notebook
from typing import (Tuple, List)
from dateutil.parser import parse

import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'serif'  # pd.plot中文
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('seaborn')

'''
最大回撤相关
    绘制回撤区间图
    回撤区间列表
    from https://github.com/quantopian/pyfolio
'''


# step 1


def get_max_drawdown_underwater(underwater):
    """
    Determines peak, valley, and recovery dates given an 'underwater'
    DataFrame.
    An underwater DataFrame is a DataFrame that has precomputed
    rolling drawdown.
    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.
    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """

    valley = underwater.idxmin()  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery


# step 2


def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    returns = returns.copy()
    df_cum = np.exp(np.log1p(returns).cumsum())
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if ((len(returns) == 0)
                or (len(underwater) == 0)
                or (np.min(underwater) == 0)):
            break

    return drawdowns


# 通过上面两个步骤 就可以取得对应的最大回撤表格


def gen_drawdown_table(returns, top=10):
    """
    Places top drawdowns in a table.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """

    df_cum = np.exp(np.log1p(returns).cumsum())
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(top)),
                                columns=['Net drawdown in %',
                                         'Peak date',
                                         'Valley date',
                                         'Recovery date',
                                         'Duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'Duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                recovery,
                                                                freq='B'))
        df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
                                            .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
                                              .strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'Recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Net drawdown in %'] = (
                                                           (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[
                                                       peak]) * 100

    df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
    df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
    df_drawdowns['Recovery date'] = pd.to_datetime(
        df_drawdowns['Recovery date'])

    return df_drawdowns


# 将上面的df转为图


def show_worst_drawdown_periods(returns, top=5):
    """
    Prints information about the worst drawdown periods.
    Prints peak dates, valley dates, recovery dates, and net
    drawdowns.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).
    """

    drawdown_df = gen_drawdown_table(returns, top=top)
    print_table(
        drawdown_df.sort_values('Net drawdown in %', ascending=False),
        name='Worst drawdown periods',
        float_format='{0:.2f}'.format,
    )


# 根据步骤1，2画图


def plot_drawdown_periods(returns, top=10, ax=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    from matplotlib.ticker import FuncFormatter
    if ax is None:
        ax = plt.gca()

    def two_dec_places(x, y):
        return '%.2f' % x

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = np.exp(np.log1p(returns).cumsum())
    df_drawdowns = gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
        ['Peak date', 'Recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])
    ax.set_ylim(lim)
    ax.set_title('Top %i drawdown periods' % top)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Portfolio'], loc='upper left',
              frameon=True, framealpha=0.5)
    ax.set_xlabel('')
    return ax


def print_table(table,
                name=None,
                float_format=None,
                formatters=None,
                header_rows=None):
    """
    Pretty print a pandas DataFrame.
    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.
    Parameters
    ----------
    table : pandas.Series or pandas.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    float_format : function, optional
        Formatter to use for displaying table elements, passed as the
        `float_format` arg to pd.Dataframe.to_html.
        E.g. `'{0:.2%}'.format` for displaying 100 as '100.00%'.
    formatters : list or dict, optional
        Formatters to use by column, passed as the `formatters` arg to
        pd.Dataframe.to_html.
    header_rows : dict, optional
        Extra rows to display at the top of the table.
    """
    from IPython.display import display, HTML
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if name is not None:
        table.columns.name = name

    html = table.to_html(float_format=float_format, formatters=formatters)

    if header_rows is not None:
        # Count the number of columns for the text to span
        n_cols = html.split('<thead>')[1].split('</thead>')[0].count('<th>')

        # Generate the HTML for the extra rows
        rows = ''
        for name, value in header_rows.items():
            rows += ('\n    <tr style="text-align: right;"><th>%s</th>' +
                     '<td colspan=%d>%s</td></tr>') % (name, n_cols, value)

        # Inject the new HTML
        html = html.replace('<thead>', '<thead>' + rows)

    display(HTML(html))


# 检验信号与滞后期收益率的相关系数
def show_corrocef(close_ser: pd.Series, signal: pd.Series, title: str = ''):
    period = np.arange(1, 32, 1)  # 滞后周期间隔

    temp = []  # 储存数据

    for i in period:
        # 收益未来收益与信号的相关系数 pearsonr
        lag_ret = close_ser.pct_change(i).shift(-i)

        lag_ret, signal = lag_ret.align(signal, join='inner')

        temp.append(
            np.corrcoef(lag_ret.fillna(0), signal.fillna(0))[0][1])

    mpl.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(18, 4))
    plt.title(title)
    plt.bar(range(len(temp)), temp, width=0.4)
    plt.xticks(range(len(temp)), period)
    plt.show()


# 交易相关分析指标

# 交易明细
class tradeAnalyze(object):

    def __init__(self, pos: pd.Series, returns: pd.Series) -> None:
        self.pos = pos
        self.returns = returns

        self.order_position = get_order_position(self.pos)  # 开平仓信息
        self._hold_ratio = CalcHoldRatio(self.order_position, returns)

    def show_all(self) -> None:
        '''展示详细信息'''
        print_table(self.hold_ratio,
                    formatters={'Hold_returns': self.myFormat},
                    name='交易明细',
                    header_rows={'交易次数': len(self.order_position),
                                 '持仓总天数': totaldays(self.pos),
                                 '胜率': self.myFormat(self.win_ratio),
                                 '日胜率': self.myFormat(self.daily_ratio)})

    @property
    def daily_ratio(self) -> float:
        '''日胜率'''
        algorithm_ret = self.pos * self.returns
        return CalcWinRatio(algorithm_ret)

    @property
    def win_ratio(self) -> float:
        '''胜率'''
        return CalcWinRatio(self._hold_ratio)

    @property
    def hold_ratio(self) -> pd.Series:
        holddays = GetHoldDay(self.order_position, self.pos)
        df = pd.concat((self.order_position, holddays, self._hold_ratio), axis=1)

        return df

    @staticmethod
    def myFormat(x: float) -> str:
        return '{:.2%}'.format(x)


# 获取开平仓信息
def get_order_position(pos: pd.Series) -> pd.DataFrame:
    '''
    pos:1表示持仓,0表示空仓
    -----------
        pos:index-date value-0,1标记
    '''
    from collections import defaultdict

    row_num = 0
    last_pos = None
    tradeDict = defaultdict(list)

    for trade, value in pos.items():

        if last_pos:

            if value == 1 and last_pos == 0:
                tmp = tradeDict[row_num]
                tmp += [trade]

            if value == 0 and last_pos == 1:
                tmp = tradeDict[row_num]
                tmp += [trade]

                row_num += 1

        else:

            if value == 0:

                pass

            else:

                tmp = tradeDict[row_num]
                tmp += [trade]

        last_pos = value

    if len(tradeDict[row_num]) < 2:
        tmp = tradeDict[row_num]
        tmp += [np.nan]
    tradeFrame = pd.DataFrame(tradeDict).T
    tradeFrame.columns = ['Buy', 'Sell']

    return tradeFrame


# check最后次是否结束
def checkLastTrade(trade: datetime.date, endDt: pd.Timestamp) -> datetime.date:
    ''''''

    if isinstance(trade, datetime.date):

        return trade
    else:

        return endDt


# 计算持有胜率
def CalcHoldRatio(tradeFrame: pd.DataFrame, returns: pd.Series) -> pd.Series:
    '''计算持有胜率'''

    endDt = returns.index[-1]
    holdRatio = tradeFrame.apply(lambda x: returns.loc[x['Buy']:checkLastTrade(x['Sell'], endDt)].sum(), axis=1)
    holdRatio.name = 'Hold_returns'
    return holdRatio


# 持有天数
def GetHoldDay(tradeFrame: pd.DataFrame, pos: pd.Series) -> pd.Series:
    '''持有天数'''
    endDt = returns.index[-1]
    holddays = tradeFrame.apply(lambda x: len(pos.loc[x['Buy']:checkLastTrade(x['Sell'], endDt)]), axis=1)
    holddays.name = 'Hold_days'
    return holddays


# 计算日胜率
def CalcWinRatio(algorithm_ret: pd.Series) -> float:
    '''日胜率'''
    return len(algorithm_ret[algorithm_ret > 0]) / len(algorithm_ret)


# 计算持有天数
def totaldays(pos: pd.Series) -> int:
    '''持有天数'''
    return pos.sum()


# 开平仓标记
def plot_trade_pos(trade_info: tradeAnalyze, benchmark: pd.Series, ax=None):
    order_position = trade_info.order_position
    sell_marker = [benchmark.index.get_loc(trade) for trade in order_position['Sell'].dropna()]
    buy_marker = [benchmark.index.get_loc(trade) for trade in order_position['Buy'].dropna()]

    if ax is None:
        ax = plt.gca()

    ax.plot(benchmark, 'o', markevery=sell_marker, color='g', label='Sell')
    ax.plot(benchmark, 'o', markevery=buy_marker, color='r', label='Buy')

    return ax


# 风险指标
def Strategy_performance(return_df: pd.DataFrame, periods='daily') -> pd.DataFrame:
    '''计算风险指标 默认为日度:日度调仓'''

    if isinstance(return_df, pd.Series):
        return_df = return_df.to_frame('ret')

    ser: pd.DataFrame = pd.DataFrame()

    ser['年化收益率'] = ep.annual_return(return_df, period=periods)
    ser['波动率'] = return_df.apply(lambda x: ep.annual_volatility(x, period=periods))
    ser['夏普'] = return_df.apply(ep.sharpe_ratio, period=periods)
    ser['最大回撤'] = return_df.apply(lambda x: ep.max_drawdown(x))

    if 'benchmark' in return_df.columns:
        select_col = [col for col in return_df.columns if col != 'benchmark']

        ser['IR'] = return_df[select_col].apply(
            lambda x: information_ratio(x, return_df['benchmark']))
        ser['Alpha'] = return_df[select_col].apply(
            lambda x: ep.alpha(x, return_df['benchmark'], period=periods))

    return ser.T


def information_ratio(returns, factor_returns):
    """
    Determines the Information ratio of a strategy.

    Parameters
    ----------
    returns : :py:class:`pandas.Series` or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns: :class:`float` / :py:class:`pandas.Series`
        Benchmark return to compare returns against.

    Returns
    -------
    :class:`float`
        The information ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/information_ratio for more details.

    """
    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error


def _adjust_returns(returns, adjustment_factor):
    """
    Returns a new :py:class:`pandas.Series` adjusted by adjustment_factor.
    Optimizes for the case of adjustment_factor being 0.

    Parameters
    ----------
    returns : :py:class:`pandas.Series`
    adjustment_factor : :py:class:`pandas.Series` / :class:`float`

    Returns
    -------
    :py:class:`pandas.Series`
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns.copy()
    return returns - adjustment_factor

#因子构造
#ama-French的构造参照特质波动率因子中的快速构造法
# 特异度因子

#class IVR(Factor):
class IVR():
    '''特异度因子'''

    import warnings
    warnings.filterwarnings("ignore")

    name = 'IVR'
    max_window = 22
    dependencies = ['market_cap', 'pb_ratio', 'close']

    def calc(self, data):
        # 获取bm
        bm_df = 1. / data['pb_ratio']
        # 收益率
        returns = data['close'].pct_change()
        # 分组标记
        market_cap_label = self._add_label(data['market_cap'], 10)
        bm_label = self._add_label(bm_df, 10)
        # 构造因子
        smb = self._calc_longshort(data, market_cap_label, returns, 1, 10)
        hml = self._calc_longshort(data, bm_label, returns, 10, 1)
        mkt = (data['market_cap'].div(data['market_cap'].sum(
            axis=1), axis=0) * returns).sum(axis=1)

        # 添加阶矩项项
        exog = sm.add_constant(pd.concat([mkt, smb, hml], axis=1))
        EXOG_COL = ['constant', 'mkt', 'smb', 'hml']
        exog.columns = EXOG_COL

        returns = returns.iloc[-(self.max_window - 1):]
        exog = exog.reindex(returns.index)  # 日期索引对齐
        # OLS回归获取残差

        rsq = returns.apply(self._calc_rsq, exog=exog)
        return 1 - rsq

    def _calc_longshort(self, data, label_df: pd.DataFrame, returns: pd.DataFrame, high: int, low: int) -> pd.Series:
        '''计算多空收益 high - low'''

        # 计算组合权重
        l_w = self._get_weight(label_df, data['market_cap'], high)
        s_w = self._get_weight(label_df, data['market_cap'], low)

        h = (l_w * returns).sum(axis=1)
        l = (s_w * returns).sum(axis=1)

        return h - l

    @staticmethod
    def _add_label(df: pd.DataFrame, N: int) -> pd.DataFrame:
        '''获取分组 分位数分组(每组数量相同)'''
        return df.apply(lambda x: pd.qcut(x, N, labels=[i for i in range(1, N + 1)]), axis=1)

    @staticmethod
    def _get_weight(label_df: pd.DataFrame, market_cap: pd.DataFrame, N: int) -> pd.DataFrame:
        '''获取目标组的权重'''

        cond = (label_df == N)
        cap = cond * market_cap  # 该组市值

        return cap.div(cap.sum(axis=1), axis=0)

    @staticmethod
    def _calc_rsq(returns: pd.Series, exog: pd.DataFrame):
        '''计算R_squared'''
        mod = sm.OLS(returns, exog)
        res = mod.fit()

        return res.rsquared

#数据获取
# 获取数据
def prepare_data(symbol: str, startDt: str, endDt: str) -> pd.DataFrame:
    # 获取沪深300每段的调仓周期
    timeRange = GetPeriodicDate(startDt, endDt)
    tmpList = []  # 临时储存容器

    for beginDt, endDt in tqdm_notebook(timeRange.get_periods):
        stocks = get_index_stocks(symbol, date=endDt)  # 获取股票池
        factor_dic = calc_factors(stocks, [IVR()], beginDt, endDt)  # 因子获取
        tmpList.append(factor_dic['IVR'])

    factor_df = pd.concat(tmpList)
    return factor_df


factor_df = prepare_data('000300.XSHG', '2013-01-01', '2021-03-17')