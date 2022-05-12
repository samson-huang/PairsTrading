# -*- coding: utf-8 -*-
# @Time : 2022/5/8 21:05
# @Author : huangtuo
# @Email : 375317196@qq.com
# @File : 复现RSRS.py
# @Project : RSRS择时改进.py
# 引入常用库
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import datetime as dt
import itertools # 迭代器工具

import pysnooper # debug
import pickle
from IPython.core.display import HTML

#from jqdata import *
#from jqfactor import *
import talib # 技术分析
# 画图
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import seaborn as sns
# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')

# 忽略报错
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("G://GitHub//PairsTrading//new_stategy//foundation_tools//")
import foundation_tushare
import json

# 使用ts
# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)

# 均值与布林策略回测情况

# 以HS300为准
#创业板指  399006.SZ
index_id='000300.SH'
# 时间范围设定
start, end = '20050408', '20220512'
# 获取数据
datas = pro.index_daily(ts_code=index_id, start_date=start,
                        end_date=end, fields='trade_date,close')
datas['trade_date'] = pd.to_datetime(datas['trade_date'])
datas.set_index('trade_date', inplace=True)
datas.sort_index(inplace=True)
# 获取每日成分股权重，后续会使用到


def get_weights(index_id, start, end):
    weight_dic = {}
    #tradeList = get_trade_days(start_date=start, end_date=end)
    #for trade in tradeList:
    #    weight_dic[trade] = get_index_weights(index_id, trade)['weight'].values
    #    print('seccuss', trade.strftime("%Y%m%d"))
    weight_temp = pro.index_weight(index_code=index_id, start_date=start,end_date=end)
    ['weight'].values

    weight_dic[trade]

    pkl_file = open('weight_dic.pkl', 'wb')
    pickle.dump(weight_dic, pkl_file)
    print('已储存')
    return weight_dic


# 粗略计算手续费
'''
read_dic如果已经调用get_weights则 true,没有则false
手续费：通过指数成分股权重*费率计算
'''


def cal_fee(datas,singal_name='singal',fee=0.006, read_dic=False):

    df = datas
    # 标记买入卖出时点
    singal = df[singal_name]*np.ones(len(df))
    next_singal = df[singal_name].shift(1)*np.ones(len(df))
    pre_singal = df[singal_name].shift(-1)*np.ones(len(df))
    # 确认时点
    df['fee'] = (singal == 1) & (
        (pre_singal < singal) | (singal > next_singal))

    # 手续费采用成分股权重*费率粗略计算
    def _get_weights(x_df,fee):
        trade = x_df.name
        # print(trade.date())
        #weights_df = get_index_weights('000300.XSHG', trade.date())
        #weights_df = pro.index_weight(index_code=index_id, trade_date=trade.date())
        cost = fee * x_df['fee'].values[0]  # 权重*费率
        return cost

    cond = df['fee'] == True

    # 如果依旧提前下载好了数据
    if read_dic:
        # 读取权重
        pkl_file=open('weight_dic.pkl','rb')
        weight_dic=pickle.load(pkl_file)
        # 权重*费率
        df['cost'] = list(map(lambda x: np.mean(
            weight_dic[x.date()]*fee) if len(weight_dic[x.date()]) != 0 else 0, df.index))*df['fee']
    else:
        cost = df[cond].groupby(level=0).apply(_get_weights,fee=fee)
        df.loc[cond, 'cost'] = cost
        df['cost'] = df['cost'].fillna(0)

    return df

# 风险指标


def risk_table(x_df):
    df = x_df[:-1].copy()
    temp = []
    for cum_name in ['CUM_RET', 'SUB_CUM_RET']:

        # 计算年华收益率
        annual_ret = df[cum_name][-1]**(244/len(df[cum_name])) - 1

        # 计算累计收益率
        cum_ret_rate = df[cum_name][-1] - 1

        # 最大回撤
        max_nv = np.maximum.accumulate(df[cum_name])
        mdd = -np.min(df[cum_name] / max_nv - 1)

        # 夏普
        if cum_name == 'CUM_RET':
            sharpe_ratio = df['RET'].mean()/df['RET'].std()*np.sqrt(250)
        elif cum_name == 'SUB_CUM_RET':
            sharpe_ratio = df['SUB_FEE_RET'].mean()/df['SUB_FEE_RET'].std()*np.sqrt(250)

        temp.append(['{:.2%}'.format(annual_ret), '{:.2%}'.format(
            cum_ret_rate), '{:.2%}'.format(mdd), '{:.2%}'.format(sharpe_ratio)])

    return pd.DataFrame(temp, index=['策略', '策略扣双向费率'], columns=['年华收益率', '累计收益', '最大回撤', '夏普'])

# 均线策略


def MA_Strategy(data):

    df = data
    df['MA20'] = df['close'].rolling(20).mean()  # 计算20日均线
    df['singal'] = df['close'] > df['MA20']  # 生产信号
    df['ret'] = df['close'].pct_change()  # 计算收益率

    # 计算费率(先标记出信号)
    df = cal_fee(df)

    # 计算持仓收益
    # 滞后一期收益：今日收盘计算信号，次日买入
    df['RET'] = df['ret'].shift(-1)*df['singal']

    # 计算扣除双边手续费收益
    df['SUB_FEE_RET'] = df['RET']-df['cost']

    # 策略净值
    df['CUM_RET'] = (1+df['RET']).cumprod()
    df['SUB_CUM_RET'] = (1+df['SUB_FEE_RET']).cumprod()

    # 基准净值
    benchmark = (1+df['ret']).cumprod()

    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(df['CUM_RET'], label='均值策略')
    ax1.plot(df['SUB_CUM_RET'], label='均值策略(扣双边手续费0.6%)')
    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('均线策略净值曲线')
    plt.show()

    # 风险报告
    display(HTML(risk_table(df).to_html()))

# 双均线策略


def BBANDS_Strategy(data):

    df = data.copy()

    df['ret'] = df['close'].pct_change()  # 计算收益率
    # 计算布林的上下轨
    upperband, middleband, lowerband = talib.BBANDS(
        df['close'].values, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)

    # 标记信号
    singal = []

    for row in range(len(df)):

        if df.iloc[row].close > upperband[row]:
            singal.append(1)
        else:
            if row != 0:
                if singal[-1] and df.iloc[row].close > lowerband[row]:
                    singal.append(1)
                else:
                    singal.append(0)
            else:
                singal.append(0)

    df['singal'] = singal
    # 计算费率(先标记出信号)
    df = cal_fee(df)

    # 计算持仓收益
    # 滞后一期收益：今日收盘计算信号，次日买入
    df['RET'] = df['ret'].shift(-1)*df['singal']

    # 计算扣除双边手续费收益
    df['SUB_FEE_RET'] = df['RET']-df['cost']

    # 策略净值
    df['CUM_RET'] = (1+df['RET']).cumprod()
    df['SUB_CUM_RET'] = (1+df['SUB_FEE_RET']).cumprod()

    # 基准净值
    benchmark = (1+df['ret']).cumprod()

    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(df['CUM_RET'], label='布林策略')
    ax1.plot(df['SUB_CUM_RET'], label='布林策略(扣双边手续费0.6%)')
    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('布林通道策略净值曲线')
    plt.show()

    # 风险报告
    display(HTML(risk_table(df).to_html()))

# 获取权重
#weight_dic=get_weights(index_id, start, end)

# 读取权重
pkl_file=open('weight_dic.pkl','rb')
weight_dic=pickle.load(pkl_file)

# 均线策略回测
MA_Strategy(datas)


# 布林通道策略回测
BBANDS_Strategy(datas)

#RSRS指标择时策略构建
# 以HS300为准
#index_id = '000300.XSHG'
# 以HS300为准
#创业板指  399006.SZ
index_id='000300.SH'

# 时间范围设定
#start, end = '2005-04-08', '2019-9-30'
# 获取数据
start, end = '20050408', '202205112'
# 获取数据
datas = pro.index_daily(ts_code=index_id, start_date=start,
                        end_date=end, fields='trade_date,close,high,low,amount')
datas['trade_date'] = pd.to_datetime(datas['trade_date'])
datas.set_index('trade_date', inplace=True)
datas.rename(columns={'amount': 'volume'}, inplace=True)
datas.sort_index(inplace=True)
df = datas.copy()


# 计算RSRS

def Cal_RSRS(df, N):
    df = df.copy()
    # 填充空缺,注意质量填充的是18，t日计算的信号，在T+1日，所以收益不需要在滞后一期
    temp = [np.nan] * N

    for row in range(len(df) - N):
        y = df['high'][row:row + N]
        x = df['low'][row:row + N]

        # 计算系数
        beta = np.polyfit(x, y, 1)[0]

        temp.append(beta)

    df['RSRS'] = temp
    return df


# 计算基础RSRS
RSRS_df = Cal_RSRS(df, 18)


# 统计画图
def stat_depict_plot(df, col, title):
    df = df[~df[col].isna()].copy()

    avgRet = np.mean(df[col])
    medianRet = np.median(df[col])
    stdRet = np.std(df[col])
    skewRet = st.skew(df[col])
    kurtRet = st.kurtosis(df[col])

    plt.style.use('ggplot')
    # 画日对数收益率分布直方图
    fig = plt.figure(figsize=(18, 9))
    plt.suptitle(title)
    v = df[col]
    x = np.linspace(avgRet - 3 * stdRet, avgRet + 3 * stdRet, 100)
    y = st.norm.pdf(x, avgRet, stdRet)
    kde = st.gaussian_kde(v)

    # plot the histogram
    plt.subplot(121)
    plt.hist(v, 50, weights=np.ones(len(v)) / len(v), alpha=0.4)
    plt.axvline(x=avgRet, color='red', linestyle='--',
                linewidth=0.8, label='Mean Count')
    plt.axvline(x=avgRet - stdRet, color='blue', linestyle='--',
                linewidth=0.8, label='-1 Standard Deviation')
    plt.axvline(x=avgRet + stdRet, color='blue', linestyle='--',
                linewidth=0.8, label='1 Standard Deviation')
    plt.ylabel('Percentage', fontsize=10)
    plt.legend(fontsize=12)

    # plot the kde and normal fit
    plt.subplot(122)
    plt.plot(x, kde(x), label='Kernel Density Estimation')
    plt.plot(x, y, color='black', linewidth=1, label='Normal Fit')
    plt.ylabel('Probability', fontsize=10)
    plt.axvline(x=avgRet, color='red', linestyle='--',
                linewidth=0.8, label='Mean Count')
    plt.legend(fontsize=12)
    return plt.show()


stat_depict_plot(RSRS_df, 'RSRS', '05年至19年斜率数据分布')

# 低阶距统计描述
def stat_depict(df, col, pr=True):
    df = df[~df[col].isna()].copy()
    # 计算总和的统计量
    avgRet = np.mean(df[col])
    medianRet = np.median(df[col])
    stdRet = np.std(df[col])
    skewRet = st.skew(df[col])
    kurtRet = st.kurtosis(df[col])
    if pr:
        print(
            """
        平均数 : %.4f
        中位数 : %.4f
        标准差 : %.4f
        偏度   : %.4f
        峰度   : %.4f
        1 Standard Deviation : %.4f
        -1 Standard Deviation : %.4f
        """ % (avgRet, medianRet, stdRet, skewRet, kurtRet, avgRet+stdRet, avgRet-stdRet)
        )
    else:
        return dict(zip('平均数,中位数,标准差,偏度,峰度,1 Standard Deviation,-1 Standard Deviation'.split(','),
                        map(lambda x: '{:.4%}'.format(x), [avgRet, medianRet, stdRet, skewRet, kurtRet,
                                                           avgRet+stdRet, avgRet-stdRet])))


print('05年至19年历史RSRS斜率数据低阶距统计')
stat_depict(RSRS_df, 'RSRS')

# 各期斜率情况
plt.figure(figsize=(18,8))
plt.title('沪深300各时期斜率均值')
plt.plot(RSRS_df['RSRS'].rolling(250).mean())


############################################
# 计算基础RSRS
RSRS_df = Cal_RSRS(df, 18)
stat_depict_plot(RSRS_df, 'RSRS', '05年至22年斜率数据分布')
# 各期斜率情况
plt.figure(figsize=(18,8))
plt.title('沪深300各时期斜率均值')
plt.plot(RSRS_df['RSRS'].rolling(5).mean())
##############################################


# 构造标准分RSRS

def Cal_RSRS_Zscore(datas,N, M):
    df=Cal_RSRS(datas,N)
    df['RSRS_temp'] = df['RSRS'].fillna(0)
    # df = Cal_RSRS(df, N)  # 计算基础斜率
    ZSCORE = (df['RSRS_temp']-df['RSRS_temp'].rolling(M).mean()
              )/df['RSRS_temp'].rolling(M).std()
    df['RSRS_Z'] = ZSCORE
    df = df.drop(columns='RSRS_temp')
    return df


# 计算标准分斜率
RSRS_Z = Cal_RSRS_Zscore(RSRS_df, 18,600)

# 查看标准分斜率分布
stat_depict_plot(RSRS_Z,'RSRS_Z','05年至19年标准分斜率数据分布')

# 查看低阶距分布
print('05年至19年历史RSRS标准分斜率低阶距统计')
stat_depict(RSRS_Z,'RSRS_Z')

#RSRS指标择时效果
# 择时指标回测
def RSRS_Strategy(datas, N, M):
    df = datas.copy()
    RSRS_Z = Cal_RSRS_Zscore(df, N, M)  # 计算标准分指标

    # 需要扣除前期计算的600日
    RSRS_Z = RSRS_Z[max(N, M):]

    print('回测起始日：', min(RSRS_Z.index))
    # 基础信号回测
    basic_singal = []
    for row in range(len(RSRS_Z)):

        if RSRS_Z['RSRS'][row] > 1.02:
            basic_singal.append(1)
        else:
            if row != 0:
                if basic_singal[-1] and RSRS_Z['RSRS'][row] > 0.78:
                    basic_singal.append(1)
                else:
                    basic_singal.append(0)
            else:
                basic_singal.append(0)

    # 储存基础信号
    RSRS_Z['basic_singal'] = basic_singal

    # 计算标准信号，S=0.7 研报给出的 我都不知到怎么来的
    z_singal = []
    S = 0.7
    for row in range(len(RSRS_Z)):

        if RSRS_Z['RSRS_Z'][row] > S:
            z_singal.append(1)

        else:
            if row != 0:
                if z_singal[-1] and RSRS_Z['RSRS_Z'][row] > -S:
                    z_singal.append(1)
                else:
                    z_singal.append(0)
            else:
                z_singal.append(0)

    # 储存标准分信号
    RSRS_Z['z_singal'] = z_singal

    # 收益
    RSRS_Z['ret'] = RSRS_Z['close'].pct_change()

    # 斜率净值
    BASIC_CUM = (1 + RSRS_Z['basic_singal'] * RSRS_Z['ret']).cumprod()
    # 标准分净值
    Z_CUM = (1 + RSRS_Z['z_singal'] * RSRS_Z['ret']).cumprod()
    # 基准净值
    benchmark = (1 + RSRS_Z['ret']).cumprod()

    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(BASIC_CUM, label='斜率指标策略')
    ax1.plot(Z_CUM, label='标准分指标策略')
    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('RSRS指标策略净值曲线')
    plt.show()

    display(HTML(summary(RSRS_Z).to_html()))


# 风险报告
def summary(df, singal_name=['basic_singal', 'z_singal']):
    summary_dic = {}
    index_name = '年华收益率,累计收益率,夏普比率,最大回撤,持仓总天数,交易次数,平均持仓天数,获利天数, \
    亏损天数,胜率(按天),平均盈利率(按天),平均亏损率(按天),平均盈亏比(按天),盈利次数,亏损次数, \
    单次最大盈利,单次最大亏损,胜率(按此),平均盈利率(按次),平均亏损率(按次),平均盈亏比(按次)'.split(
        ',')

    col_dic = dict(zip(['RSRS_singal', 'RSRS_Z_singal', 'RSRS_Revise_singal', 'RSRS_Positive_singal']
                       , ['斜率指标策略', '标准分指标策略', '修正标准分策略', '右偏标准分策略']))

    # 判断是否是默认的singal_name
    if singal_name[0] in col_dic:
        col_name = [col_dic[x] for x in singal_name]
    else:
        col_name = '斜率指标策略,标准分指标策略'.split(',')

    def format_x(x):
        return '{:.2%}'.format(x)

    for singal in singal_name:
        RET = df['ret'] * df[singal]
        CUM_RET = (1 + RET).cumprod()

        # 计算年华收益率
        annual_ret = CUM_RET[-1] ** (250 / len(RET)) - 1

        # 计算累计收益率
        cum_ret_rate = CUM_RET[-1] - 1

        # 最大回撤
        max_nv = np.maximum.accumulate(np.nan_to_num(CUM_RET))
        mdd = -np.min(CUM_RET / max_nv - 1)

        # 夏普
        sharpe_ratio = np.mean(RET) / np.nanstd(RET, ddof=1) * np.sqrt(250)

        # 标记买入卖出时点
        mark = df[singal]
        pre_mark = np.nan_to_num(df[singal].shift(-1))
        # 买入时点
        trade = (mark == 1) & (pre_mark < mark)

        # 交易次数
        trade_count = np.nansum(trade)

        # 持仓总天数
        total = np.sum(mark)
        # 平均持仓天数
        mean_hold = total / trade_count
        # 获利天数
        win = np.sum(np.where(RET > 0, 1, 0))
        # 亏损天数
        lose = np.sum(np.where(RET < 0, 1, 0))
        # 胜率
        win_ratio = win / total
        # 平均盈利率（天）
        mean_win_ratio = np.sum(np.where(RET > 0, RET, 0)) / win
        # 平均亏损率（天）
        mean_lose_ratio = np.sum(np.where(RET < 0, RET, 0)) / lose
        # 盈亏比(天)
        win_lose = win / lose

        # 盈利次数
        temp_df = df.copy()
        diff = temp_df[singal] != temp_df[singal].shift(1)
        temp_df['mark'] = diff.cumsum()
        # 每次开仓的收益率情况
        temp_df = temp_df.query(singal + '==1').groupby('mark')['ret'].sum()

        # 盈利次数
        win_count = np.sum(np.where(temp_df > 0, 1, 0))
        # 亏损次数
        lose_count = np.sum(np.where(temp_df < 0, 1, 0))
        # 单次最大盈利
        max_win = np.max(temp_df)
        # 单次最大亏损
        max_lose = np.min(temp_df)
        # 胜率
        win_rat = win_count / len(temp_df)
        # 平均盈利率（次）
        mean_win = np.sum(np.where(temp_df > 0, temp_df, 0)) / len(temp_df)
        # 平均亏损率（天）
        mean_lose = np.sum(np.where(temp_df < 0, temp_df, 0)) / len(temp_df)
        # 盈亏比(次)
        mean_wine_lose = win_count / lose_count

        summary_dic[singal] = [format_x(annual_ret), format_x(cum_ret_rate), sharpe_ratio, format_x(
            mdd), total, trade_count, mean_hold, win, lose, format_x(win_ratio), format_x(mean_win_ratio),
                               format_x(mean_lose_ratio), win_lose, win_count, lose_count, format_x(
                max_win), format_x(max_lose),
                               format_x(win_rat), format_x(mean_win), format_x(mean_lose), mean_wine_lose]

    summary_df = pd.DataFrame(summary_dic, index=index_name)
    summary_df.columns = col_name
    return summary_df

# 无双边手续费情况
RSRS_Strategy(df,18,600)


# 含费率
# 择时指标回测
def RSRS_Strategy_FEE(datas, N, M):
    df = datas.copy()
    RSRS_Z = Cal_RSRS_Zscore(df, N, M)  # 计算标准分指标

    # 需要扣除前期计算的600日
    RSRS_Z = RSRS_Z[max(N, M):]

    print('回测起始日：', min(RSRS_Z.index))

    # 计算标准信号，S=0.7 研报给出的 我都不知到怎么来的
    z_singal = []
    S = 0.7
    for row in range(len(RSRS_Z)):

        if RSRS_Z['RSRS_Z'][row] > S:
            z_singal.append(1)

        else:
            if row != 0:
                if z_singal[-1] and RSRS_Z['RSRS_Z'][row] > -S:
                    z_singal.append(1)
                else:
                    z_singal.append(0)
            else:
                z_singal.append(0)

    # 储存标准分信号
    RSRS_Z['z_singal'] = z_singal

    RSRS_Z = cal_fee(RSRS_Z, 'z_singal')

    # 收益
    RSRS_Z['ret'] = RSRS_Z['close'].pct_change()

    # 标准分净值 不含费
    Z_CUM = (1 + RSRS_Z['z_singal'] * RSRS_Z['ret']).cumprod()
    # 标准分净值 含费
    Z_CUM_fee = (1 + RSRS_Z['z_singal'] * RSRS_Z['ret'] - RSRS_Z['cost']).cumprod()
    # 基准净值
    benchmark = (1 + RSRS_Z['ret']).cumprod()
    print('不含费净值：%2.2f,含费净值：%2.2f' % (Z_CUM[-1], Z_CUM_fee[-1]))
    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(Z_CUM, label='标准分指标策略')
    ax1.plot(Z_CUM_fee, label='标准分指标策略(扣手续费)')
    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('RSRS标准分策略净值曲线')
    plt.show()


RSRS_Strategy_FEE(df, 18, 600)

#RSRS标准分指标优化
# 计算修正RSRS
def Cal_RSRS_Revise(datas, N, M):
    df = datas.copy()
    df = Cal_RSRS_Zscore(df, N, M)
    # 获取R方
    temp = [np.nan]*N
    for row in range(len(df)-N):
        x = sm.add_constant(df['low'][row:row+N])
        y = df['high'][row:row+N]
        r = sm.OLS(y, x).fit().rsquared
        temp.append(r)

    df['rsquared'] = temp
    df['RSRS_Revise'] = df['rsquared']*df['RSRS_Z']
    return df


# 获取修正标准分
RSRS_Revise = Cal_RSRS_Revise(df, 18, 600)


# 修正标准分(含费率)
# 修正标准分择时指标回测
def RSRS_Strategy_Revise(datas, N, M, pl=True):
    df = datas.copy()
    RSRS_Z = Cal_RSRS_Revise(df, N, M)  # 计算标准分指标

    # 需要扣除前期计算的600日
    RSRS_Z = RSRS_Z[max(N, M):]

    print('回测起始日：', min(RSRS_Z.index))

    # 计算标准信号，S=0.7 研报给出的 我都不知到怎么来的
    z_singal = []
    S = 0.7
    for row in range(len(RSRS_Z)):

        if RSRS_Z['RSRS_Revise'][row] > S:
            z_singal.append(1)

        else:
            if row != 0:
                if z_singal[-1] and RSRS_Z['RSRS_Revise'][row] > -S:
                    z_singal.append(1)
                else:
                    z_singal.append(0)
            else:
                z_singal.append(0)

    # 储存标准分信号
    RSRS_Z['z_singal'] = z_singal

    RSRS_Z = cal_fee(RSRS_Z, 'z_singal')

    # 收益
    RSRS_Z['ret'] = RSRS_Z['close'].pct_change()

    # 标准分净值 不含费
    Z_CUM = (1 + RSRS_Z['z_singal'] * RSRS_Z['ret']).cumprod()
    # 标准分净值 含费
    Z_CUM_fee = (1 + RSRS_Z['z_singal'] * RSRS_Z['ret'] - RSRS_Z['cost']).cumprod()
    # 基准净值
    benchmark = (1 + RSRS_Z['ret']).cumprod()
    print('不含费净值：%2.2f,含费净值：%2.2f' % (Z_CUM[-1], Z_CUM_fee[-1]))
    if pl:
        # 画图
        plt.figure()
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(Z_CUM, label='修正标准分指标策略')
        ax1.plot(Z_CUM_fee, label='修正标准分指标策略(扣手续费)')
        ax1.plot(benchmark, label='沪深300')

        ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
        plt.legend(loc='best')
        plt.xlabel('时间')
        plt.ylabel('净值')
        plt.title('RSRS修正标准分策略净值曲线')
        plt.show()
    else:
        return RSRS_Z


# 修正标准分
RSRS_Strategy_Revise(df, 18, 600)

# 标准分斜率分布
stat_depict_plot(RSRS_Z,'RSRS_Z','05年至19年标准分斜率数据分布')

# 修正标准分分布
stat_depict_plot(RSRS_Revise,'RSRS_Revise','05年至19年修正标准分斜率数据分布')

# RSRS标准分与RSRS修正标准分分布数据比较
stat_depict_df = pd.concat([pd.Series(stat_depict(RSRS_Revise, 'RSRS_Revise', pr=False)),
                            pd.Series(stat_depict(RSRS_Z, 'RSRS_Z', pr=False))], axis=1)

stat_depict_df.columns = ['修正标准分', '标准分']
print('RSRS标准分与RSRS修正标准分分布数据比较')
display(HTML(stat_depict_df.T.to_html()))


# 获取因子后N日的上涨概率/收益期望
def plot_average_cumulative_return(df, factor_name, after=10, title=None, probability=True, prt=True):
    '''
    df 因子计算后的df
    factor_name df中因子的名称
    title 图标的标题
    after 之后N日
    probability True计算概率 False 计算平均收益
    '''
    RSRS = df[['close', factor_name]].copy()
    # 计算未来N日收益率
    RSRS['ret'] = RSRS.close.pct_change(after).shift(-after)
    group = pd.cut(RSRS[factor_name], 50)
    RSRS['group'] = group
    if probability:
        # 计算上涨概率
        after_ret = RSRS.groupby('group')['ret'].apply(lambda x: np.sum(np.where(x > 0, 1, 0)) / len(x))
    else:
        after_ret = RSRS.groupby('group')['ret'].mean()

    if prt:
        # 画图
        plt.figure(figsize=(18, 6))
        # 设置标题
        plt.title(title)
        size = len(after_ret)
        plt.bar(range(size), after_ret.values, width=0.8, alpha=0.5)
        # rotation旋转x轴标签
        plt.xticks(range(size), after_ret.index.categories.right, rotation=30)
        # 设置y轴标题
        plt.ylabel('上涨概率')
    else:
        return after_ret

# 标准分未来10日上涨概率
plot_average_cumulative_return(RSRS_Z,'RSRS_Z',after=10,title='标准分未来10日上涨概率')

# 标准分未来10日期望收益
plot_average_cumulative_return(RSRS_Z,'RSRS_Z',after=10,title='修正标准分未来10日期望收益',probability=False)

# 标准分与相关系数
print('-'*30, '\n右侧标准分:\n')
zr_df = plot_average_cumulative_return(
    RSRS_Z, 'RSRS_Z', after=10, prt=False, probability=False)
zr_df.index = zr_df.index.categories.right
zr_df = zr_df.fillna(0)
zr_df = zr_df.reset_index()
zr_df.columns = ['factor', 'ret']
rq = zr_df.query('factor>0')

zp_df = plot_average_cumulative_return(
    RSRS_Z, 'RSRS_Z', after=10, prt=False, probability=True)
zp_df.index = zp_df.index.categories.right
zp_df = zp_df.fillna(0)
zp_df = zp_df.reset_index()
zp_df.columns = ['factor', 'ret']
pq = zp_df.query('factor>0')


print('标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['factor'], rq['ret'])[0][1])

print('标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(pq['factor'], pq['ret'])[0][1])


print('-'*30, '\n左侧标准分:\n')
rq = zr_df.query('factor<0')
pq = zp_df.query('factor<0')

print('标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['factor'], rq['ret'])[0][1])

print('标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(pq['factor'], pq['ret'])[0][1])

print('-'*30, '\n标准分整体:\n')
rq = zr_df
pq = zp_df

print('标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['ret'], rq['factor'])[0][1])
print('标准分与上涨概率相关系数:%.2f'
      % np.corrcoef(pq['ret'], pq['factor'])[0][1])

# 修正标准分与相关系数
print('-'*30, '\n右侧修正标准分:\n')
zr_df = plot_average_cumulative_return(
    RSRS_Revise, 'RSRS_Revise', after=10, prt=False, probability=False)
zr_df.index = zr_df.index.categories.right
zr_df = zr_df.fillna(0)
zr_df = zr_df.reset_index()
zr_df.columns = ['factor', 'ret']
rq = zr_df.query('factor>0')

zp_df = plot_average_cumulative_return(
    RSRS_Revise, 'RSRS_Revise', after=10, prt=False, probability=True)
zp_df.index = zp_df.index.categories.right
zp_df = zp_df.fillna(0)
zp_df = zp_df.reset_index()
zp_df.columns = ['factor', 'ret']
pq = zp_df.query('factor>0')


print('修正标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['factor'], rq['ret'])[0][1])

print('修正标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(pq['factor'], pq['ret'])[0][1])


print('-'*30, '\n左侧标准分:\n')
rq = zr_df.query('factor<0')
pq = zp_df.query('factor<0')

print('修正修正标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['factor'], rq['ret'])[0][1])

print('修正标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(pq['factor'], pq['ret'])[0][1])

print('-'*30, '\n修正标准分整体:\n')
rq = zr_df
pq = zp_df

print('修正标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['ret'], rq['factor'])[0][1])
print('修正标准分与上涨概率相关系数:%.2f'
      % np.corrcoef(pq['ret'], pq['factor'])[0][1])

# 信号与相关系数
'''
corr_df=RSRS_Revise.copy()
print('-'*30,'\n右侧标准分:\n')
# 计算标准分与未来10日期望收益的相关系数
## 计算未来10日收益
corr_df['ret'] = corr_df.close.pct_change(10).shift(-10)
## 标准分大于0
corr_df1 = corr_df.query('RSRS_Revise>0').dropna()
# 标准分>0，与未来10日的收益率相关系数
print('标准分与期望收益相关系数:%.2f'
      % np.corrcoef(corr_df1['ret'], corr_df1['RSRS_Revise'])[1][0])

# 计算未来10日上涨概率
## 计算每日收益率
corr_df_p=corr_df.copy()
corr_df_p['ret']=corr_df_p.close.pct_change()
corr_df_p['P']=corr_df_p['ret'].rolling(10).apply(lambda x:np.sum(np.where(x>0,1,0))/len(x)).shift(-10)
corr_df2 = corr_df_p.query('RSRS_Revise>0').dropna()
# 标准分>0，与未来10日的上涨概率相关系数
print('标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(corr_df2['ret'], corr_df2['P'])[1][0])

print('-'*30,'\n左侧标准分:\n')

# 计算标准分与未来10日上涨概率相关系数
## 标准分大于0
corr_df3 = corr_df.query('RSRS_Revise<0').dropna()
# 标准分>0，与未来10日的收益率相关系数
print('标准分与期望收益相关系数:%.2f'
      % np.corrcoef(corr_df3['ret'], corr_df3['RSRS_Revise'])[1][0])

# 计算未来10日上涨概率
## 计算每日收益率
corr_df_4 = corr_df_p.query('RSRS_Revise<0').dropna()
# 标准分>0，与未来10日的上涨概率相关系数
print('标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(corr_df_4['ret'], corr_df_4['P'])[1][0])

print('-'*30,'\n标准分整体:\n')
corr_df_5=corr_df.dropna()
print('标准分与期望收益相关系数:%.2f'
      % np.corrcoef(corr_df_5['ret'], corr_df_5['RSRS_Revise'])[1][0])
'''

# 修正标准分未来10日上涨概率
plot_average_cumulative_return(RSRS_Revise,'RSRS_Revise',after=10,title='修正标准分未来10日上涨概率')

# 修正标准分未来10日期望收益
plot_average_cumulative_return(RSRS_Revise,'RSRS_Revise',after=10,title='修正标准分未来10日期望收益',probability=False)


# 比较修正标准分与标准分策略
def Compare_RSRS_Strategy(datas, N, M):
    df = datas.copy()

    RSRS_Z = Cal_RSRS_Revise(df, N, M)  # 计算标准分指标

    singal_name = ['{}_singal'.format(x) for x in 'RSRS_Z,RSRS_Revise'.split(',')]
    # 需要扣除前期计算的600日
    RSRS_Z = RSRS_Z[max(N, M):]

    print('回测起始日：', min(RSRS_Z.index))
    # 获取标准分信号
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Z', 0.7, -0.7)
    # 获取修正标准分
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Revise', 0.7, -0.7)

    # 收益
    RSRS_Z['ret'] = RSRS_Z['close'].pct_change()

    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    # 斜率净值
    for n in singal_name:
        CUM = (1 + RSRS_Z[n] * RSRS_Z['ret']).cumprod()
        ax1.plot(CUM, label=n)

    # 基准净值
    benchmark = (1 + RSRS_Z['ret']).cumprod()

    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('RSRS指标策略净值曲线')
    plt.show()

    display(HTML(summary(RSRS_Z, singal_name).to_html()))


# 标准信号
def get_singal(datas, singal_name, upper, lower):
    # 储存信号
    basic_singal = []
    for row in range(len(datas)):

        if datas[singal_name][row] > upper:
            basic_singal.append(1)
        else:
            if row != 0:
                if basic_singal[-1] and datas[singal_name][row] > lower:
                    basic_singal.append(1)
                else:
                    basic_singal.append(0)
            else:
                basic_singal.append(0)
    datas[singal_name + '_singal'] = basic_singal
    return datas


Compare_RSRS_Strategy(df, 18, 600)

# 计算右偏标准分
def Cal_RSRS_Positive(datas,N,M):
    df=Cal_RSRS_Revise(datas,N,M)
    df['RSRS_Positive']=df['RSRS_Revise']*df['RSRS']
    return df
RSRS_Positive=Cal_RSRS_Positive(df,18,600)

stat_depict_plot(RSRS_Positive,'RSRS_Positive','05年至19年右偏标准分斜率数据分布')

print('05年至19年历史RSRS右偏标准分斜率低阶距统计')
stat_depict(RSRS_Positive,'RSRS_Positive')

# 右偏标准分未来10日上涨概率
plot_average_cumulative_return(
    RSRS_Positive, 'RSRS_Positive', after=10, title='右偏标准分分未来10日上涨概率')

# 右偏标准分未来10日期望收益
plot_average_cumulative_return(
    RSRS_Positive, 'RSRS_Positive', after=10, title='右偏标准分分未来10日期望收益', probability=False)

# 右偏标准分与相关系数
print('-'*30, '\n右侧修正标准分:\n')
zr_df = plot_average_cumulative_return(
    RSRS_Positive, 'RSRS_Positive', after=10, prt=False, probability=False)
zr_df.index = zr_df.index.categories.right
zr_df = zr_df.fillna(0)
zr_df = zr_df.reset_index()
zr_df.columns = ['factor', 'ret']
rq = zr_df.query('factor>0')

zp_df = plot_average_cumulative_return(
    RSRS_Positive, 'RSRS_Positive', after=10, prt=False, probability=True)
zp_df.index = zp_df.index.categories.right
zp_df = zp_df.fillna(0)
zp_df = zp_df.reset_index()
zp_df.columns = ['factor', 'ret']
pq = zp_df.query('factor>0')


print('修正标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['factor'], rq['ret'])[0][1])

print('修正标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(pq['factor'], pq['ret'])[0][1])


print('-'*30, '\n左侧标准分:\n')
rq = zr_df.query('factor<0')
pq = zp_df.query('factor<0')

print('修正修正标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['factor'], rq['ret'])[0][1])

print('修正标准分与未来上涨概率相关系数:%.2f'
      % np.corrcoef(pq['factor'], pq['ret'])[0][1])

print('-'*30, '\n修正标准分整体:\n')
rq = zr_df
pq = zp_df

print('修正标准分与期望收益相关系数:%.2f'
      % np.corrcoef(rq['ret'], rq['factor'])[0][1])
print('修正标准分与上涨概率相关系数:%.2f'
      % np.corrcoef(pq['ret'], pq['factor'])[0][1])

# 右偏标准分择时指标回测
'''
研报中并没把标准分、修正标准分放在同一参数上回测，标准分和修正标准依旧用的原参数(18，600)
在这里我将三个策略放在同一参数下回测
'''


# @pysnooper.snoop()
def RSRS_Positive_Strategy(datas, N, M):
    df = datas.copy()

    RSRS_Z = Cal_RSRS_Positive(df, N, M)  # 计算标准分指标

    singal_name = ['{}_singal'.format(x) for x in 'RSRS_Z,RSRS_Revise,RSRS_Positive'.split(',')]
    # 需要扣除前期计算的600日
    RSRS_Z = RSRS_Z[max(N, M) - 1:]

    print('回测起始日：', min(RSRS_Z.index))
    # 获取标准分信号
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Z', 0.7, -0.7)
    # 获取修正标准分
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Revise', 0.7, -0.7)
    # 获取右偏标准分
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Positive', 0.7, -0.7)

    # 收益
    RSRS_Z['ret'] = RSRS_Z['close'].pct_change()

    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    # 斜率净值
    for n in singal_name:
        CUM = (1 + RSRS_Z[n] * RSRS_Z['ret']).cumprod()
        ax1.plot(CUM, label=n)

    # 基准净值
    benchmark = (1 + RSRS_Z['ret']).cumprod()

    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('RSRS指标策略净值曲线')
    plt.show()

    display(HTML(summary(RSRS_Z, singal_name).to_html()))


# 右偏标准分
RSRS_Positive_Strategy(df, 16, 300)


# 在同等时间长度，不同策略参数下回测
def RSRS_Pam_Strategy(datas):
    df = datas.copy()
    RSRS = Cal_RSRS_Revise(df, 18, 600)
    RSRS_Z = Cal_RSRS_Positive(df, 16, 300)  # 计算标准分指标
    RSRS_Z.loc[:, ['RSRS_Z', 'RSRS_Revise']] = RSRS.loc[:, ['RSRS_Z', 'RSRS_Revise']]

    singal_name = ['{}_singal'.format(x) for x in 'RSRS_Z,RSRS_Revise,RSRS_Positive'.split(',')]

    # 添加20日均线
    RSRS_Z['MA'] = RSRS_Z['close'].rolling(20).mean()
    # 需要扣除前期计算的600日
    RSRS_Z = RSRS_Z[599:]

    print('回测起始日：', min(RSRS_Z.index))
    # 获取标准分信号
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Z', 0.7, -0.7)
    # 获取修正标准分
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Revise', 0.7, -0.7)
    # 获取右偏标准分
    RSRS_Z = get_singal(RSRS_Z, 'RSRS_Positive', 0.7, -0.7)

    # 收益
    RSRS_Z['ret'] = RSRS_Z['close'].pct_change()

    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    # 斜率净值
    for n in singal_name:
        CUM = (1 + RSRS_Z[n] * RSRS_Z['ret']).cumprod()
        ax1.plot(CUM, label=n)

    # 基准净值
    benchmark = (1 + RSRS_Z['ret']).cumprod()

    ax1.plot(benchmark, label='沪深300')

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('RSRS指标策略净值曲线')
    plt.show()

    display(HTML(summary(RSRS_Z, singal_name).to_html()))


# 基于当前市场价格趋势的优化
RSRS_Pam_Strategy(df)


