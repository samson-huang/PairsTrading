# 引入库
#from jqdata import *
#from jqfactor import get_factor_values

# 日常处理
import datetime
import calendar
from dateutil.relativedelta import relativedelta

# 常用库
import numpy as np
import pandas as pd


# 爬虫用
import json
import time
import re
import requests

# 其他
from alphalens.utils import print_table
from tqdm import * # 进度条
import itertools
import copy
import pickle

# 线性模型库
import statsmodels.api as sm

# 计算
import scipy.stats as ss
from scipy.stats import zscore,spearmanr

# 机器学习
from sklearn import linear_model,svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 画图
from pylab import mpl
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.family'] = 'serif'

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn')

#3.1 数据提取
## step1：获取回测区间每月最后一个交易日
def GetTradePerid(start_date: str, end_date: str, freq: str = 'M') -> list:
    '''
    start_date/end_date:str YYYY-MM-DD
    freq:M月末，Q季末,Y年末 默认M
    ================
    return datetime.date list
    '''
    days = [x.date() for x in pd.date_range(start_date, end_date, freq=freq)]

    return [
        d if d in get_trade_days(start_date, end_date) else get_trade_days(
            end_date=d, count=1)[0] for d in days
    ]


## step2:获取回测区间每月最后一个交易日的满足筛选条件的股票池
def GetStocks(symbol: str, trDate: datetime.date, limit: int = 6) -> list:
    '''
    symobl:指数代码
    trDate:交易日期
    limit:上市不足N月 默认未上市不足6月
    ================
    return list
    '''
    stocks = get_index_stocks(symbol, date=trDate)

    # 1.过滤ST
    is_st = get_extras('is_st', stocks, end_date=trDate, count=1).iloc[-1]

    stocks = is_st[is_st == False].index.tolist()

    # 2.过滤上市不足6月股票
    stocks = [
        s for s in stocks
        if get_security_info(symbol, date=trDate).start_date < trDate -
           datetime.timedelta(limit * 30)
    ]

    # 3.过滤当日未交易股票
    pause = get_price(
        stocks, end_date=trDate, fields='paused', count=1, panel=False)
    stocks = pause.query('paused==0')['code'].values.tolist()

    return stocks


## step3: 提取因子
def GetFactors(dates: list):
    factors_list = []

    for date in tqdm(dates, desc='Download Factors'):
        ## 提取每月末的股票池
        stocks = GetStocks('000300.XSHG', date)

        ## 提取每月末对应的因子值
        factors = factors = [
            'book_to_price_ratio', 'PEG', 'market_cap', 'cfo_to_ev',
            'total_asset_growth_rate', 'roe_ttm', 'LVGI', 'ROC20', 'ROC60',
            'sharpe_ratio_20', 'VOL20', 'Volume1M'
        ]

        f_dict = get_factor_values(
            stocks, factors, end_date=date, count=1)

        f_df = pd.concat(f_dict, axis=1).stack()

        ## 提取辅助因子：申万一级行业名称,用于行业中性化
        ind = get_industry(security=stocks, date=date)

        ind = {
            x: v.get('sw_l1').get('industry_name', np.nan)
            for x in ind.keys() for v in ind.values() if 'sw_l1' in v.keys()
        }

        f_df['INDUSTRY'] = list(
            map(lambda x: ind.get(x, np.nan), f_df.index.get_level_values(1)))

        factors_list.append(f_df)  # 将每月末提取的因子数据存list中

    factors_df = pd.concat(factors_list)
    factors_df.index.names = ['date', 'code']
    return factors_df


## step1：获取回测区间每月最后一个交易日
def GetTradePerid(start_date: str, end_date: str, freq: str = 'M') -> list:
    '''
    start_date/end_date:str YYYY-MM-DD
    freq:M月末，Q季末,Y年末 默认M
    ================
    return datetime.date list
    '''
    days = [x.date() for x in pd.date_range(start_date, end_date, freq=freq)]

    return [
        d if d in get_trade_days(start_date, end_date) else get_trade_days(
            end_date=d, count=1)[0] for d in days
    ]


## step2:获取回测区间每月最后一个交易日的满足筛选条件的股票池
def GetStocks(symbol: str, trDate: datetime.date, limit: int = 6) -> list:
    '''
    symobl:指数代码
    trDate:交易日期
    limit:上市不足N月 默认未上市不足6月
    ================
    return list
    '''
    stocks = get_index_stocks(symbol, date=trDate)

    # 1.过滤ST
    is_st = get_extras('is_st', stocks, end_date=trDate, count=1).iloc[-1]

    stocks = is_st[is_st == False].index.tolist()

    # 2.过滤上市不足6月股票
    stocks = [
        s for s in stocks
        if get_security_info(symbol, date=trDate).start_date < trDate -
           datetime.timedelta(limit * 30)
    ]

    # 3.过滤当日未交易股票
    pause = get_price(
        stocks, end_date=trDate, fields='paused', count=1, panel=False)
    stocks = pause.query('paused==0')['code'].values.tolist()

    return stocks


## step3: 提取因子
def GetFactors(dates: list):
    factors_list = []

    for date in tqdm(dates, desc='Download Factors'):
        ## 提取每月末的股票池
        stocks = GetStocks('000300.XSHG', date)

        ## 提取每月末对应的因子值
        factors = factors = [
            'book_to_price_ratio', 'PEG', 'market_cap', 'cfo_to_ev',
            'total_asset_growth_rate', 'roe_ttm', 'LVGI', 'ROC20', 'ROC60',
            'sharpe_ratio_20', 'VOL20', 'Volume1M'
        ]

        f_dict = get_factor_values(
            stocks, factors, end_date=date, count=1)

        f_df = pd.concat(f_dict, axis=1).stack()

        ## 提取辅助因子：申万一级行业名称,用于行业中性化
        ind = get_industry(security=stocks, date=date)

        ind = {
            x: v.get('sw_l1').get('industry_name', np.nan)
            for x in ind.keys() for v in ind.values() if 'sw_l1' in v.keys()
        }

        f_df['INDUSTRY'] = list(
            map(lambda x: ind.get(x, np.nan), f_df.index.get_level_values(1)))

        factors_list.append(f_df)  # 将每月末提取的因子数据存list中

    factors_df = pd.concat(factors_list)
    factors_df.index.names = ['date', 'code']
    return factors_df

# 数据获取
datas = GetFactors(dates)
datas.to_csv('../Data/SVM.csv')

datas.head()

#3.2 数据清洗
# 数据读取
factors = pd.read_csv('../Data/SVM.csv',index_col=[0,1],parse_dates=True)
factors.head()


## step1:构建绝对中位数处理法函数
# data为输入的数据集，如果数值超过num个判断标准则使其等于num个标准
def extreme_process_MAD(df:pd.DataFrame, num:int=3)->pd.DataFrame:

    # 为不破坏原始数据，先对其进行拷贝
    df_ = df.copy()
    feature_names = [
        i for i in df_.columns.tolist() if i not in ['INDUSTRY']
    ]  #获取数据集中需测试的因子名

    # 获取中位数
    median = df_[feature_names].median(axis=0)

    # 按列索引匹配，并在行中广播
    MAD = abs(df_[feature_names].sub(median, axis=1)).median(axis=0)

    # 利用clip()函数，将因子取值限定在上下限范围内，即用上下限来代替异常值
    df_.loc[:, feature_names] = df_.loc[:, feature_names].clip(
        lower=median - num * 1.4826 * MAD,
        upper=median + num * 1.4826 * MAD,
        axis=1)

    return df_


## step2:构建缺失值处理函数
def factors_null_process(df:pd.DataFrame)->pd.DataFrame:

    # 删除行业缺失值
    df = df[df['INDUSTRY'].notnull()]

    # 变化索引，以行业为第一索引，股票代码为第二索引
    df_ = df.reset_index().set_index(['INDUSTRY', 'code']).sort_index()

    # 用行业中位数填充
    df_ = df_.groupby(
        level=0).apply(lambda factor: factor.fillna(factor.median()))

    # 将索引换回
    df_ = df_.reset_index().set_index('code').sort_index()

    return df_.drop('date', axis=1)


## step3:构建标准化处理函数
def data_scale_Z_Score(df:pd.DataFrame)->pd.DataFrame:
    # 为不破坏原始数据，先对其进行拷贝

    df_ = df.copy()
    feature_names = [
        i for i in df_.columns.tolist() if i not in ['INDUSTRY']
    ]  #获取数据集中需测试的因子名

    df_.loc[:, feature_names] = (
        df_.loc[:, feature_names] -
        df_.loc[:, feature_names].mean()) / df_.loc[:, feature_names].std()

    return df_


## step4:构建对称正交变换函数
def lowdin_orthogonal(df:pd.DataFrame)->pd.DataFrame:

    df_ = df.copy()
    # 除去第一列行业指标,将数据框转化为矩阵
    col = [i for i in df_.columns.tolist() if i not in ['INDUSTRY']]
    F = np.array(df_[col])
    M = np.dot(F.T, F)
    a, U = np.linalg.eig(M)  #U为特征向量，a为特征值
    one = np.identity(len(col))
    D = one * a  #生成有特征值组成的对角矩阵
    D_inv = np.linalg.inv(D)
    S = U.dot(np.sqrt(D_inv)).dot(U.T)
    df_[col] = df_[col].dot(S)
    return df_


factors1 = factors.groupby(level='date').apply(extreme_process_MAD)  #去极值
factors2 = factors1.groupby(level='date').apply(factors_null_process)  #去缺失值
factors3 = factors2.groupby(level='date').apply(data_scale_Z_Score)  #标准化处理
factors4 = factors3.groupby(level='date').apply(lowdin_orthogonal)  #对称正交化

print(factors.info())
print('去缺失值:\n%s'%factors2.info())


#四、以对称正交后的因子收益作为预测目标变量
#4.1 对称正交前后因子横截面相关系数对比
# 构建计算横截面因子载荷相关系数均值函数
def get_relations(datas: pd.DataFrame) -> pd.DataFrame:
    '''
    datas:MultiIndex date,code columns->factor_name
    '''
    dates = set(datas.index.get_level_values(0))

    relations = 0

    for date, data in datas.groupby(level='date'):
        # data为提取横截面因子数据
        relations = relations + data.corr()  # 计算相关系数

    return relations / len(dates)  # relations_mean计算横截面因子载荷相关系数均值

#绘制因子正交前的相关性的热力图
fig = plt.figure(figsize=(26, 7))
relations = get_relations(factors3)  #计算对称正交之前的相关系数矩阵

sns.heatmap(
    relations,
    annot=True,
    linewidths=0.05,
    linecolor='white',
    annot_kws={
        'size': 8,
        'weight': 'bold'
    })

# 绘制因子正交后的相关性热力图
fig = plt.figure(figsize=(18, 8))
relations = get_relations(factors4)  #计算对称正交之后的相关系数矩阵

sns.heatmap(
    relations,
    annot=True,
    linewidths=0.05,
    linecolor='white',
    annot_kws={
        'size': 8,
        'weight': 'bold'
    })

#4.2 计算对称正交后的因子收益
# 提取市值因子并计算未来一期对数收益率
def GetRetCap(factors: pd.DataFrame) -> pd.DataFrame:
    '''
    factors:因子值的df MultiIndex date,code
    ==================
    return df next_ret,market_cap
    '''

    dates = [x.date() for x in factors.index.levels[0]]
    start_date = min(dates)
    end_date = max(dates)

    # 将最后一天的日期再往后推30天，这样能将最后一个调仓日往后推一个月
    target = end_date + relativedelta(months=1)
    monthCountDay = calendar.monthrange(target.year, target.month)[1]
    offset_day = datetime.date(target.year, target.month, day=monthCountDay)
    dates.append(offset_day)

    datas_dic = {}
    n = len(dates) - 1

    for i in tqdm(range(n), desc='DownLoad NetRet'):
        date = dates[i]

        date_next = dates[i + 1]

        stocks = factors.loc[date].index.tolist()  # 提取股票池

        # 计算对数收益率
        close_df = get_price(
            stocks, end_date=date, count=1, fields='close',
            panel=False).set_index('code')['close']

        next_df = get_price(
            stocks, end_date=date_next, count=1, fields='close',
            panel=False).set_index('code')['close']

        df = np.log(next_df / close_df)  # 计算对数收益率
        df = df.to_frame('log_ret')

        # 提取总市值
        df['cap'] = get_valuation(
            stocks, end_date=date, fields='market_cap',
            count=1).set_index('code')['market_cap']

        # 存储数
        datas_dic[date] = df[['cap', 'log_ret']]

    datas_df = pd.concat(datas_dic)
    datas_df.index.names = ['date', 'code']
    datas_df = datas_df.reset_index()
    datas_df['date'] = pd.to_datetime(datas_df['date'])
    datas_df.set_index(['date', 'code'], inplace=True)

    return datas_df

ret_cap = GetRetCap(factors)  # 对数收益计算较慢，代码运行时间较长

# 存储对称正交变换后的数据
datas_all = pd.concat([factors4, ret_cap], axis=1, join='inner')  # 将数据合并入原始数据中
datas_all.index.names = ['date', 'code']
datas_all.to_csv('SVM_timing_datas.csv')  # 将数据存入数据文件中
# 存储对称正交前的数据
datas_all = pd.concat([factors3, ret_cap], axis=1, join='inner')  # 将数据合并入原始数据中
datas_all.index.names = ['date', 'code']
datas_all.to_csv('noorth_SVM_timing_datas.csv')  # 将数据存入数据文件中

# 读取对称正交后的数据集用于计算因子收益
datas_all = pd.read_csv(
    '../Data/SVM_timing_datas.csv', index_col=[0, 1])
datas_all.tail()


# 计算因子收益
def Neu_Ret(datas_all: pd.DataFrame) -> pd.DataFrame:
    '''
    datas_all:df MultiIndex date,code
    ==========
    return ser 因子收益
    index-date columns-factor_name
    '''

    # 提取因子名
    factors = [
        i for i in datas_all.columns.tolist()
        if i not in ['INDUSTRY', 'cap', 'log_ret']
    ]

    dates = datas_all.index.get_level_values(0).unique()

    # 创建初始因子收益矩阵
    ret = pd.DataFrame(index=dates, columns=factors)

    for trDate, data in datas_all.groupby(level='date'):

        data = data.fillna(data.mean())  # 均值填充缺失值
        dfswsdummies = pd.get_dummies(data['INDUSTRY'])  # 得到申万一级行业虚拟变量
        ret_dic = {}

        for label, factor_df in data.loc[:, factors].items():
            X = pd.concat([factor_df, dfswsdummies], axis=1)  # 提取回归时的自变量
            Y = data[['log_ret']]  # 提取回归的因变量
            ret.loc[trDate, label] = sm.regression.linear_model.OLS(
                Y, X, missing='drop').fit().params[0]  # 进行OLS回归

    return ret

Ret_mat=Neu_Ret(datas_all)  #此处使用对称正交后的因子计算的因子收益
Ret_mat.tail()

#五、择时变量的选择
# 获取择时因子
def get_Timing_variables(begdate: str, enddate: str) -> pd.DataFrame:
    Bond_yield_3M = Get_Bond_yield_3M(begdate, enddate)
    M1 = GetMacro_M1(begdate, enddate)
    CPI = GetMacro_CPI(begdate, enddate)
    PPI = GetMacro_PPI(begdate, enddate)

    RISK = GetRiskIndex(begdate, enddate)
    TS = Get_TS(begdate, enddate)
    CS = Get_CS(begdate, enddate)
    RET_300 = GetIndexMRet('000300.XSHG', begdate, enddate, 'RET_300')
    RET_1000 = GetIndexMRet('000852.XSHG', begdate, enddate, 'RET_1000')

    STD_300 = GetIndexMSTD('000300.XSHG', begdate, enddate, 'STD_300')
    STD_1000 = GetIndexMSTD('000852.XSHG', begdate, enddate, 'STD_1000')

    RET_Spread = RET_300['RET_300'] - RET_1000['RET_1000']
    RET_Spread.name = 'RET_Spread'

    STD_Spread = STD_300['STD_300'] - STD_1000['STD_1000']
    STD_Spread.name = 'STD_Spread'

    return pd.concat([
        Bond_yield_3M, M1, CPI, PPI, TS, CS, RET_300, RET_1000, STD_300,
        STD_1000, RET_Spread, STD_Spread, RISK
    ],
        axis=1)


def GetMacro_M1(start_date: str, end_date: str) -> pd.DataFrame:
    '''
    start_date/end_date:YYYY-MM-dd
    '''
    # 获取M1同比
    q = query(macro.MAC_MONEY_SUPPLY_MONTH.stat_month,
              macro.MAC_MONEY_SUPPLY_MONTH.m1_yoy).filter(
        macro.MAC_MONEY_SUPPLY_MONTH.stat_month >= start_date[:7],
        macro.MAC_MONEY_SUPPLY_MONTH.stat_month <= end_date[:7])

    df = macro.run_query(q)

    df['stat_month'] = pd.to_datetime(df['stat_month'])

    dates = GetTradePerid(start_date, end_date)
    idx = pd.DataFrame({'stat_month': pd.to_datetime(dates)})

    return pd.merge_asof(idx, df.sort_values('stat_month'), on='stat_month').set_index('stat_month')


def GetMacro_CPI(start_date: str, end_date: str) -> pd.DataFrame:
    '''
    start_date/end_date:YYYY-MM-DD
    '''
    # 获取cpi同比
    q = query(macro.MAC_CPI_MONTH.stat_month,
              macro.MAC_CPI_MONTH.yoy).filter(
        macro.MAC_CPI_MONTH.stat_month >= start_date[:7],
        macro.MAC_CPI_MONTH.stat_month <= end_date[:7])

    df = macro.run_query(q)
    df.rename(columns={'yoy': 'cpi_yoy'}, inplace=True)
    df['stat_month'] = pd.to_datetime(df['stat_month'])

    dates = GetTradePerid(start_date, end_date)
    idx = pd.DataFrame({'stat_month': pd.to_datetime(dates)})

    return pd.merge_asof(idx, df.sort_values('stat_month'), on='stat_month').set_index('stat_month')


def GetMacro_PPI(start_date: str, end_date: str) -> pd.DataFrame:
    df = macro_china_ppi_yearly()
    df = df.loc[start_date:end_date]
    df.index.names = ['date']

    dates = GetTradePerid(start_date, end_date)
    idx = pd.DataFrame({'date': pd.to_datetime(dates)})

    return pd.merge_asof(idx, df.reset_index(), on='date').set_index('date')


def GetRiskIndex(start_date: str, end_date: str) -> pd.DataFrame:
    # 获取10年国债到期收益
    yeild = get_bond_yield(start_date, end_date, 10, 'hzsylqx')
    yeild.columns = ['n', 'date', 'yeild']
    yeild.index = pd.to_datetime(yeild['date'])

    # 股息率使用中证红利代表全市场
    DividendRatio = getIndexDividendRatio('000922', start_date, end_date)
    risk = DividendRatio['DividendRatio'] - yeild['yeild']
    dates = GetTradePerid(start_date, end_date)
    risk = risk.reindex(dates).to_frame('Risk')
    return risk.fillna(0)


def Get_Bond_yield_3M(start_date: str, end_date: str) -> pd.DataFrame:
    '''
    单位:%
    '''
    df = get_bond_yield(start_date, end_date, 0.25, 'hzsylqx')
    df.columns = ['n', 'date', 'yeild']
    df.index = pd.to_datetime(df['date'])
    dates = GetTradePerid(start_date, end_date)

    return df.reindex(dates)['yeild'].to_frame('Bond_yield_3M')


def Get_TS(start_date: str, end_date: str) -> pd.DataFrame:
    '''
    单位:%
    '''
    long = get_bond_yield(start_date, end_date, 10, 'hzsylqx')
    long.columns = ['n', 'date', 'yeild']
    short = get_bond_yield(start_date, end_date, 1, 'hzsylqx')
    short.columns = ['n', 'date', 'yeild']

    long = long.set_index('date')
    short = short.set_index('date')

    ts_ser = long['yeild'] - short['yeild']
    ts_ser.index = pd.to_datetime(ts_ser.index)
    dates = GetTradePerid(start_date, end_date)

    return ts_ser.reindex(dates).to_frame('TS')


def Get_CS(start_date: str, end_date: str) -> pd.DataFrame:
    '''
    单位:%
    '''
    long = get_bond_yield(start_date, end_date, 1, 'syyhsylqx')
    long.columns = ['n', 'date', 'yeild']
    short = get_bond_yield(start_date, end_date, 1, 'hzsylqx')
    short.columns = ['n', 'date', 'yeild']

    long = long.set_index('date')
    short = short.set_index('date')

    ts_ser = long['yeild'] - short['yeild']
    ts_ser.index = pd.to_datetime(ts_ser.index)
    dates = GetTradePerid(start_date, end_date)

    return ts_ser.reindex(dates).to_frame('CS')


def GetIndexMRet(symbol: str, start_date: str, end_date: str, name: str) -> pd.DataFrame:
    '''
    name :设置收益名称
    '''
    begin_date = pd.date_range(end=start_date, periods=1, freq='M')[0].strftime('%Y-%m-%d')
    dates = GetTradePerid(begin_date, end_date)
    index_price = get_price(symbol, begin_date, end_date, fields='close', panel=False)

    index_price = index_price.reindex(dates)

    return index_price['close'].pct_change().dropna().to_frame(name)


def GetIndexMSTD(symbol: str, start_date: str, end_date: str, name: str) -> pd.DataFrame:
    index_price = get_price(symbol, start_date, end_date, fields='close', panel=False)
    std_df = index_price.groupby(pd.Grouper(freq='M'), as_index=False).apply(
        lambda x: np.std(x.pct_change()) * np.sqrt(20))
    dates = GetTradePerid(start_date, end_date)
    std_df.index = dates
    return std_df.rename(columns={'close': name})


# 单次返回所有
# 金十数据中心-经济指标-中国-国民经济运行状况-物价水平-中国PPI年率报告
def macro_china_ppi_yearly():
    """
    中国年度PPI数据, 数据区间从19950801-至今
    https://datacenter.jin10.com/reportType/dc_chinese_ppi_yoy
    :return: pandas.Series
    """
    t = time.time()

    JS_CHINA_PPI_YEARLY_URL = (
        "https://cdn.jin10.com/dc/reports/dc_chinese_ppi_yoy_all.js?v={}&_={}")

    res = requests.get(
        JS_CHINA_PPI_YEARLY_URL.format(
            str(int(round(t * 1000))), str(int(round(t * 1000)) + 90)
        )
    )
    json_data = json.loads(res.text[res.text.find("{"): res.text.rfind("}") + 1])
    date_list = [item["date"] for item in json_data["list"]]
    value_list = [item["datas"]["中国PPI年率报告"] for item in json_data["list"]]
    value_df = pd.DataFrame(value_list)
    value_df.columns = json_data["kinds"]
    value_df.index = pd.to_datetime(date_list)
    temp_df = value_df["今值(%)"]
    temp_df.name = "ppi"
    return temp_df


def bond_china_yield(start_date: str, end_date: str, gjqx: int, qxId: str = "hzsylqx"):
    """
    中国债券信息网-国债及其他债券收益率曲线
    https://www.chinabond.com.cn/
    http://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery?startDate=2019-02-07&endDate=2020-02-04&gjqx=0&qxId=ycqx&locale=cn_ZH
    注意: end_date - start_date 应该小于一年
    :param start_date: 需要查询的日期, 返回在该日期之后一年内的数据
        gjqx 为收益率的年限
    hzsylqx是中债登国债收益曲线、syyhsylqx是中债登商业银行普通债收益率曲线、zdqpjsylqx是中债登短期票据收
    :type start_date: str
    :param end_date: 需要查询的日期, 返回在该日期之前一年内的数据
    :type end_date: str
    :return: 返回在指定日期之间之前一年内的数据
    :rtype: pandas.DataFrame
    """
    url = "http://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "gjqx": str(gjqx),
        "qxId": qxId,
        "locale": "cn_ZH",
    }
    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
    }
    res = requests.get(url, params=params, headers=headers)
    data_text = res.text.replace("&nbsp", "")
    data_df = pd.read_html(data_text, header=0)[1]
    return data_df


def get_bond_yield(start_date: str, end_date: str, periods: int, bond_type: str):
    '''
    periods:债券期限
    bond_type:债券类型
    '''
    dates = get_trade_days(start_date, end_date)
    n_days = len(dates)
    limit = 244

    if n_days > limit:

        n = n_days // limit
        df_list = []
        i = 0
        pos1, pos2 = n * i, n * (i + 1) - 1
        while pos2 < n_days:
            # print(pos2)
            df = bond_china_yield(start_date=dates[pos1], end_date=dates[pos2], gjqx=periods, qxId=bond_type)
            df_list.append(df)
            i += 1
            pos1, pos2 = n * i, n * (i + 1) - 1

        if pos1 < n_days:
            df = bond_china_yield(start_date=dates[pos1], end_date=dates[-1], gjqx=periods, qxId=bond_type)
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
    else:
        df = bond_china_yield(start_date=start_date, end_date=end_date, gjqx=periods, qxId=bond_type)

    return df.dropna(axis=1)


# 查询指数股息率
# 估值衍生
def getIndexDividendRatio(symbol: str, start_date: str, end_date: str):
    # 查询内部编码
    InnerCode_id = jy.run_query(
        query(jy.LC_IndexRelationship.InnerCode).filter(
            jy.LC_IndexRelationship.SecuCode == symbol))['InnerCode'][0]

    # 查询数据
    q = query(jy.LC_IndexDerivative.TradingDay,
              jy.LC_IndexDerivative.DividendRatio).filter(
        jy.LC_IndexDerivative.IndexCode == InnerCode_id,
        jy.LC_IndexDerivative.TradingDay >= start_date,
        jy.LC_IndexDerivative.TradingDay <= end_date)

    return jy.run_query(q).set_index('TradingDay')

## 设置回测区间
start_date = '2014-01-01'
end_date = '2019-12-31'

# 前推36个月
begin_date = pd.date_range(end=start_date,periods=36,freq='M')[0].strftime('%Y-%m-%d')

df_m_shifted_=get_Timing_variables(begin_date,end_date)
df_m_shifted_.to_csv('../Data/Timing_variables.csv')

df_m_shifted_=pd.read_csv('../Data/Timing_variables.csv',index_col=[0])
df_m_shifted_.tail()

#六、正交变换后外部变量解释能力显著增强
# 构建拟合优度函数
# 输入参数：df_m_shifted_i（某段时间的择时变量），Ret_mat（对应时段的因子收益）
def R_squared(df_m_shifted_i: pd.DataFrame, Ret_mat: pd.DataFrame) -> pd.DataFrame:
    # 提取输入的择时变量对应的索引值（即时间维度）
    datelist = df_m_shifted_i.index.unique()

    # 提取与择时变量相同维度的因子收益
    Ret_mat = Ret_mat.loc[datelist]

    # 计算因子个数
    num_factor = len(Ret_mat.columns)

    # 设置用于择时的变量
    macro_factor_selected = [
        'TS', 'CS', 'cpi_yoy', 'ppi', 'STD_Spread', 'RET_Spread', 'm1_yoy',
        'Bond_yield_3M'
    ]

    # 构建初始的拟合优度数据框
    R2_mat = pd.DataFrame(index=Ret_mat.columns, columns=['R_square'])

    for label, factor in Ret_mat.items():
        # 提取回归因变量（因子收益）
        factor_ret = factor.reset_index(drop=True)

        # 提取回归自变量（择时因子）
        macro_df = df_m_shifted_i.loc[datelist, macro_factor_selected].reset_index(drop=True)

        # 进行回归并提取拟合优度值
        r2 = sm.regression.linear_model.OLS(
            factor_ret.astype(float), macro_df.astype(float)).fit().rsquared

        R2_mat.loc[label, :] = r2

    return R2_mat  # 得到的R2是一个以因子名为索引，以'R_square'为列名的数据框


# 构建滚动计算拟合优度及拟合优度均值
# 输入参数：df_m_shifted_（为全部测试区间段择时因子数据集）,Ret_mat（为全部时间段的因子收益）,window=24（滚动周期）
def rolling_R2(
        df_m_shifted_: pd.DataFrame, Ret_mat: pd.DataFrame, window: int = 24
) -> pd.DataFrame:
    # 提取日期数
    datelist = df_m_shifted_.index.unique()

    # 初始化拟合优度数据框中，以因子名为索引值
    R2_mat_all = pd.DataFrame(
        index=Ret_mat.columns.tolist())

    for i in range(window - 1, len(df_m_shifted_)):
        df_m_shifted_i = df_m_shifted_.iloc[(i - window + 1):i, :]

        # 计算每期的拟合优度
        R2_mat_i = R_squared(df_m_shifted_i, Ret_mat)

        # 将计算的第i期数据合并入初始化拟合优度数据框中
        R2_mat_all[datelist[i]] = R2_mat_i

    return R2_mat_all.T

# 构建滚动计算拟合优度及拟合优度均值
R2_mat_all=rolling_R2(df_m_shifted_.fillna(0),Ret_mat.fillna(0),window=24)
R2_mat_all.head()

#同理计算对称正交变换前的拟合优度
noorth_factors=pd.read_csv('../Data/noorth_SVM_timing_datas.csv',index_col=[0,1])  # 读取对称正交前的数据
noorth_Ret_mat=Neu_Ret(noorth_factors)  # 计算因子收益
noorth_R2_mat_all=rolling_R2(df_m_shifted_.fillna(0),noorth_Ret_mat.fillna(0),window=24) # 滚动计算拟合优度
noorth_R2_mat_all.head()

# 绘制对称正交前后择时因子解释能力变化对比柱状图
df = pd.DataFrame()
df['before_orthogonal'] = noorth_R2_mat_all.mean()  # 计算对称正交前的平均拟合优度
df['after_orthogonal'] = R2_mat_all.mean()  # 计算对称正交后的平均拟合优度
df.plot.bar(figsize=(18, 7))

#七、回测流程及参数设置
# step1：构建因子收益的加权移动平均函数，作为基础权重
# 输入因子收益和半衰期
def Ret_mat_emw(Ret_mat: pd.DataFrame, period: int) -> pd.DataFrame:
    # 指数加权移动平均
    return Ret_mat.ewm(halflife=period, min_periods=2).mean()

weight_in = Ret_mat_emw(Ret_mat,period=3)
weight_in.head()

# step2:构建分类模型(可通过输入不同的方法method，选择不同的分类模型),预测因子受益方向
def run_predict_models(Ret_mat: pd.DataFrame,
                       df_m_shifted_: pd.DataFrame,
                       method: str,
                       window: int = 24) -> pd.DataFrame:
    '''
    输入因子收益Ret_mat，择时变量df_m_shifted_，
    滚动窗口window为24（每次以24期的数据进行预测，
    其中前23期作为训练集，最后一期作为测试集）
    method:SVM,Logistic,DecisionTree,RandomForest
    '''
    ##初始化数据
    # 获取需要预测的日期序列
    datelist = df_m_shifted_.index.tolist()[window - 1:]

    # 由于因子收益的时间跨度比择时变量的时间跨度大，为保持预测时间上的一致性，
    # 截取与择时变量相同时间跨度的因子收益
    Ret_mat_ = Ret_mat.loc[df_m_shifted_.index.tolist()]

    # 因子收益大于0的赋值为1，小于0的赋值为-1
    Ret_mat_sign = np.sign(Ret_mat_)

    # 获取待预测的因子个数
    num_factor = len(Ret_mat.columns)

    # 提取用于预测的择时因子
    macro_df = df_m_shifted_[[
        'TS', 'CS', 'cpi_yoy', 'ppi', 'STD_Spread', 'RET_Spread', 'm1_yoy',
        'Bond_yield_3M'
    ]]

    ## 用支持向量机进行预测
    # 初始化因子预测数据框,其中起始预测时间为start_date
    predicted_mat = pd.DataFrame(index=datelist, columns=Ret_mat_.columns)

    for j in range(num_factor):

        # 提取分类模型的预测变量，即第j个因子收益
        factor_sign_i = Ret_mat_sign.iloc[:, j]

        # 初始化单个因子的测试值
        predict = pd.Series(0, index=predicted_mat.index.tolist())

        for i in range(len(predict)):

            # 不进行变量缩减,提取预测日期前推window的择时变量数据
            x_design = macro_df.iloc[i:i + window, :]

            # 对数据进行标准化处理
            x_std = x_design.apply(lambda x: (x - x.mean()) / x.std())

            # 以前23期为训练集
            x_train = x_std.iloc[:-1, :]

            # 以最后一期为测试集
            x_test = x_std.iloc[-1:, :]

            # 提取y集
            y_design_logit = factor_sign_i.iloc[i:i + window - 1]

            # 判断选用哪个预测模型
            if method == 'SVM':
                regr = svm.SVC(kernel='rbf')  # 支持向量机核函数选择rbf

            elif method == 'Logistic':
                regr = linear_model.LogisticRegression()  # 导入逻辑回归模型

            elif method == 'DecisionTree':
                regr = DecisionTreeClassifier(max_depth=3)  # 导入决策树模型

            elif method == 'RandomForest':
                regr = RandomForestClassifier(
                    n_estimators=20, max_depth=3,
                    random_state=0)  # 随机森林最大深度为3，n_estimators=20

            regr.fit(x_train.fillna(0), y_design_logit.astype(float))
            predict.iloc[i] = regr.predict(x_test.fillna(0))

        print(Ret_mat.columns[j], "success")  # 打印测试成功的因子

        predicted_mat.iloc[:, j] = predict  # 将各个因子滚动预测的因子收益存入初始化因子预测数据框中

    return predicted_mat

predict_mat = run_predict_models(
    Ret_mat, df_m_shifted_, method='SVM', window=24)
predict_mat.head()


#step5:构建权重调整函数
def weight_timing_threshold(Ret_mat: pd.DataFrame,
                            predict_mat: pd.DataFrame,
                            weight_in: pd.DataFrame,
                            r2: pd.DataFrame,
                            th: float,
                            z: float = 0.1) -> pd.DataFrame:
    '''
    输入参数：Ret_mat（因子收益数据集），
    predict_mat（预测的因子收益方向），
    weight_in（初始权重），
    r2(拟合优度R2)，
    th(阈值),
    z(权重调整系数)
    '''
    predict_mat.dropna(inplace=True)  # 删除预期收益的缺失值

    # 计算过去36个月的因子收益均值
    factor_sign = np.sign(Ret_mat.rolling(min_periods=1, window=36).mean())

    # 初始化新的因子权重
    weight_new = pd.DataFrame(
        0, index=predict_mat.index, columns=weight_in.columns)

    # 提取预期因子收益时间对应的原始因子权重集
    weight_in_chunk = weight_in.loc[predict_mat.index, :]

    for j in range(len(weight_new.index)):
        # 判断预测的因子收益方向是否发生变化，若发生变化则需要调整权重
        iftrue = pd.Series(predict_mat.loc[weight_new.index[j], :] ==
                           factor_sign.loc[weight_new.index[j], :])

        time_weight = iftrue.apply(lambda x: 1 if x == True else z)
        # 调整因子权重
        weight_new.iloc[j, :] = weight_in_chunk.iloc[j, :] * time_weight

        # 根据拟合优度判断是否有变换权重的资格
        aa = r2.loc[weight_new.index[j], :]  # 提取对应日期的过去n期的r2
        for name in weight_new.columns:
            if aa[name] < th:
                weight_new.loc[:, name] = weight_in_chunk.loc[:, name].astype(
                    float)  # 当拟合优度小于阈值时，权重保持不变

    return weight_new

weight_new = weight_timing_threshold(
    Ret_mat, predict_mat, weight_in, r2, th=0.05, z=0.1)
weight_new.head()


#八、 三种分类模型的参数选取

#九、SVM预测能力较强，随机森林表现稳定

# 定义计算预测准确度函数
def get_accuracy(predict_mat: pd.DataFrame, Ret_mat: pd.DataFrame,
                 method: str) -> pd.DataFrame:
    ret_mat_sign = np.sign(Ret_mat.loc[predict_mat.index, :])
    accuracy = pd.DataFrame(index=[method])

    for name in predict_mat.columns:
        iftrue = pd.Series(predict_mat[name] == ret_mat_sign[name])
        accuracy[name] = iftrue.sum() / len(predict_mat)
    return accuracy.T

#计算SVM的预测准确率
predict_mat = run_predict_models(
    Ret_mat, df_m_shifted_, method='SVM', window=24)
SVM_accuracy = get_accuracy(predict_mat, Ret_mat, method='SVM')

#计算随机森林的预测准确率
predict_mat = run_predict_models(
    Ret_mat, df_m_shifted_, method='RandomForest', window=20)
RF_accuracy = get_accuracy(predict_mat, Ret_mat, method='RandomForest')

#计算逻辑回归的预测准确率
predict_mat = run_predict_models(
    Ret_mat, df_m_shifted_, method='Logistic', window=36)
LOG_accuracy = get_accuracy(predict_mat, Ret_mat, method='Logistic')


#绘制预测准确度对比柱状图
df = pd.concat([SVM_accuracy, RF_accuracy, LOG_accuracy], axis=1)
df.plot.bar(figsize=(18, 8))


#十、 随机森林和SVM的收益提升较为明显
datas_all = pd.read_csv(
    '../Data/SVM_timing_datas.csv', index_col=[0, 1])  #读取对称正交后的数据集

datas = datas_all[[
    i for i in datas_all.columns if i not in ['cap', 'log_ret', 'INDUSTRY']
]]  #提取因子数据

datas.head()

# 用于回测调用
Ret_mat.to_csv('../Data/Ret_mat.csv')
datas.to_csv('../Data/datas.csv')
df_m_shifted_.to_csv('../Data/df_m_shifted_.csv')
weight_in.to_csv('../Data/weight_in.csv')


# 定义类'参数分析'
class parameter_analysis(object):

    # 定义函数中不同的变量
    def __init__(self, algorithm_id=None):
        self.algorithm_id = algorithm_id  # 回测id

        self.params_df = pd.DataFrame(
        )  # 回测中所有调参备选值的内容，列名字为对应修改面两名称，对应回测中的 g.XXXX
        self.results = {}  # 回测结果的回报率，key 为 params_df 的行序号，value 为
        self.evaluations = {
        }  # 回测结果的各项指标，key 为 params_df 的行序号，value 为一个 dataframe
        self.backtest_ids = {}  # 回测结果的 id

        # 新加入的基准的回测结果 id，可以默认为空 ''，则使用回测中设定的基准
        self.benchmark_id = 'f16629492d6b6f4040b2546262782c78'

        self.benchmark_returns = []  # 新加入的基准的回测回报率
        self.returns = {}  # 记录所有回报率
        self.excess_returns = {}  # 记录超额收益率
        self.log_returns = {}  # 记录收益率的 log 值
        self.log_excess_returns = {}  # 记录超额收益的 log 值
        self.dates = []  # 回测对应的所有日期
        self.excess_max_drawdown = {}  # 计算超额收益的最大回撤
        self.excess_annual_return = {}  # 计算超额收益率的年化指标
        self.evaluations_df = pd.DataFrame()  # 记录各项回测指标，除日回报率外
        self.failed_list = []
        self.nav_df = pd.DataFrame()

    # 定义排队运行多参数回测函数
    def run_backtest(
            self,  #
            algorithm_id=None,  # 回测策略id
            running_max=10,  # 回测中同时巡行最大回测数量
            start_date='2006-01-01',  # 回测的起始日期
            end_date='2016-11-30',  # 回测的结束日期
            frequency='day',  # 回测的运行频率
            initial_cash='1000000',  # 回测的初始持仓金额
            param_names=[],  # 回测中调整参数涉及的变量
            param_values=[],  # 回测中每个变量的备选参数值
            python_version=2,  # 回测的python版本
            use_credit=False  # 是否允许消耗积分继续回测
    ):
        # 当此处回测策略的 id 没有给出时，调用类输入的策略 id
        if algorithm_id == None:
            algorithm_id = self.algorithm_id

        # 生成所有参数组合并加载到 df 中
        # 包含了不同参数具体备选值的排列组合中一组参数的 tuple 的 list
        param_combinations = list(itertools.product(*param_values))
        # 生成一个 dataframe， 对应的列为每个调参的变量，每个值为调参对应的备选值
        to_run_df = pd.DataFrame(param_combinations, dtype='object')
        # 修改列名称为调参变量的名字
        to_run_df.columns = param_names

        # 设定运行起始时间和保存格式
        start = time.time()
        # 记录结束的运行回测
        finished_backtests = {}
        # 记录运行中的回测
        running_backtests = {}
        # 计数器
        pointer = 0
        # 总运行回测数目，等于排列组合中的元素个数
        total_backtest_num = len(param_combinations)
        # 记录回测结果的回报率
        all_results = {}
        # 记录回测结果的各项指标
        all_evaluations = {}

        # 在运行开始时显示
        print(('【已完成|运行中|待运行】:'), end=' ')
        # 当运行回测开始后，如果没有全部运行完全的话：
        while len(finished_backtests) < total_backtest_num:
            # 显示运行、完成和待运行的回测个数
            print(('[%s|%s|%s].' %
                   (len(finished_backtests), len(running_backtests),
                    (total_backtest_num - len(finished_backtests) -
                     len(running_backtests)))),
                  end=' ')
            # 记录当前运行中的空位数量
            to_run = min(
                running_max - len(running_backtests), total_backtest_num -
                len(running_backtests) - len(finished_backtests))
            # 把可用的空位进行跑回测
            for i in range(pointer, pointer + to_run):
                # 备选的参数排列组合的 df 中第 i 行变成 dict，每个 key 为列名字，value 为 df 中对应的值
                params = to_run_df.iloc[i].to_dict()
                # 记录策略回测结果的 id，调整参数 extras 使用 params 的内容
                backtest = create_backtest(
                    algorithm_id=algorithm_id,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    initial_cash=initial_cash,
                    extras=params,
                    # 再回测中把改参数的结果起一个名字，包含了所有涉及的变量参数值
                    name=str(params),
                    python_version=python_version,
                    use_credit=use_credit)
                # 记录运行中 i 回测的回测 id
                running_backtests[i] = backtest
            # 计数器计数运行完的数量
            pointer = pointer + to_run

            # 获取回测结果
            failed = []
            finished = []
            # 对于运行中的回测，key 为 to_run_df 中所有排列组合中的序数
            for key in list(running_backtests.keys()):
                # 研究调用回测的结果，running_backtests[key] 为运行中保存的结果 id
                back_id = running_backtests[key]
                bt = get_backtest(back_id)
                # 获得运行回测结果的状态，成功和失败都需要运行结束后返回，如果没有返回则运行没有结束
                status = bt.get_status()
                # 当运行回测失败
                if status == 'failed':
                    # 失败 list 中记录对应的回测结果 id
                    print('')
                    print((
                            '回测失败 : https://www.joinquant.com/algorithm/backtest/detail?backtestId='
                            + back_id))
                    failed.append(key)
                # 当运行回测成功时
                elif status == 'done':
                    # 成功 list 记录对应的回测结果 id，finish 仅记录运行成功的
                    finished.append(key)
                    # 回测回报率记录对应回测的回报率 dict， key to_run_df 中所有排列组合中的序数， value 为回报率的 dict
                    # 每个 value 一个 list 每个对象为一个包含时间、日回报率和基准回报率的 dict
                    all_results[key] = bt.get_results()
                    # 回测回报率记录对应回测结果指标 dict， key to_run_df 中所有排列组合中的序数， value 为回测结果指标的 dataframe
                    all_evaluations[key] = bt.get_risk()
            # 记录运行中回测结果 id 的 list 中删除失败的运行
            for key in failed:
                finished_backtests[key] = running_backtests.pop(key)
            # 在结束回测结果 dict 中记录运行成功的回测结果 id，同时在运行中的记录中删除该回测
            for key in finished:
                finished_backtests[key] = running_backtests.pop(key)
            #                 print (finished_backtests)
            # 当一组同时运行的回测结束时报告时间
            if len(finished_backtests) != 0 and len(
                    finished_backtests) % running_max == 0 and to_run != 0:
                # 记录当时时间
                middle = time.time()
                # 计算剩余时间，假设没工作量时间相等的话
                remain_time = (middle - start) * (
                        total_backtest_num -
                        len(finished_backtests)) / len(finished_backtests)
                # print 当前运行时间
                print(
                    ('[已用%s时,尚余%s时,请不要关闭浏览器].' %
                     (str(round((middle - start) / 60.0 / 60.0,
                                3)), str(round(remain_time / 60.0 / 60.0, 3)))),
                    end=' ')
            self.failed_list += failed
            # 5秒钟后再跑一下
            time.sleep(5)
        # 记录结束时间
        end = time.time()
        print('')
        print(
            ('【回测完成】总用时：%s秒(即%s小时)。' %
             (str(int(end - start)), str(round(
                 (end - start) / 60.0 / 60.0, 2)))),
            end=' ')
        #         print (to_run_df,all_results,all_evaluations,finished_backtests)
        # 对应修改类内部对应
        #         to_run_df = {key:value for key,value in returns.items() if key not in faild}
        self.params_df = to_run_df
        #         all_results = {key:value for key,value in all_results.items() if key not in faild}
        self.results = all_results
        #         all_evaluations = {key:value for key,value in all_evaluations.items() if key not in faild}
        self.evaluations = all_evaluations
        #         finished_backtests = {key:value for key,value in finished_backtests.items() if key not in faild}
        self.backtest_ids = finished_backtests

    # 7 最大回撤计算方法
    def find_max_drawdown(self, returns):
        # 定义最大回撤的变量
        result = 0
        # 记录最高的回报率点
        historical_return = 0
        # 遍历所有日期
        for i in range(len(returns)):
            # 最高回报率记录
            historical_return = max(historical_return, returns[i])
            # 最大回撤记录
            drawdown = 1 - (returns[i] + 1) / (historical_return + 1)
            # 记录最大回撤
            result = max(drawdown, result)
        # 返回最大回撤值
        return result

    # log 收益、新基准下超额收益和相对与新基准的最大回撤
    def organize_backtest_results(self, benchmark_id=None):
        # 若新基准的回测结果 id 没给出
        if benchmark_id == None:
            # 使用默认的基准回报率，默认的基准在回测策略中设定
            self.benchmark_returns = [
                x['benchmark_returns'] for x in self.results[0]
            ]
        # 当新基准指标给出后
        else:
            # 基准使用新加入的基准回测结果
            self.benchmark_returns = [
                x['returns'] for x in get_backtest(benchmark_id).get_results()
            ]
        # 回测日期为结果中记录的第一项对应的日期
        self.dates = [x['time'] for x in self.results[0]]

        # 对应每个回测在所有备选回测中的顺序 （key），生成新数据
        # 由 {key：{u'benchmark_returns': 0.022480100091729405,
        #           u'returns': 0.03184566700000002,
        #           u'time': u'2006-02-14'}} 格式转化为：
        # {key: []} 格式，其中 list 为对应 date 的一个回报率 list
        for key in list(self.results.keys()):
            self.returns[key] = [x['returns'] for x in self.results[key]]
        # 生成对于基准（或新基准）的超额收益率
        for key in list(self.results.keys()):
            self.excess_returns[key] = [
                (x + 1) / (y + 1) - 1
                for (x, y) in zip(self.returns[key], self.benchmark_returns)
            ]
        # 生成 log 形式的收益率
        for key in list(self.results.keys()):
            self.log_returns[key] = [log(x + 1) for x in self.returns[key]]
        # 生成超额收益率的 log 形式
        for key in list(self.results.keys()):
            self.log_excess_returns[key] = [
                log(x + 1) for x in self.excess_returns[key]
            ]
        # 生成超额收益率的最大回撤
        for key in list(self.results.keys()):
            self.excess_max_drawdown[key] = self.find_max_drawdown(
                self.excess_returns[key])
        # 生成年化超额收益率
        for key in list(self.results.keys()):
            self.excess_annual_return[key] = (self.excess_returns[key][-1] +
                                              1) ** (252. /
                                                     float(len(self.dates))) - 1
        # 把调参数据中的参数组合 df 与对应结果的 df 进行合并
        self.evaluations_df = pd.concat(
            [self.params_df, pd.DataFrame(self.evaluations).T], axis=1)

    #         self.evaluations_df =

    # 获取最总分析数据，调用排队回测函数和数据整理的函数

    def get_backtest_data(
            self,
            algorithm_id=None,  # 回测策略id
            benchmark_id=None,  # 新基准回测结果id
            file_name='results.pkl',  # 保存结果的 pickle 文件名字
            running_max=10,  # 最大同时运行回测数量
            start_date='2006-01-01',  # 回测开始时间
            end_date='2016-11-30',  # 回测结束日期
            frequency='day',  # 回测的运行频率
            initial_cash='1000000',  # 回测初始持仓资金
            param_names=[],  # 回测需要测试的变量
            param_values=[],  # 对应每个变量的备选参数
            python_version=2,
            use_credit=False):
        # 调运排队回测函数，传递对应参数
        self.run_backtest(
            algorithm_id=algorithm_id,
            running_max=running_max,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            initial_cash=initial_cash,
            param_names=param_names,
            param_values=param_values,
            python_version=python_version,
            use_credit=use_credit,
        )
        # 回测结果指标中加入 log 收益率和超额收益率等指标
        self.organize_backtest_results(benchmark_id)
        # 生成 dict 保存所有结果。
        results = {
            'returns': self.returns,
            'excess_returns': self.excess_returns,
            'log_returns': self.log_returns,
            'log_excess_returns': self.log_excess_returns,
            'dates': self.dates,
            'benchmark_returns': self.benchmark_returns,
            'evaluations': self.evaluations,
            'params_df': self.params_df,
            'backtest_ids': self.backtest_ids,
            'excess_max_drawdown': self.excess_max_drawdown,
            'excess_annual_return': self.excess_annual_return,
            'evaluations_df': self.evaluations_df,
            "failed_list": self.failed_list
        }
        # 保存 pickle 文件
        pickle_file = open(file_name, 'wb')
        pickle.dump(results, pickle_file)
        pickle_file.close()

    # 读取保存的 pickle 文件，赋予类中的对象名对应的保存内容
    def read_backtest_data(self, file_name='results.pkl'):
        pickle_file = open(file_name, 'rb')
        results = pickle.load(pickle_file)
        self.returns = results['returns']
        self.excess_returns = results['excess_returns']
        self.log_returns = results['log_returns']
        self.log_excess_returns = results['log_excess_returns']
        self.dates = results['dates']
        self.benchmark_returns = results['benchmark_returns']
        self.evaluations = results['evaluations']
        self.params_df = results['params_df']
        self.backtest_ids = results['backtest_ids']
        self.excess_max_drawdown = results['excess_max_drawdown']
        self.excess_annual_return = results['excess_annual_return']
        self.evaluations_df = results['evaluations_df']
        self.failed_list = results['failed_list']
        self.nav_df = self.GetNavDf()

    # 回报率折线图
    def plot_returns(self):
        # 通过figsize参数可以指定绘图对象的宽度和高度，单位为英寸；
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111)
        # 作图
        for key in list(self.returns.keys()):
            ax.plot(
                list(range(len(self.returns[key]))),
                self.returns[key],
                label=key)
        # 设定benchmark曲线并标记
        ax.plot(
            list(range(len(self.benchmark_returns))),
            self.benchmark_returns,
            label='benchmark',
            c='k',
            linestyle='--')
        ticks = [int(x) for x in np.linspace(0, len(self.dates) - 1, 11)]
        plt.xticks(ticks, [self.dates[i] for i in ticks])
        # 设置图例样式
        ax.legend(loc=2, fontsize=10)
        # 设置y标签样式
        ax.set_ylabel('returns', fontsize=20)
        # 设置x标签样式
        ax.set_yticklabels([str(x * 100) + '% ' for x in ax.get_yticks()])
        # 设置图片标题样式
        ax.set_title(
            "Strategy's performances with different parameters", fontsize=21)
        plt.xlim(0, len(self.returns[0]))

    # 超额收益率图
    def plot_excess_returns(self):
        # 通过figsize参数可以指定绘图对象的宽度和高度，单位为英寸；
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111)
        # 作图
        for key in list(self.returns.keys()):
            ax.plot(
                list(range(len(self.excess_returns[key]))),
                self.excess_returns[key],
                label=key)
        # 设定benchmark曲线并标记
        ax.plot(
            list(range(len(self.benchmark_returns))),
            [0] * len(self.benchmark_returns),
            label='benchmark',
            c='k',
            linestyle='--')
        ticks = [int(x) for x in np.linspace(0, len(self.dates) - 1, 11)]
        plt.xticks(ticks, [self.dates[i] for i in ticks])
        # 设置图例样式
        ax.legend(loc=2, fontsize=10)
        # 设置y标签样式
        ax.set_ylabel('excess returns', fontsize=20)
        # 设置x标签样式
        ax.set_yticklabels([str(x * 100) + '% ' for x in ax.get_yticks()])
        # 设置图片标题样式
        ax.set_title(
            "Strategy's performances with different parameters", fontsize=21)
        plt.xlim(0, len(self.excess_returns[0]))

    # 回测的4个主要指标，包括总回报率、最大回撤夏普率和波动
    def get_eval4_bar(self, sort_by=[]):

        sorted_params = self.params_df
        for by in sort_by:
            sorted_params = sorted_params.sort(by)
        indices = sorted_params.index
        indices = set(sorted_params.index) - set(self.failed_list)
        fig = plt.figure(figsize=(20, 7))

        # 定义位置
        ax1 = fig.add_subplot(221)
        # 设定横轴为对应分位，纵轴为对应指标
        ax1.bar(
            list(range(len(indices))),
            [self.evaluations[x]['algorithm_return'] for x in indices],
            0.6,
            label='Algorithm_return')
        plt.xticks([x + 0.3 for x in range(len(indices))], indices)
        # 设置图例样式
        ax1.legend(loc='best', fontsize=15)
        # 设置y标签样式
        ax1.set_ylabel('Algorithm_return', fontsize=15)
        # 设置y标签样式
        ax1.set_yticklabels([str(x * 100) + '% ' for x in ax1.get_yticks()])
        # 设置图片标题样式
        ax1.set_title(
            "Strategy's of Algorithm_return performances of different quantile",
            fontsize=15)
        # x轴范围
        plt.xlim(0, len(indices))

        # 定义位置
        ax2 = fig.add_subplot(224)
        # 设定横轴为对应分位，纵轴为对应指标
        ax2.bar(
            list(range(len(indices))),
            [self.evaluations[x]['max_drawdown'] for x in indices],
            0.6,
            label='Max_drawdown')
        plt.xticks([x + 0.3 for x in range(len(indices))], indices)
        # 设置图例样式
        ax2.legend(loc='best', fontsize=15)
        # 设置y标签样式
        ax2.set_ylabel('Max_drawdown', fontsize=15)
        # 设置x标签样式
        ax2.set_yticklabels([str(x * 100) + '% ' for x in ax2.get_yticks()])
        # 设置图片标题样式
        ax2.set_title(
            "Strategy's of Max_drawdown performances of different quantile",
            fontsize=15)
        # x轴范围
        plt.xlim(0, len(indices))

        # 定义位置
        ax3 = fig.add_subplot(223)
        # 设定横轴为对应分位，纵轴为对应指标
        ax3.bar(
            list(range(len(indices))),
            [self.evaluations[x]['sharpe'] for x in indices],
            0.6,
            label='Sharpe')
        plt.xticks([x + 0.3 for x in range(len(indices))], indices)
        # 设置图例样式
        ax3.legend(loc='best', fontsize=15)
        # 设置y标签样式
        ax3.set_ylabel('Sharpe', fontsize=15)
        # 设置x标签样式
        ax3.set_yticklabels([str(x * 100) + '% ' for x in ax3.get_yticks()])
        # 设置图片标题样式
        ax3.set_title(
            "Strategy's of Sharpe performances of different quantile",
            fontsize=15)
        # x轴范围
        plt.xlim(0, len(indices))

        # 定义位置
        ax4 = fig.add_subplot(222)
        # 设定横轴为对应分位，纵轴为对应指标
        ax4.bar(
            list(range(len(indices))),
            [self.evaluations[x]['algorithm_volatility'] for x in indices],
            0.6,
            label='Algorithm_volatility')
        plt.xticks([x + 0.3 for x in range(len(indices))], indices)
        # 设置图例样式
        ax4.legend(loc='best', fontsize=15)
        # 设置y标签样式
        ax4.set_ylabel('Algorithm_volatility', fontsize=15)
        # 设置x标签样式
        ax4.set_yticklabels([str(x * 100) + '% ' for x in ax4.get_yticks()])
        # 设置图片标题样式
        ax4.set_title(
            "Strategy's of Algorithm_volatility performances of different quantile",
            fontsize=15)
        # x轴范围
        plt.xlim(0, len(indices))

    # 14 年化回报和最大回撤，正负双色表示
    def get_eval(self, sort_by=[]):

        sorted_params = self.params_df
        for by in sort_by:
            sorted_params = sorted_params.sort(by)
        indices = sorted_params.index
        indices = set(sorted_params.index) - set(self.failed_list)
        # 大小
        fig = plt.figure(figsize=(20, 8))
        # 图1位置
        ax = fig.add_subplot(111)
        # 生成图超额收益率的最大回撤
        ax.bar([x + 0.3 for x in range(len(indices))],
               [-self.evaluations[x]['max_drawdown'] for x in indices],
               color='#32CD32',
               width=0.6,
               label='Max_drawdown',
               zorder=10)
        # 图年化超额收益
        ax.bar([x for x in range(len(indices))],
               [self.evaluations[x]['annual_algo_return'] for x in indices],
               color='r',
               width=0.6,
               label='Annual_return')
        plt.xticks([x + 0.3 for x in range(len(indices))], indices)
        # 设置图例样式
        ax.legend(loc='best', fontsize=15)
        # 基准线
        plt.plot([0, len(indices)], [0, 0], c='k', linestyle='--', label='zero')
        # 设置图例样式
        ax.legend(loc='best', fontsize=15)
        # 设置y标签样式
        ax.set_ylabel('Max_drawdown', fontsize=15)
        # 设置x标签样式
        ax.set_yticklabels([str(x * 100) + '% ' for x in ax.get_yticks()])
        # 设置图片标题样式
        ax.set_title(
            "Strategy's performances of different quantile", fontsize=15)
        #   设定x轴长度
        plt.xlim(0, len(indices))

    # 14 超额收益的年化回报和最大回撤
    # 加入新的benchmark后超额收益和
    def get_excess_eval(self, sort_by=[]):

        sorted_params = self.params_df
        for by in sort_by:
            sorted_params = sorted_params.sort(by)
        indices = sorted_params.index
        indices = set(sorted_params.index) - set(self.failed_list)
        # 大小
        fig = plt.figure(figsize=(20, 8))
        # 图1位置
        ax = fig.add_subplot(111)
        # 生成图超额收益率的最大回撤
        ax.bar([x + 0.3 for x in range(len(indices))],
               [-self.excess_max_drawdown[x] for x in indices],
               color='#32CD32',
               width=0.6,
               label='Excess_max_drawdown')
        # 图年化超额收益
        ax.bar([x for x in range(len(indices))],
               [self.excess_annual_return[x] for x in indices],
               color='r',
               width=0.6,
               label='Excess_annual_return')
        plt.xticks([x + 0.3 for x in range(len(indices))], indices)
        # 设置图例样式
        ax.legend(loc='best', fontsize=15)
        # 基准线
        plt.plot([0, len(indices)], [0, 0], c='k', linestyle='--', label='zero')
        # 设置图例样式
        ax.legend(loc='best', fontsize=15)
        # 设置y标签样式
        ax.set_ylabel('Max_drawdown', fontsize=15)
        # 设置x标签样式
        ax.set_yticklabels([str(x * 100) + '% ' for x in ax.get_yticks()])
        # 设置图片标题样式
        ax.set_title(
            "Strategy's performances of different quantile", fontsize=15)
        #   设定x轴长度
        plt.xlim(0, len(indices))

    def GetNavDf(self):

        df = 1 + pd.DataFrame(self.returns, index=self.dates)
        df.columns = ['SVM', 'Logistic', 'RandomForest', 'not_timing']
        df['benchmark'] = self.benchmark_returns
        df.index = pd.to_datetime(df.index)
        return df

    # 计算组合收益率分析:年化收益率、收益波动率、夏普比率、最大回撤
    def strategy_performance(self, nav_df=None):

        if isinstance(nav_df, pd.DataFrame):

            nav_df = nav_df
        else:
            nav_df = self.nav_df
        ##part1:根据回测净值计算相关指标的数据准备（日度数据）
        nav_next = nav_df.shift(1)
        return_df = (nav_df - nav_next) / nav_next  # 计算净值变化率，即为日收益率,包含组合与基准
        return_df = return_df.dropna()  # 在计算净值变化率时，首日得到的是缺失值，需将其删除

        analyze = pd.DataFrame()  # 用于存储计算的指标

        ##part2:计算年化收益率
        cum_return = np.exp(np.log1p(return_df).cumsum()) - 1  # 计算整个回测期内的复利收益率
        annual_return_df = (1 + cum_return) ** (252 / len(return_df)) - 1  # 计算年化收益率
        analyze['annual_return'] = annual_return_df.iloc[-1]  # 将年化收益率的Series赋值给数据框

        # part3:计算收益波动率（以年为基准）
        analyze['return_volatility'] = return_df.std() * np.sqrt(
            252)  # return中的收益率为日收益率，所以计算波动率转化为年时，需要乘上np.sqrt(252)

        # part4:计算夏普比率
        risk_free = 0
        return_risk_adj = return_df - risk_free
        analyze['sharpe_ratio'] = return_risk_adj.mean() / np.std(
            return_risk_adj, ddof=1)

        # prat5:计算最大回撤
        cumulative = np.exp(np.log1p(return_df).cumsum()) * 100  # 计算累计收益率
        max_return = cumulative.cummax()  # 计算累计收益率的在各个时间段的最大值
        analyze['max_drawdown'] = cumulative.sub(max_return).div(
            max_return).min()  # 最大回撤一般小于0，越小，说明离1越远，各时间点与最大收益的差距越大

        # part6:计算相对指标
        analyze['relative_return'] = analyze['annual_return'] - analyze.loc[
            'benchmark', 'annual_return']  # 计算相对年化波动率
        analyze['relative_volatility'] = analyze['return_volatility'] - analyze.loc[
            'benchmark', 'return_volatility']  # 计算相对波动
        analyze['relative_drawdown'] = analyze['max_drawdown'] - analyze.loc[
            'benchmark', 'max_drawdown']  # 计算相对最大回撤

        # part6:计算信息比率
        return_diff = return_df.sub(
            return_df['benchmark'], axis=0).std() * np.sqrt(
            252)  # 计算策略与基准日收益差值的年化标准差
        analyze['info_ratio'] = analyze['relative_return'].div(return_diff)

        return analyze.T

    # 构建每年的收益表现函数
    def get_return_year(self, method):

        nav = self.nav_df[['benchmark', method]]
        result_dic = {}  # 用于存储每年计算的各项指标
        for y, nav_df in nav.groupby(pd.Grouper(level=0, freq='Y')):
            result = self.strategy_performance(nav_df)
            result_dic[str(y)[:4]] = result.iloc[:, -1]

        result_df = pd.DataFrame(result_dic)

        return result_df.T

# 2 设定回测的 策略id
pa = parameter_analysis('a46bf5ddd29f2ce407d284e4bb01a6ee')

#3 运行回测
pa.get_backtest_data(file_name = 'results.pkl',  # 保存回测结果的Pickle文件名
                          running_max = 2,      # 同时回测的最大个数,可以通过积分商城兑换
                          benchmark_id = None,   # 基准的回测ID,注意是回测ID而不是策略ID,为None时为策略中使用的基准
                          start_date = '2014-01-01', #回测开始时间
                          end_date = '2019-12-31',   #回测结束时间
                          frequency = 'day',         #回测频率,支持 day, minute, tick
                          initial_cash = '5000000',  #初始资金
                          param_names = ['method'],  #变量名称
                          param_values = [['SVM','Logistic','RandomForest','not_timing']],  #变量对应的参数
                          python_version = 3   # 回测python版本
                          )

#4 数据读取 已经运行过直接读取就可以
pa.read_backtest_data('results.pkl')

#6 查看回测结果指标
print_table(pa.strategy_performance())
print_table(pa.get_return_year('SVM'))
print_table(pa.evaluations_df.T)

 #7 回报率折线图
pa.plot_returns()







