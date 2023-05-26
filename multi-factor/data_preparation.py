import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
from send_mail_tool import *
from technical_analysis_patterns import (rolling_patterns2pool,plot_patterns_chart,rolling_patterns)
from typing import (List, Tuple, Dict, Callable, Union)
from tqdm.notebook import tqdm
import json
import pandas as pd
import numpy as np
import empyrical as ep
import seaborn as sns
import matplotlib as mpl
import mplfinance as mpf
import matplotlib.pyplot as plt
import datetime
import os
import statsmodels.api as sm
test_circ_mv = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//circ_mv.pkl')
test_close = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//close.pkl')
test_dv_ratio = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//dv_ratio.pkl')
test_dv_ttm = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//dv_ttm.pkl')
test_float_share = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//float_share.pkl')
test_free_share = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//free_share.pkl')

test_pb= pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//pb.pkl')
test_pe = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//pe.pkl')
test_pe_ttm = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//pe_ttm.pkl')
test_ps = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//ps.pkl')
test_total_mv = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//total_mv.pkl')
test_total_share = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//total_share.pkl')

test_turnover_rate= pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//turnover_rate.pkl')
test_turnover_rate_f = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//turnover_rate_f.pkl')
test_volume_ratio = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//volume_ratio.pkl')

#['close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio','pe', 'pe_ttm',
#                          'pb','ps','ps_ttm','dv_ratio','dv_ttm','total_share','float_share','free_share',
#                          'total_mv','circ_mv']

global dataBase
curPath = os.path.abspath(os.path.dirname('c:\\temp\\multi_factor_data\\'))
# rootPath = curPath[:curPath.find("多因子框架\\")+len("多因子框架\\")]
rootPath = curPath
dataBase = rootPath + '\\base_data\\'

#转换城alphalens-example方式
class transformer_data:
    def __init__(self, start_date='20230320', end_date='20230320'):
        self.start_date = start_date
        self.end_date = end_date
        self.local_url = dataBase + 'mkt\\'
    @staticmethod
    def data_melt(para_name, *args, **kwds):
        para_name_return = pd.read_pickle(dataBase+'mkt//'+para_name+'.pkl')
        para_name_return=para_name_return.reset_index()
        para_name_return=para_name_return.melt(id_vars=['trade_date'],var_name='ts_code',value_name=para_name)
        return para_name_return

    @staticmethod
    def data_merge(name_list, *args, **kwds):
        count_num = 1
        for data_name in name_list:
            return_data_indermediate=transformer_data.data_melt(data_name)
            if count_num==1:
                return_data=return_data_indermediate
            else:
               return_data=pd.merge(return_data, return_data_indermediate, on=['trade_date','ts_code'])
            count_num=count_num+1
        return return_data


columns = ['close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio','pe', 'pe_ttm',
                   'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share','free_share',
                  'total_mv','circ_mv']
data_final=transformer_data.data_merge(columns)


data_final.set_index(['trade_date', 'ts_code'])


data_path = dataBase + 'all_stock_basic.pkl'
df = pd.read_pickle(data_path)

test123 = pd.merge(data_final, df[['ts_code','industry']], on=['ts_code'])

#test123[(test123["ts_code"] == "000001.SZ")&(test123["trade_date"] == "20230320")]
test123 = test123.set_index(['trade_date', 'ts_code'])
test123.info()


import datacompy

# step1:构建缺失值处理函数
def factors_null_process(data: pd.DataFrame ) -> pd.DataFrame:
    # 删除行业缺失值
    data = data[data['industry'].notnull()]
    # 变化索引，以行业为第一索引，股票代码为第二索引
    data_ = data.reset_index().set_index(
        ['industry', 'ts_code']).sort_index()
    # 用行业中位数填充
    data_ = data_.groupby(level=0).apply(
        lambda factor: factor.fillna(factor.median()))

    # 有些行业可能只有一两个个股却都为nan此时使用0值填充
    data_ = data_.fillna(0)
    # 将索引换回
    data_ = data_.reset_index().set_index('ts_code').sort_index()
    return data_.drop('trade_date', axis=1)

#3.2去极值
# step2:构建绝对中位数处理法函数
def extreme_process_MAD(data: pd.DataFrame, num: int = 3) -> pd.DataFrame:
    ''' data为输入的数据集，如果数值超过num个判断标准则使其等于num个标准'''

    # 为不破坏原始数据，先对其进行拷贝
    data_ = data.copy()

    # 获取数据集中需测试的因子名
    feature_names = [i for i in data_.columns.tolist() if i not in [
        'industry','total_mv']]

    # 获取中位数
    median = data_[feature_names].median(axis=0)
    # 按列索引匹配，并在行中广播
    MAD = abs(data_[feature_names].sub(median, axis=1)
              ).median(axis=0)
    # 利用clip()函数，将因子取值限定在上下限范围内，即用上下限来代替异常值
    data_.loc[:, feature_names] = data_.loc[:, feature_names].clip(
        lower=median-num * 1.4826 * MAD, upper=median + num * 1.4826 * MAD, axis=1)
    return data_

##3.3标准化
##step3:构建标准化处理函数
def data_scale_Z_Score(data: pd.DataFrame) -> pd.DataFrame:

    # 为不破坏原始数据，先对其进行拷贝
    data_ = data.copy()
    # 获取数据集中需测试的因子名
    feature_names = [i for i in data_.columns.tolist() if i not in [
        'industry','total_mv']]
    data_.loc[:, feature_names] = (
        data_.loc[:, feature_names] - data_.loc[:, feature_names].mean()) / data_.loc[:, feature_names].std()
    return data_

##3.4市值和行业中性化
# step4:因子中性化处理函数
def neutralization(data: pd.DataFrame) -> pd.DataFrame:
    '''按市值、行业进行中性化处理 ps:处理后无行业市值信息'''
    factor_name = [i for i in data.columns.tolist() if i not in [
        'industry', 'total_mv']]

    # 回归取残差
    def _calc_resid(x: pd.DataFrame, y: pd.Series) -> float:
        result = sm.OLS(y, x).fit()

        return result.resid

    X = pd.get_dummies(data['industry'])
    # 总市值单位为亿元
    X['total_mv'] = np.log(data['total_mv'] * 100000000)

    df = pd.concat([_calc_resid(X.fillna(0), data[i])
                    for i in factor_name], axis=1)

    df.columns = factor_name

    df['industry'] = data['industry']
    df['total_mv'] = data['total_mv']

    return df

## 其实可以直接pipe处理但是这里为了后续灵活性没有选择pipe化

# 去极值
factors1 = factors.groupby(level='trade_date').apply(extreme_process_MAD)
# 缺失值处理
factors2 = factors1.groupby(level='trade_date').apply(factors_null_process)
# 中性化
factors3 = factors2.groupby(level='trade_date').apply(neutralization)
# 标准化
factors4 = factors3.groupby(level='trade_date').apply(data_scale_Z_Score)


print(factors.info())
print('去缺失值:')
print(factors2.info())

#四、因子多重共线性的处理
## step5:构建对称正交变换函数
def lowdin_orthogonal(data: pd.DataFrame) -> pd.DataFrame:
    data_ = data.copy()  # 创建副本不影响原数据
    col = [col for col in data_.columns if col not in ['industry', 'total_mv']]

    F = np.mat(data_[col])  # 除去行业指标,将数据框转化为矩阵
    M = F.T @ F  # 等价于 (F.shape[0] - 1) * np.cov(F.T)
    a, U = np.linalg.eig(M)  # a为特征值，U为特征向量
    D_inv = np.linalg.inv(np.diag(a))
    S = U @ np.sqrt(D_inv) @ U.T
    data_[col] = data_[col].dot(S)

    return data_

#对称正交化
factors5 = factors4.groupby(level='trade_date').apply(lowdin_orthogonal)
factors5.info()


# 构建计算横截面因子载荷相关系数均值函数
def get_relations(datas: pd.DataFrame) -> pd.DataFrame:
    relations = 0
    for trade, d in datas.groupby(level='trade_date'):
        relations += d.corr()

    relations_mean = relations / len(datas.index.levels[0])

    return relations_mean

#未进行正交处理,各因子的相关性如下
# 绘制因子正交前的相关性的热力图
fig = plt.figure(figsize=(26, 18))
# 计算对称正交之前的相关系数矩阵
relations = get_relations(factors4.iloc[:,:-2])
sns.heatmap(relations, annot=True, linewidths=0.05,
            linecolor='white', annot_kws={'size': 8, 'weight': 'bold'})

#正交处理后各因子相关性明显下降
#绘制因子正交后的相关性热力图
fig=plt.figure(figsize=(26,18))
#计算对称正交之后的相关系数矩阵
relations= get_relations(factors5.iloc[:,:-2])
sns.heatmap(relations,annot=True,linewidths=0.05,
            linecolor='white',annot_kws={'size':8,'weight':'bold'})

#################################################################################


#五、打分法计算预期收益

#5.1计算ICIR权重
def get_next_ret(factor: pd.DataFrame, keep_last_term: bool = False, last_term_next_date: str = None) -> pd.Series:
    '''
    keep_last_term:是否保留最后一期数据
    last_term_next_date:如果keep_last_term=True,则此参数为计算最后一期下期收益时的截止时间,必须时交易日
    '''
    securities = factor.index.levels[1].tolist()  # 股票代码
    periods = [i.strftime('%Y-%m-%d') for i in factor.index.levels[0]]  # 日期

    if keep_last_term:
        end = last_term_next_date
        periods = periods + [end]

        if not end:
            raise ValueError('如果keep_last_term=True,则必须有last_term_next_date参数')

    close = pd.concat([get_price(securities, end_date=i, count=1, fields='close', panel=False)
                       for i in periods])

    close = pd.pivot_table(close, index='time', columns='code', values='close')
    ret = close.pct_change().shift(-1)
    ret = ret.iloc[:-1]
    return ret.stack()

# 获取下期收益率
next_ret = get_next_ret(factors5,True,'2020-10-14')

factors5['NEXT_RET'] = next_ret


# 根据IR计算因子权重

# step1:计算rank_IC

def calc_rank_IC(factor: pd.DataFrame) -> pd.DataFrame:
    factor_col = [x for x in factor.columns if x not in [
        'INDUSTRY_CODE', 'market_cap', 'NEXT_RET']]

    IC = factor.groupby(level='date').apply(lambda x: [st.spearmanr(
        x[factor], x['NEXT_RET'])[0] for factor in factor_col])

    return pd.DataFrame(IC.tolist(), index=IC.index, columns=factor_col)


## step2: 计算IR权重
def IR_weight(factor: pd.DataFrame) -> pd.DataFrame:
    data_ = factor.copy()
    # 计算ic值，得到ic的
    IC = calc_rank_IC(data_)

    # 计算ic的绝对值
    abs_IC = IC.abs()
    # rolling为移动窗口函数,滚动12个月
    rolling_ic = abs_IC.rolling(12, min_periods=1).mean()
    # 当滚动计算标准差时，起始日期得到的是缺失值，所以算完权重后，起始日期的值任用原值IC代替
    rolling_ic_std = abs_IC.rolling(12, min_periods=1).std()
    IR = rolling_ic / rolling_ic_std  # 计算IR值
    IR.iloc[0, :] = rolling_ic.iloc[0, :]
    weight = IR.div(IR.sum(axis=1), axis=0)  # 计算IR权重,按行求和,按列相除

    return weight

# 获取权重
weights = IR_weight(factors5)

# 获取因子名称
factor_names = [name for name in factors5.columns if name not in [
    'INDUSTRY_CODE', 'market_cap', 'NEXT_RET']]

# 计算因子分数
factors5['SCORE'] = (factors5[factor_names].mul(weights)).sum(axis=1)

factors5.head()

#六、分层抽样指数增强策略实现

#6.1 策略概述
#市值和行业是很重要的风险因子，分层抽样策略的核心是使投资组合在这两个风险维度上与基准指数保持一致，
#然后在市值、行业属性比较相似的若干只股票里优选一只预期收益最高的进行投资，以获取超额收益。

#策略中所使用的行业分类如下： 以申万34个一级行业为蓝本
#策略步骤：
#step1:将基准指数成份股按以上行业划分成34个子集，在每个子集中用市值因子将股票划分为数目相等的三组；
#step2:计算每个小组内所有股票在基准指数中的总权重；
#step3:在每个小组中选择预期收益（打分法）最高的一只股票，令它在投资组合中的权重等于它所处小组的权重。这样就能选出包含102只股票的分层抽样组合。

#本策略的基准指数选沪深300指数，且在基准指数成份股内选股。
#参考于华泰证券《指数增强方法汇总及实例——量化多因子指数增强策略实证》中的第一个简单策略。
def get_weighs(symbol: str, start: str, end: str, method: str = 'cons') -> pd.DataFrame:
    '''
    获取月度指数成份权重
    --------
        mehtod:ind 输出 行业权重
               cons 输出 成份股权重
    '''
    periods = GetTradePeriod(start, end, 'ME')

    ser_dic = {}
    if method == 'ind':
        for d in periods:
            # 获取当日成份及权重
            index_w = get_index_weights(symbol, date=d)
            # 获取行业
            index_w['ind'] = get_stock_ind(index_w.index.tolist(), d)
            # 计算行业所占权重
            weight = index_w.groupby('ind')['weight'].sum() / 100

            ser_dic[d] = weight

        ser = pd.concat(ser_dic, names=['date', 'industry']).reset_index()
        ser['date'] = pd.to_datetime(ser['date'])
        return ser.set_index(['date', 'industry'])

    elif method == 'cons':

        df = pd.concat([get_index_weights(symbol, date=d) for d in periods])
        df.drop(columns='display_name', inplace=True)

        df.set_index('date', append=True, inplace=True)
        df = df.swaplevel()
        df['weight'] = df['weight'] / 100
        return df


def get_group(ser: pd.Series, N: int = 3, ascend: bool = True) -> pd.Series:
    '''默认分三组 升序'''
    ranks = ser.rank(ascending=ascend)
    label = ['G' + str(i) for i in range(1, N + 1)]

    return pd.cut(ranks, bins=N, labels=label)


def stratified_sampling(symbol: str, START_DATE: str, END_DATE: str, factors: pd.DataFrame) -> pd.DataFrame:
    factors_ = factors.copy()
    ind_weight = get_weighs(symbol, START_DATE, END_DATE)
    # 市值等量分三组
    k1 = [pd.Grouper(level='date'),
          pd.Grouper(key='INDUSTRY_CODE')]

    factors_['GROUP'] = factors_.groupby(
        k1)['market_cap'].apply(lambda x: get_group(x, 3))

    # 获取每组得分最大的
    k2 = [pd.Grouper(level='date'),
          pd.Grouper(key='INDUSTRY_CODE'),
          pd.Grouper(key='GROUP')]

    industry_kfold_stock = factors_.groupby(
        k2)['SCORE'].apply(lambda x: x.idxmax()[1])

    # 格式调整
    industry_kfold_stock = industry_kfold_stock.reset_index()
    industry_kfold_stock = industry_kfold_stock.set_index(['date', 'SCORE'])
    industry_kfold_stock.index.names = ['date', 'code']

    # 加入权重
    industry_kfold_stock['weight'] = ind_weight['weight']

    # 令权重加总为1
    industry_kfold_stock['w'] = industry_kfold_stock.groupby(
        level='date')['weight'].transform(lambda x: x / x.sum())

    industry_kfold_stock['NEXT_RET'] = factors['NEXT_RET']

    return industry_kfold_stock

# 获取分层数据
result_df = stratified_sampling('000300.XSHG', START_DATE, END_DATE, factors5[[
                                'INDUSTRY_CODE', 'market_cap', 'SCORE', 'NEXT_RET']])

# 储存,用于回测
result_df.to_csv('../../result_df.csv')
result_df.head()







