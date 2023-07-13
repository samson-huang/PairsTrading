############################
#20230710 重载多因子指数增强
#######################
import talib
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.optimize import minimize

from tqdm import tqdm_notebook
from dateutil.parser import parse

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.style.use('seaborn')

from pprint import pprint
from pathlib import Path
import pandas as pd
import qlib


import tushare as ts
import warnings
import json
warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api()

#暂时不从qlib获取数据
#qlib.init()

MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "tutorial_exp"


#step1: 构建缺失值处理函数
def factors_null_process(data: pd.DataFrame) -> pd.DataFrame:
    # 删除行业缺失值
    data = data[data['INDUSTRY_CODE'].notnull()]

    # 变化索引，以行业为第一索引，股票代码为第二索引
    data_ = data.reset_index().set_index(
        ['INDUSTRY_CODE', 'code']).sort_index()
    # 用行业中位数填充
    data_ = data_.groupby(level=0).apply(
        lambda factor: factor.fillna(factor.median()))

    # 有些行业可能只有一两个个股却都为nan此时使用0值填充
    data_ = data_.fillna(0)
    #删除一列index20230710
    data_ = data_.reset_index(level=0, drop=True)
    # 将索引换回
    data_ = data_.reset_index().set_index('code').sort_index()
    return data_.drop('date', axis=1)

# step2:构建绝对中位数处理法函数
def extreme_process_MAD(data: pd.DataFrame, num: int = 3) -> pd.DataFrame:
    ''' data为输入的数据集，如果数值超过num个判断标准则使其等于num个标准'''

    # 为不破坏原始数据，先对其进行拷贝
    data_ = data.copy()

    # 获取数据集中需测试的因子名
    feature_names = [i for i in data_.columns.tolist() if i not in [
        'INDUSTRY_CODE','market_cap']]

    # 获取中位数
    median = data_[feature_names].median(axis=0)
    # 按列索引匹配，并在行中广播
    MAD = abs(data_[feature_names].sub(median, axis=1)
              ).median(axis=0)
    # 利用clip()函数，将因子取值限定在上下限范围内，即用上下限来代替异常值
    data_.loc[:, feature_names] = data_.loc[:, feature_names].clip(
        lower=median-num * 1.4826 * MAD, upper=median + num * 1.4826 * MAD, axis=1)
    return data_

##step3:构建标准化处理函数
def data_scale_Z_Score(data: pd.DataFrame) -> pd.DataFrame:

    # 为不破坏原始数据，先对其进行拷贝
    data_ = data.copy()
    # 获取数据集中需测试的因子名
    feature_names = [i for i in data_.columns.tolist() if i not in [
        'INDUSTRY_CODE','market_cap']]
    data_.loc[:, feature_names] = (
        data_.loc[:, feature_names] - data_.loc[:, feature_names].mean()) / data_.loc[:, feature_names].std()
    return data_


# step4:因子中性化处理函数
def neutralization(data: pd.DataFrame) -> pd.DataFrame:
    '''按市值、行业进行中性化处理 ps:处理后无行业市值信息'''

    factor_name = [i for i in data.columns.tolist() if i not in [
        'INDUSTRY_CODE', 'market_cap','close']]

    # 回归取残差
    def _calc_resid(x: pd.DataFrame, y: pd.Series,i) -> float:
        # 将输入数据转换为NumPy数组
        print(i+'/n')
        x = np.asarray(x , dtype=np.float64)
        y = np.asarray(y , dtype=np.float64)
        result = sm.OLS(y, x).fit()

        #return pd.Series(result.resid)
        return pd.DataFrame(result.resid, columns=[i])

    X = pd.get_dummies(data['INDUSTRY_CODE'])
    # 总市值单位为亿元
    X['market_cap'] = np.log(data['market_cap'] * 100000000)

    df = pd.concat([_calc_resid(X.fillna(0), data[i],i)
                    for i in factor_name], axis=1)

    df.columns = factor_name
    df=df.set_index(data.index)
    df['INDUSTRY_CODE'] = data['INDUSTRY_CODE']
    df['market_cap'] = data['market_cap']

    return df

#####################################


ts_daily_basic_temp=pd.read_csv('c://temp/ts_daily_basic_temp.csv',index_col=[0,1],parse_dates=[0],dtype={'industry_code':str})
#ts_daily_basic_temp.set_index(['date', 'code'], inplace=True)
ts_daily_basic_temp = ts_daily_basic_temp.rename(columns={'industry_code': 'INDUSTRY_CODE'})
factors = ts_daily_basic_temp

# 去极值
factors1 = factors.groupby(level='date').apply(extreme_process_MAD)
factors1 = factors1.reset_index(level=0, drop=True)
# 缺失值处理
factors2 = factors1.groupby(level='date').apply(factors_null_process)
# 中性化
factors3 = factors2.groupby(level='date').apply(neutralization)
factors3 = factors3.reset_index(level=0, drop=True)
# 标准化
factors4 = factors3.groupby(level='date').apply(data_scale_Z_Score)
factors4 = factors4.reset_index(level=0, drop=True)


print(factors.info())
print('去缺失值:')
print(factors2.info())


## step5:构建对称正交变换函数
def lowdin_orthogonal(data: pd.DataFrame) -> pd.DataFrame:
    data_ = data.copy()  # 创建副本不影响原数据
    col = [col for col in data_.columns if col not in ['INDUSTRY_CODE', 'market_cap']]

    F = np.mat(data_[col])  # 除去行业指标,将数据框转化为矩阵
    M = F.T @ F  # 等价于 (F.shape[0] - 1) * np.cov(F.T)
    a, U = np.linalg.eig(M)  # a为特征值，U为特征向量
    D_inv = np.linalg.inv(np.diag(a))
    S = U @ np.sqrt(D_inv) @ U.T
    data_[col] = data_[col].dot(S)

    return data_

# 数据读取
factors4 = pd.read_csv('factors4.csv',index_col=[0,1],parse_dates=[0],dtype={'INDUSTRY_CODE':str})

#对称正交化
factors5 = factors4.groupby(level='date').apply(lowdin_orthogonal)
factors5 = factors5.reset_index(level=0, drop=True)
factors5.info()


# 构建计算横截面因子载荷相关系数均值函数
def get_relations(datas: pd.DataFrame) -> pd.DataFrame:
    relations = 0
    for trade, d in datas.groupby(level='date'):
        relations += d.corr()

    relations_mean = relations / len(datas.index.levels[0])

    return relations_mean

# 绘制因子正交前的相关性的热力图
fig = plt.figure(figsize=(26, 18))
# 计算对称正交之前的相关系数矩阵
relations = get_relations(factors4.iloc[:,:-2])
sns.heatmap(relations, annot=True, linewidths=0.05,
            linecolor='white', annot_kws={'size': 8, 'weight': 'bold'})


#绘制因子正交后的相关性热力图
fig=plt.figure(figsize=(26,18))
#计算对称正交之后的相关系数矩阵
relations= get_relations(factors5.iloc[:,:-2])
sns.heatmap(relations,annot=True,linewidths=0.05,
            linecolor='white',annot_kws={'size':8,'weight':'bold'})

#五、打分法计算预期收益
#5.1计算ICIR权重
# 获取下期收益率
#daily_NEXT_RET = pro.daily(trade_date='20230710')
daily_NEXT_RET =ts_daily_basic_temp[['close']]
daily_NEXT_RET['pct_change'] = daily_NEXT_RET.groupby(level='code').close.pct_change()

daily_NEXT_RET.head()

merged_df = pd.concat([daily_NEXT_RET['pct_change'], factors5], axis=1)

merged_df = merged_df.rename(columns={'pct_change': 'NEXT_RET'})
factors5 = merged_df

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

#六、分层抽样指数增强策略实现
#6.1 策略概述

#step1:将基准指数成份股按以上行业划分成34个子集，在每个子集中用市值因子将股票划分为数目相等的三组；
#step2:计算每个小组内所有股票在基准指数中的总权重；
#step3:在每个小组中选择预期收益（打分法）最高的一只股票，令它在投资组合中的权重等于它所处小组的权重。这样就能选出包含102只股票的分层抽样组合。
