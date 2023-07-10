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

qlib.init()

MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "tutorial_exp"

hd=pd.read_csv('c://temp/Alpha158_test.csv')
hd.head()
hd.set_index(['datetime', 'instrument'], inplace=True)
#修改索引名字
new_names = ['date', 'code']
hd.index.set_names(new_names, inplace=True)


hd.loc[:, ['KMID','KLEN']].head()

# 获取申万二级行业列表
SW2021 = pro.index_classify(level='L2', src='SW2021')

# 初始化一个字典来存储每个行业的股票代码
industry_stocks = {}

# 遍历二级行业列表
for _, row in SW2021.iterrows():
    industry_code = row['index_code']
    industry_name = row['industry_name']

    # 获取指定行业的股票列表
    stocks = pro.index_member(index_code=industry_code)

    # 提取股票代码，并添加到字典中
    stock_codes = list(stocks['con_code'])
    industry_stocks[industry_name] = stock_codes

# 打印各行业的股票代码
# 将字典转换为 DataFrame，并添加行业名称列
df = pd.DataFrame([(k, v) for k, val in industry_stocks.items() for v in val], columns=['industry_name', 'stock_code'])

df = df.rename(columns={'stock_code': 'code', 'industry_name': 'industry_code'})

df = df.dropna(axis=0, how='any')
df = df.drop_duplicates(subset=['code'])
df.describe()
# 将 A 列设置为索引
industry_code=df
industry_code['code'] = industry_code['code'].str[7:9]+industry_code['code'].str[0:6]
industry_code = industry_code.set_index('code')


# 合并两个 DataFrame，并根据 date 和 code 进行排序
merged_df = pd.merge(hd, industry_code, left_index=True, right_index=True).sort_index()


merged_df=pd.read_csv('c://temp/merged_df.csv')
merged_df.set_index(['date', 'code'], inplace=True)



merged_df_new = pd.merge(merged_df, SW2021_temp, on='key', how='left')


merged_df_new.to_csv('c://temp//merged_df_new.csv')
factors=merged_df_new.astype(float)
factors['industry_code'] = factors['industry_code'].astype(int)

##########总股本处理########
df = df.rename(columns={'industry_code':'industry_name' })
SW2021_temp = pd.merge(df, SW2021, on='industry_name', how='left')
test123 = SW2021_temp.loc[:, ['code','industry_code']]
test123 = test123.rename(columns={'code':'ts_code' })


ts_daily_basic = pro.daily_basic(ts_code='', trade_date='20230707')
ts_daily_basic_temp=pd.merge(ts_daily_basic, test123, on='ts_code', how='left')
ts_daily_basic_temp_1 = ts_daily_basic_temp
ts_daily_basic_temp = ts_daily_basic_temp.rename(columns={'total_mv': 'market_cap'})
ts_daily_basic_temp = ts_daily_basic_temp.rename(columns={'ts_code': 'code'})
ts_daily_basic_temp = ts_daily_basic_temp.rename(columns={'trade_date': 'date'})
ts_daily_basic_temp = ts_daily_basic_temp.rename(columns={'industry_code': 'INDUSTRY_CODE'})
ts_daily_basic_temp.set_index(['date', 'code'], inplace=True)
#某一列为 NaN 的行
ts_daily_basic_temp[ts_daily_basic_temp['industry_code'].isna()]


merged_df_new=pd.read_csv('c://temp/merged_df_new.csv')
ts_daily_basic = pro.daily_basic(ts_code='', trade_date='20120707')
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

ts_daily_basic_temp=pd.read_csv('c://temp/ts_daily_basic_temp.csv',index_col=[0,1],parse_dates=[0],dtype={'industry_code':str})
#ts_daily_basic_temp.set_index(['date', 'code'], inplace=True)
ts_daily_basic_temp = ts_daily_basic_temp.rename(columns={'industry_code': 'INDUSTRY_CODE'})
factors=ts_daily_basic_temp
#
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
        'INDUSTRY_CODE', 'market_cap']]

    # 回归取残差
    def _calc_resid(x: pd.DataFrame, y: pd.Series) -> float:
        result = sm.OLS(y, x).fit()

        return result.resid

    X = pd.get_dummies(data['INDUSTRY_CODE'])
    # 总市值单位为亿元
    X['market_cap'] = np.log(data['market_cap'] * 100000000)

    df = pd.concat([_calc_resid(X.fillna(0), data[i])
                    for i in factor_name], axis=1)

    df.columns = factor_name

    df['INDUSTRY_CODE'] = data['INDUSTRY_CODE']
    df['market_cap'] = data['market_cap']

    return df

#####################################
###########################################
## 其实可以直接pipe处理但是这里为了后续灵活性没有选择pipe化

# 去极值
factors1 = factors.groupby(level='date').apply(extreme_process_MAD)
factors1 = factors1.reset_index(level=0, drop=True)

# 缺失值处理
factors2 = factors1.groupby(level='date').apply(factors_null_process)
# 中性化
factors3 = factors2.groupby(level='date').apply(neutralization)
# 标准化
factors4 = factors3.groupby(level='date').apply(data_scale_Z_Score)

