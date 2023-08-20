import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config
# 实现一个自定义的特征集，MACD、RSI

from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDatasetH
from qlib.contrib.model.pytorch_alstm_ts import ALSTM
from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

import seaborn as sns
import matplotlib.pyplot as plt

import json
import tushare as ts
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api(timeout=5)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.style.use('seaborn')

# 配置数据
train_period = ("2019-01-01", "2021-12-31")
valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2023-01-01", "2023-08-04")

# 导入QlibDataLoader
from qlib.data.dataset.loader import QlibDataLoader

# 这次我们加载沪深300成分股的10日和30日收盘价指数加权均价
market = 'all_fund' # 沪深300股票池代码，在instruments文件夹下有对应的sh000300.txt
close_ma = ['$change'] # EMA($close, 10)表示计算close的10日指数加权均线
ma_names = ['change']
qdl_ma = QlibDataLoader(config=(close_ma, ma_names))
total_fund = qdl_ma.load(instruments=market, start_time='20190101', end_time='20211231')

from qlib.data.dataset.processor import Fillna

# 数据处理
fillna_processor = Fillna()
total_fund = fillna_processor(total_fund)

df = total_fund.reset_index()
df = df.set_index('datetime')
df = df.pivot(columns='instrument', values='change')
df = fillna_processor(df)

# 绘制因子正交前的相关性的热力图
fig = plt.figure(figsize=(26, 18))
# 计算对称正交之前的相关系数矩阵
relations = df.corr()
sns.heatmap(relations, annot=True, linewidths=0.05,
            linecolor='white', annot_kws={'size': 8, 'weight': 'bold'})

corr_df = df.corr()
# 求每一列的和
df_sum = corr_df.sum()

# 对求和后的Series排序
df_sorted = df_sum.sort_values(ascending=False)

df_sorted_new = df_sorted.to_frame()
df_sorted_new = df_sorted_new.reset_index()
df_sorted_new = df_sorted_new.rename(columns={'instrument': 'ts_code',0: 'score'})


fund_basic_e=pd.read_csv("d:\\data\\temp\\fund_basic_e.csv")
fund_basic_e.drop(columns=['Unnamed: 0'], inplace=True)
fund_basic_e['ts_code'] = fund_basic_e['ts_code'].apply(lambda x: x.split('.')[1] + x.split('.')[0])
df_merge = pd.merge(df_sorted_new, fund_basic_e, on='ts_code')



#将相关系数矩阵换成名字显示
# 构建一个映射词典
mapping = dict(zip(fund_basic_e.ts_code, fund_basic_e.name))

# 使用map方法应用映射替换index
corr_df.index = corr_df.index.map(mapping)

# 使用map方法应用映射替换columns
corr_df.columns = corr_df.columns.map(mapping)
